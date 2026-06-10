"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import abc
import datetime
import os
import pprint
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
from jax.experimental import multihost_utils
from skimage.metrics import structural_similarity as ssim

from maxdiffusion import max_logging, max_utils, train_utils
from maxdiffusion.generate_wan import inference_generate_video
from maxdiffusion.generate_wan import run as generate_wan
from maxdiffusion.pipelines.wan.wan_pipeline import WanPipeline
from maxdiffusion.schedulers import FlaxFlowMatchScheduler
from maxdiffusion.train_utils import _metrics_queue, _tensorboard_writer_worker, load_next_batch
from maxdiffusion.utils import load_video
from maxdiffusion.video_processor import VideoProcessor


class TrainState(train_state.TrainState):
    graphdef: nnx.GraphDef
    rest_of_state: nnx.State
    ema_params: Any = None  # EMA teacher params; None when self-distillation is disabled


def _to_array(x):
    if not isinstance(x, jax.Array):
        x = jnp.asarray(x)
    return x



def _distill_swap(state: TrainState, is_distill: bool) -> TrainState:
    """Return state with params=teacher and ema_params=student for distillation checkpoints.

    Swapping ensures the standard load path (which reads wan_state["params"]) loads
    the teacher (EMA) for inference, while the student is preserved under ema_params
    so training can be resumed correctly.
    """
    if not is_distill or state.ema_params is None:
        return state
    return state.replace(params=state.ema_params, ema_params=state.params)


def _log_param_shapes(params, tag: str = "SHAPE"):
    """Log shapes of params leaves on process 0 to diagnose multi-host shape corruption."""
    if jax.process_index() != 0:
        return
    for path, leaf in jax.tree_util.tree_leaves_with_path(params):
        if not isinstance(leaf, jax.Array):
            continue
        path_str = "/".join(str(k) for k in path)
        if "condition_embedder" in path_str and "kernel" in path_str:
            max_logging.log(f"[{tag}] {path_str}: global_shape={leaf.shape} sharding={getattr(leaf, 'sharding', None)}")


def _state_to_save_dict(state: TrainState, is_distill: bool) -> dict:
    """Build a plain dict of serializable fields for checkpointing.

    Only array-bearing fields are included; non-serializable TrainState fields
    (apply_fn, tx, graphdef) are excluded so PyTreeSave doesn't choke on them.
    The params/ema_params swap for distillation is applied here.
    """
    s = _distill_swap(state, is_distill)
    _log_param_shapes(s.params, tag="PRE_SAVE")
    d = {"params": s.params, "step": s.step}
    if s.opt_state is not None:
        d["opt_state"] = s.opt_state
    if s.ema_params is not None:
        d["ema_params"] = s.ema_params
    return d


def generate_sample(config, pipeline, filename_prefix):
    """
    Generates a video to validate training did not corrupt the model
    """
    if not hasattr(pipeline, "vae"):
        wan_vae, vae_cache = WanPipeline.load_vae(
            pipeline.mesh.devices, pipeline.mesh, nnx.Rngs(jax.random.key(config.seed)), config
        )
        pipeline.vae = wan_vae
        pipeline.vae_cache = vae_cache
    return generate_wan(config, pipeline, filename_prefix)


def print_ssim(pretrained_video_path, posttrained_video_path):
    video_processor = VideoProcessor()
    pretrained_video = load_video(pretrained_video_path[0])
    pretrained_video = video_processor.preprocess_video(pretrained_video)
    pretrained_video = np.array(pretrained_video)
    pretrained_video = np.transpose(pretrained_video, (0, 2, 3, 4, 1))
    pretrained_video = np.uint8((pretrained_video + 1) * 255 / 2)

    posttrained_video = load_video(posttrained_video_path[0])
    posttrained_video = video_processor.preprocess_video(posttrained_video)
    posttrained_video = np.array(posttrained_video)
    posttrained_video = np.transpose(posttrained_video, (0, 2, 3, 4, 1))
    posttrained_video = np.uint8((posttrained_video + 1) * 255 / 2)

    ssim_compare = ssim(pretrained_video[0], posttrained_video[0], multichannel=True, channel_axis=-1, data_range=255)

    max_logging.log(f"SSIM score after training is {ssim_compare}")


class BaseWanTrainer(abc.ABC):
    _profiler: max_utils.Profiler | None = None

    def __init__(self, config):
        if config.train_text_encoder:
            raise ValueError("this script currently doesn't support training text_encoders")
        self.config = config
        self.checkpointer = self._get_checkpointer()

    @abc.abstractmethod
    def _get_checkpointer(self):
        """Returns the checkpointer for the trainer."""

    def post_training_steps(self, pipeline, params, train_states, msg=""):
        pass

    def post_train_step(self, state, step: int):
        """Hook called after every JIT train step. Override to inject per-step logic
        (e.g. EMA updates) that should run on the host outside the JIT boundary.
        Must return the (possibly updated) state."""
        return state

    def create_scheduler(self):
        """Creates and initializes the Flow Match scheduler for training."""
        noise_scheduler = FlaxFlowMatchScheduler(dtype=jnp.float32)
        noise_scheduler_state = noise_scheduler.create_state()
        noise_scheduler_state = noise_scheduler.set_timesteps(
            noise_scheduler_state, num_inference_steps=1000, training=True
        )
        return noise_scheduler, noise_scheduler_state

    @staticmethod
    def calculate_tflops(pipeline):
        maxdiffusion_config = pipeline.config
        # Model configuration
        height = pipeline.config.height
        width = pipeline.config.width
        num_frames = pipeline.config.num_frames

        # Transformer dimensions
        transformer_config = pipeline.transformer.config
        num_layers = transformer_config.num_layers
        heads = pipeline.transformer.config.num_attention_heads
        head_dim = pipeline.transformer.config.attention_head_dim
        ffn_dim = transformer_config.ffn_dim
        seq_len = int(((height / 8) * (width / 8) * ((num_frames - 1) // pipeline.vae_scale_factor_temporal + 1)) / 4)
        text_encoder_dim = 512
        # Attention FLOPS
        # Self
        self_attn_qkv_proj_flops = 3 * (2 * seq_len * (heads * head_dim) ** 2)
        self_attn_qk_v_flops = 2 * (2 * seq_len**2 * (heads * head_dim))
        # Cross
        cross_attn_kv_proj_flops = 3 * (2 * text_encoder_dim * (heads * head_dim) ** 2)
        cross_attn_q_proj_flops = 1 * (2 * seq_len * (heads * head_dim) ** 2)
        cross_attention_qk_v_flops = 2 * (2 * seq_len * text_encoder_dim * (heads * head_dim))

        # Output_projection from attention
        attn_output_proj_flops = 2 * (2 * seq_len * (heads * head_dim) ** 2)

        total_attn_flops = (
            self_attn_qkv_proj_flops
            + self_attn_qk_v_flops
            + cross_attn_kv_proj_flops
            + cross_attn_q_proj_flops
            + cross_attention_qk_v_flops
            + attn_output_proj_flops
        )

        # FFN
        ffn_flops = 2 * (2 * seq_len * (heads * head_dim) * ffn_dim)

        flops_per_block = total_attn_flops + ffn_flops

        total_transformer_flops = flops_per_block * num_layers

        tflops = maxdiffusion_config.per_device_batch_size * total_transformer_flops / 1e12
        train_tflops = 3 * tflops

        max_logging.log(f"Calculated TFLOPs per pass: {train_tflops:.4f}")
        return train_tflops, total_attn_flops, seq_len

    @abc.abstractmethod
    def get_data_shardings(self, mesh):
        """Returns data shardings for training."""

    @abc.abstractmethod
    def get_eval_data_shardings(self, mesh):
        """Returns data shardings for evaluation."""

    @abc.abstractmethod
    def load_dataset(self, mesh, pipeline=None, is_training=True):
        """Loads the dataset."""

    @abc.abstractmethod
    def get_train_step(self, pipeline, mesh, state_shardings, data_shardings):
        """Returns the training step function."""

    @abc.abstractmethod
    def get_eval_step(self, pipeline, mesh, state_shardings, eval_data_shardings):
        """Returns the evaluation step function."""

    def preprocess_batch(self, batch, pipeline):
        """Optional hook for on-the-fly batch encoding.

        Override in subclasses that use raw-pixel datasets (e.g. dataset_type="droid")
        to encode frames into latents before the training step. The default
        implementation is a no-op for pre-encoded TFRecord datasets.
        """
        return batch

    def start_training(self):
        with nn_partitioning.axis_rules(self.config.logical_axis_rules):
            pipeline, opt_state, step, extra_state = self.checkpointer.load_checkpoint()
        restore_args = {}
        if opt_state is not None and step is not None:
            restore_args = {"opt_state": opt_state, "step": step}
            del opt_state
        if extra_state.get("student_params") is not None:
            restore_args["student_params"] = extra_state["student_params"]
        if self.config.enable_ssim:
            # Generate a sample before training to compare against generated sample after training.
            pretrained_video_path = generate_sample(self.config, pipeline, filename_prefix="pre-training-")

        needs_vae_for_training = getattr(self.config, "dataset_type", "") == "droid"
        if not needs_vae_for_training and (
            self.config.eval_every == -1 or (not self.config.enable_generate_video_for_eval)
        ):
            # save some memory.
            del pipeline.vae
            del pipeline.vae_cache

        mesh = pipeline.mesh
        train_data_iterator = self.load_dataset(mesh, pipeline=pipeline, is_training=True)

        # Load FlowMatch scheduler
        scheduler, scheduler_state = self.create_scheduler()
        pipeline.scheduler = scheduler
        pipeline.scheduler_state = scheduler_state
        optimizer, learning_rate_scheduler = self.checkpointer._create_optimizer(
            pipeline.transformer, self.config, self.config.learning_rate
        )
        # Returns pipeline with trained transformer state
        pipeline = self.training_loop(pipeline, optimizer, learning_rate_scheduler, train_data_iterator, restore_args)

        if self.config.enable_ssim:
            posttrained_video_path = generate_sample(self.config, pipeline, filename_prefix="post-training-")
            print_ssim(pretrained_video_path, posttrained_video_path)

    def eval(self, mesh, eval_rng_key, step, p_eval_step, state, scheduler_state, writer):
        eval_data_iterator = self.load_dataset(mesh, is_training=False)
        eval_rng = eval_rng_key
        eval_losses_by_timestep = {}
        # Loop indefinitely until the iterator is exhausted
        while True:
            try:
                eval_start_time = datetime.datetime.now()
                eval_batch = load_next_batch(eval_data_iterator, None, self.config)
                with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
                    metrics, eval_rng = p_eval_step(state, eval_batch, eval_rng, scheduler_state)
                    metrics["scalar"]["learning/eval_loss"].block_until_ready()
                losses = metrics["scalar"]["learning/eval_loss"]
                timesteps = eval_batch["timesteps"]
                gathered_losses = multihost_utils.process_allgather(losses, tiled=True)
                gathered_losses = jax.device_get(gathered_losses)
                gathered_timesteps = multihost_utils.process_allgather(timesteps, tiled=True)
                gathered_timesteps = jax.device_get(gathered_timesteps)
                if jax.process_index() == 0:
                    for t, l in zip(gathered_timesteps.flatten(), gathered_losses.flatten()):
                        timestep = int(t)
                        if timestep not in eval_losses_by_timestep:
                            eval_losses_by_timestep[timestep] = []
                        eval_losses_by_timestep[timestep].append(l)
                    eval_end_time = datetime.datetime.now()
                    eval_duration = eval_end_time - eval_start_time
                    max_logging.log(f"Eval time: {eval_duration.total_seconds():.2f} seconds.")
            except StopIteration:
                # This block is executed when the iterator has no more data
                break
        # Check if any evaluation was actually performed
        if eval_losses_by_timestep and jax.process_index() == 0:
            mean_per_timestep = []
            if jax.process_index() == 0:
                max_logging.log(f"Step {step}, calculating mean loss per timestep...")
            for timestep, losses in sorted(eval_losses_by_timestep.items()):
                losses = jnp.array(losses)
                losses = losses[: min(self.config.eval_max_number_of_samples_in_bucket, len(losses))]
                mean_loss = jnp.mean(losses)
                max_logging.log(f"  Mean eval loss for timestep {timestep}: {mean_loss:.4f}")
                mean_per_timestep.append(mean_loss)
            final_eval_loss = jnp.mean(jnp.array(mean_per_timestep))
            max_logging.log(f"Step {step}, Final Average Eval loss: {final_eval_loss:.4f}")
            if writer:
                writer.add_scalar("learning/eval_loss", final_eval_loss, step)

    def training_loop(
        self, pipeline, optimizer, learning_rate_scheduler, train_data_iterator, restore_args: dict = {}
    ):
        mesh = pipeline.mesh
        graphdef, loaded_params, rest_of_state = nnx.split(pipeline.transformer, nnx.Param, ...)

        ema_decay = getattr(self.config, "ema_decay", 0.0)
        distill = getattr(self.config, "distill", False)

        # When resuming from a distillation checkpoint the pipeline was loaded with
        # teacher weights (saved under "params" after the save-time swap).  The
        # actual student weights are returned under "student_params" in restore_args.
        student_params_from_ckpt = restore_args.pop("student_params", None)
        if distill and ema_decay > 0.0 and student_params_from_ckpt is not None:
            params = student_params_from_ckpt   # student → receives gradient updates
            ema_params = loaded_params           # teacher → provides distillation targets
        else:
            params = loaded_params
            if ema_decay > 0.0:
                # Must copy inside jax.jit so the result has the correct GLOBAL array
                # shape on multi-host setups.  jnp.copy called outside jit materialises
                # only the local per-host shard, giving shard shape (e.g. [16, dim])
                # instead of the global shape (e.g. [256, dim]), which causes
                # dot_general shape mismatches on the first train step and corrupt
                # shapes in saved checkpoints.  jit also ensures new buffer objects so
                # p_train_step can donate state without the "donate same buffer twice"
                # error that a plain identity copy (lambda x: x) would trigger.
                ema_params = jax.jit(lambda p: jax.tree_util.tree_map(jnp.copy, p))(params)
            else:
                ema_params = None

        # When distilling, always save the full TrainState so both student (params)
        # and teacher (ema_params) are preserved.  Otherwise honour save_optimizer.
        _save_full_state = distill and ema_params is not None

        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
            state = TrainState.create(
                apply_fn=graphdef.apply,
                params=params,
                tx=optimizer,
                graphdef=graphdef,
                rest_of_state=rest_of_state,
                ema_params=ema_params,
            )
            if restore_args:
                step = restore_args.get("step", 0)
                max_logging.log(f"Restoring optimizer and resuming from step {step}")
                state = state.replace(opt_state=restore_args.get("opt_state"), step=step)
                del restore_args["opt_state"]
                del optimizer
            state = jax.tree.map(_to_array, state)
            _state_spec = nnx.get_partition_spec(state)
            _state_shardings = nnx.get_named_sharding(state, mesh)
            # nnx.get_partition_spec returns P() for arrays without logical-axis
            # annotations (optax mu/nu, ema_params copies). For arrays already on
            # TPU with a NamedSharding, use their actual sharding as the target so
            # that state_shardings is correct for p_train_step compilation and
            # _to_tpu_if_cpu does not try to reshard them. CPU arrays (from a
            # checkpoint restore) keep the annotation-derived target sharding.
            def _use_actual_for_tpu(x, computed):
                if not isinstance(x, jax.Array):
                    return computed
                if any(d.platform == "cpu" for d in x.devices()):
                    return computed
                actual = getattr(x, "sharding", None)
                return actual if isinstance(actual, jax.sharding.NamedSharding) else computed
            _state_shardings = jax.tree.map(_use_actual_for_tpu, state, _state_shardings)
            def _to_tpu_if_cpu(x, target_sharding):
                if not isinstance(x, jax.Array):
                    return x
                on_cpu = any(d.platform == "cpu" for d in x.devices())
                if on_cpu:
                    full = np.asarray(x.addressable_data(0))
                    return jax.make_array_from_callback(x.shape, target_sharding, lambda idx: full[idx])
                # Only reshard replicated TPU arrays. addressable_data(0) equals the
                # full tensor for replicated arrays but only a local shard for sharded
                # arrays, so make_array_from_callback is unsafe on sharded arrays.
                # PartitionSpec() and PartitionSpec(None,...,None) compare unequal in
                # Python despite being semantically identical; test with all() instead.
                spec = getattr(getattr(x, "sharding", None), "spec", None)
                is_replicated = spec is not None and all(dim is None for dim in spec)
                target_spec = getattr(target_sharding, "spec", None)
                if not (is_replicated and target_spec is not None and target_spec != spec):
                    return x
                full = np.asarray(x.addressable_data(0))
                return jax.make_array_from_callback(x.shape, target_sharding, lambda idx: full[idx])
            state = jax.tree.map(_to_tpu_if_cpu, state, _state_shardings)
            state_shardings = _state_shardings
            if jax.process_index() == 0 and restore_args:
                state_spec = nnx.get_partition_spec(state)
                max_logging.log("--- Optimizer State Sharding Spec (opt_state) ---")
                pretty_string = pprint.pformat(state_spec.opt_state, indent=4, width=60)
                max_logging.log(pretty_string)
                max_logging.log("------------------------------------------------")
        data_shardings = self.get_data_shardings(mesh)
        eval_data_shardings = self.get_eval_data_shardings(mesh)

        writer = max_utils.initialize_summary_writer(self.config)
        writer_thread = threading.Thread(target=_tensorboard_writer_worker, args=(writer, self.config), daemon=True)
        writer_thread.start()

        self._wandb_run = None
        if jax.process_index() == 0 and getattr(self.config, "wandb_project", ""):
            import wandb
            self._wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=getattr(self.config, "wandb_entity", None) or None,
                name=self.config.run_name or None,
                settings=wandb.Settings(start_method="thread"),
            )

        num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
        max_utils.add_text_to_summary_writer("number_model_parameters", str(num_model_parameters), writer)
        max_utils.add_text_to_summary_writer("libtpu_init_args", os.environ.get("LIBTPU_INIT_ARGS", ""), writer)
        max_utils.add_config_to_summary_writer(self.config, writer)

        if jax.process_index() == 0:
            max_logging.log("***** Running training *****")
            max_logging.log(f"  Instantaneous batch size per device = {self.config.per_device_batch_size}")
            max_logging.log(
                f"  Total train batch size (w. parallel & distributed) = {self.config.global_batch_size_to_train_on}"
            )
            max_logging.log(f"  Total optimization steps = {self.config.max_train_steps}")

        _log_param_shapes(state.params, tag="PRE_TRAIN_STEP0")
        p_train_step = self.get_train_step(pipeline, mesh, state_shardings, data_shardings)
        p_eval_step = self.get_eval_step(pipeline, mesh, state_shardings, eval_data_shardings)

        rng = jax.random.key(self.config.seed)
        rng, eval_rng_key = jax.random.split(rng)
        start_step = 0
        last_step_completion = datetime.datetime.now()
        local_metrics_file = open(self.config.metrics_file, "a", encoding="utf8") if self.config.metrics_file else None
        running_gcs_metrics = [] if self.config.gcs_metrics else None
        first_profiling_step = self.config.skip_first_n_steps_for_profiler
        if max_utils.profiler_enabled(self.config) and first_profiling_step >= self.config.max_train_steps:
            raise ValueError("Profiling requested but initial profiling step set past training final step")
        last_profiling_step = np.clip(
            first_profiling_step + self.config.profiler_steps - 1,
            first_profiling_step,
            self.config.max_train_steps - 1,
        )
        if restore_args.get("step", 0):
            max_logging.log(f"Resuming training from step {step}")
        start_step = restore_args.get("step", 0)
        if start_step > 0:
            train_data_iterator = self.load_dataset(
                pipeline.mesh, pipeline=pipeline, is_training=True,
                seed=self.config.seed + start_step,
            )
        per_device_tflops, _, _ = BaseWanTrainer.calculate_tflops(pipeline)
        scheduler_state = pipeline.scheduler_state
        example_batch = load_next_batch(train_data_iterator, None, self.config)

        with ThreadPoolExecutor(max_workers=1) as executor:
            for step in np.arange(start_step, self.config.max_train_steps):
                if max_utils.profiler_enabled(self.config) and step == first_profiling_step:
                    self._profiler = max_utils.Profiler(self.config)
                    self._profiler.start()
                start_step_time = datetime.datetime.now()

                next_batch_future = executor.submit(load_next_batch, train_data_iterator, example_batch, self.config)
                example_batch = self.preprocess_batch(example_batch, pipeline)
                with (
                    jax.profiler.StepTraceAnnotation("train", step_num=step),
                    pipeline.mesh,
                    nn_partitioning.axis_rules(self.config.logical_axis_rules),
                ):
                    state, scheduler_state, train_metric, rng = p_train_step(
                        state, example_batch, rng, scheduler_state
                    )
                    state = self.post_train_step(state, int(step))
                    train_metric["scalar"]["learning/loss"].block_until_ready()
                last_step_completion = datetime.datetime.now()

                if max_utils.profiler_enabled(self.config) and step == last_profiling_step:
                    if self._profiler:
                        self._profiler.stop()

                train_utils.record_scalar_metrics(
                    train_metric,
                    last_step_completion - start_step_time,
                    per_device_tflops,
                    learning_rate_scheduler(step),
                )
                if self.config.write_metrics:
                    train_utils.write_metrics(
                        writer, local_metrics_file, running_gcs_metrics, train_metric, step, self.config
                    )
                if self._wandb_run is not None and step % self.config.log_period == 0:
                    self._wandb_run.log({
                        "train/loss": float(jax.device_get(train_metric["scalar"]["learning/loss"])),
                        "train/lr": float(train_metric["scalar"].get("learning/current_learning_rate", 0)),
                        "train/steps_per_sec": 1.0 / train_metric["scalar"].get("perf/step_time_seconds", 1),
                    }, step=step)

                if self.config.eval_every > 0 and (step + 1) % self.config.eval_every == 0:
                    if self.config.enable_generate_video_for_eval:
                        # Use teacher weights for inference video when distilling.
                        inference_params = state.ema_params if _save_full_state else state.params
                        pipeline.transformer = nnx.merge(state.graphdef, inference_params, state.rest_of_state)
                        inference_generate_video(self.config, pipeline, filename_prefix=f"{step + 1}-train_steps-")
                    # Re-create the iterator each time you start evaluation to reset it
                    # This assumes your data loading logic can be called to get a fresh iterator.
                    self.eval(mesh, eval_rng_key, step, p_eval_step, state, scheduler_state, writer)

                example_batch = next_batch_future.result()
                if step != 0 and self.config.checkpoint_every != -1 and step % self.config.checkpoint_every == 0:
                    max_logging.log(f"Saving checkpoint for step {step}")
                    if self.config.save_optimizer or _save_full_state:
                        self.checkpointer.save_checkpoint(step, pipeline, _state_to_save_dict(state, _save_full_state))
                    else:
                        self.checkpointer.save_checkpoint(step, pipeline, {"params": state.params})

            _metrics_queue.put(None)
            writer_thread.join()
            if writer:
                writer.flush()
            if self._wandb_run is not None:
                self._wandb_run.finish()
            if self.config.save_final_checkpoint:
                max_logging.log(f"Saving final checkpoint for step {step}")
                if _save_full_state:
                    self.checkpointer.save_checkpoint(
                        self.config.max_train_steps - 1, pipeline, _state_to_save_dict(state, _save_full_state)
                    )
                else:
                    self.checkpointer.save_checkpoint(self.config.max_train_steps - 1, pipeline, {"params": state.params})
                self.checkpointer.checkpoint_manager.wait_until_finished()
            # Load trained transformer — use teacher (EMA) when distilling.
            final_params = state.ema_params if _save_full_state else state.params
            pipeline.transformer = nnx.merge(state.graphdef, final_params, state.rest_of_state)
            return pipeline
