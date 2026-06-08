# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import time
import os
import subprocess
import glob
from maxdiffusion.checkpointing.wan_checkpointer_2_1 import WanCheckpointer2_1
from maxdiffusion.checkpointing.wan_checkpointer_2_2 import WanCheckpointer2_2
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p1 import WanCheckpointerI2V_2_1
from maxdiffusion.checkpointing.wan_checkpointer_i2v_2p2 import WanCheckpointerI2V_2_2
from maxdiffusion.checkpointing.wan_checkpointer_ti2v_2p2 import WanCheckpointerTI2V_2_2
from maxdiffusion import pyconfig, max_logging, max_utils
from absl import app
from maxdiffusion.train_utils import transformer_engine_context
from maxdiffusion.utils import export_to_video
from maxdiffusion.utils.loading_utils import load_image, load_video
from google.cloud import storage
import flax
from maxdiffusion.common_types import WAN2_1, WAN2_2
from maxdiffusion.loaders.wan_lora_nnx_loader import Wan2_1NNXLoraLoader, Wan2_2NNXLoraLoader


def upload_video_to_gcs(output_dir: str, video_path: str):
  """
  Uploads a local video file to a specified Google Cloud Storage bucket.
  """
  try:
    path_without_scheme = output_dir.removeprefix("gs://")
    parts = path_without_scheme.split("/", 1)
    bucket_name = parts[0]
    folder_name = parts[1] if len(parts) > 1 else ""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    source_file_path = f"./{video_path}"
    destination_blob_name = os.path.join(folder_name, "videos", video_path)

    blob = bucket.blob(destination_blob_name)

    max_logging.log(f"Uploading {source_file_path} to {bucket_name}/{destination_blob_name}...")
    blob.upload_from_filename(source_file_path)
    max_logging.log(f"Upload complete {source_file_path}.")

  except Exception as e:
    max_logging.log(f"An error occurred: {e}")


def delete_file(file_path: str):
  if os.path.exists(file_path):
    try:
      os.remove(file_path)
      max_logging.log(f"Successfully deleted file: {file_path}")
    except OSError as e:
      max_logging.log(f"Error deleting file '{file_path}': {e}")
  else:
    max_logging.log(f"The file '{file_path}' does not exist.")


def get_git_commit_hash():
  """Tries to get the current Git commit hash."""
  try:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    return commit_hash
  except subprocess.CalledProcessError:
    max_logging.log("Warning: 'git rev-parse HEAD' failed. Not running in a git repo?")
    return None
  except FileNotFoundError:
    max_logging.log("Warning: 'git' command not found.")
    return None


jax.config.update("jax_use_shardy_partitioner", True)


def load_val_sample_latents(val_data_dir: str, window_size: int, sample_index: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Load one sample from val TFRecords and return (oracle_latents, text_embed).

  Reads TFRecords formatted identically to training data (latent_cam0/1/2 + text_embed).
  Cameras are stacked on H.  Returns:
    oracle_latents: (1, window_size, H_lat*3, W_lat, C_z) channels-last, float32, normalized.
    text_embed:     (1, 512, 4096) float32.
  """
  import tensorflow as tf

  pattern = os.path.join(val_data_dir, "*.tfrecord*")
  files = sorted(glob.glob(pattern))
  if not files:
    raise FileNotFoundError(f"No .tfrecord files found in {val_data_dir}")

  feature_description = {
      "latent_cam0": tf.io.FixedLenFeature([], tf.string),
      "latent_cam1": tf.io.FixedLenFeature([], tf.string),
      "latent_cam2": tf.io.FixedLenFeature([], tf.string),
      "text_embed":  tf.io.FixedLenFeature([], tf.string),
      "traj_len":    tf.io.FixedLenFeature([], tf.int64),
  }

  ds = tf.data.TFRecordDataset(files).map(
      lambda x: tf.io.parse_single_example(x, feature_description)
  ).skip(sample_index).take(1)

  for raw in ds:
    cam0 = tf.cast(tf.io.parse_tensor(raw["latent_cam0"], out_type=tf.float16), tf.float32).numpy()
    cam1 = tf.cast(tf.io.parse_tensor(raw["latent_cam1"], out_type=tf.float16), tf.float32).numpy()
    cam2 = tf.cast(tf.io.parse_tensor(raw["latent_cam2"], out_type=tf.float16), tf.float32).numpy()
    text = tf.cast(tf.io.parse_tensor(raw["text_embed"],  out_type=tf.float16), tf.float32).numpy()

  # Latents are (F_lat, C, H_lat, W_lat); stack cameras on H.
  latent = np.concatenate([cam0, cam1, cam2], axis=2)   # (F_lat, C, H_lat*3, W_lat)
  f_total = latent.shape[0]
  start = max(0, f_total - window_size)
  latent = latent[start : start + window_size]           # (window_size, C, H_lat*3, W_lat)

  # Transpose to channels-last and add batch dim: (1, window_size, H_lat*3, W_lat, C)
  latent = latent.transpose(0, 2, 3, 1)                 # (window_size, H_lat*3, W_lat, C)
  oracle_latents = jnp.array(latent[None])              # (1, window_size, H_lat*3, W_lat, C)
  text_embed = jnp.array(text[None])                    # (1, 512, 4096)
  return oracle_latents, text_embed


def call_pipeline(config, pipeline, prompt, negative_prompt):
  model_key = config.model_name
  model_type = config.model_type
  if model_type == "I2V":
    image = load_image(config.image_url)
    if model_key == WAN2_1:
      return pipeline(
          prompt=prompt,
          image=image,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=config.num_inference_steps,
          guidance_scale=config.guidance_scale,
          use_magcache=config.use_magcache,
          magcache_thresh=config.magcache_thresh,
          magcache_K=config.magcache_K,
          retention_ratio=config.retention_ratio,
      )
    elif model_key == WAN2_2:
      return pipeline(
          prompt=prompt,
          image=image,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=config.num_inference_steps,
          guidance_scale_low=config.guidance_scale_low,
          guidance_scale_high=config.guidance_scale_high,
          use_cfg_cache=config.use_cfg_cache,
          use_sen_cache=config.use_sen_cache,
      )
    else:
      raise ValueError(f"Unsupported model_name for I2V in config: {model_key}")
  elif model_type == "TI2V":
    val_data_dir = getattr(config, "val_data_dir", "")
    if val_data_dir:
      window_size = 1 + config.num_frames // 4
      sample_index = getattr(config, "val_sample_index", 0)
      oracle_latents, text_embed = load_val_sample_latents(val_data_dir, window_size, sample_index)
      # oracle_latents: (1, F, H_lat, W_lat, C) — derive pixel dims from latent spatial shape.
      vae_spatial = pipeline.vae_scale_factor_spatial
      latent_h = int(oracle_latents.shape[2])
      latent_w = int(oracle_latents.shape[3])
      if model_key == WAN2_2:
        return pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=latent_h * vae_spatial,
            width=latent_w * vae_spatial,
            num_frames=config.num_frames,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            use_cfg_cache=config.use_cfg_cache,
            use_sen_cache=config.use_sen_cache,
            prompt_embeds=text_embed,
            preencoded_oracle_latents=oracle_latents,
        )
      else:
        raise ValueError(f"Unsupported model_name for TI2V val inference: {model_key}")

    image = load_image(config.image_url)
    conditioning_video = None
    if hasattr(config, "conditioning_video") and config.conditioning_video:
      import PIL.Image
      frames = load_video(config.conditioning_video)  # List[PIL.Image]
      # Resize to target resolution to match generation latent spatial dims
      frames = [f.convert("RGB").resize((config.width, config.height), PIL.Image.LANCZOS) for f in frames]
      # Use frame 0 of the conditioning video as the anchor image so the VAE-encoded
      # frame_0 latent is identical across the image path and oracle path.
      image = frames[0]
      # Stack to [T, H, W, C], normalize to [-1, 1], then reshape to [B, C, T, H, W]
      arr = np.stack([np.array(f) for f in frames], axis=0).astype(np.float32)
      arr = arr / 127.5 - 1.0  # [T, H, W, C]
      arr = arr.transpose(3, 0, 1, 2)  # [C, T, H, W]
      conditioning_video = jnp.array(arr)[None]  # [1, C, T, H, W]
    if model_key == WAN2_2:
      return pipeline(
          prompt=prompt,
          image=image,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=config.num_inference_steps,
          guidance_scale=config.guidance_scale,
          use_cfg_cache=config.use_cfg_cache,
          use_sen_cache=config.use_sen_cache,
          conditioning_video=conditioning_video,
          privileged=conditioning_video is not None and getattr(config, "num_privileged_frames", 0) != 0,
      )
    else:
      raise ValueError(f"Unsupported model_name for TI2V in config: {model_key}")
  elif model_type == "T2V":
    if model_key == WAN2_1:
      return pipeline(
          prompt=prompt,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=config.num_inference_steps,
          guidance_scale=config.guidance_scale,
          use_cfg_cache=config.use_cfg_cache,
          use_magcache=config.use_magcache,
          magcache_thresh=config.magcache_thresh,
          magcache_K=config.magcache_K,
          retention_ratio=config.retention_ratio,
      )
    elif model_key == WAN2_2:
      return pipeline(
          prompt=prompt,
          negative_prompt=negative_prompt,
          height=config.height,
          width=config.width,
          num_frames=config.num_frames,
          num_inference_steps=config.num_inference_steps,
          guidance_scale_low=config.guidance_scale_low,
          guidance_scale_high=config.guidance_scale_high,
          use_cfg_cache=config.use_cfg_cache,
          use_sen_cache=config.use_sen_cache,
      )
    else:
      raise ValueError(f"Unsupported model_name for T2Vin config: {model_key}")


def inference_generate_video(config, pipeline, filename_prefix=""):
  s0 = time.perf_counter()
  prompt = [config.prompt] * config.global_batch_size_to_train_on
  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}, video: {filename_prefix}"
  )

  videos = call_pipeline(config, pipeline, prompt, negative_prompt)

  max_logging.log(f"video {filename_prefix}, compile time: {(time.perf_counter() - s0)}")
  for i in range(len(videos)):
    video_path = f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
    export_to_video(videos[i], video_path, fps=config.fps)
    if config.output_dir.startswith("gs://"):
      upload_video_to_gcs(os.path.join(config.output_dir, config.run_name), video_path)
      # Delete local files to avoid storing too manys videos
      delete_file(f"./{video_path}")
  return


def run(config, pipeline=None, filename_prefix="", commit_hash=None):
  model_key = config.model_name
  writer = max_utils.initialize_summary_writer(config)
  if jax.process_index() == 0 and writer:
    max_logging.log(f"TensorBoard logs will be written to: {config.tensorboard_dir}")

    if commit_hash:
      writer.add_text("inference/git_commit_hash", commit_hash, global_step=0)
      max_logging.log(f"Git Commit Hash: {commit_hash}")
    else:
      max_logging.log("Could not retrieve Git commit hash.")

  if pipeline is None:
    load_start = time.perf_counter()
    model_type = config.model_type
    if model_key == WAN2_1:
      if model_type == "I2V":
        checkpoint_loader = WanCheckpointerI2V_2_1(config=config)
      else:
        checkpoint_loader = WanCheckpointer2_1(config=config)
    elif model_key == WAN2_2:
      if model_type == "I2V":
        checkpoint_loader = WanCheckpointerI2V_2_2(config=config)
      elif model_type == "TI2V":
        checkpoint_loader = WanCheckpointerTI2V_2_2(config=config)
      else:
        checkpoint_loader = WanCheckpointer2_2(config=config)
    else:
      raise ValueError(f"Unsupported model_name for checkpointer: {model_key}")
    pipeline, _, _, _ = checkpoint_loader.load_checkpoint()
    load_time = time.perf_counter() - load_start
    max_logging.log(f"load_time: {load_time:.1f}s")
  else:
    load_time = 0.0

  # If LoRA is specified, inject layers and load weights.
  if (
      config.enable_lora
      and hasattr(config, "lora_config")
      and config.lora_config
      and config.lora_config["lora_model_name_or_path"]
  ):
    if model_key == WAN2_1:
      lora_loader = Wan2_1NNXLoraLoader()
      lora_config = config.lora_config
      for i in range(len(lora_config["lora_model_name_or_path"])):
        pipeline = lora_loader.load_lora_weights(
            pipeline,
            lora_config["lora_model_name_or_path"][i],
            transformer_weight_name=lora_config["weight_name"][i],
            rank=lora_config["rank"][i],
            scale=lora_config["scale"][i],
            scan_layers=config.scan_layers,
            dtype=config.weights_dtype,
        )

    if model_key == WAN2_2:
      lora_loader = Wan2_2NNXLoraLoader()
      lora_config = config.lora_config
      for i in range(len(lora_config["lora_model_name_or_path"])):
        pipeline = lora_loader.load_lora_weights(
            pipeline,
            lora_config["lora_model_name_or_path"][i],
            high_noise_weight_name=lora_config["high_noise_weight_name"][i],
            low_noise_weight_name=lora_config["low_noise_weight_name"][i],
            rank=lora_config["rank"][i],
            scale=lora_config["scale"][i],
            scan_layers=config.scan_layers,
            dtype=config.weights_dtype,
        )

  s0 = time.perf_counter()

  # Using global_batch_size_to_train_on so not to create more config variables
  prompt = [config.prompt] * config.global_batch_size_to_train_on
  negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on

  max_logging.log(
      f"Num steps: {config.num_inference_steps}, height: {config.height}, width: {config.width}, frames: {config.num_frames}"
  )
  videos = call_pipeline(config, pipeline, prompt, negative_prompt)

  max_logging.log("===================== Model details =======================")
  max_logging.log(f"model name: {config.model_name}")
  max_logging.log(f"model path: {config.pretrained_model_name_or_path}")
  max_logging.log(f"model type: {config.model_type}")
  max_logging.log(f"hardware: {jax.devices()[0].platform}")
  max_logging.log(f"number of devices: {jax.device_count()}")
  max_logging.log(f"per_device_batch_size: {config.per_device_batch_size}")
  max_logging.log("============================================================")

  compile_time = time.perf_counter() - s0
  max_logging.log(f"compile_time: {compile_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/compile_time", compile_time, global_step=0)
  saved_video_path = []
  for i in range(len(videos)):
    video_path = f"{filename_prefix}wan_output_{config.seed}_{i}.mp4"
    export_to_video(videos[i], video_path, fps=config.fps)
    saved_video_path.append(video_path)
    if config.output_dir.startswith("gs://"):
      upload_video_to_gcs(os.path.join(config.output_dir, config.run_name), video_path)

  s0 = time.perf_counter()
  videos = call_pipeline(config, pipeline, prompt, negative_prompt)
  generation_time = time.perf_counter() - s0
  max_logging.log(f"generation_time: {generation_time}")
  if writer and jax.process_index() == 0:
    writer.add_scalar("inference/generation_time", generation_time, global_step=0)
    num_devices = jax.device_count()
    num_videos = num_devices * config.per_device_batch_size
    if num_videos > 0:
      generation_time_per_video = generation_time / num_videos
      writer.add_scalar("inference/generation_time_per_video", generation_time_per_video, global_step=0)
      max_logging.log(f"generation time per video: {generation_time_per_video}")
    else:
      max_logging.log("Warning: Number of videos is zero, cannot calculate generation_time_per_video.")
  max_logging.log(
      f"\n{'=' * 50}\n"
      f"  TIMING SUMMARY\n"
      f"{'=' * 50}\n"
      f"  Load (checkpoint):   {load_time:>7.1f}s\n"
      f"  Compile:             {compile_time:>7.1f}s\n"
      f"  {'─' * 40}\n"
      f"  Inference:           {generation_time:>7.1f}s\n"
      f"{'=' * 50}"
  )

  s0 = time.perf_counter()
  if max_utils.profiler_enabled(config):
    with max_utils.Profiler(config):
      videos = call_pipeline(config, pipeline, prompt, negative_prompt)
    generation_time_with_profiler = time.perf_counter() - s0
    max_logging.log(f"generation_time_with_profiler: {generation_time_with_profiler}")
    if writer and jax.process_index() == 0:
      writer.add_scalar("inference/generation_time_with_profiler", generation_time_with_profiler, global_step=0)

  return saved_video_path


def main(argv: Sequence[str]) -> None:
  commit_hash = get_git_commit_hash()
  pyconfig.initialize(argv)
  try:
    flax.config.update("flax_always_shard_variable", False)
  except LookupError:
    pass
  max_utils.ensure_machinelearning_job_runs(pyconfig.config)
  run(pyconfig.config, commit_hash=commit_hash)


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
