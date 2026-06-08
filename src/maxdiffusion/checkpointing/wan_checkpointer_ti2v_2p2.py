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

import json
import os
from typing import Optional, Tuple
import jax
from etils import epath
from maxdiffusion.checkpointing.checkpointing_utils import add_sharding_to_struct, get_cpu_mesh_and_sharding
from maxdiffusion.checkpointing.wan_checkpointer import WanCheckpointer
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_manager import CheckpointManager, CheckpointManagerOptions
from .. import max_logging
from ..pipelines.wan.wan_pipeline_ti2v_2p2 import WanPipelineTI2V_2_2


def _log(msg: str):
  if jax.process_index() == 0:
    max_logging.log(msg)


class WanCheckpointerTI2V_2_2(WanCheckpointer):

  def __init__(self, config):
    self.config = config
    self.opt_state = None

    checkpoint_dir = config.checkpoint_dir if config.checkpoint_dir.startswith("gs://") else os.path.abspath(config.checkpoint_dir)
    self.checkpoint_manager: CheckpointManager = CheckpointManager(
        epath.Path(checkpoint_dir),
        item_names=("wan_config", "wan_state"),
        item_handlers={
            "wan_config": ocp.JsonCheckpointHandler(),
            "wan_state": ocp.StandardCheckpointHandler(),
        },
        options=CheckpointManagerOptions(
            create=True,
            max_to_keep=3,
            enable_async_checkpointing=True,
            keep_period=getattr(config, "checkpoint_keep_period", -1) or None,
        ),
    )

  def load_wan_configs_from_orbax(self, step: Optional[int]) -> Tuple[Optional[dict], Optional[int]]:
    if step is None:
      step = self.checkpoint_manager.latest_step()
      _log(f"Latest WAN checkpoint step: {step}")
      if step is None:
        _log("No WAN checkpoint found.")
        return None, None
    _log(f"Loading WAN checkpoint from step {step}")

    mesh, replicated_sharding = get_cpu_mesh_and_sharding()
    metadatas = self.checkpoint_manager.item_metadata(step)
    state = metadatas.wan_state

    target_shardings = jax.tree_util.tree_map(lambda x: replicated_sharding, state)

    with mesh:
      abstract_train_state_with_sharding = jax.tree_util.tree_map(add_sharding_to_struct, state, target_shardings)

    _log("Restoring WAN TI2V 2.2 checkpoint")
    restored_checkpoint = self.checkpoint_manager.restore(
        step=step,
        args=ocp.args.Composite(
            wan_config=ocp.args.JsonRestore(),
            wan_state=ocp.args.StandardRestore(abstract_train_state_with_sharding),
        ),
    )
    _log(f"restored checkpoint {restored_checkpoint.keys()}")
    _log(f"restored checkpoint wan_state {restored_checkpoint.wan_state.keys()}")
    _log(f"optimizer found in checkpoint {'opt_state' in restored_checkpoint.wan_state.keys()}")
    _log(f"optimizer state saved in attribute self.opt_state {self.opt_state}")
    return restored_checkpoint, step

  def load_diffusers_checkpoint(self):
    pipeline = WanPipelineTI2V_2_2.from_pretrained(self.config)
    return pipeline

  def load_checkpoint(self, step=None) -> Tuple[WanPipelineTI2V_2_2, Optional[dict], Optional[int], dict]:
    restored_checkpoint, step = self.load_wan_configs_from_orbax(step)
    opt_state = None
    extra_state = {}
    if restored_checkpoint:
      _log("Loading WAN TI2V pipeline from checkpoint")
      pipeline = WanPipelineTI2V_2_2.from_checkpoint(self.config, restored_checkpoint)
      wan_state_keys = restored_checkpoint.wan_state.keys()
      if "opt_state" in wan_state_keys:
        opt_state = restored_checkpoint.wan_state["opt_state"]
      if "ema_params" in wan_state_keys:
        extra_state["student_params"] = restored_checkpoint.wan_state["ema_params"]
        _log("Distillation checkpoint detected: restoring student params from ema_params field.")
    else:
      _log("No checkpoint found, loading default pipeline.")
      pipeline = self.load_diffusers_checkpoint()

    return pipeline, opt_state, step, extra_state

  def save_checkpoint(self, train_step, pipeline: WanPipelineTI2V_2_2, train_states: dict):
    """Saves the training state and model configurations."""

    def config_to_json(model_or_config):
      return json.loads(model_or_config.to_json_string())

    _log(f"Saving checkpoint for step {train_step}")
    items = {
        "wan_config": ocp.args.JsonSave(config_to_json(pipeline.transformer)),
        "wan_state": ocp.args.StandardSave(train_states),
    }

    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    _log(f"Checkpoint for step {train_step} saved.")
