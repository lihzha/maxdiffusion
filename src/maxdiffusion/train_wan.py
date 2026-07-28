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

from typing import Sequence

import jax
from absl import app
from maxdiffusion import max_logging, pyconfig, max_utils
from maxdiffusion.train_utils import (
    validate_train_config,
    transformer_engine_context,
)
import flax


def train(config):
  if config.model_type == "I2V":
    from maxdiffusion.trainers.wan_i2v_trainer import WanI2VTrainer
    trainer = WanI2VTrainer(config)
  elif config.model_type == "TI2V":
    from maxdiffusion.trainers.wan_ti2v_trainer import WanTI2VTrainer
    trainer = WanTI2VTrainer(config)
  elif config.model_type == "AC_TI2V":
    from maxdiffusion.trainers.wan_ctrl_world_trainer import WanCtrlWorldTrainer
    trainer = WanCtrlWorldTrainer(config)
  elif config.model_type == "SIDE_ADAPTER_TI2V":
    from maxdiffusion.trainers.wan_ti2v_side_adapter_trainer import WanTI2VSideAdapterTrainer
    trainer = WanTI2VSideAdapterTrainer(config)
  elif config.model_type == "FULL_FT_TI2V":
    from maxdiffusion.trainers.wan_ti2v_full_ft_trainer import WanTI2VFullFTTrainer
    trainer = WanTI2VFullFTTrainer(config)
  else:
    from maxdiffusion.trainers.wan_trainer import WanTrainer
    trainer = WanTrainer(config)
  trainer.start_training()


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv, validate_training=True)
  config = pyconfig.config
  max_utils.ensure_machinelearning_job_runs(pyconfig.config)
  validate_train_config(config)
  max_logging.log(f"Found {jax.device_count()} devices.")
  try:
    flax.config.update("flax_always_shard_variable", False)
  except LookupError:
    pass
  train(config)


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
