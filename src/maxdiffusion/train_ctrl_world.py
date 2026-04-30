"""Action-conditioned SVD (Ctrl-World) training entry point.

Mirrors src/maxdiffusion/train.py but instantiates CtrlWorldTrainer.

Usage::

    python src/maxdiffusion/train_ctrl_world.py \
        src/maxdiffusion/configs/base_ctrl_world.yml \
        run_name=my-run \
        output_dir=gs://<bucket>/ctrl-world-runs/my-run \
        train_data_dir=gs://<bucket>/ctrl_world_droid/train \
        eval_data_dir=gs://<bucket>/ctrl_world_droid/val \
        stats_path=gs://<bucket>/ctrl_world_droid/stats.json
"""

from typing import Sequence

import jax
from absl import app

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.train_utils import transformer_engine_context, validate_train_config
from maxdiffusion.trainers.ctrl_world_trainer import CtrlWorldTrainer


def train(config):
    trainer = CtrlWorldTrainer(config)
    trainer.start_training()


def main(argv: Sequence[str]) -> None:
    pyconfig.initialize(argv)
    config = pyconfig.config
    max_utils.ensure_machinelearning_job_runs(config)
    validate_train_config(config)
    max_logging.log(f"[ctrl_world] found {jax.device_count()} devices")
    train(config)


if __name__ == "__main__":
    with transformer_engine_context():
        app.run(main)
