"""exp_06 `rollout_adapter` — T4: the dispatch target for `model_type: POS_ROLLOUT_TI2V`.

**One job: turn a config into the loop's arguments.** T3b-4 built the loop with the model, the
optimizer and the DEV instrument as injected seams, so this class is the thing that resolves those
from configuration — and nothing else. It is constructible from the config ALONE (exp_05's S8
standard), because that is all `train_wan.train` has.

**What is wired here:** the schedule (arm, k, cadence, batching, seed), the DEV cohort — bound to
exp_04's published J0 manifest by path AND digest, so a misconfigured run fails at construction
rather than after an hour of TPU time — and the checkpoint trees.

**What is NOT wired here, stated plainly.** The pipeline load, the TFRecord iterator, the sharded
state and the optimizer are the `WanTI2VSideAdapterTrainer` seams the plan (§3) says to build on, and
they are a round's work in their own right — T4's contract is dispatch and configuration. So
:meth:`start_training` resolves and validates everything configuration can decide, then raises with
a message naming exactly the one thing still missing. It does NOT silently no-op, and it does not
pretend to train: exp_05's S7 shipped a dispatched `start_training` that raised for the same reason,
and the reviewer recorded that as a fact about scope rather than a defect.
"""

from __future__ import annotations

from maxdiffusion.pos_rollout_arms import ARMS, PRODUCTION_DROPOUT_RATE
from maxdiffusion.pos_rollout_dev_instrument import load_dev_cohort
from maxdiffusion.pos_rollout_loop import LoopSchedule, build_checkpoint_manager, build_selection_manager
from maxdiffusion.run_wan_null_inversion import optional_config_value

POS_ROLLOUT_MODEL_TYPE = "POS_ROLLOUT_TI2V"


class WanPosRolloutTrainer:
    """Config in, loop arguments out."""

    def __init__(self, config):
        self.config = config
        self.schedule = LoopSchedule.from_config(config)
        if self.schedule.arm not in ARMS:
            raise ValueError(f"unknown arm {self.schedule.arm!r}; declared arms are {list(ARMS)}")
        # Dropout is enforced at arm construction (T3b-2), but a run whose YAML asks for a nonzero
        # rate should die here rather than after the pipeline load.
        self.dropout_rate = float(optional_config_value(config, "dropout", PRODUCTION_DROPOUT_RATE))
        if self.dropout_rate != PRODUCTION_DROPOUT_RATE:
            raise ValueError(
                f"dropout must be {PRODUCTION_DROPOUT_RATE} for exp_06: got {self.dropout_rate}. matched-C0's "
                f"dropout key is inert only at zero (plan §3b)."
            )
        self.dev_manifest = str(optional_config_value(config, "pos_dev_manifest", ""))
        self.dev_manifest_sha256 = str(optional_config_value(config, "pos_dev_manifest_sha256", ""))
        self.checkpoint_dir = str(optional_config_value(config, "checkpoint_dir", "") or "")

    def load_dev_cohort(self):
        """Bind selection to the approved manifest, by path AND digest, before anything expensive.

        A directory is not accepted here and cannot be: the instrument takes a manifest file, hashes
        its bytes, and refuses anything that is not the published DEV-64 cohort (T3b-3).
        """
        if not self.dev_manifest:
            raise ValueError(
                "pos_dev_manifest must name exp_04's published J0 DEV-64 manifest: selection binds to a "
                "manifest file, never a directory (plan §3d)"
            )
        return load_dev_cohort(self.dev_manifest, expected_sha256=self.dev_manifest_sha256)

    def checkpoint_managers(self):
        if not self.checkpoint_dir:
            return None, None
        return build_checkpoint_manager(self.checkpoint_dir), build_selection_manager(self.checkpoint_dir)

    def start_training(self) -> None:
        # Everything configuration can decide is decided and validated first, so a misconfigured run
        # fails in seconds rather than after a pipeline load.
        self.load_dev_cohort()
        raise NotImplementedError(
            "exp_06's model/data wiring is not part of T4 (dispatch + config). The schedule, the DEV "
            "cohort binding and the checkpoint trees resolve; still to come are the pipeline load, the "
            "TFRecord iterator, the sharded state and the optimizer — the WanTI2VSideAdapterTrainer "
            "seams the plan §3 names. Until then this arm dispatches and validates but does not train."
        )
