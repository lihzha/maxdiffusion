"""exp_06 `rollout_adapter` — T4: the dispatch target for `model_type: POS_ROLLOUT_TI2V`.

**One job: turn a config into the loop's arguments.** T3b-4 built the loop with the model, the
optimizer and the DEV instrument as injected seams, so this class is the thing that resolves those
from configuration — and nothing else. It is constructible from the config ALONE (exp_05's S8
standard), because that is all `train_wan.train` has.

**What is wired here:** the schedule (arm, k, cadence, batching, seed), the DEV cohort — bound to
exp_04's published J0 manifest by path AND digest, so a misconfigured run fails at construction
rather than after an hour of TPU time — the M1 authorization gate, the run-level RECIPE LOCK that
makes matched-C0 a real control, the SHA-bound resume adoption, and the checkpoint trees.

**Three things review pass 3 forced in here, all structural:**

* **M1's authorization is checked against a context this process DERIVES** (``running_context``),
  not against nothing. The reviewer published an authorization carrying a foreign SHA, a foreign
  model and the wrong device kind and production accepted it, because ``assert_cell_authorized`` was
  called without any current context to compare against. The cell now carries the ARM too, so a
  matched-C0 measurement no longer authorizes R-B.
* **The recipe lock (T6-1).** Yixun's decision-1 control is only a control if the two arms differ by
  ONE variable. A launcher can emit both arms from one array, but two queue submissions a day apart
  can still differ in a seed, a learning rate or a data directory, and nothing refused that. So the
  FIRST arm of a run publishes the normalized recipe — every declared config value except the arm
  and the four destinations derived from it — and the second arm ADOPTS it or is refused by name,
  with the differing keys listed. Divergence is not detected downstream; it cannot start.
* **Resume is an immutable INPUT, distinct from the fresh attempt's OUTPUT (T6-3).** A retry adopts
  the latest COMPLETE publication whose recorded SHA is the code now running, and writes into its
  own attempt root; it never reuses one mutable tree, and it never adopts a half-written one,
  because the publication marker is written LAST and is what "complete" means (issue #10/#13).

**What is NOT wired here, stated plainly.** The pipeline load, the TFRecord iterator, the sharded
state and the optimizer are the `WanTI2VSideAdapterTrainer` seams the plan (§3) says to build on, and
they are a round's work in their own right — T4's contract is dispatch and configuration. So
:meth:`start_training` resolves and validates everything configuration can decide, then raises with
a message naming exactly the one thing still missing. It does NOT silently no-op, and it does not
pretend to train.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Mapping

from maxdiffusion.pos_rollout_arms import ARMS, PRODUCTION_DROPOUT_RATE
from maxdiffusion.pos_rollout_dev_instrument import J0_DEV64_SHA256, load_dev_cohort
from maxdiffusion.pos_rollout_fit_probe import (
    FitCell,
    ProbeContext,
    assert_cell_authorized,
    derive_probe_context,
    load_authorization,
)
from maxdiffusion.pos_rollout_loop import LoopSchedule, build_checkpoint_manager, build_selection_manager
from maxdiffusion.pos_rollout_support import is_remote, storage_exists, storage_read_bytes, storage_write_bytes
from maxdiffusion.run_wan_null_inversion import optional_config_value

POS_ROLLOUT_MODEL_TYPE = "POS_ROLLOUT_TI2V"

RECIPE_LOCK_PROTOCOL = "exp06.recipe_lock.v1"
PUBLICATION_PROTOCOL = "exp06.attempt_publication.v1"
#: The publication marker is written LAST, so its presence is what COMPLETE means (issue #10).
PUBLICATION_NAME = "publication.json"

#: The ONLY values the two arms of a matched pair may differ in: the arm, and the destinations
#: derived from it. Everything else is the recipe, and the lock refuses a run that changes any of it.
ARM_SCOPED_KEYS = (
    "pos_rollout_arm",
    "checkpoint_dir",
    "base_output_directory",
    "pos_resume_parent",
    "pos_recipe_lock",
)

_ATTEMPT_RE = re.compile(r"^att-[A-Za-z0-9._-]+$")


def _digest(payload: Mapping) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _plain(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    return str(value)


def _config_keys(config) -> dict:
    """Every key this config declares, on each config object the dispatch path sees.

    ``HyperParameters`` exposes ``get_keys()`` -- the only DECLARED way to ask it what it has -- and
    a plain object exposes ``__dict__``. Two-argument ``getattr`` only (issue #11).
    """
    try:
        return dict(config.get_keys())
    except (AttributeError, TypeError, ValueError):
        pass
    if isinstance(config, Mapping):
        return dict(config)
    return dict(vars(config))


def _list_children(parent: str) -> list[str]:
    """The child names under an artifact root, on the storage layer the rest of exp_06 publishes to.

    ``pathlib`` silently turns ``gs://bucket/x`` into the local path ``gs:/bucket/x``, so a remote
    parent never falls back — it raises, exactly as the read/write/exists helpers do.
    """
    try:
        from tensorflow.io import gfile
    except Exception as error:  # noqa: BLE001 -- a local path must not need tensorflow
        if is_remote(parent):
            raise RuntimeError(
                f"{parent} is a remote URI and tensorflow.io.gfile is unavailable ({error}); refusing to "
                f"fall back to a local listing, which is how a gs:// resume root silently becomes empty"
            ) from error
        root = pathlib.Path(str(parent))
        return sorted(child.name for child in root.iterdir()) if root.is_dir() else []
    if not gfile.exists(str(parent)):
        return []
    return sorted(name.rstrip("/") for name in gfile.listdir(str(parent)))


# =================================================================================================
# The recipe lock: matched-C0 as a fact about the run, not a hope about two submissions.
# =================================================================================================


def normalized_recipe(config) -> dict:
    """Every declared config value except the arm and the destinations derived from it.

    This is the "normalized recipe" the reviewer asked the two arm commands to be compared as. The
    exclusions are exactly the permitted differences: the arm itself, and the four paths the launcher
    derives from the arm so that R-B and matched-C0 cannot share checkpoint state.
    """
    return {key: _plain(value) for key, value in sorted(_config_keys(config).items()) if key not in ARM_SCOPED_KEYS}


def publish_recipe_lock(path: str, config, *, arm: str) -> dict:
    """Publish this run's recipe, or ADOPT the identical one the paired arm already published.

    The first arm to start writes the lock; the second must match it in every key outside
    :data:`ARM_SCOPED_KEYS` or it does not start. The refusal names the keys, because "the arms
    differed somewhere" discovered after 10,000 steps is not a finding anybody can use.
    """
    recipe = normalized_recipe(config)
    payload = {"protocol": RECIPE_LOCK_PROTOCOL, "recipe": recipe, "recipe_digest": _digest(recipe)}
    published = {**payload, "sha256": _digest(payload)}
    if storage_exists(path):
        stored = json.loads(storage_read_bytes(path).decode("utf-8"))
        existing = stored.get("payload") or {}
        if _digest(existing) != stored.get("sha256"):
            raise ValueError(f"{path}: the recipe lock's digest does not describe its payload; it was edited")
        if existing.get("recipe_digest") != payload["recipe_digest"]:
            locked = existing.get("recipe") or {}
            differing = sorted(
                key for key in set(locked) | set(recipe) if locked.get(key, "<absent>") != recipe.get(key, "<absent>")
            )
            raise ValueError(
                f"arm {arm!r} asks to join run {path} with a DIFFERENT recipe: {differing} differ from the "
                f"locked one (e.g. "
                + ", ".join(
                    f"{key}: {locked.get(key, '<absent>')!r} -> {recipe.get(key, '<absent>')!r}"
                    for key in differing[:4]
                )
                + "). exp_06's causal claim is that the arms differ by the OBJECTIVE and nothing else, so the "
                "pair is refused rather than run and reconciled afterwards (Yixun's decision 1)."
            )
        return {**existing, "sha256": stored["sha256"], "adopted": True}
    storage_write_bytes(
        path, json.dumps({"payload": payload, "sha256": published["sha256"]}, sort_keys=True).encode("utf-8")
    )
    return {**published, "adopted": False}


# =================================================================================================
# Attempt publications: what "the latest COMPLETE checkpoint" means, and how a retry adopts it.
# =================================================================================================


def publish_attempt(
    parent: str,
    *,
    attempt: str,
    arm: str,
    code_sha: str,
    context_digest: str,
    step: int,
    checkpoint_dir: str,
) -> dict:
    """Mark one attempt COMPLETE, LAST — the marker is the only thing a resume will adopt.

    Written after the checkpoint tree it describes, so a crash mid-write leaves an attempt that a
    later run will correctly ignore rather than half-adopt.
    """
    if not _ATTEMPT_RE.match(str(attempt)):
        raise ValueError(f"{attempt!r} is not an attempt id: attempt roots are derived, never caller-shaped paths")
    payload = {
        "protocol": PUBLICATION_PROTOCOL,
        "attempt": str(attempt),
        "arm": str(arm),
        "code_sha": str(code_sha),
        "context_digest": str(context_digest),
        "step": int(step),
        "checkpoint_dir": str(checkpoint_dir),
        "complete": True,
    }
    path = f"{str(parent).rstrip('/')}/{attempt}/{PUBLICATION_NAME}"
    published = {**payload, "sha256": _digest(payload)}
    if storage_exists(path):
        stored = json.loads(storage_read_bytes(path).decode("utf-8"))
        if stored.get("sha256") != published["sha256"]:
            raise ValueError(
                f"{path} was already published with digest {stored.get('sha256')!r}; an attempt is published "
                f"once and adopted, never rewritten (issue #10/#13)"
            )
        return {**(stored.get("payload") or {}), "sha256": stored["sha256"]}
    storage_write_bytes(
        path, json.dumps({"payload": payload, "sha256": published["sha256"]}, sort_keys=True).encode("utf-8")
    )
    return published


def load_publication(path: str) -> dict:
    stored = json.loads(storage_read_bytes(path).decode("utf-8"))
    payload = stored.get("payload")
    if not isinstance(payload, Mapping) or _digest(payload) != stored.get("sha256"):
        raise ValueError(f"{path}: the publication marker's digest does not describe its payload; it was edited")
    if payload.get("protocol") != PUBLICATION_PROTOCOL:
        raise ValueError(f"{path}: protocol {payload.get('protocol')!r} is not {PUBLICATION_PROTOCOL}")
    for field in ("attempt", "arm", "code_sha", "checkpoint_dir"):
        if not str(payload.get(field, "")):
            raise ValueError(f"{path}: a publication marker without {field} cannot be adopted")
    if payload.get("complete") is not True:
        raise ValueError(f"{path}: this attempt is not marked complete, so there is nothing to resume from")
    return {**dict(payload), "sha256": stored["sha256"]}


def select_resume_publication(parent: str, *, code_sha: str, arm: str) -> dict | None:
    """The latest COMPLETE publication for THIS arm whose recorded SHA is the code now running.

    Three filters, each one a way a resume silently corrupts a run: an incomplete attempt (adopting a
    half-written tree), another arm's attempt (matched-C0 restoring R-B's parameters — the reviewer's
    scientifically critical finding), and another commit's attempt (resuming optimizer state built by
    a different program). Anything unreadable is skipped rather than fatal: a damaged old attempt
    must not stop a run that has a good one.
    """
    candidates = []
    for name in _list_children(parent):
        if not _ATTEMPT_RE.match(name):
            continue
        path = f"{str(parent).rstrip('/')}/{name}/{PUBLICATION_NAME}"
        if not storage_exists(path):
            continue
        try:
            publication = load_publication(path)
        except (ValueError, json.JSONDecodeError, OSError):
            continue
        if str(publication["arm"]) != str(arm) or str(publication["code_sha"]) != str(code_sha):
            continue
        candidates.append(publication)
    if not candidates:
        return None
    return max(candidates, key=lambda entry: (int(entry.get("step", 0)), str(entry["attempt"])))


# =================================================================================================
# The trainer.
# =================================================================================================


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
        self.fit_authorization = str(optional_config_value(config, "pos_fit_authorization", ""))
        self.resume_parent = str(optional_config_value(config, "pos_resume_parent", "") or "")
        self.recipe_lock = str(optional_config_value(config, "pos_recipe_lock", "") or "")

    def load_dev_cohort(self):
        """Bind selection to the approved manifest, by path AND digest, before anything expensive.

        A directory is not accepted here and cannot be: the instrument takes a manifest file, hashes
        its bytes, and refuses anything that is not the published DEV-64 cohort (T3b-3).

        The YAML's digest is a DECLARATION of which manifest the run intends, checked against the
        pinned constant — it is not passed to the loader. A caller-settable digest was the second of
        the reviewer's executed forgeries (T3b-4 review, BLOCKER A-B1): it let a DEV-labelled
        manifest whose first row was the genuine TEST row be loaded under its own hash.
        """
        if not self.dev_manifest:
            raise ValueError(
                "pos_dev_manifest must name exp_04's published J0 DEV-64 manifest: selection binds to a "
                "manifest file, never a directory (plan §3d)"
            )
        if self.dev_manifest_sha256 and self.dev_manifest_sha256 != J0_DEV64_SHA256:
            raise ValueError(
                f"pos_dev_manifest_sha256 declares {self.dev_manifest_sha256}, but the DEV instrument is "
                f"pinned to {J0_DEV64_SHA256}. The config declares which manifest a run intends; it cannot "
                f"choose what the instrument will accept."
            )
        return load_dev_cohort(self.dev_manifest)

    def checkpoint_managers(self):
        if not self.checkpoint_dir:
            return None, None
        return build_checkpoint_manager(self.checkpoint_dir), build_selection_manager(self.checkpoint_dir)

    def running_context(self) -> ProbeContext:
        """This job's provenance, DERIVED here rather than read off the artifact being checked.

        The whole point of the pass-3 rework: training derives its own context — the code SHA of the
        checkout it is running from, the resolved model snapshot on this machine, the devices this
        process actually holds, the tensor geometry and the footprint-bearing recipe — and then
        requires the authorization to have been measured on exactly that. Two independent
        derivations of the same program, compared; not a claim, believed.
        """
        return derive_probe_context(self.config)

    def authorized_cell(self, context: ProbeContext | None = None) -> FitCell:
        """The (arm, microbatch, k) this config asks for — and M1's proof that it was measured.

        This is the structural half of plan §4-P1: **M1 authorizes only the cells it measured**, and
        this is the only door into a training run, so an unmeasured cell has no route rather than a
        discouraged one. A 64-chip job that discovers a miss after the reservation is the failure
        this prevents; exp_03's C arm missed fit by 0.1% and that is the precedent.
        """
        if not self.fit_authorization:
            raise ValueError(
                "pos_fit_authorization must name M1's published fit-probe authorization: a training "
                "run executes only an (arm, microbatch, k) cell M1 measured and found to fit (plan §4-P1)"
            )
        cell = FitCell(
            arm=str(self.schedule.arm),
            microbatch=int(self.schedule.microbatch),
            k_b=int(self.schedule.k_b),
        )
        assert_cell_authorized(
            load_authorization(self.fit_authorization),
            cell,
            context=self.running_context() if context is None else context,
        )
        return cell

    def assert_paired_recipe(self) -> dict:
        """Publish or adopt the run's recipe lock, so the pair differs by the objective and nothing else."""
        if not self.recipe_lock:
            raise ValueError(
                "pos_recipe_lock must name this run's recipe lock: matched-C0 is a control only if the two "
                "arms share every value outside the arm and its derived destinations, and the lock is what "
                "makes a divergent second arm refuse to start (Yixun's decision 1)"
            )
        return publish_recipe_lock(self.recipe_lock, self.config, arm=str(self.schedule.arm))

    def resume_source(self, context: ProbeContext | None = None) -> dict | None:
        """The immutable INPUT a retry adopts — never this attempt's own output tree.

        Returns ``None`` for the first attempt of a run, which is the honest answer: there is nothing
        complete to resume from yet. Returning the fresh output tree instead is what made the old
        launcher reuse one mutable directory across attempts and arms.
        """
        if not self.resume_parent:
            raise ValueError(
                "pos_resume_parent must name this arm's attempts root: a resume adopts the latest COMPLETE "
                "publication under it, which is a different thing from the fresh attempt's output tree "
                "(issue #13)"
            )
        if self.checkpoint_dir and self.checkpoint_dir.rstrip("/") == self.resume_parent.rstrip("/"):
            raise ValueError(
                f"the resume input and this attempt's output are the same root ({self.checkpoint_dir}); one "
                f"mutable tree shared by every attempt is exactly the arrangement that let a retry inherit "
                f"a half-written state"
            )
        derived = self.running_context() if context is None else context
        return select_resume_publication(self.resume_parent, code_sha=derived.code_sha, arm=str(self.schedule.arm))

    def start_training(self) -> None:
        # Everything configuration can decide is decided and validated first, so a misconfigured run
        # fails in seconds rather than after a pipeline load. Order: the cheap manifest binding, then
        # M1's gate (which resolves devices and the model snapshot), then the pair lock — so no
        # artifact is published for a job M1 would have refused — then the resume adoption.
        self.load_dev_cohort()
        context = self.running_context()
        self.authorized_cell(context)
        self.assert_paired_recipe()
        self.resume_source(context)
        raise NotImplementedError(
            "exp_06's model/data wiring is not part of T4 (dispatch + config). The schedule, the DEV "
            "cohort binding, M1's authorization, the run's recipe lock, the resume adoption and the "
            "checkpoint trees resolve; still to come are the pipeline load, the TFRecord iterator, the "
            "sharded state and the optimizer — the WanTI2VSideAdapterTrainer seams the plan §3 names. "
            "Until then this arm dispatches and validates but does not train."
        )
