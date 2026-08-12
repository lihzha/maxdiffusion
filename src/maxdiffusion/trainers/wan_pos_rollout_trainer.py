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

**W2 — the model/data wiring (review A2).** :meth:`start_training` now runs the job. The four things
it needs that only the settled production trainer has — the pipeline load under ``axis_rules``, the
deployed null context, the deployed scheduler and the deployed TFRecord iterator — are taken from
``WanTI2VSideAdapterTrainer`` *as they are*, and everything else is built from the SHARED factories
M1 also calls (``build_adapter_stack`` / ``build_optimizer`` / ``build_logical_update``), so
"the probe measured the trainer's step" is a fact about the call graph. The settled file is read,
never edited.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Mapping


from maxdiffusion.pos_rollout_arms import ARMS, PRODUCTION_DROPOUT_RATE, ArmContext
from maxdiffusion.pos_rollout_dev_instrument import J0_DEV64_SHA256, load_dev_cohort, score_dev_cohort
from maxdiffusion.pos_rollout_fit_probe import (
    FitCell,
    ProbeContext,
    assert_cell_authorized,
    derive_probe_context,
    load_authorization,
)
from maxdiffusion.pos_rollout_loop import (
    DEV_METRIC,
    LoopSchedule,
    RolloutTrainState,
    RunReport,
    build_checkpoint_manager,
    build_selection_manager,
    read_checkpoint_json,
    restore_checkpoint,
    restore_eval_history,
    run_loop,
    save_checkpoint,
)
from maxdiffusion.pos_rollout_support import is_remote, storage_exists, storage_read_bytes, storage_write_bytes
from maxdiffusion.pos_rollout_update import (
    LoadedBackbone,
    TrainingProgram,
    arm_context,
    build_training_program,
    load_backbone,
)
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
    # `context_digest` is REQUIRED (review F5c, BLOCKER 2). The field has been recorded since T6-3
    # and demanded by nothing, so a marker written by an older publisher -- or by hand -- read as
    # adoptable while carrying no identity at all. Two git-less deployments can share a `COMMIT`
    # label and differ in their running bytes, and this is the field that separates them.
    for field in ("attempt", "arm", "code_sha", "context_digest", "checkpoint_dir"):
        if not str(payload.get(field, "")):
            raise ValueError(f"{path}: a publication marker without {field} cannot be adopted")
    if payload.get("complete") is not True:
        raise ValueError(f"{path}: this attempt is not marked complete, so there is nothing to resume from")
    return {**dict(payload), "sha256": stored["sha256"]}


def describe_resume_candidates(parent: str, *, code_sha: str, arm: str) -> list[dict]:
    """Every COMPLETE publication for this arm at this SHA — a REPORT, never a decision (F5c).

    The launcher's preflight runs AFTER the HF prefetch but BEFORE distributed initialization and the
    model load, so ``jax.devices()`` would report that host's LOCAL chip count -- wrong by a factor of
    eight on a v6e-64 job -- and the config has not been through ``pyconfig`` there, so the recipe
    fingerprint is unavailable. It therefore cannot derive the context that decides adoption. It gets
    this instead: the candidates, each carrying the context it was measured under, and no claim about
    which one (if any) the running process will accept. A preflight that predicted adoption from a
    commit label would be the very error :func:`select_resume_publication` was just fixed for.
    """
    found = []
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
        if str(publication["arm"]) == str(arm) and str(publication["code_sha"]) == str(code_sha):
            found.append(publication)
    return sorted(found, key=lambda entry: (int(entry.get("step", 0)), str(entry["attempt"])))


def select_resume_publication(parent: str, *, code_sha: str, arm: str, context_digest: str) -> dict | None:
    """The latest COMPLETE publication for THIS arm measured by THIS PROGRAM.

    Four filters, each one a way a resume silently corrupts a run: an incomplete attempt (adopting a
    half-written tree), another arm's attempt (matched-C0 restoring R-B's parameters — the reviewer's
    scientifically critical finding), another commit's attempt, and — since review F5c — another
    CONTEXT's attempt. Anything unreadable is skipped rather than fatal: a damaged old attempt must
    not stop a run that has a good one.

    **Why the fourth filter exists.** ``code_sha`` is a commit LABEL, and on a git-less worker it is
    whatever ``COMMIT`` said; the reviewer published two same-SHA attempts whose contexts differed and
    watched this selector take the higher-step foreign one, so one deployment could resume another's
    optimizer state. ``context_digest`` is the same digest cell adoption compares — code manifest,
    model revision, device topology, geometry and recipe — and it is a REQUIRED keyword precisely
    because the identity was lost by being droppable.
    """
    if not str(context_digest or ""):
        raise ValueError(
            "select_resume_publication needs the digest of the context THIS process derived: a resume that "
            "matches only a commit label adopts state built by a program it cannot identify (review F5c)"
        )
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
        if str(publication["context_digest"]) != str(context_digest):
            continue
        candidates.append(publication)
    if not candidates:
        return None
    return max(candidates, key=lambda entry: (int(entry.get("step", 0)), str(entry["attempt"])))


# =================================================================================================
# The trainer.
# =================================================================================================


class WanPosRolloutTrainer:
    """Config in, loop arguments out — and, since W2, the loop itself."""

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

    def authorized_context(self) -> ProbeContext:
        """M1's gate, as ONE statement, so it can be the FIRST one `start_training` executes.

        Deriving the running context is not a step BEFORE the gate — it IS the gate's first half:
        there is nothing to check an authorization against until this process has said what program
        it is. Returning it means the gate happens once and the rest of the run reuses the answer.
        """
        context = self.running_context()
        self.authorized_cell(context)
        return context

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
        # The WHOLE derived context, not one field of it (review F5c, BLOCKER 2): this method used to
        # derive the complete identity and then hand the selector `code_sha` alone.
        return select_resume_publication(
            self.resume_parent,
            code_sha=derived.code_sha,
            arm=str(self.schedule.arm),
            context_digest=derived.digest(),
        )

    def assert_loader_yields_the_logical_batch(self) -> int:
        """The iterator must hand the loop EXACTLY ``pos_logical_batch`` examples per step.

        ``checked_logical_batch`` already refuses a wrong-width batch — but it refuses it at step 1,
        with the 5B on device and the reservation spent. The width is a CONFIGURATION fact
        (``pyconfig`` derives ``global_batch_size_to_load = device_count x per_device_batch_size``),
        so it is decidable here, in seconds, and the refusal says which knob to turn.
        """
        loaded = int(optional_config_value(self.config, "global_batch_size_to_load", 0))
        if loaded != int(self.schedule.logical_batch):
            raise ValueError(
                f"the input pipeline loads {loaded} examples per step but pos_logical_batch is "
                f"{self.schedule.logical_batch}. pyconfig derives global_batch_size_to_load = "
                f"device_count x per_device_batch_size, so set per_device_batch_size = "
                f"{self.schedule.logical_batch}/device_count for this job. Accumulation preserves the "
                f"logical batch and never adapts to what the loader hands it (plan §3b), so a run that "
                f"started would die at step 1 with the pipeline already loaded."
            )
        return loaded

    def load_backbone(self) -> LoadedBackbone:
        """The frozen 5B and the deployed null context/scheduler/mesh — THE shared loader (W3).

        The body moved to ``pos_rollout_update.load_backbone`` so that M1 enters the same one. It was
        already the settled trainer's four seams; what changed is that there is now exactly one
        caller-visible way to reach them, and the fit probe uses it too.
        """
        return load_backbone(self.config)

    def arm_context(self, backbone: LoadedBackbone) -> ArmContext:
        """The one context BOTH arms and BOTH PATHS are handed — the shared builder (W3)."""
        return arm_context(self.config, backbone, k_b=int(self.schedule.k_b), num_steps=int(self.schedule.num_steps))

    def build_program(self, backbone: LoadedBackbone) -> TrainingProgram:
        """THE program — adapter, optimizer, shardings, axis-rule scope and jitted step.

        Every one of those is now produced by ``build_training_program``, which M1 also calls. The
        reviewer's blocker was that sharing the three factories did not make the two sides compile the
        same thing: the trainer replicated its parameters and compiled under the logical axis rules
        while M1 did neither, so M1 could report a per-device peak for a program training never runs.
        The boundary moved up a level and this method is now a thin call, which is the point.
        """
        return build_training_program(
            self.config,
            backbone,
            arm=str(self.schedule.arm),
            k_b=int(self.schedule.k_b),
            num_steps=int(self.schedule.num_steps),
            dropout_rate=self.dropout_rate,
        )

    def batches(self, backbone: LoadedBackbone):
        """``run_loop``'s iterator factory: a fresh TFRecord iterator opened at the seed it is given.

        The seed IS the dataloader position (``resume_seed``), because Orbax serializes params,
        opt_state and step but not the loader's cursor; the loop derives it and this factory must
        pass it straight through to the settled loader rather than substituting ``config.seed``.
        """

        def open_at(seed):
            from maxdiffusion.train_utils import load_next_batch

            iterator = backbone.loaders._load_dataset(backbone.mesh, is_training=True, seed=int(seed))
            while True:
                yield load_next_batch(iterator, None, self.config)

        return open_at

    def dev_metric_fn(self, cohort, program: TrainingProgram):
        """plan §3d's selection estimand: the fixed-draw DEV-64 instrument, never the train stream.

        The instrument owns the draws (a pinned ``(support, epsilon, t_idx)`` per example, from its
        own seed and purpose) and it owns the batches (its own manifest-bound reader). This function
        supplies only the parameters being scored, so two checkpoints differ in their scores by their
        parameters and by nothing else.
        """

        def measure(state, global_step) -> float:
            measured = score_dev_cohort(
                cohort,
                program.dev_loss_fn,
                params=state.params,
                context=program.context,
                example_shape=program.example_shape,
                eval_index=int(global_step),
                arm=str(self.schedule.arm),
            )
            return float(measured["metric"])

        return measure

    def attempt_identity(self) -> tuple[str, str]:
        """``(attempts parent, attempt id)`` DERIVED from the tree this job is actually writing to.

        The launcher builds ``<pos_resume_parent>/<attempt>/checkpoints``, so the attempt id is a
        fact about where the checkpoints land rather than a second value travelling beside them —
        stamped-≠-bound applied to the publication marker. Any other shape is REFUSED: a marker
        naming an attempt whose tree it does not describe is how a resume adopts the wrong state.
        """
        head, _, leaf = self.checkpoint_dir.rstrip("/").rpartition("/")
        parent, _, attempt = head.rpartition("/")
        if (
            leaf != "checkpoints"
            or not _ATTEMPT_RE.match(attempt)
            or parent.rstrip("/") != self.resume_parent.rstrip("/")
        ):
            raise ValueError(
                f"checkpoint_dir {self.checkpoint_dir!r} is not this attempt's derived output root: "
                f"exp_06 writes <pos_resume_parent>/<att-...>/checkpoints (issue #13), and the "
                f"publication marker names the attempt it can be read out of. pos_resume_parent is "
                f"{self.resume_parent!r}."
            )
        return parent, attempt

    def publish_this_attempt(self, *, context: ProbeContext, step: int) -> dict:
        """Mark THIS attempt COMPLETE — last, after the loop's own eval-boundary saves.

        The marker is what ``select_resume_publication`` adopts, and it is written once, after the
        checkpoints it describes exist, so a retry can never half-adopt a tree that was still being
        written (issue #10/#13).
        """
        parent, attempt = self.attempt_identity()
        return publish_attempt(
            parent,
            attempt=attempt,
            arm=str(self.schedule.arm),
            code_sha=str(context.code_sha),
            context_digest=context.digest(),
            step=int(step),
            checkpoint_dir=self.checkpoint_dir,
        )

    def adopt_publication(self, publication: Mapping, *, manager, selection_manager, template) -> RolloutTrainState:
        """Copy the adopted attempt's COMPLETE state into THIS attempt's fresh roots.

        Plan §4's runbook rule is "SHA-bound adoption of the latest COMPLETE published checkpoint
        **into a fresh attempt root** on resume", and this is that sentence as code. The published
        tree is opened READ-ONLY and its state, its eval history and its selection artifact are
        written into this attempt's own trees; the loop then restores from its own root and every
        later decision — the stop rule's history, the reconciliation, the retention — belongs to one
        self-contained attempt. Nothing is ever written back into the attempt being adopted.
        """
        source = str(publication["checkpoint_dir"])
        adopted = build_checkpoint_manager(source)
        restored, step = restore_checkpoint(adopted, template)
        history = restore_eval_history(adopted)
        if step <= 0:
            return template
        save_checkpoint(
            manager,
            restored,
            dev_metric=history[-1].dev_metric if history else None,
            history=history,
            arm=str(self.schedule.arm),
            k_b=int(self.schedule.k_b),
        )
        source_selection = build_selection_manager(source)
        incumbent = source_selection.latest_step()
        if selection_manager is not None and incumbent is not None and history:
            # The sibling travels WITH the resume tree: without it, a resume whose historical best is
            # an earlier step fails `reconcile_selection` closed at startup — correctly, because the
            # parameters that could repair it are gone.
            selected, _ = restore_checkpoint(source_selection, template)
            save_checkpoint(
                selection_manager,
                selected,
                dev_metric=read_checkpoint_json(source_selection, incumbent).get(DEV_METRIC),
                history=history,
                arm=str(self.schedule.arm),
                k_b=int(self.schedule.k_b),
            )
        print(f"[pos-rollout] adopted attempt {publication['attempt']} at step {step} from {source}")
        return restored

    def start_training(self) -> RunReport:
        # Everything configuration can decide is decided and validated first, so a misconfigured run
        # fails in seconds rather than after a pipeline load. M1's gate is LITERALLY FIRST (final
        # review, LOW 5): the DEV manifest used to load ahead of it, which left the stated ordering
        # and the test claim disagreeing with the code by one line. Then the DEV binding, then the
        # pair lock — so no artifact is published for a job M1 would have refused — then the resume
        # adoption, then the loader width. NOTHING below this block runs for a job M1 did not
        # authorize, and nothing above it costs more than a hash.
        context = self.authorized_context()
        cohort = self.load_dev_cohort()
        self.assert_paired_recipe()
        publication = self.resume_source(context)
        self.assert_loader_yields_the_logical_batch()

        backbone = self.load_backbone()
        program = self.build_program(backbone)
        manager, selection_manager = self.checkpoint_managers()

        state = RolloutTrainState(params=program.params, opt_state=program.opt_state, step=0)
        if publication is not None and manager is not None:
            state = self.adopt_publication(
                publication, manager=manager, selection_manager=selection_manager, template=state
            )

        print(
            f"[pos-rollout] arm={self.schedule.arm} k={self.schedule.k_b} "
            f"logical_batch={self.schedule.logical_batch} microbatch={self.schedule.microbatch} "
            f"steps={self.schedule.max_train_steps} eval_every={self.schedule.eval_every}"
        )
        report = run_loop(
            state,
            self.schedule,
            batches=self.batches(backbone),
            update_fn=program.update_fn,
            dev_metric_fn=self.dev_metric_fn(cohort, program),
            manager=manager,
            selection_manager=selection_manager,
        )
        if manager is not None:
            # LAST, and only on a normal return: the marker is what "this attempt is complete and may
            # be adopted" means, so it must not exist for a tree that is still being written.
            self.publish_this_attempt(context=context, step=int(report.state.step))
        print(
            f"[pos-rollout] finished at step {int(report.state.step)}: {report.verdict.reason}; "
            f"selected step {report.retained_step}"
        )
        return report
