"""Build the committed exp_02 episode manifest (plan v4 §2 D1/D2/D5).

exp_02 memorizes the windows of 100 successful DROID trajectories. *Which* 100, and which
of each episode's up-to-three language annotations, is the experiment's identity -- so it is
frozen here, once, into `overfit100_manifest.json` with complete provenance rather than being
re-derived (the source bucket was still filling during the plan probes, so "seed 0" alone is
not a reproducibility claim).

Selection (D1): walk `numpy.random.default_rng(selection_seed).permutation(69723)` and accept an
episode iff `success == 1`, it has >= 1 non-empty `texts` entry, its view-0 MP4 exists, and that
MP4 has >= 33 frames. Stop at 100 accepted. Every drawn candidate -- accepted or not -- is
recorded in the ordered draw log with its rejection reason.

Instruction (D2): among the NON-EMPTY texts only, pick with `fold_in(selection_seed, episode_id)`,
so the choice is stable per episode and independent of call order. The full `texts` list, the
chosen index into it, the raw text and the stripped `used_text` are all recorded.

Provenance (D5): per episode, the GCS generation/md5/size of BOTH the annotation JSON and the
MP4, plus ffprobe geometry; globally, the builder commit, tool versions and the V1-fixture
fingerprint (from `extract_v1_fixture`).

Integrity contracts added after the Codex cycle-A review:

* **A1 -- the manifest must name the code that built it.** A production build refuses to run
  unless every implementation file is committed and clean at HEAD (`assert_implementation_committed`),
  and it records ffmpeg/numpy/jax versions alongside python/gsutil/ffprobe.
* **A2 -- content is bound to the recorded fingerprint.** Objects are statted FIRST, then
  downloaded at that exact generation (`uri#generation`) and re-hashed against the stat before
  the bytes are used; the annotation's embedded `episode_id` must equal the drawn candidate.
* **A3 -- absence and failure are different facts.** Per-object outcomes are `found` /
  `absent` / `error`; only a confirmed absence becomes a rejection reason, and any error aborts
  the build (`SourceError`). Errors on candidates prefetched past the stopping acceptance are
  never consumed and therefore cannot influence the walk.
* **A4 -- verification fails closed.** `validate_manifest_structure` is a pure structural gate
  (required keys, contiguous indices, draw-log/tally/totals reconciliation, non-empty chosen
  text, URI patterns, `n_windows` vs counted frames) that runs BEFORE any remote stat.

CLI:
    python -m maxdiffusion.data_preprocessing.build_overfit100_manifest \
        --seed 0 --n 100 \
        --out docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json \
        --fixture-fingerprint docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_fixture_fingerprint.json
    # --dry-run stops at the annotation-level decisions (no video stats, no MP4s, nothing written)

IO is deliberately thin and SEQUENTIAL (`gsutil` without `-m`, which stalls on the build host).
Stats and annotation downloads are batched one BLOCK of candidates per invocation -- a pure
invocation-count optimization: the accept/stop walk stays strictly ordered and MP4s are fetched
only for candidates the walk actually consumes, so the draw log is identical to a
one-candidate-at-a-time walk (asserted by `test_block_size_never_changes_the_walk`).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import jax
import numpy as np

from maxdiffusion.data_preprocessing.extract_v1_fixture import (
    STATUS_ABSENT,
    STATUS_ERROR,
    STATUS_FOUND,
    TARGET_NAMES,
    Resolved,
    gsutil_stat_many,
    pinned_uri,
    run_gsutil,
    verify_payload_binding,
)

DATASET_ROOT = "gs://v6_east1d/datasets/droid_ctrl_world_aligned"
DEFAULT_VAE_REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
# The launcher stamps the sha it verified clean into the job environment; workers that run
# from an uploaded tarball (no .git) can only relay it. See `deployed_code_commit`.
DEPLOYED_COMMIT_ENV = "COMMIT"

# T1 (tarball-guard review): every variable from `git help environment` that can redirect
# WHICH repository / worktree / index / object store git answers about. Inheriting them lets
# an ambient `GIT_DIR=/nowhere` make a real, DIRTY worktree look like a deployed tarball,
# downgrading the clean-commit guard to "trust whatever COMMIT says". Stripped from every git
# subprocess this module runs.
GIT_ENV_STRIP = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_INDEX_FILE",
    "GIT_INDEX_VERSION",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
    "GIT_TOPLEVEL",
)
GIT_MARKER = ".git"  # a DIRECTORY in a normal clone, a FILE in a linked worktree or submodule

N_EPISODES = 69723  # annotation ids 0..69722
VIEW_INDEX = 0  # Query 2: one camera view per episode, the first exterior view
MIN_FRAMES = 33  # one window = 33 consecutive frames -> 9 latent frames
WINDOW_STRIDE = 4  # window starts 0, 4, 8, ...
DEFAULT_BLOCK_SIZE = 25  # candidates whose annotations/stats are fetched per gsutil invocation
VERIFY_CHUNK = 50  # objects per `gsutil stat` invocation during verification

# The corpus geometry proven over all 100 accepted episodes; pinned so a manifest that silently
# changed resolution/fps cannot pass verification (plan D4 assumes exactly this).
EXPECTED_WIDTH, EXPECTED_HEIGHT = 320, 192
EXPECTED_FPS = 5.0
EXPECTED_PIX_FMT = "yuv420p"

REASONS = ("missing_annotation", "not_success", "no_nonempty_text", "missing_video", "too_short", "accepted")

IMPLEMENTATION_PATHS = (
    "src/maxdiffusion/data_preprocessing/build_overfit100_manifest.py",
    "src/maxdiffusion/data_preprocessing/extract_v1_fixture.py",
    "src/maxdiffusion/tests/worklogs_yixun/test_overfit100_fixture.py",
    "src/maxdiffusion/tests/worklogs_yixun/test_overfit100_manifest.py",
    "src/maxdiffusion/tests/worklogs_yixun/test_overfit100_selection.py",
)

FFPROBE_ARGS = (
    "-v",
    "error",
    "-select_streams",
    "v:0",
    "-count_frames",
    "-show_entries",
    "stream=width,height,nb_frames,nb_read_frames,r_frame_rate,pix_fmt",
    "-of",
    "json",
)

_TOP_LEVEL_KEYS = {
    "selection_seed",
    "builder_commit",
    "created_utc",
    "tool_versions",
    "provisional",
    "fixture",
    "vae_fingerprint",
    "episodes",
    "draw_log",
    "rejection_tally",
    "totals",
}
_EPISODE_KEYS = {
    "episode_index",
    "episode_id",
    "texts",
    "chosen_text_index",
    "chosen_text_raw",
    "used_text",
    "annotation_fingerprint",
    "video_fingerprint",
    "ffprobe",
    "n_windows",
}
_FINGERPRINT_KEYS = {"uri", "generation", "md5", "size"}
_FIXTURE_KEYS = {"uri", "generation", "md5", "size_bytes", "names", "shapes", "dtypes"}
_FFPROBE_KEYS = {"width", "height", "nb_frames", "fps", "pix_fmt"}
_TOOL_KEYS = ("python", "gsutil", "ffprobe", "ffmpeg", "numpy", "jax")
# B1 (cycle-B review): the VAE that encodes the latents is part of the dataset's identity.
VAE_PIN_KEYS = ("hf_repo", "revision", "vae_config_sha256")
_AMENDMENT_KEYS = {"reason", "date", "commit", "fields"}
VAE_CONFIG_RELPATH = ("vae", "config.json")

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class SourceError(RuntimeError):
    """A source object could not be resolved or failed its binding -- abort, never record."""


class DirtyImplementationError(RuntimeError):
    """The builder is uncommitted/modified, so `builder_commit` could not honestly name it."""


# ----------------------------------------------------------------------------------
# Pure selection rules (D1 / D2 / D4) -- no IO
# ----------------------------------------------------------------------------------


def annotation_uri(episode_id: int) -> str:
    return f"{DATASET_ROOT}/annotation/train/{int(episode_id)}.json"


def video_uri(episode_id: int, view: int = VIEW_INDEX) -> str:
    return f"{DATASET_ROOT}/videos/train/{int(episode_id)}/{int(view)}.mp4"


def candidate_order(selection_seed: int, n_episodes: int = N_EPISODES) -> np.ndarray:
    """The locked draw order: a seeded permutation of every episode id, without replacement."""
    return np.random.default_rng(selection_seed).permutation(n_episodes)


def nonempty_text_indices(texts: Sequence[str]) -> list[int]:
    """Indices of the annotations that survive the empty-instruction filter (Query 1)."""
    return [i for i, text in enumerate(texts) if isinstance(text, str) and text.strip() != ""]


def pick_instruction_index(selection_seed: int, episode_id: int, texts: Sequence[str]) -> int:
    """Pick one instruction among the NON-EMPTY texts; returns its index in the original list."""
    candidates = nonempty_text_indices(texts)
    if not candidates:
        raise ValueError(f"episode {episode_id} has no non-empty texts to pick from")
    key = jax.random.fold_in(jax.random.key(int(selection_seed)), int(episode_id))
    return int(candidates[int(jax.random.randint(key, (), 0, len(candidates)))])


def decide_candidate(
    annotation: dict | None,
    video_exists: bool | None = None,
    nb_frames: int | None = None,
) -> tuple[bool, str]:
    """Acceptance predicate. Probe results left as ``None`` yield a PROVISIONAL accept (dry run)."""
    if annotation is None:
        return False, "missing_annotation"
    if int(annotation.get("success", 0)) != 1:
        return False, "not_success"
    if not nonempty_text_indices(annotation.get("texts") or []):
        return False, "no_nonempty_text"
    if video_exists is False:
        return False, "missing_video"
    if nb_frames is not None and int(nb_frames) < MIN_FRAMES:
        return False, "too_short"
    return True, "accepted"


def n_windows(nb_frames: int) -> int:
    """Number of window starts `s = 0, 4, 8, ...` with `s + 33 <= nb_frames`."""
    nb_frames = int(nb_frames)
    if nb_frames < MIN_FRAMES:
        raise ValueError(f"clip of {nb_frames} frames is shorter than one window ({MIN_FRAMES})")
    return 1 + (nb_frames - MIN_FRAMES) // WINDOW_STRIDE


def parse_ffprobe_json(text: str) -> dict:
    """Pull geometry + the COUNTED frame total out of `ffprobe -count_frames ... -of json`."""
    streams = json.loads(text).get("streams") or []
    if not streams:
        raise ValueError("ffprobe reported no video stream")
    stream = streams[0]
    # Every miss raises ValueError (not KeyError) so `ManifestIO.probe_video` reports one uniform
    # source error instead of crashing mid-build.
    missing = [key for key in ("width", "height", "r_frame_rate", "pix_fmt") if key not in stream]
    if missing:
        raise ValueError(f"ffprobe stream is missing {missing}")
    frames = stream.get("nb_read_frames", stream.get("nb_frames"))
    if frames is None:
        raise ValueError("ffprobe reported neither nb_read_frames nor nb_frames")
    numerator, _, denominator = str(stream["r_frame_rate"]).partition("/")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "nb_frames": int(frames),
        "fps": float(numerator) / float(denominator or 1),
        "pix_fmt": str(stream["pix_fmt"]),
    }


def verify_annotation_binding(episode_id: int, annotation) -> list[str]:
    """The annotation must actually describe the drawn candidate (Codex review A2)."""
    if not isinstance(annotation, dict):
        return [f"episode {episode_id}: annotation is not a JSON object"]
    errors: list[str] = []
    embedded = annotation.get("episode_id")
    if embedded is None:
        errors.append(f"episode {episode_id}: annotation carries no embedded episode_id")
    else:
        try:
            if int(embedded) != int(episode_id):
                errors.append(f"episode {episode_id}: annotation embedded episode_id is {embedded}")
        except (TypeError, ValueError):
            errors.append(f"episode {episode_id}: annotation embedded episode_id {embedded!r} is not an int")
    if "success" not in annotation:
        errors.append(f"episode {episode_id}: annotation carries no success field")
    if not isinstance(annotation.get("texts", []), list):
        errors.append(f"episode {episode_id}: annotation texts is not a list")
    return errors


# ----------------------------------------------------------------------------------
# VAE pin (B1) -- which weights encoded the dataset is part of the manifest's identity
# ----------------------------------------------------------------------------------


def vae_pin_errors(pin) -> list[str]:
    """Fail-closed shape gate for `manifest["vae_fingerprint"]` (empty list = usable)."""
    if not isinstance(pin, dict):
        return ["vae_fingerprint is missing or is not an object"]
    errors: list[str] = []
    extra = set(pin) - set(VAE_PIN_KEYS)
    missing = set(VAE_PIN_KEYS) - set(pin)
    if missing:
        errors.append(f"vae_fingerprint is missing {sorted(missing)}")
    if extra:
        # Exactly these three keys: a mutable field (e.g. a machine-local snapshot path)
        # inside the pin would make the manifest's identity host-dependent.
        errors.append(f"vae_fingerprint carries unexpected keys {sorted(extra)}")
    if not str(pin.get("hf_repo") or "").strip():
        errors.append("vae_fingerprint.hf_repo is empty")
    revision = str(pin.get("revision") or "")
    if not _SHA_RE.fullmatch(revision):
        errors.append(f"vae_fingerprint.revision {revision!r} is not a 40-hex commit sha")
    digest = str(pin.get("vae_config_sha256") or "")
    if not _SHA256_RE.fullmatch(digest):
        errors.append(f"vae_fingerprint.vae_config_sha256 {digest!r} is not a 64-hex sha256")
    return errors


def vae_config_sha256(path: str | Path) -> str:
    """sha256 of the VAE `config.json` bytes -- the architecture half of the pin."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_vae_snapshot(hf_repo: str, revision: str | None = None, local_files_only: bool = False) -> dict:
    """Resolve a repo (+ optional pinned revision) to ONE local snapshot directory.

    Returns `{"snapshot_path": str, "pin": {hf_repo, revision, vae_config_sha256}}`. The
    caller passes `snapshot_path` to BOTH the fingerprint check and the model loader, so the
    weights that get loaded are the fingerprinted ones by construction (B1).

    A local directory cannot prove a revision, so the caller's `revision` is echoed back --
    that path exists for tests and for pre-staged snapshots, never for the production build.
    """
    local = Path(hf_repo)
    if local.is_dir():
        snapshot_path, resolved_revision = local, revision
    else:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(
            snapshot_download(
                repo_id=hf_repo,
                revision=revision,
                allow_patterns=["model_index.json", "vae/*"],  # VAE-only job: nothing else is fetched
                local_files_only=local_files_only,
            )
        )
        parts = snapshot_path.parts  # NOT resolved: snapshots/<revision>/ is a symlink farm
        resolved_revision = parts[parts.index("snapshots") + 1] if "snapshots" in parts else revision
    config_path = snapshot_path.joinpath(*VAE_CONFIG_RELPATH)
    if not config_path.is_file():
        raise FileNotFoundError(f"{snapshot_path}: no {'/'.join(VAE_CONFIG_RELPATH)} in the resolved snapshot")
    return {
        "snapshot_path": str(snapshot_path),
        "pin": {
            "hf_repo": str(hf_repo),
            "revision": str(resolved_revision or ""),
            "vae_config_sha256": vae_config_sha256(config_path),
        },
    }


def amend_manifest_vae_pin(manifest: dict, pin: dict, *, reason: str, commit: str, now: str) -> dict:
    """Return a copy of `manifest` carrying `vae_fingerprint` plus an explicit amendment log.

    Pure and idempotent: re-applying the SAME pin is a no-op, and a DIFFERENT pin raises --
    an amendment may add provenance the artifact was missing, never silently restate what the
    dataset was built against. Selection content (episodes/draw log/totals) is untouched.
    """
    errors = vae_pin_errors(pin)
    if errors:
        raise ValueError("refusing to amend with a malformed VAE pin:\n  " + "\n  ".join(errors))
    existing = manifest.get("vae_fingerprint")
    if existing == pin:
        return copy.deepcopy(manifest)
    if existing is not None:
        raise ValueError(f"manifest is already pinned to {existing!r}; refusing to replace it with {pin!r}")
    amended = copy.deepcopy(manifest)
    amended["vae_fingerprint"] = copy.deepcopy(pin)
    amended["amended"] = list(amended.get("amended") or []) + [
        {"reason": str(reason), "date": str(now), "commit": str(commit), "fields": ["vae_fingerprint"]}
    ]
    return amended


# ----------------------------------------------------------------------------------
# Builder provenance (A1)
# ----------------------------------------------------------------------------------


def parse_git_porcelain(text: str) -> list[str]:
    """Paths reported dirty by `git status --porcelain` (untracked, modified, staged, renamed)."""
    paths: list[str] = []
    for line in (text or "").splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if " -> " in path:  # rename: the post-rename path is the one that matters
            path = path.split(" -> ", 1)[1].strip()
        if path:
            paths.append(path.strip('"'))
    return paths


def implementation_provenance_errors(
    tracked_at_head: Iterable[str],
    dirty_paths: Iterable[str],
    paths: Iterable[str] = IMPLEMENTATION_PATHS,
) -> list[str]:
    """Every implementation file must exist at HEAD and carry no local modifications."""
    tracked = set(tracked_at_head)
    dirty = set(dirty_paths)
    errors: list[str] = []
    for path in paths:
        if path not in tracked:
            errors.append(f"{path} is not committed at HEAD")
        if path in dirty:
            errors.append(f"{path} has uncommitted changes")
    return errors


def sanitized_git_env(env: dict | None = None) -> dict:
    """`env` (default `os.environ`) without any git repository/worktree/index selector (T1)."""
    sanitized = dict(os.environ if env is None else env)
    for key in GIT_ENV_STRIP:
        sanitized.pop(key, None)
    return sanitized


def _git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
        env=sanitized_git_env(),
    )
    return proc.stdout


def is_git_worktree(repo_root: str | Path) -> bool:
    """Is this directory inside a real git worktree?

    The TPU queue deploys an uploaded code TARBALL with no `.git`, so every git call on the
    worker exits 128 (probe failure 20260729-062523). Detect that explicitly instead of
    letting a `CalledProcessError` surface as a mystery crash mid-build.

    Two hardenings from the tarball-guard review (T1), because THIS ANSWER decides whether the
    dirty check runs at all:

    * discovery runs with a sanitized environment, so an ambient `GIT_DIR` / `GIT_WORK_TREE`
      cannot make a real worktree look deployed (nor let a plain directory borrow a foreign
      repository's answers);
    * a `.git` marker at the root (a DIRECTORY in a clone, a FILE in a linked worktree or
      submodule) whose discovery still fails is FATAL. Code shipped with a repository marker is
      not an uploaded tarball, and "we could not tell" must never resolve to the weaker contract.
    """
    root = Path(repo_root)
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
            env=sanitized_git_env(),
        )
        inside = proc.returncode == 0 and proc.stdout.strip() == "true"
        detail = (proc.stderr or proc.stdout or "").strip()[:400]
    except (OSError, subprocess.SubprocessError) as exc:  # no git binary at all
        inside, detail = False, repr(exc)
    if inside:
        return True
    if (root / GIT_MARKER).exists():
        raise DirtyImplementationError(
            f"{root / GIT_MARKER} exists but git could not open a worktree there ({detail}). "
            "Refusing to fall back to deployed-code mode: code shipped with a repository marker "
            "is not an uploaded tarball, and an unreadable repository cannot prove a clean tree."
        )
    return False


def deployed_code_commit(env: dict | None = None, log: Callable = print) -> str:
    """The launch-time sha for code that arrived as a tarball, or a fail-closed refusal.

    A worker cannot verify cleanliness -- there is no repository to inspect. The honest
    contract is the one the LAUNCHER established: it refused to submit a dirty tree and
    shipped `COMMIT` with the job. Absent or malformed, this raises: the fallback may relay
    provenance, never invent it.
    """
    env = os.environ if env is None else env
    commit = str(env.get(DEPLOYED_COMMIT_ENV) or "").strip()
    if not _SHA_RE.fullmatch(commit):
        raise DirtyImplementationError(
            f"deployed-code mode (no git worktree): {DEPLOYED_COMMIT_ENV} must carry the 40-hex sha the "
            f"launcher verified clean before uploading the tarball; got {commit!r}. "
            f"Pass it through the queue (e.g. `--env {DEPLOYED_COMMIT_ENV}=$(git rev-parse HEAD)`)."
        )
    log(
        f"[provenance] deployed-code mode: clean-commit guard was enforced at launch on the "
        f"submitting machine; {DEPLOYED_COMMIT_ENV}={commit} from env"
    )
    return commit


def head_commit(repo_root: str | Path | None = None, env: dict | None = None, log: Callable = print) -> str:
    """HEAD's sha, or the launcher's `COMMIT` when there is no worktree to ask.

    Records provenance WITHOUT asserting cleanliness -- used where a sha is being written into
    an artifact (e.g. an amendment log) rather than gating a production build.
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    if not is_git_worktree(root):
        return deployed_code_commit(env, log=log)
    return _git(root, "rev-parse", "HEAD").strip()


def assert_implementation_committed(
    repo_root: str | Path | None = None,
    paths: Sequence[str] = IMPLEMENTATION_PATHS,
    env: dict | None = None,
    log: Callable = print,
) -> str:
    """Return the sha this artifact may honestly claim to come from.

    In a git worktree that is HEAD, and only once every implementation file is proven
    committed and clean -- unchanged behavior, and `COMMIT` in the environment is NOT a way to
    bypass it. Where there is no worktree (deployed tarball) it is the launcher's `COMMIT`.

    ``paths`` lets a later cycle reuse the same fail-closed guard for its own implementation
    files (cycle B's dataset builder passes `CYCLE_B_IMPLEMENTATION_PATHS`).
    """
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    if not is_git_worktree(root):
        return deployed_code_commit(env, log=log)
    paths = tuple(paths)
    tracked = {line.strip() for line in _git(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines()}
    dirty = parse_git_porcelain(_git(root, "status", "--porcelain", "--", *paths))
    errors = implementation_provenance_errors(tracked, dirty, paths=paths)
    if errors:
        raise DirtyImplementationError(
            "refusing to build a production artifact from an uncommitted implementation "
            "(the recorded commit would not name the code that ran):\n  " + "\n  ".join(errors)
        )
    return _git(root, "rev-parse", "HEAD").strip()


def _first_line(text: str) -> str:
    return (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""


def _tool_version(*command: str) -> str:
    """First line of a `--version` probe, or "" if the tool is absent/unusable.

    `check=False` suppresses a non-zero EXIT, not a missing executable: without the
    FileNotFoundError guard, writing the provenance summary would crash on a host that simply
    lacks one of these tools (probe attempt 2 -- the TPU image has no ffmpeg/ffprobe).
    """
    try:
        proc = subprocess.run(list(command), capture_output=True, text=True, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return _first_line(proc.stdout) or _first_line(proc.stderr)


def collect_tool_versions() -> dict:
    """Record the versions of every tool/library that shaped this manifest."""
    return {
        "python": platform.python_version(),
        "gsutil": _tool_version("gsutil", "version"),
        "ffprobe": _tool_version("ffprobe", "-version"),
        "ffmpeg": _tool_version("ffmpeg", "-version"),
        "numpy": np.__version__,  # drives candidate_order
        "jax": jax.__version__,  # drives pick_instruction_index
    }


# ----------------------------------------------------------------------------------
# IO layer -- sequential gsutil + ffprobe, every object resolved explicitly
# ----------------------------------------------------------------------------------


class ManifestIO:
    """Resolves annotations / videos to `Resolved` outcomes. One gsutil invocation at a time.

    Order is always stat -> pinned download -> hash check, so the bytes a decision is made from
    are provably the bytes whose fingerprint the manifest records (A2).
    """

    def __init__(self, tmp_dir: str | Path | None = None, view: int = VIEW_INDEX, log: Callable = print):
        self.view = view
        self.log = log
        self.tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="exp02_manifest_"))
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def stat_annotations(self, episode_ids: Iterable[int]) -> dict[int, Resolved]:
        episode_ids = [int(e) for e in episode_ids]
        stats = gsutil_stat_many([annotation_uri(e) for e in episode_ids])
        return {e: stats[annotation_uri(e)] for e in episode_ids}

    def stat_videos(self, episode_ids: Iterable[int]) -> dict[int, Resolved]:
        episode_ids = [int(e) for e in episode_ids]
        stats = gsutil_stat_many([video_uri(e, self.view) for e in episode_ids])
        return {e: stats[video_uri(e, self.view)] for e in episode_ids}

    def fetch_annotations(self, episode_ids: Iterable[int], fingerprints: dict[int, dict]) -> dict[int, Resolved]:
        """Download a block at its statted generation, verify md5 + size, parse JSON."""
        episode_ids = [int(e) for e in episode_ids]
        if not episode_ids:
            return {}
        dest = self.tmp_dir / "annotations"
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        pinned = [pinned_uri(annotation_uri(e), fingerprints[e]) for e in episode_ids]
        proc = run_gsutil(["cp", *pinned, f"{dest}/"], check=False)
        detail = proc.stderr.decode("utf-8", "replace").strip()[:400]

        out: dict[int, Resolved] = {}
        for episode_id in episode_ids:
            uri = annotation_uri(episode_id)
            path = dest / f"{episode_id}.json"
            if not path.exists():
                # The stat proved this generation exists, so a missing file is a FAILURE to
                # fetch -- never evidence of absence.
                out[episode_id] = Resolved.failed(
                    f"{uri}: pinned download produced no file (gsutil exit {proc.returncode}): {detail}"
                )
                continue
            data = path.read_bytes()
            binding = verify_payload_binding(uri, data, fingerprints[episode_id])
            if binding:
                out[episode_id] = Resolved.failed("; ".join(binding))
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                out[episode_id] = Resolved.failed(f"{uri}: downloaded bytes are not valid JSON ({exc})")
                continue
            out[episode_id] = Resolved.found(fingerprints[episode_id], payload=payload)
        shutil.rmtree(dest, ignore_errors=True)
        return out

    def probe_video(self, episode_id: int, fingerprint: dict) -> Resolved:
        """Download the statted generation, verify md5 + size, ffprobe it, delete it."""
        uri = video_uri(episode_id, self.view)
        local = self.tmp_dir / f"ep{int(episode_id)}_v{self.view}.mp4"
        try:
            proc = run_gsutil(["cp", pinned_uri(uri, fingerprint), str(local)], check=False)
            if proc.returncode != 0 or not local.exists():
                detail = proc.stderr.decode("utf-8", "replace").strip()[:400]
                return Resolved.failed(f"{uri}: pinned download failed (gsutil exit {proc.returncode}): {detail}")
            binding = verify_payload_binding(uri, local.read_bytes(), fingerprint)
            if binding:
                return Resolved.failed("; ".join(binding))
            probe = subprocess.run(["ffprobe", *FFPROBE_ARGS, str(local)], capture_output=True, text=True, check=False)
            if probe.returncode != 0:
                return Resolved.failed(f"{uri}: ffprobe exited {probe.returncode}: {probe.stderr.strip()[:400]}")
            try:
                return Resolved.found(fingerprint, payload=parse_ffprobe_json(probe.stdout))
            except (ValueError, json.JSONDecodeError) as exc:
                return Resolved.failed(f"{uri}: unusable ffprobe output ({exc})")
        finally:
            local.unlink(missing_ok=True)


# ----------------------------------------------------------------------------------
# Manifest assembly
# ----------------------------------------------------------------------------------


def _episode_record(index, episode_id, annotation, seed, annotation_fp, video_fp, probe) -> dict:
    texts = list(annotation.get("texts") or [])
    chosen = pick_instruction_index(seed, episode_id, texts)
    raw = texts[chosen]
    return {
        "episode_index": index,
        "episode_id": episode_id,
        "texts": texts,
        "chosen_text_index": chosen,
        "chosen_text_raw": raw,
        "used_text": raw.strip(),
        "annotation_fingerprint": annotation_fp,
        "video_fingerprint": video_fp,
        "ffprobe": probe,
        "n_windows": None if probe is None else n_windows(probe["nb_frames"]),
    }


def _resolve_video(io, episode_id, annotation, video_stats, dry_run):
    """Second-stage decision for a provisionally accepted candidate. Raises on source failure."""
    if dry_run:
        return True, "accepted", None, None
    video_stat = video_stats[episode_id]
    if video_stat.status == STATUS_ERROR:
        raise SourceError(f"episode {episode_id}: video stat unresolved -- {video_stat.error}")
    if video_stat.status == STATUS_ABSENT:
        accepted, reason = decide_candidate(annotation, video_exists=False)
        return accepted, reason, None, None
    probed = io.probe_video(episode_id, video_stat.fingerprint)
    if probed.status != STATUS_FOUND:
        # Seam f: the object exists but could not be read. That is a source failure, not a
        # `missing_video` rejection that would silently shrink the corpus.
        raise SourceError(f"episode {episode_id}: video unreadable -- {probed.error}")
    probe = probed.payload
    accepted, reason = decide_candidate(annotation, video_exists=True, nb_frames=probe["nb_frames"])
    return accepted, reason, video_stat.fingerprint, probe


def build_manifest(
    io,
    *,
    seed: int,
    n_target: int,
    fixture: dict,
    vae_fingerprint: dict,
    builder_commit: str,
    created_utc: str,
    tool_versions: dict,
    order: Sequence[int] | None = None,
    dry_run: bool = False,
    block_size: int = DEFAULT_BLOCK_SIZE,
    log: Callable = print,
) -> dict:
    """Walk the seeded draw order until `n_target` episodes are accepted; return the manifest.

    Per block the annotations are statted and downloaded up front; MP4s are fetched only inside
    the ordered walk, for candidates actually consumed. Any `error` outcome raises `SourceError`
    the moment the walk CONSUMES that candidate -- so a failure on a candidate prefetched past
    the stopping acceptance can never change the result.

    ``order`` overrides the seeded permutation (test seam only -- production passes None).
    """
    pin_errors = vae_pin_errors(vae_fingerprint)
    if pin_errors:
        raise ValueError("refusing to build a manifest without a valid VAE pin:\n  " + "\n  ".join(pin_errors))
    order = [int(e) for e in (candidate_order(seed) if order is None else order)]
    episodes: list[dict] = []
    draw_log: list[dict] = []
    tally: Counter = Counter()
    position = 0

    while len(episodes) < n_target and position < len(order):
        block = order[position : position + block_size]
        position += block_size

        annotation_stats = io.stat_annotations(block)
        present = [e for e in block if annotation_stats[e].status == STATUS_FOUND]
        fetched = (
            io.fetch_annotations(present, {e: annotation_stats[e].fingerprint for e in present}) if present else {}
        )
        provisional = [
            e
            for e in block
            if fetched.get(e) is not None and fetched[e].ok and decide_candidate(fetched[e].payload)[0]
        ]
        video_stats = io.stat_videos(provisional) if (provisional and not dry_run) else {}

        for episode_id in block:
            if len(episodes) >= n_target:
                break

            stat = annotation_stats[episode_id]
            if stat.status == STATUS_ERROR:
                raise SourceError(f"episode {episode_id}: annotation stat unresolved -- {stat.error}")

            annotation = annotation_fp = video_fp = probe = None
            if stat.status == STATUS_ABSENT:
                accepted, reason = decide_candidate(None)
            else:
                resolved = fetched.get(episode_id)
                if resolved is None or resolved.status == STATUS_ERROR:
                    detail = resolved.error if resolved is not None else "no result returned"
                    raise SourceError(f"episode {episode_id}: annotation fetch failed -- {detail}")
                annotation = resolved.payload
                binding = verify_annotation_binding(episode_id, annotation)
                if binding:
                    raise SourceError("; ".join(binding))
                annotation_fp = resolved.fingerprint
                accepted, reason = decide_candidate(annotation)
                if accepted:
                    accepted, reason, video_fp, probe = _resolve_video(
                        io, episode_id, annotation, video_stats, dry_run
                    )

            draw_log.append({"episode_id": episode_id, "reason": reason})
            tally[reason] += 1
            if not accepted:
                log(f"[manifest] draw {len(draw_log):4d}  ep {episode_id:<6d} {reason}")
                continue
            episodes.append(
                _episode_record(len(episodes), episode_id, annotation, seed, annotation_fp, video_fp, probe)
            )
            log(
                f"[manifest] draw {len(draw_log):4d}  ep {episode_id:<6d} accepted "
                f"({len(episodes)}/{n_target}, n_windows={episodes[-1]['n_windows']})"
            )

    if len(episodes) < n_target:
        raise RuntimeError(
            f"candidate pool exhausted after {len(draw_log)} draws: only {len(episodes)}/{n_target} accepted"
        )

    return {
        "selection_seed": int(seed),
        "builder_commit": builder_commit,
        "created_utc": created_utc,
        "tool_versions": tool_versions,
        "provisional": bool(dry_run),
        "fixture": fixture,
        "vae_fingerprint": copy.deepcopy(vae_fingerprint),
        "episodes": episodes,
        "draw_log": draw_log,
        "rejection_tally": dict(sorted(tally.items())),
        "totals": {
            "episodes": len(episodes),
            "windows": sum(episode["n_windows"] or 0 for episode in episodes),
        },
    }


# ----------------------------------------------------------------------------------
# Verification -- structure (fail closed) then live fingerprints
# ----------------------------------------------------------------------------------


def _validate_fingerprint(tag: str, key: str, fingerprint, expected_uri: str | None) -> list[str]:
    if not isinstance(fingerprint, dict):
        return [f"{tag}: {key} is missing"]
    missing = _FINGERPRINT_KEYS - set(fingerprint)
    if missing:
        return [f"{tag}: {key} is missing {sorted(missing)}"]
    errors: list[str] = []
    if expected_uri is not None and fingerprint["uri"] != expected_uri:
        errors.append(f"{tag}: {key} uri {fingerprint['uri']} != expected {expected_uri}")
    if not isinstance(fingerprint["generation"], int):
        errors.append(f"{tag}: {key} generation {fingerprint['generation']!r} is not an int")
    if not str(fingerprint["md5"] or "").strip():
        errors.append(f"{tag}: {key} has an empty md5")
    if not isinstance(fingerprint["size"], int) or fingerprint["size"] <= 0:
        errors.append(f"{tag}: {key} size {fingerprint['size']!r} is not a positive int")
    return errors


def _validate_episode(position: int, episode, seen_ids: set) -> list[str]:
    tag = f"episodes[{position}]"
    if not isinstance(episode, dict):
        return [f"{tag} is not an object"]
    missing = _EPISODE_KEYS - set(episode)
    if missing:
        return [f"{tag} is missing {sorted(missing)}"]

    errors: list[str] = []
    episode_id = episode["episode_id"]
    if episode["episode_index"] != position:
        errors.append(f"{tag}: episode_index {episode['episode_index']} != position {position}")
    if not isinstance(episode_id, int):
        errors.append(f"{tag}: episode_id {episode_id!r} is not an int")
        episode_id = None
    elif episode_id in seen_ids:
        errors.append(f"{tag}: duplicate episode_id {episode_id}")

    texts = episode["texts"]
    if not isinstance(texts, list) or not texts:
        errors.append(f"{tag}: texts must be a non-empty list")
    else:
        index = episode["chosen_text_index"]
        if not isinstance(index, int) or not 0 <= index < len(texts):
            errors.append(f"{tag}: chosen_text_index {index!r} is out of range for {len(texts)} texts")
        else:
            raw = texts[index]
            if not str(raw).strip():
                errors.append(f"{tag}: chosen_text_index {index} lands on an empty text")
            if episode["chosen_text_raw"] != raw:
                errors.append(f"{tag}: chosen_text_raw does not equal texts[{index}]")
            if episode["used_text"] != str(episode["chosen_text_raw"]).strip():
                errors.append(f"{tag}: used_text is not the stripped chosen_text_raw")

    errors += _validate_fingerprint(
        tag,
        "annotation_fingerprint",
        episode["annotation_fingerprint"],
        annotation_uri(episode_id) if episode_id is not None else None,
    )
    errors += _validate_fingerprint(
        tag,
        "video_fingerprint",
        episode["video_fingerprint"],
        video_uri(episode_id) if episode_id is not None else None,
    )

    probe = episode["ffprobe"]
    if not isinstance(probe, dict):
        errors.append(f"{tag}: ffprobe block is missing")
        return errors
    probe_missing = _FFPROBE_KEYS - set(probe)
    if probe_missing:
        errors.append(f"{tag}: ffprobe is missing {sorted(probe_missing)}")
        return errors
    if (probe["width"], probe["height"]) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        errors.append(f"{tag}: geometry {probe['width']}x{probe['height']} != {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}")
    if float(probe["fps"]) != EXPECTED_FPS:
        errors.append(f"{tag}: fps {probe['fps']} != {EXPECTED_FPS}")
    if probe["pix_fmt"] != EXPECTED_PIX_FMT:
        errors.append(f"{tag}: pix_fmt {probe['pix_fmt']} != {EXPECTED_PIX_FMT}")
    frames = probe["nb_frames"]
    if not isinstance(frames, int) or frames < MIN_FRAMES:
        errors.append(f"{tag}: nb_frames {frames!r} is below the {MIN_FRAMES}-frame window")
    elif episode["n_windows"] != n_windows(frames):
        errors.append(f"{tag}: n_windows {episode['n_windows']} != {n_windows(frames)} for {frames} frames")
    return errors


def _validate_amendments(amended) -> list[str]:
    """`amended` is optional, but if present every entry must name why/when/what/from-where."""
    if amended is None:
        return []
    if not isinstance(amended, list) or not amended:
        return ["amended must be a non-empty list when present"]
    errors: list[str] = []
    for position, entry in enumerate(amended):
        tag = f"amended[{position}]"
        if not isinstance(entry, dict) or set(entry) != _AMENDMENT_KEYS:
            errors.append(f"{tag} must be exactly {sorted(_AMENDMENT_KEYS)}")
            continue
        if not str(entry["reason"] or "").strip():
            errors.append(f"{tag}: reason is empty")
        if not str(entry["date"] or "").strip():
            errors.append(f"{tag}: date is empty")
        if not _SHA_RE.fullmatch(str(entry["commit"] or "")):
            errors.append(f"{tag}: commit {entry['commit']!r} is not a 40-hex git sha")
        if not isinstance(entry["fields"], list) or not entry["fields"]:
            errors.append(f"{tag}: fields must be a non-empty list of amended manifest keys")
    return errors


def validate_manifest_structure(manifest, expected_episodes: int | None = None) -> list[str]:
    """Pure fail-closed gate over every internal invariant of the manifest (Codex review A4).

    Returns the list of violations (empty = structurally usable). Cycle B calls this BEFORE any
    remote stat, so a malformed artifact cannot pass verification merely because its objects
    still exist in the bucket.
    """
    if not isinstance(manifest, dict):
        return ["manifest is not a JSON object"]
    missing = _TOP_LEVEL_KEYS - set(manifest)
    if missing:
        return [f"manifest is missing top-level keys {sorted(missing)}"]

    errors: list[str] = []
    if manifest["provisional"] is not False:
        errors.append("manifest is provisional (a --dry-run artifact); it must not be consumed")
    seed = manifest["selection_seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        errors.append(f"selection_seed {seed!r} is not an int")
    commit = manifest["builder_commit"]
    if not (isinstance(commit, str) and _SHA_RE.fullmatch(commit)):
        errors.append(f"builder_commit {commit!r} is not a 40-hex git sha")
    if not str(manifest["created_utc"] or "").strip():
        errors.append("created_utc is empty")

    tools = manifest["tool_versions"] if isinstance(manifest["tool_versions"], dict) else {}
    for key in _TOOL_KEYS:
        if not str(tools.get(key) or "").strip():
            errors.append(f"tool_versions is missing a non-empty {key} version")

    # B1: the pin is mandatory -- a manifest that cannot name the VAE it was built against
    # cannot bind the build to those weights, so the build has nothing to verify.
    errors += vae_pin_errors(manifest["vae_fingerprint"])
    errors += _validate_amendments(manifest.get("amended"))

    fixture = manifest["fixture"] if isinstance(manifest["fixture"], dict) else {}
    fixture_missing = _FIXTURE_KEYS - set(fixture)
    if fixture_missing:
        errors.append(f"fixture fingerprint is missing {sorted(fixture_missing)}")
    else:
        if list(fixture["names"]) != list(TARGET_NAMES):
            errors.append(f"fixture names {fixture['names']} != {list(TARGET_NAMES)}")
        if not isinstance(fixture["generation"], int):
            errors.append("fixture generation is not an int (was the fixture uploaded?)")

    episodes = manifest["episodes"]
    if not isinstance(episodes, list) or not episodes:
        errors.append("episodes must be a non-empty list")
        episodes = []
    if expected_episodes is not None and len(episodes) != expected_episodes:
        errors.append(f"expected {expected_episodes} episodes, manifest carries {len(episodes)}")
    seen_ids: set = set()
    for position, episode in enumerate(episodes):
        errors += _validate_episode(position, episode, seen_ids)
        if isinstance(episode, dict) and isinstance(episode.get("episode_id"), int):
            seen_ids.add(episode["episode_id"])

    draw_log = manifest["draw_log"]
    if not isinstance(draw_log, list) or not draw_log:
        errors.append("draw_log must be a non-empty list")
        draw_log = []
    drawn: list = []
    accepted_ids: list = []
    tally: Counter = Counter()
    for position, draw in enumerate(draw_log):
        if not isinstance(draw, dict) or set(draw) != {"episode_id", "reason"}:
            errors.append(f"draw_log[{position}] must be exactly {{episode_id, reason}}")
            continue
        if draw["reason"] not in REASONS:
            errors.append(f"draw_log[{position}]: unknown reason {draw['reason']!r}")
        drawn.append(draw["episode_id"])
        tally[draw["reason"]] += 1
        if draw["reason"] == "accepted":
            accepted_ids.append(draw["episode_id"])
    if len(set(drawn)) != len(drawn):
        errors.append("draw_log draws the same episode more than once")
    if draw_log and draw_log[-1].get("reason") != "accepted":
        errors.append("draw_log must end on the accepting draw that stopped the walk")
    if accepted_ids != [e.get("episode_id") for e in episodes if isinstance(e, dict)]:
        errors.append("draw_log accepted ids do not match the episodes list, in order")
    recorded_tally = manifest["rejection_tally"] if isinstance(manifest["rejection_tally"], dict) else {}
    if dict(sorted(tally.items())) != dict(sorted(recorded_tally.items())):
        errors.append("rejection_tally does not reconcile with the draw log")

    totals = manifest["totals"] if isinstance(manifest["totals"], dict) else {}
    if totals.get("episodes") != len(episodes):
        errors.append(f"totals.episodes {totals.get('episodes')!r} != {len(episodes)}")
    expected_windows = sum(e.get("n_windows") or 0 for e in episodes if isinstance(e, dict))
    if totals.get("windows") != expected_windows:
        errors.append(f"totals.windows {totals.get('windows')!r} != {expected_windows}")
    return errors


def verify_manifest(
    manifest: dict,
    stat_fn: Callable | None = None,
    chunk_size: int = VERIFY_CHUNK,
    expected_episodes: int | None = None,
) -> list[str]:
    """Structural gate first (no network), then re-stat every fingerprinted object."""
    structural = validate_manifest_structure(manifest, expected_episodes=expected_episodes)
    if structural:
        return structural

    stat_fn = stat_fn or gsutil_stat_many
    expected: dict[str, dict] = {manifest["fixture"]["uri"]: manifest["fixture"]}
    for episode in manifest["episodes"]:
        for key in ("annotation_fingerprint", "video_fingerprint"):
            fingerprint = episode[key]
            expected[fingerprint["uri"]] = fingerprint

    uris = list(expected)
    actual: dict[str, Resolved] = {}
    for start in range(0, len(uris), chunk_size):
        actual.update(stat_fn(uris[start : start + chunk_size]))

    errors: list[str] = []
    for uri in uris:
        outcome = actual.get(uri)
        if outcome is None or outcome.status == STATUS_ERROR:
            errors.append(f"{uri}: stat unresolved -- {outcome.error if outcome else 'no result'}")
            continue
        if outcome.status == STATUS_ABSENT:
            errors.append(f"{uri}: absent (gsutil stat found no such object)")
            continue
        got, want = outcome.fingerprint, expected[uri]
        want_size = want.get("size", want.get("size_bytes"))
        if int(got["generation"]) != int(want["generation"]):
            errors.append(f"{uri}: generation drift ({want['generation']} -> {got['generation']})")
        if got["md5"] != want["md5"]:
            errors.append(f"{uri}: md5 drift ({want['md5']} -> {got['md5']})")
        if int(got["size"]) != int(want_size):
            errors.append(f"{uri}: size drift ({want_size} -> {got['size']})")
    return errors


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the exp_02 overfit100 episode manifest.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", required=True, help="manifest JSON path")
    parser.add_argument("--fixture-fingerprint", default=None, help="fixture_fingerprint.json from A1")
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--tmp-dir", default=None, help="scratch dir for annotation/MP4 downloads")
    parser.add_argument("--dry-run", action="store_true", help="annotation-level decisions only; writes nothing")
    parser.add_argument("--vae-repo", default=DEFAULT_VAE_REPO, help="HF repo whose VAE encodes the dataset")
    parser.add_argument("--vae-revision", default=None, help="pin this exact revision (default: resolve current)")
    parser.add_argument(
        "--amend-vae-pin",
        action="store_true",
        help="add the resolved VAE pin to the EXISTING manifest at --out (no selection rebuild)",
    )
    parser.add_argument("--amend-reason", default="cycle-B review B1: VAE pin made mandatory")
    return parser.parse_args(argv)


def amend_vae_pin_file(path: str | Path, pin: dict, *, reason: str, commit: str, log: Callable = print) -> dict:
    """Apply `amend_manifest_vae_pin` to a manifest ON DISK, validating before and after."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    amended = amend_manifest_vae_pin(
        manifest, pin, reason=reason, commit=commit, now=datetime.now(timezone.utc).isoformat()
    )
    errors = validate_manifest_structure(amended, expected_episodes=len(amended["episodes"]))
    if errors:
        raise RuntimeError("the amended manifest failed its structural gate:\n  " + "\n  ".join(errors))
    path.write_text(json.dumps(amended, indent=2) + "\n")
    log(f"[manifest] amended {path} with vae_fingerprint {pin['revision']}")
    return amended


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    if args.amend_vae_pin:
        resolved = resolve_vae_snapshot(args.vae_repo, args.vae_revision)
        log(f"[manifest] resolved {args.vae_repo} -> {resolved['snapshot_path']}")
        amend_vae_pin_file(
            args.out,
            resolved["pin"],
            reason=args.amend_reason,
            commit=head_commit(log=log),
            log=log,
        )
        return 0

    if args.dry_run:
        fixture, builder_commit = {}, "dry-run"
        vae_pin = {"hf_repo": args.vae_repo, "revision": "0" * 40, "vae_config_sha256": "0" * 64}
    else:
        if not args.fixture_fingerprint:
            raise SystemExit("--fixture-fingerprint is required (the manifest embeds it verbatim)")
        fixture = json.loads(Path(args.fixture_fingerprint).read_text())
        fixture_missing = _FIXTURE_KEYS - set(fixture)
        if fixture_missing:
            raise SystemExit(f"fixture fingerprint is missing {sorted(fixture_missing)}")
        # B1: pin the VAE the dataset will be encoded with, before any selection work.
        vae_pin = resolve_vae_snapshot(args.vae_repo, args.vae_revision)["pin"]
        log(f"[manifest] VAE pin: {vae_pin['hf_repo']}@{vae_pin['revision']}")
        # A1: a production manifest may only claim a commit that actually contains the builder.
        builder_commit = assert_implementation_committed()
        log(f"[manifest] implementation is committed and clean at {builder_commit}")

    io = ManifestIO(tmp_dir=args.tmp_dir, log=log)
    manifest = build_manifest(
        io,
        seed=args.seed,
        n_target=args.n,
        fixture=fixture,
        vae_fingerprint=vae_pin,
        builder_commit=builder_commit,
        created_utc=datetime.now(timezone.utc).isoformat(),
        tool_versions=collect_tool_versions(),
        dry_run=args.dry_run,
        block_size=args.block_size,
        log=log,
    )

    log(f"[manifest] accepted {manifest['totals']['episodes']} episodes / {manifest['totals']['windows']} windows")
    log(f"[manifest] draws={len(manifest['draw_log'])} tally={manifest['rejection_tally']}")
    if args.dry_run:
        log("[manifest] --dry-run: provisional accepts (annotation-level only), nothing written")
        log(json.dumps([episode["episode_id"] for episode in manifest["episodes"]]))
        return 0

    structural = validate_manifest_structure(manifest, expected_episodes=args.n)
    if structural:
        raise RuntimeError("the built manifest failed its own structural gate:\n  " + "\n  ".join(structural))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"[manifest] structural gate passed; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
