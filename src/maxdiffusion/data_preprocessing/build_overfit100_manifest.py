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
import json
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

_SHA_RE = re.compile(r"[0-9a-f]{40}")


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


def _git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True, check=True)
    return proc.stdout


def assert_implementation_committed(repo_root: str | Path | None = None) -> str:
    """Return HEAD's sha, or raise if the manifest could not honestly claim to come from it."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[3]
    tracked = {line.strip() for line in _git(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines()}
    dirty = parse_git_porcelain(_git(root, "status", "--porcelain", "--", *IMPLEMENTATION_PATHS))
    errors = implementation_provenance_errors(tracked, dirty)
    if errors:
        raise DirtyImplementationError(
            "refusing to build a production manifest from an uncommitted implementation "
            "(builder_commit would not name the code that ran):\n  " + "\n  ".join(errors)
        )
    return _git(root, "rev-parse", "HEAD").strip()


def _first_line(text: str) -> str:
    return (text or "").strip().splitlines()[0].strip() if (text or "").strip() else ""


def collect_tool_versions() -> dict:
    """Record the versions of every tool/library that shaped this manifest."""
    gsutil = subprocess.run(["gsutil", "version"], capture_output=True, text=True, check=False)
    ffprobe = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True, check=False)
    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=False)
    return {
        "python": platform.python_version(),
        "gsutil": _first_line(gsutil.stdout) or _first_line(gsutil.stderr),
        "ffprobe": _first_line(ffprobe.stdout) or _first_line(ffprobe.stderr),
        "ffmpeg": _first_line(ffmpeg.stdout) or _first_line(ffmpeg.stderr),
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    def log(message: str) -> None:
        print(message, flush=True)

    if args.dry_run:
        fixture, builder_commit = {}, "dry-run"
    else:
        if not args.fixture_fingerprint:
            raise SystemExit("--fixture-fingerprint is required (the manifest embeds it verbatim)")
        fixture = json.loads(Path(args.fixture_fingerprint).read_text())
        fixture_missing = _FIXTURE_KEYS - set(fixture)
        if fixture_missing:
            raise SystemExit(f"fixture fingerprint is missing {sorted(fixture_missing)}")
        # A1: a production manifest may only claim a commit that actually contains the builder.
        builder_commit = assert_implementation_committed()
        log(f"[manifest] implementation is committed and clean at {builder_commit}")

    io = ManifestIO(tmp_dir=args.tmp_dir, log=log)
    manifest = build_manifest(
        io,
        seed=args.seed,
        n_target=args.n,
        fixture=fixture,
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
