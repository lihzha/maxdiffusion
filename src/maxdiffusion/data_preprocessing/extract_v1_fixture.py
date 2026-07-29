"""Extract and publish the exp_02 V1 encoder-validation fixture (plan v4 §2 D4/H1).

The V1 gate of the cycle-B dataset build re-encodes three *cached* exp_01 windows through
the new Wan-VAE encode path and compares against the cached latents. Those three windows
(`ep0_v0_s00000`, `ep0_v0_s00004`, `ep0_v0_s00008`) previously existed only as a throwaway
probe, so the gate was not reproducible. This tool materializes them:

    stream the first records of
      gs://v6_east1d/datasets/droid_wan_side_adapter/train/train-00000-of-00704.tfrecord
    -> verify names, shapes, dtypes and the cache contract `z_i0 == z_video[:, :1]` (bitwise)
    -> save one .npz (`<name>_z_i0`, `<name>_z_video`, `names`)
    -> upload to gs://v6_east1d/datasets/exp02_overfit100/fixtures/v1_cache_windows.npz
    -> `gsutil stat` it and write `fixture_fingerprint.json` {uri, generation, md5, size_bytes,
       names, shapes, dtypes}

The fingerprint is embedded verbatim in `overfit100_manifest.json`; the build job's preflight
calls `verify_fixture()` on the downloaded copy and refuses to encode anything if it drifts.

This module also hosts the thin `gsutil` primitives (`run_gsutil`, `parse_gsutil_stat`,
`gsutil_stat_many`, `md5_b64`) shared with `build_overfit100_manifest`; keeping them here
avoids a third cycle-A file while leaving that module free of a TensorFlow dependency
(TensorFlow is imported lazily, only on the TFRecord path).

CLI:
    python -m maxdiffusion.data_preprocessing.extract_v1_fixture \
        --out-fingerprint docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_fixture_fingerprint.json
    # add --skip-upload to stop at a local .npz (no GCS write)
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

SOURCE_TFRECORD_URI = "gs://v6_east1d/datasets/droid_wan_side_adapter/train/train-00000-of-00704.tfrecord"
FIXTURE_URI = "gs://v6_east1d/datasets/exp02_overfit100/fixtures/v1_cache_windows.npz"

TARGET_NAMES = ("ep0_v0_s00000", "ep0_v0_s00004", "ep0_v0_s00008")
Z_I0_SHAPE = (48, 1, 12, 20)
Z_VIDEO_SHAPE = (48, 9, 12, 20)
FIXTURE_DTYPE = np.float16
# The three targets are the first three records of shard 0; scan a small prefix and fail loudly.
MAX_RECORDS_SCANNED = 16

GSUTIL_TIMEOUT_S = 900
# gsutil surfaces an expired/absent reauth session with these markers. They are fatal: the
# user must re-authenticate, so never retry in a loop (exp_02 cycle-A operating rule).
_REAUTH_MARKERS = ("ReauthUnattendedError", "ReauthFailedError", "Reauthentication failed")


class ReauthRequiredError(RuntimeError):
    """gcloud/gsutil needs an interactive reauth -- stop, do not retry."""


class GsutilError(RuntimeError):
    """A gsutil invocation failed for a non-auth reason."""


# Per-object outcome vocabulary. Codex cycle-A review A3: "confirmed absent" and "we could not
# find out" are different facts. Only the former may become a recorded rejection reason; the
# latter must abort the build, never silently shrink the corpus.
STATUS_FOUND = "found"
STATUS_ABSENT = "absent"
STATUS_ERROR = "error"

_ABSENT_RE = re.compile(r"No URLs matched:\s*(\S+)")


@dataclasses.dataclass(frozen=True)
class Resolved:
    """The outcome of resolving one remote object: found / confirmed-absent / error."""

    status: str
    fingerprint: dict | None = None
    payload: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == STATUS_FOUND

    @classmethod
    def found(cls, fingerprint: dict, payload: Any = None) -> "Resolved":
        return cls(STATUS_FOUND, fingerprint=fingerprint, payload=payload)

    @classmethod
    def absent(cls) -> "Resolved":
        return cls(STATUS_ABSENT)

    @classmethod
    def failed(cls, message: str) -> "Resolved":
        # NOT named `error`: a classmethod of that name would shadow the `error` FIELD's default,
        # leaving every non-error outcome with a truthy bound method in `.error`.
        return cls(STATUS_ERROR, error=message)


# ----------------------------------------------------------------------------------
# gsutil primitives (shared with build_overfit100_manifest)
# ----------------------------------------------------------------------------------


def run_gsutil(args: Iterable[str], *, check: bool = True, timeout: int = GSUTIL_TIMEOUT_S):
    """Run one sequential `gsutil` invocation. Never uses `-m` (it stalls on this host)."""
    argv = ["gsutil", *[str(a) for a in args]]
    proc = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
    stderr = proc.stderr.decode("utf-8", "replace")
    if any(marker in stderr for marker in _REAUTH_MARKERS):
        raise ReauthRequiredError(f"gsutil requires reauthentication; re-run `gcloud auth login`.\n{stderr.strip()}")
    if check and proc.returncode != 0:
        raise GsutilError(f"`{' '.join(argv)}` exited {proc.returncode}\n{stderr.strip()}")
    return proc


def parse_gsutil_stat(text: str) -> dict[str, dict]:
    """Parse `gsutil stat` output (one or many objects) into `uri -> {uri, generation, md5, size}`."""
    parsed: dict[str, dict] = {}
    current: str | None = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith("gs://") and line.rstrip().endswith(":"):
            current = line.rstrip()[:-1]
            parsed[current] = {"uri": current}
            continue
        if current is None:
            continue
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key, value = key.strip(), value.strip()
        if key == "Content-Length":
            parsed[current]["size"] = int(value)
        elif key == "Hash (md5)":
            parsed[current]["md5"] = value
        elif key == "Generation":
            parsed[current]["generation"] = int(value)
    for uri, fields in parsed.items():
        missing = {"size", "md5", "generation"} - set(fields)
        if missing:
            raise ValueError(f"gsutil stat output for {uri} is missing {sorted(missing)}")
    return parsed


def parse_absent_uris(stderr: str) -> set[str]:
    """URIs gsutil explicitly reported as non-existent (`No URLs matched: <uri>`).

    Exact per-object evidence -- deliberately NOT a substring test over the whole stderr, so a
    transient error whose text happens to contain `404` is never read as "object is absent".
    """
    return set(_ABSENT_RE.findall(stderr or ""))


def classify_stat_batch(uris: Iterable[str], stdout: str, stderr: str, returncode: int) -> dict[str, Resolved]:
    """Classify every requested URI as found / confirmed-absent / unresolved-error."""
    uris = list(uris)
    parsed = parse_gsutil_stat(stdout)
    absent = parse_absent_uris(stderr)
    detail = (stderr or "").strip()[:400] or f"gsutil exited {returncode} without explanation"
    resolved: dict[str, Resolved] = {}
    for uri in uris:
        if uri in parsed:
            resolved[uri] = Resolved.found(parsed[uri])
        elif uri in absent:
            resolved[uri] = Resolved.absent()
        else:
            resolved[uri] = Resolved.failed(f"{uri}: unresolved by `gsutil stat` (exit {returncode}): {detail}")
    return resolved


def gsutil_stat_many(uris: Iterable[str], retry_individually: bool = True) -> dict[str, Resolved]:
    """Stat a batch in ONE sequential gsutil call, then retry any unresolved member ONCE alone.

    A batch invocation can fail for reasons that have nothing to do with a given member, so an
    unresolved object is retried on its own before being reported as an error.
    """
    uris = list(uris)
    if not uris:
        return {}
    proc = run_gsutil(["stat", *uris], check=False)
    resolved = classify_stat_batch(
        uris,
        proc.stdout.decode("utf-8", "replace"),
        proc.stderr.decode("utf-8", "replace"),
        proc.returncode,
    )
    if retry_individually:
        for uri in [u for u in uris if resolved[u].status == STATUS_ERROR]:
            retry = run_gsutil(["stat", uri], check=False)
            resolved[uri] = classify_stat_batch(
                [uri],
                retry.stdout.decode("utf-8", "replace"),
                retry.stderr.decode("utf-8", "replace"),
                retry.returncode,
            )[uri]
    return resolved


def md5_b64(data: bytes) -> str:
    """base64-encoded MD5 digest -- the exact form `gsutil stat` reports as `Hash (md5)`."""
    return base64.b64encode(hashlib.md5(data).digest()).decode()


def pinned_uri(uri: str, fingerprint: dict) -> str:
    """`<uri>#<generation>` -- downloads the EXACT object generation that was statted."""
    generation = (fingerprint or {}).get("generation")
    if generation is None:
        raise ValueError(f"cannot pin {uri}: fingerprint has no generation")
    return f"{uri}#{int(generation)}"


def verify_payload_binding(uri: str, data: bytes, fingerprint: dict) -> list[str]:
    """Bind downloaded bytes to the fingerprint that was recorded (Codex review A2)."""
    errors: list[str] = []
    actual_md5, actual_size = md5_b64(data), len(data)
    if actual_md5 != fingerprint.get("md5"):
        errors.append(f"{uri}: md5 mismatch (recorded {fingerprint.get('md5')}, downloaded {actual_md5})")
    if actual_size != int(fingerprint.get("size", -1)):
        errors.append(f"{uri}: size mismatch (recorded {fingerprint.get('size')}, downloaded {actual_size})")
    return errors


# ----------------------------------------------------------------------------------
# TFRecord parsing
# ----------------------------------------------------------------------------------


def _tf():
    """Import TensorFlow lazily: only the TFRecord path needs it."""
    import tensorflow as tf

    return tf


def _feature_bytes(feature, key: str) -> bytes:
    if key not in feature or not feature[key].bytes_list.value:
        raise ValueError(f"record is missing the bytes feature {key!r}")
    return feature[key].bytes_list.value[0]


def _decode_array(feature, key: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = _feature_bytes(feature, key)
    expected = int(np.prod(shape)) * FIXTURE_DTYPE().itemsize
    if len(raw) != expected:
        raise ValueError(f"{key}: expected {expected} bytes for shape {tuple(shape)} float16, got {len(raw)}")
    return np.frombuffer(raw, dtype=FIXTURE_DTYPE).reshape(shape).copy()


def parse_record(
    raw: bytes,
    *,
    z_i0_shape: tuple[int, ...] = Z_I0_SHAPE,
    z_video_shape: tuple[int, ...] = Z_VIDEO_SHAPE,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Decode one cache record and assert the `z_i0 == z_video[:, :1]` contract bitwise."""
    tf = _tf()
    feature = tf.train.Example.FromString(raw).features.feature
    name = _feature_bytes(feature, "name").decode()
    z_i0 = _decode_array(feature, "z_i0", tuple(z_i0_shape))
    z_video = _decode_array(feature, "z_video", tuple(z_video_shape))
    if z_i0.tobytes() != z_video[:, :1].copy().tobytes():
        raise ValueError(f"{name}: z_i0 is not bitwise equal to z_video[:, :1] (cache contract violated)")
    return name, z_i0, z_video


def iter_tfrecord(uri: str | Path) -> Iterator[bytes]:
    """Yield raw records from a TFRecord file; `tf.data` reads `gs://` URIs directly."""
    tf = _tf()
    for record in tf.data.TFRecordDataset([str(uri)]):
        yield bytes(record.numpy())


def extract_windows(
    raw_records: Iterable[bytes],
    names: Iterable[str] = TARGET_NAMES,
    max_records: int = MAX_RECORDS_SCANNED,
    *,
    z_i0_shape: tuple[int, ...] = Z_I0_SHAPE,
    z_video_shape: tuple[int, ...] = Z_VIDEO_SHAPE,
) -> dict[str, dict[str, np.ndarray]]:
    """Pull the named windows out of a bounded record prefix, in `names` order."""
    wanted = list(names)
    found: dict[str, dict[str, np.ndarray]] = {}
    scanned = 0
    for raw in raw_records:
        if scanned >= max_records:
            break
        scanned += 1
        name, z_i0, z_video = parse_record(raw, z_i0_shape=z_i0_shape, z_video_shape=z_video_shape)
        if name in wanted and name not in found:
            found[name] = {"z_i0": z_i0, "z_video": z_video}
        if len(found) == len(wanted):
            break
    missing = [name for name in wanted if name not in found]
    if missing:
        raise ValueError(f"scanned {scanned} record(s) (max {max_records}) without finding {missing}")
    return {name: found[name] for name in wanted}


# ----------------------------------------------------------------------------------
# .npz fixture + fingerprint
# ----------------------------------------------------------------------------------


def save_fixture_npz(path: str | Path, windows: dict[str, dict[str, np.ndarray]]) -> Path:
    """Write one .npz holding `names` plus `<name>_z_i0` / `<name>_z_video` per window."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {"names": np.array(list(windows))}
    for name, arrays in windows.items():
        payload[f"{name}_z_i0"] = arrays["z_i0"]
        payload[f"{name}_z_video"] = arrays["z_video"]
    with open(path, "wb") as handle:
        np.savez(handle, **payload)
    return path


def load_fixture_npz(path: str | Path) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    """Inverse of `save_fixture_npz`; no pickle (names are stored as a unicode array)."""
    with np.load(path) as data:
        names = [str(name) for name in data["names"]]
        windows = {name: {"z_i0": data[f"{name}_z_i0"], "z_video": data[f"{name}_z_video"]} for name in names}
    return names, windows


def build_fingerprint(
    local_path: str | Path,
    uri: str,
    windows: dict[str, dict[str, np.ndarray]],
    stat: dict,
) -> dict:
    """Assemble the committed fingerprint, asserting the remote hash/size match the local file."""
    local_path = Path(local_path)
    data = local_path.read_bytes()
    local_md5, local_size = md5_b64(data), len(data)
    if stat.get("md5") != local_md5:
        raise ValueError(f"uploaded md5 {stat.get('md5')!r} != local md5 {local_md5!r} for {uri}")
    if int(stat.get("size", -1)) != local_size:
        raise ValueError(f"uploaded size {stat.get('size')} != local size {local_size} for {uri}")
    reference = next(iter(windows.values()))
    generation = stat.get("generation")
    return {
        "uri": uri,
        "generation": None if generation is None else int(generation),
        "md5": local_md5,
        "size_bytes": local_size,
        "names": list(windows),
        "shapes": {key: list(reference[key].shape) for key in ("z_i0", "z_video")},
        "dtypes": {key: str(reference[key].dtype) for key in ("z_i0", "z_video")},
    }


def validate_fixture_structure(
    names: Iterable[str],
    windows: dict[str, dict[str, np.ndarray]],
    fingerprint: dict,
    required_names: Iterable[str] = TARGET_NAMES,
) -> list[str]:
    """Fail-closed structural gate for the V1 fixture (Codex review A4).

    Requires the EXACT ordered name set, shapes/dtypes for every array of every window, and the
    `z_i0 == z_video[:, :1]` bitwise contract the fixture exists to preserve -- so a structurally
    wrong fixture can never reach the cycle-B V1 gate just because its md5 happens to match.
    """
    errors: list[str] = []
    required = list(required_names)
    fingerprint_names = list((fingerprint or {}).get("names") or [])
    if fingerprint_names != required:
        errors.append(f"fixture names {fingerprint_names} != required names in order {required}")
    if list(names) != fingerprint_names:
        errors.append(f".npz names {list(names)} != fingerprint names {fingerprint_names} (exact order required)")

    shapes = (fingerprint or {}).get("shapes") or {}
    dtypes = (fingerprint or {}).get("dtypes") or {}
    for key in ("z_i0", "z_video"):
        if key not in shapes or key not in dtypes:
            errors.append(f"fixture fingerprint is missing shape/dtype for {key}")

    for name in fingerprint_names:
        window = windows.get(name)
        if window is None:
            errors.append(f"missing window {name}")
            continue
        for key in ("z_i0", "z_video"):
            array = window.get(key)
            if array is None:
                errors.append(f"{name}: missing array {key}")
                continue
            if key in shapes and list(array.shape) != list(shapes[key]):
                errors.append(f"{name}/{key}: shape {list(array.shape)} != {list(shapes[key])}")
            if key in dtypes and str(array.dtype) != dtypes[key]:
                errors.append(f"{name}/{key}: dtype {array.dtype} != {dtypes[key]}")
        z_i0, z_video = window.get("z_i0"), window.get("z_video")
        if z_i0 is None or z_video is None:
            continue
        try:
            if z_i0.tobytes() != z_video[:, :1].copy().tobytes():
                errors.append(f"{name}: z_i0 is not bitwise equal to z_video[:, :1]")
        except Exception as exc:  # noqa: BLE001 -- any comparison failure is a fixture failure
            errors.append(f"{name}: cannot compare z_i0 with z_video[:, :1] ({exc})")
    return errors


def verify_fixture(path: str | Path, fingerprint: dict) -> list[str]:
    """Preflight: hash/size binding PLUS the fail-closed structural gate. Empty list = usable."""
    path = Path(path)
    if not path.exists():
        return [f"fixture file missing: {path}"]
    data = path.read_bytes()
    errors: list[str] = []
    if md5_b64(data) != fingerprint["md5"]:
        errors.append(f"{path}: md5 mismatch (expected {fingerprint['md5']}, got {md5_b64(data)})")
    if len(data) != int(fingerprint["size_bytes"]):
        errors.append(f"{path}: size mismatch (expected {fingerprint['size_bytes']}, got {len(data)})")
    try:
        names, windows = load_fixture_npz(path)
    except Exception as exc:  # noqa: BLE001 -- surfaced verbatim as a preflight error
        errors.append(f"{path}: unreadable .npz ({exc})")
        return errors
    errors.extend(f"{path}: {error}" for error in validate_fixture_structure(names, windows, fingerprint))
    return errors


# ----------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract + publish the exp_02 V1 cache-window fixture.")
    parser.add_argument("--out-fingerprint", required=True, help="where to write fixture_fingerprint.json")
    parser.add_argument("--npz-path", default=None, help="local .npz path (default: a temp file)")
    parser.add_argument("--source", default=SOURCE_TFRECORD_URI)
    parser.add_argument("--fixture-uri", default=FIXTURE_URI)
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS_SCANNED)
    parser.add_argument("--skip-upload", action="store_true", help="write the .npz locally, do not touch GCS")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    npz_path = (
        Path(args.npz_path)
        if args.npz_path
        else Path(tempfile.mkdtemp(prefix="exp02_fixture_")) / Path(FIXTURE_URI).name
    )

    print(f"[fixture] reading {args.source}")
    windows = extract_windows(iter_tfrecord(args.source), max_records=args.max_records)
    for name, arrays in windows.items():
        print(
            f"[fixture]   {name}: z_i0 {arrays['z_i0'].shape} {arrays['z_i0'].dtype}, z_video {arrays['z_video'].shape}"
        )
    save_fixture_npz(npz_path, windows)
    print(f"[fixture] wrote {npz_path} ({npz_path.stat().st_size} bytes)")

    if args.skip_upload:
        data = npz_path.read_bytes()
        stat = {"generation": None, "md5": md5_b64(data), "size": len(data)}
        print("[fixture] --skip-upload: no GCS write, generation left null")
    else:
        print(f"[fixture] uploading to {args.fixture_uri}")
        run_gsutil(["cp", str(npz_path), args.fixture_uri])
        resolved = gsutil_stat_many([args.fixture_uri])[args.fixture_uri]
        if not resolved.ok:
            raise GsutilError(
                f"upload succeeded but `gsutil stat {args.fixture_uri}` returned "
                f"{resolved.status} ({resolved.error or 'object not found'})"
            )
        stat = resolved.fingerprint

    fingerprint = build_fingerprint(npz_path, args.fixture_uri, windows, stat)
    errors = verify_fixture(npz_path, fingerprint)
    if errors:
        raise RuntimeError("fixture preflight failed:\n  " + "\n  ".join(errors))

    out = Path(args.out_fingerprint)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fingerprint, indent=2) + "\n")
    print(f"[fixture] preflight OK; fingerprint -> {out}")
    print(json.dumps(fingerprint, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
