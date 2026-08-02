"""Write per-sample RESIDUAL videos for a pulled full-FT val/train step directory.

Part II (Query 10): for every ``sample_*`` dir that ``generate_wan_side_adapter.py`` wrote
(each holds ``ground_truth.mp4`` + ``sample.mp4`` + ``comparison_gt_top_pred_bottom.mp4`` +
``metrics.json``), this produces ``residual_gainN.mp4`` -- a visualization of the per-pixel
absolute difference between the prediction and the ground truth, amplified by an integer
``gain`` and clipped. Brighter = larger error. The gallery
(``make_wan_val_gallery.py``) then renders it as a fourth card video.

Stdlib only; the actual encode is one ffmpeg subprocess per sample. Usage::

    python -m maxdiffusion.make_wan_residual_videos <step_dir> [--gain 4] [--overwrite]

The exact filter chain (single source of truth: ``build_ffmpeg_cmd``)
================================================================================
Per sample, with input 0 = ground truth and input 1 = prediction::

    [0:v]format=rgb24[gt];
    [1:v]format=rgb24[pred];
    [gt][pred]blend=all_mode=difference:shortest=1:repeatlast=0,
              lutrgb=r='clip(val*N,0,255)':g='clip(val*N,0,255)':b='clip(val*N,0,255)',
              format=yuv420p[out]

encoded with ``-c:v libx264 -pix_fmt yuv420p -movflags +faststart`` at the input fps
(blend inherits input 0's frame rate, so "same fps" holds without forcing ``-r``).

Strictness & atomicity (Codex strengthen review F1-F3)
------------------------------------------------------
* **Equal-length preflight (F1).** ffmpeg's framesync defaults (``shortest=0``,
  ``repeatlast=1``) silently repeat the last frame of the shorter input -- a 4-frame GT vs
  8-frame pred yields 8 residual frames whose tail is a frozen-frame artifact (reproduced).
  ``process_step_dir`` therefore ffprobes BOTH inputs first and refuses unequal frame counts
  or fps (compared by value: ``16/1`` == ``16000/1000``) with an error naming both. The
  ``shortest=1:repeatlast=0`` options in the chain are belt-and-suspenders for callers who
  bypass the preflight by invoking ``build_ffmpeg_cmd`` directly.
* **One gain per sample dir (F2).** Requesting gain N while a dir holds a
  ``residual_gainM.mp4`` (M != N) fails fast, listing the stale files: ``--overwrite`` only
  re-encodes the requested gain's filename, so it can never heal a mixed-gain dir (and the
  gallery refuses dirs with more than one residual). Delete stale residuals explicitly,
  then re-run.
* **Atomic writes (F3).** Each encode targets a sibling ``residual_gainN.tmp.mp4`` and is
  ``os.replace``d onto the final name only when ffmpeg exits 0 (temp unlinked on failure).
  An interrupted encode therefore never leaves a partial ``residual_gainN.mp4`` for the
  skip-existing path to trust; a crash-leftover temp matches neither the gallery's nor this
  tool's residual pattern and is simply overwritten (``-y``) on the next run.

Colorspace reasoning (why ``format=rgb24`` before ``blend``, and ``lutrgb`` not ``lutyuv``)
--------------------------------------------------------------------------------
The source videos are H.264 ``yuv420p`` (320x192, 33 frames, fps 16; color primaries /
transfer / matrix are unspecified in the stream, so ffmpeg decodes them with its SD default,
BT.601). If ``blend=difference`` ran directly on the ``yuv420p`` frames it would subtract the
planes independently: |Y_pred - Y_gt| on the gamma-encoded luma plane and |U-U|,|V-V| on the
half-resolution chroma planes. That is NOT the error a viewer sees -- it under-weights pure
chroma shifts, is taken on subsampled chroma, and mixes luma/chroma rather than giving a
per-channel RGB error.

So both inputs are first converted to packed ``rgb24`` (``format=rgb24`` triggers the
YUV->RGB matrix using the stream's color metadata). ``blend=all_mode=difference`` then
computes |pred - gt| independently on the R, G, B channels of the DECODED (display-space,
gamma-encoded) image -- exactly the difference the human eye integrates in the side-by-side
comparison. We deliberately do NOT linearize first: the goal is to visualize display-space
error, matching what report 03's comparison video shows, not linear-light radiometric error.
Because the frames are RGB at this point, the amplify+clip lut must be ``lutrgb`` (operating
on r/g/b); ``lutyuv`` there would be a bug. ``clip(val*N,0,255)`` multiplies each 8-bit
channel by the gain and clips into range, per channel. Finally ``format=yuv420p`` re-encodes
for broad browser/H.264 compatibility.

A flat-gray sanity check pins this: GT=100, pred=40, gain=3 yields a near-uniform field at
|100-40|*3 = 180 on decoded RGB (measured 179; 1 LSB of YUV<->RGB round-trip). Any of the
plausible mistakes -- addition instead of difference, a dropped gain, or a YUV-plane
difference -- moves that number far outside tolerance (see the end-to-end test).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from typing import Sequence

GT_NAME = "ground_truth.mp4"
PRED_NAME = "sample.mp4"
SAMPLE_GLOB = "sample_*"
# A valid residual filename is exactly residual_gain<int>.mp4 (same contract as the gallery);
# the .tmp.mp4 sibling used for atomic writes deliberately does NOT match.
_RESIDUAL_NAME_RE = re.compile(r"^residual_gain(\d+)\.mp4$")


def _validate_gain(gain) -> int:
    """Gain must be a positive integer -- it names the output (``residual_gainN.mp4``) and is
    the RGB multiply in the lut. Reject bool (a subclass of int) and non-ints so the
    filename<->label round trip stays exact."""
    if isinstance(gain, bool) or not isinstance(gain, int):
        raise ValueError(f"gain must be a positive integer (each residual is amplified by it); got {gain!r}")
    if gain <= 0:
        raise ValueError(f"gain must be a positive integer > 0; got {gain!r}")
    return gain


def residual_filename(gain: int) -> str:
    return f"residual_gain{_validate_gain(gain)}.mp4"


def build_ffmpeg_cmd(gt_path, pred_path, out_path, gain) -> list[str]:
    """Return the single ffmpeg argv that writes one residual video. See module docstring
    for the filter chain + colorspace justification. Input 0 is GT, input 1 is prediction.

    Raises ``ValueError`` for a non-positive / non-integer gain.
    """
    gain = _validate_gain(gain)
    lut = f"lutrgb=r='clip(val*{gain},0,255)':g='clip(val*{gain},0,255)':b='clip(val*{gain},0,255)'"
    filter_complex = (
        "[0:v]format=rgb24[gt];"
        "[1:v]format=rgb24[pred];"
        # shortest=1:repeatlast=0 (F1 belt-and-suspenders): never pad the shorter stream --
        # process_step_dir refuses unequal inputs up front, but a direct caller is safe too.
        f"[gt][pred]blend=all_mode=difference:shortest=1:repeatlast=0,{lut},format=yuv420p[out]"
    )
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(gt_path),  # input 0 = ground truth
        "-i",
        str(pred_path),  # input 1 = prediction
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]


def ensure_ffmpeg() -> str:
    """Startup guard: return the ffmpeg path, or raise an actionable error naming the dep(s).

    ffprobe is guarded too: the F1 preflight uses it to verify equal-length inputs. Both
    binaries ship together in every standard ffmpeg install.
    """
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise RuntimeError(
            f"{' and '.join(missing)} not found on PATH -- residual-video generation requires the ffmpeg CLI "
            "(ffprobe ships with it and verifies equal-length inputs). Install it, e.g. `brew install ffmpeg` "
            "on macOS or `apt-get install ffmpeg` on Debian/Ubuntu."
        )
    return shutil.which("ffmpeg")


def probe_video_stream(path) -> dict:
    """ffprobe the first video stream: ``{"nb_frames": int, "r_frame_rate": str}`` (F1 preflight).

    Raises ``RuntimeError`` (with ffprobe's stderr) if ffprobe fails or reports no usable
    frame count -- an unverifiable input must not reach the encoder.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path} (exit {result.returncode}) -- cannot verify frame count/fps "
            f"before building a residual.\n--- ffprobe stderr ---\n{result.stderr}"
        )
    meta = {}
    for line in (result.stdout or "").strip().splitlines():
        key, _, value = line.partition("=")
        meta[key.strip()] = value.strip()
    if not meta.get("nb_frames", "").isdigit():
        raise RuntimeError(
            f"{path}: ffprobe reported no usable nb_frames (stream entries: {meta!r}) -- cannot verify "
            "equal-length inputs; is this a valid mp4 video stream?"
        )
    return {"nb_frames": int(meta["nb_frames"]), "r_frame_rate": meta.get("r_frame_rate", "")}


def _same_rate(a: str, b: str) -> bool:
    """Compare frame rates by value ('16/1' == '16000/1000'), falling back to string equality."""
    try:
        return Fraction(a) == Fraction(b)
    except (ValueError, ZeroDivisionError):
        return a == b


def _tmp_output_path(out: str) -> str:
    """Sibling temp target for the atomic write (F3): ``residual_gainN.tmp.mp4``.

    Still ends in ``.mp4`` so ffmpeg infers the mp4 muxer, but matches neither
    ``_RESIDUAL_NAME_RE`` nor the gallery's residual pattern, and skip-existing checks only
    the final name -- so a crash-leftover temp is invisible to every consumer and simply
    overwritten (``-y``) then replaced on the next run.
    """
    return out[: -len(".mp4")] + ".tmp.mp4"


def process_step_dir(step_dir, gain: int = 4, overwrite: bool = False) -> list[str]:
    """Write ``residual_gainN.mp4`` for every ``sample_*`` dir under ``step_dir``.

    Idempotent: an existing output is skipped unless ``overwrite``. Fails fast -- before ANY
    encode -- on: a sample dir missing either input (path-naming error); a dir holding
    residuals at a DIFFERENT gain (F2: ``--overwrite`` only re-encodes the requested gain's
    filename, so stale gains must be deleted explicitly); and input pairs whose frame counts
    or fps disagree (F1: ffmpeg's framesync would otherwise silently repeat/pad the shorter
    stream). Encodes are atomic (F3): ffmpeg writes ``residual_gainN.tmp.mp4``, which is
    ``os.replace``d onto the final name only on rc==0; on failure the temp is unlinked and
    ffmpeg's stderr surfaced. Prints a written/skipped tally and returns the written paths.
    """
    gain = _validate_gain(gain)

    sample_dirs = sorted(d for d in glob.glob(os.path.join(step_dir, SAMPLE_GLOB)) if os.path.isdir(d))
    if not sample_dirs:
        raise ValueError(
            f"{step_dir}: found no {SAMPLE_GLOB} directories -- not a per-sample validation "
            f"step dir (expected sample_XXXX/ dirs each with {GT_NAME} and {PRED_NAME})."
        )

    # Fail fast (1/3): every sample dir has both inputs before any subprocess work.
    for d in sample_dirs:
        for fn in (GT_NAME, PRED_NAME):
            p = os.path.join(d, fn)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{d}: missing required input {fn}: {p} (each {SAMPLE_GLOB} dir needs both "
                    f"{GT_NAME} and {PRED_NAME} to build a residual)."
                )

    # Fail fast (2/3, F2): no dir may hold residuals at a different gain than requested.
    for d in sample_dirs:
        stale = []
        for p in sorted(glob.glob(os.path.join(d, "residual_gain*.mp4"))):
            m = _RESIDUAL_NAME_RE.match(os.path.basename(p))
            if m and int(m.group(1)) != gain:
                stale.append(os.path.basename(p))
        if stale:
            raise ValueError(
                f"{d}: found existing residual(s) at a different gain than the requested ×{gain}: "
                f"{', '.join(stale)}. A sample dir may hold at most one residual_gainN.mp4 (the gallery "
                f"refuses ambiguous dirs), and --overwrite only re-encodes the requested gain's filename. "
                f"Delete the stale file(s) explicitly, then re-run."
            )

    # Fail fast (3/3, F1): both inputs of every pair must agree on frame count and fps --
    # ffmpeg's framesync default would silently repeat/pad the shorter stream otherwise.
    for d in sample_dirs:
        gt_meta = probe_video_stream(os.path.join(d, GT_NAME))
        pred_meta = probe_video_stream(os.path.join(d, PRED_NAME))
        mismatch = gt_meta["nb_frames"] != pred_meta["nb_frames"] or not _same_rate(
            gt_meta["r_frame_rate"], pred_meta["r_frame_rate"]
        )
        if mismatch:
            raise ValueError(
                f"{d}: {GT_NAME} ({gt_meta['nb_frames']} frames @ {gt_meta['r_frame_rate']}) and "
                f"{PRED_NAME} ({pred_meta['nb_frames']} frames @ {pred_meta['r_frame_rate']}) disagree -- "
                f"a residual requires equal frame counts and rates. ffmpeg's framesync would otherwise "
                f"silently repeat/pad the shorter stream, i.e. trailing garbage in an error visualization. "
                f"Re-generate this pair or remove the sample dir."
            )

    written: list[str] = []
    skipped: list[str] = []
    for d in sample_dirs:
        gt = os.path.join(d, GT_NAME)
        pred = os.path.join(d, PRED_NAME)
        out = os.path.join(d, residual_filename(gain))
        if os.path.exists(out) and not overwrite:
            skipped.append(out)
            continue
        tmp_out = _tmp_output_path(out)
        cmd = build_ffmpeg_cmd(gt, pred, tmp_out, gain)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            if os.path.exists(tmp_out):
                os.unlink(tmp_out)  # F3: never leave a partial encode behind
            raise RuntimeError(
                f"ffmpeg failed for {d} (exit {result.returncode}).\n"
                f"command: {' '.join(cmd)}\n"
                f"--- ffmpeg stderr ---\n{result.stderr}"
            )
        if not os.path.exists(tmp_out):
            raise RuntimeError(f"ffmpeg reported success but wrote no output: {tmp_out} (cannot finalize {out}).")
        os.replace(tmp_out, out)  # F3: atomic promotion -- skip-existing only ever sees complete files
        written.append(out)

    print(f"residual videos: wrote {len(written)}, skipped {len(skipped)} existing (gain ×{gain}) under {step_dir}")
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m maxdiffusion.make_wan_residual_videos",
        description="Write per-sample residual_gainN.mp4 (|pred - GT|, amplified xN) into a step dir.",
    )
    parser.add_argument("step_dir", help="local path to a pulled step_XXXXXX directory with sample_*/ dirs")
    parser.add_argument("--gain", type=int, default=4, help="integer amplification of the difference (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="re-encode residuals that already exist")
    args = parser.parse_args(argv)

    ensure_ffmpeg()
    written = process_step_dir(args.step_dir, gain=args.gain, overwrite=args.overwrite)
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
