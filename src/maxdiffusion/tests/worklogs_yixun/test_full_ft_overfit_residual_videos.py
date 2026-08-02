"""CPU tests for per-sample residual videos + gallery residual rendering (Part II, cycle D).

Query 10 asks for per-sample RESIDUAL videos -- a visualization of |prediction - ground
truth| -- added to the step-dir galleries (the 6 val clips of report 03 and the 16 train
clips the user pulled locally). Two units are under test:

  * ``make_wan_residual_videos`` -- stdlib-only orchestration that shells out to ffmpeg
    once per ``sample_*`` dir to write ``residual_gainN.mp4``. ``build_ffmpeg_cmd`` is the
    single source of the command line so the exact filter chain is asserted without running
    ffmpeg; ``process_step_dir`` is exercised with ``subprocess.run`` monkeypatched (no
    ffmpeg needed) for the call-count / skip / overwrite / missing-input / failure paths.
  * ``make_wan_val_gallery`` -- a minimal, backward-compatible extension: when a sample dir
    holds exactly one ``residual_gain*.mp4`` it renders as a FOURTH card video labeled
    ``residual |pred - GT| xN`` plus one definition line near the provenance sentence. With
    no residual file the output is BYTE-IDENTICAL to the pre-cycle behavior -- pinned here
    by a sha256 of the current renderer's output (the golden) and by an exact ``_card``
    string, and by the untouched ``test_full_ft_overfit_val_gallery.py`` suite staying green.

The colorspace-sensitive END-TO-END test (skipped when ffmpeg is absent) synthesizes solid
grays, runs the real pipeline, and probes decoded RGB: a residual of two flat grays A,B must
be a near-uniform field of |A-B|*gain (clipped). That single mean+uniformity assertion is
what catches a wrong filter (addition vs difference), a dropped gain, or a difference taken
on gamma-space luma/chroma planes instead of decoded RGB.

Strengthen-phase contracts (Codex review F1-F3) also pinned here:

  * F1 -- ffmpeg's framesync defaults (shortest=0, repeatlast=1) silently repeat/pad the
    shorter input (reproduced: 4-frame GT + 8-frame pred -> 8 residual frames, the tail a
    frozen-pred artifact). ``process_step_dir`` PREFLIGHTS both inputs with ffprobe and
    REJECTS unequal frame counts / fps, naming both; the filter chain additionally carries
    ``shortest=1:repeatlast=0`` belt-and-suspenders for direct ``build_ffmpeg_cmd`` callers.
  * F2 -- a dir holding residuals at a DIFFERENT gain than requested is refused fast with
    the stale files listed. ``--overwrite`` only re-encodes the requested gain's filename,
    so it can never heal a mixed-gain dir; explicit deletion is the documented fix (the
    gallery's multiple-residual advice says the same).
  * F3 -- encodes land on a sibling ``residual_gainN.tmp.mp4`` and are ``os.replace``d onto
    the final name only on ffmpeg rc==0 (temp unlinked on failure), so an interrupted encode
    can never leave a partial file that skip-existing later trusts. The fake subprocess
    below deliberately CREATES the output file even on failure to model exactly that
    partial-write hazard.

Stdlib only -- the modules import no jax/tf, so this file does too.
"""

from __future__ import annotations

import hashlib
import os
import types

import pytest

import maxdiffusion.make_wan_residual_videos as rv
import maxdiffusion.make_wan_val_gallery as gallery

# ffmpeg + ffprobe are genuine runtime dependencies of the end-to-end tests only; pure-logic
# tests run everywhere. shutil.which is the importorskip-style guard the SOP asks for.
HAVE_FFMPEG = all(rv.shutil.which(b) for b in ("ffmpeg", "ffprobe"))
requires_ffmpeg = pytest.mark.skipif(
    not HAVE_FFMPEG, reason="ffmpeg/ffprobe not installed (end-to-end residual probe)"
)


# ==========================================================================================
# Fixture builders.
# ==========================================================================================


def _write_stub_mp4(path: str) -> None:
    """A placeholder file for the subprocess-monkeypatched tests (never decoded)."""
    with open(path, "wb") as f:
        f.write(b"\x00\x00")


def _make_step_tree(tmp_path, *, n=3, drop_inputs=None, preexisting_residual=None, gain=4):
    """Build ``step/sample_XXXX/{ground_truth,sample}.mp4`` stubs.

    ``drop_inputs`` maps sample idx -> filename to omit; ``preexisting_residual`` is a set of
    idx that already carry ``residual_gain{gain}.mp4`` (for the skip/overwrite paths).
    """
    drop_inputs = drop_inputs or {}
    preexisting_residual = preexisting_residual or set()
    step = tmp_path / "step_020000"
    step.mkdir()
    for i in range(n):
        d = step / f"sample_{i:04d}_clip{i}"
        d.mkdir()
        for fn in ("ground_truth.mp4", "sample.mp4"):
            if drop_inputs.get(i) == fn:
                continue
            _write_stub_mp4(str(d / fn))
        if i in preexisting_residual:
            _write_stub_mp4(str(d / f"residual_gain{gain}.mp4"))
    return step


class _FakeRun:
    """Dispatching ``subprocess.run`` stand-in (no real binaries needed).

    ``ffprobe`` argv -> canned per-path ``nb_frames``/``r_frame_rate`` stdout via
    ``probe_meta(path)`` (default: equal 4-frame 16/1 streams). ``ffmpeg`` argv -> writes a
    stub file at argv[-1] EVEN WHEN FAILING (modeling the real partial-output hazard F3
    guards against) and returns the configured rc/stderr.
    """

    def __init__(self, ffmpeg_returncode=0, ffmpeg_stderr="", probe_meta=None, probe_returncode=0, probe_stderr=""):
        self.calls = []
        self.ffmpeg_returncode = ffmpeg_returncode
        self.ffmpeg_stderr = ffmpeg_stderr
        self.probe_meta = probe_meta or (lambda path: (4, "16/1"))
        self.probe_returncode = probe_returncode
        self.probe_stderr = probe_stderr

    def __call__(self, cmd, *args, **kwargs):
        self.calls.append(list(cmd))
        prog = os.path.basename(cmd[0])
        if prog == "ffprobe":
            if self.probe_returncode != 0:
                return types.SimpleNamespace(returncode=self.probe_returncode, stdout="", stderr=self.probe_stderr)
            frames, fps = self.probe_meta(cmd[-1])
            stdout = f"nb_frames={frames}\nr_frame_rate={fps}\n"
            return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        # ffmpeg: create the target named by argv[-1] even on failure (partial-write model).
        with open(cmd[-1], "wb") as f:
            f.write(b"\x00")
        return types.SimpleNamespace(returncode=self.ffmpeg_returncode, stdout="", stderr=self.ffmpeg_stderr)

    @property
    def ffmpeg_calls(self):
        return [c for c in self.calls if os.path.basename(c[0]) == "ffmpeg"]

    @property
    def ffprobe_calls(self):
        return [c for c in self.calls if os.path.basename(c[0]) == "ffprobe"]


# ==========================================================================================
# (1) build_ffmpeg_cmd -- the single source of the command line.
# ==========================================================================================


def _filter_arg(cmd):
    i = cmd.index("-filter_complex")
    return cmd[i + 1]


def test_build_ffmpeg_cmd_core_structure():
    cmd = rv.build_ffmpeg_cmd("/a/gt.mp4", "/a/pred.mp4", "/a/residual_gain4.mp4", 4)
    assert cmd[0] == "ffmpeg"
    filt = _filter_arg(cmd)
    # Difference is computed on DECODED RGB (both inputs forced to rgb24 before blend), not
    # on gamma-space luma: the two format=rgb24 conversions must precede the blend.
    assert filt.count("format=rgb24") == 2
    assert "blend=all_mode=difference" in filt
    assert filt.index("format=rgb24") < filt.index("blend=all_mode=difference")
    # F1 belt-and-suspenders: framesync must never pad the shorter stream, even for a caller
    # that bypasses process_step_dir's ffprobe preflight.
    assert "blend=all_mode=difference:shortest=1:repeatlast=0" in filt
    # Gain/clip is a per-channel RGB lut, and the browser-compatible re-encode is yuv420p.
    assert "lutrgb" in filt
    assert filt.rstrip().endswith("format=yuv420p[out]")


def test_build_ffmpeg_cmd_input_order_gt_then_pred():
    cmd = rv.build_ffmpeg_cmd("/x/GROUND.mp4", "/x/PRED.mp4", "/x/out.mp4", 4)
    i_positions = [i for i, a in enumerate(cmd) if a == "-i"]
    assert len(i_positions) == 2, "exactly two inputs"
    assert cmd[i_positions[0] + 1] == "/x/GROUND.mp4"  # GT is input 0
    assert cmd[i_positions[1] + 1] == "/x/PRED.mp4"  # pred is input 1
    # And the filter binds [0:v] to GT, [1:v] to pred (order is load-bearing for the label).
    filt = _filter_arg(cmd)
    assert "[0:v]format=rgb24[gt]" in filt
    assert "[1:v]format=rgb24[pred]" in filt


def test_build_ffmpeg_cmd_gain_appears_in_lut_stage():
    for gain in (2, 4, 8):
        cmd = rv.build_ffmpeg_cmd("/gt.mp4", "/pred.mp4", f"/residual_gain{gain}.mp4", gain)
        filt = _filter_arg(cmd)
        # The gain multiplies the decoded difference inside the lut, once per RGB channel.
        assert filt.count(f"val*{gain}") == 3
        # The multiply lives in the lutrgb stage (not e.g. a stray token elsewhere).
        lut = filt[filt.index("lutrgb") :]
        assert f"val*{gain}" in lut


def test_build_ffmpeg_cmd_output_last_faststart_yuv420p():
    cmd = rv.build_ffmpeg_cmd("/gt.mp4", "/pred.mp4", "/out/residual_gain4.mp4", 4)
    assert cmd[-1] == "/out/residual_gain4.mp4"  # output path is last
    assert "+faststart" in cmd
    assert cmd[cmd.index("+faststart") - 1] == "-movflags"
    # yuv420p appears both as the container pixel format and inside the filter's final stage.
    assert "yuv420p" in cmd
    assert cmd[cmd.index("yuv420p") - 1] == "-pix_fmt"


@pytest.mark.parametrize("bad", [0, -1, -4])
def test_build_ffmpeg_cmd_nonpositive_gain_raises(bad):
    with pytest.raises(ValueError) as ei:
        rv.build_ffmpeg_cmd("/gt.mp4", "/pred.mp4", "/out.mp4", bad)
    assert "gain" in str(ei.value)


@pytest.mark.parametrize("bad", [2.5, "4", True, None])
def test_build_ffmpeg_cmd_non_integer_gain_raises(bad):
    # residual_gainN.mp4 needs an integer N; reject non-int (bool included) to keep the name
    # deterministic and the filename<->label round trip exact.
    with pytest.raises(ValueError):
        rv.build_ffmpeg_cmd("/gt.mp4", "/pred.mp4", "/out.mp4", bad)


# ==========================================================================================
# (2) process_step_dir -- fake subprocess (no ffmpeg needed).
# ==========================================================================================


def test_process_step_dir_one_ffmpeg_call_per_complete_sample(tmp_path, monkeypatch, capsys):
    step = _make_step_tree(tmp_path, n=3)
    fake = _FakeRun()
    monkeypatch.setattr(rv.subprocess, "run", fake)
    written = rv.process_step_dir(str(step), gain=4, overwrite=False)
    assert len(fake.ffmpeg_calls) == 3  # one encode per complete sample dir
    assert len(fake.ffprobe_calls) == 2 * 3  # F1 preflight probes BOTH inputs of every dir
    assert len(written) == 3
    for p in written:
        assert p.endswith("residual_gain4.mp4") and os.path.exists(p)  # finals materialized
    # Encodes target the sibling temp (F3); the command is build_ffmpeg_cmd's output.
    for call in fake.ffmpeg_calls:
        assert call[0] == "ffmpeg" and call[-1].endswith("residual_gain4.tmp.mp4")
    out = capsys.readouterr().out
    assert "wrote 3" in out and "skipped 0" in out


def test_process_step_dir_skips_existing_by_default(tmp_path, monkeypatch, capsys):
    step = _make_step_tree(tmp_path, n=3, preexisting_residual={1}, gain=4)
    fake = _FakeRun()
    monkeypatch.setattr(rv.subprocess, "run", fake)
    written = rv.process_step_dir(str(step), gain=4, overwrite=False)
    assert len(fake.ffmpeg_calls) == 2  # sample 1 already has residual_gain4.mp4 -> skipped
    assert len(written) == 2
    assert all("sample_0001" not in p for p in written)
    assert "skipped 1" in capsys.readouterr().out


def test_process_step_dir_overwrite_reruns_existing(tmp_path, monkeypatch):
    step = _make_step_tree(tmp_path, n=3, preexisting_residual={1}, gain=4)
    fake = _FakeRun()
    monkeypatch.setattr(rv.subprocess, "run", fake)
    written = rv.process_step_dir(str(step), gain=4, overwrite=True)
    assert len(fake.ffmpeg_calls) == 3  # --overwrite re-runs the pre-existing one too
    assert len(written) == 3


def test_process_step_dir_gain_flows_into_filename_and_cmd(tmp_path, monkeypatch):
    step = _make_step_tree(tmp_path, n=1)
    fake = _FakeRun()
    monkeypatch.setattr(rv.subprocess, "run", fake)
    written = rv.process_step_dir(str(step), gain=8, overwrite=False)
    assert written[0].endswith("residual_gain8.mp4")
    encode = fake.ffmpeg_calls[0]
    assert "val*8" in encode[encode.index("-filter_complex") + 1]


def test_process_step_dir_missing_input_is_actionable(tmp_path, monkeypatch):
    step = _make_step_tree(tmp_path, n=3, drop_inputs={2: "sample.mp4"})
    fake = _FakeRun()
    monkeypatch.setattr(rv.subprocess, "run", fake)
    with pytest.raises(FileNotFoundError) as ei:
        rv.process_step_dir(str(step), gain=4, overwrite=False)
    msg = str(ei.value)
    assert "sample.mp4" in msg  # names the missing input
    assert "sample_0002" in msg  # and the exact offending dir
    assert fake.calls == []  # validated before ANY subprocess work


def test_process_step_dir_ffmpeg_failure_surfaces_stderr(tmp_path, monkeypatch):
    step = _make_step_tree(tmp_path, n=1)
    fake = _FakeRun(ffmpeg_returncode=1, ffmpeg_stderr="x264 [error]: bogus filter chain")
    monkeypatch.setattr(rv.subprocess, "run", fake)
    with pytest.raises(RuntimeError) as ei:
        rv.process_step_dir(str(step), gain=4, overwrite=False)
    msg = str(ei.value)
    assert "x264 [error]: bogus filter chain" in msg  # ffmpeg's own stderr is surfaced
    assert "sample_0000" in msg
    # F3: the failed encode leaves NO final and NO temp -- even though the fake subprocess
    # DID create the temp file (modeling ffmpeg's real partial output on failure).
    assert list(step.rglob("residual_gain4.mp4")) == []
    assert list(step.rglob("*.tmp.mp4")) == []


# ---- F1: strict duration/fps preflight (mock ffprobe -- no binaries needed) ----


def test_process_step_dir_unequal_frame_counts_rejected(tmp_path, monkeypatch):
    step = _make_step_tree(tmp_path, n=2)
    # GT decodes to 4 frames, pred to 8 -- framesync would silently pad; we must refuse.
    fake = _FakeRun(probe_meta=lambda p: (8, "16/1") if p.endswith("sample.mp4") else (4, "16/1"))
    monkeypatch.setattr(rv.subprocess, "run", fake)
    with pytest.raises(ValueError) as ei:
        rv.process_step_dir(str(step), gain=4, overwrite=False)
    msg = str(ei.value)
    assert "4" in msg and "8" in msg  # names BOTH frame counts
    assert "ground_truth.mp4" in msg and "sample.mp4" in msg
    assert "sample_0000" in msg  # names the offending dir
    assert fake.ffmpeg_calls == []  # rejected before ANY encode


def test_process_step_dir_unequal_fps_rejected(tmp_path, monkeypatch):
    step = _make_step_tree(tmp_path, n=1)
    fake = _FakeRun(probe_meta=lambda p: (4, "30/1") if p.endswith("sample.mp4") else (4, "16/1"))
    monkeypatch.setattr(rv.subprocess, "run", fake)
    with pytest.raises(ValueError) as ei:
        rv.process_step_dir(str(step), gain=4, overwrite=False)
    msg = str(ei.value)
    assert "16/1" in msg and "30/1" in msg  # names both rates
    assert fake.ffmpeg_calls == []


def test_process_step_dir_equivalent_fps_notation_not_rejected(tmp_path, monkeypatch):
    # 16/1 and 16000/1000 are the same rate -- the comparison is by value, not by string.
    step = _make_step_tree(tmp_path, n=1)
    fake = _FakeRun(probe_meta=lambda p: (4, "16000/1000") if p.endswith("sample.mp4") else (4, "16/1"))
    monkeypatch.setattr(rv.subprocess, "run", fake)
    written = rv.process_step_dir(str(step), gain=4, overwrite=False)
    assert len(written) == 1


def test_process_step_dir_probe_failure_is_actionable(tmp_path, monkeypatch):
    step = _make_step_tree(tmp_path, n=1)
    fake = _FakeRun(probe_returncode=1, probe_stderr="moov atom not found")
    monkeypatch.setattr(rv.subprocess, "run", fake)
    with pytest.raises(RuntimeError) as ei:
        rv.process_step_dir(str(step), gain=4, overwrite=False)
    msg = str(ei.value)
    assert "ffprobe" in msg and "moov atom not found" in msg
    assert fake.ffmpeg_calls == []


# ---- F2: mixed-gain dirs are refused fast ----


def test_process_step_dir_mixed_gain_rejected(tmp_path, monkeypatch):
    # Dir 0 still holds a gain-8 residual; requesting gain 4 must refuse and list it
    # (--overwrite could only ever rewrite residual_gain4.mp4, never clean the stale 8).
    step = _make_step_tree(tmp_path, n=2, preexisting_residual={0}, gain=8)
    fake = _FakeRun()
    monkeypatch.setattr(rv.subprocess, "run", fake)
    with pytest.raises(ValueError) as ei:
        rv.process_step_dir(str(step), gain=4, overwrite=False)
    msg = str(ei.value)
    assert "residual_gain8.mp4" in msg  # lists the stale file
    assert "sample_0000" in msg  # names the dir
    assert "delete" in msg.lower()  # instructs explicit removal
    assert fake.ffmpeg_calls == []


def test_process_step_dir_mixed_gain_rejected_even_with_overwrite(tmp_path, monkeypatch):
    step = _make_step_tree(tmp_path, n=1, preexisting_residual={0}, gain=8)
    fake = _FakeRun()
    monkeypatch.setattr(rv.subprocess, "run", fake)
    with pytest.raises(ValueError):
        rv.process_step_dir(str(step), gain=4, overwrite=True)
    assert fake.ffmpeg_calls == []


def test_process_step_dir_same_gain_existing_is_not_mixed(tmp_path, monkeypatch):
    # A residual at the REQUESTED gain is the normal skip/overwrite case, never an error.
    step = _make_step_tree(tmp_path, n=2, preexisting_residual={0}, gain=4)
    fake = _FakeRun()
    monkeypatch.setattr(rv.subprocess, "run", fake)
    written = rv.process_step_dir(str(step), gain=4, overwrite=False)
    assert len(written) == 1  # dir 0 skipped, dir 1 written


# ---- F3: atomic temp-then-replace ----


def test_process_step_dir_atomic_temp_then_replace_flow(tmp_path, monkeypatch):
    step = _make_step_tree(tmp_path, n=2)
    fake = _FakeRun()
    monkeypatch.setattr(rv.subprocess, "run", fake)
    written = rv.process_step_dir(str(step), gain=4, overwrite=False)
    # Every encode targets the sibling temp, never the final name directly...
    for call in fake.ffmpeg_calls:
        assert call[-1].endswith("residual_gain4.tmp.mp4")
    # ...and success leaves exactly the finals, with no temp anywhere.
    for p in written:
        assert os.path.exists(p) and p.endswith("residual_gain4.mp4")
    assert list(step.rglob("*.tmp.mp4")) == []


def test_process_step_dir_no_sample_dirs_is_actionable(tmp_path):
    empty = tmp_path / "step_empty"
    empty.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)) as ei:
        rv.process_step_dir(str(empty), gain=4, overwrite=False)
    assert "sample_" in str(ei.value)


def test_ensure_ffmpeg_missing_is_actionable(monkeypatch):
    monkeypatch.setattr(rv.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError) as ei:
        rv.ensure_ffmpeg()
    msg = str(ei.value)
    assert "ffmpeg" in msg
    assert "ffprobe" in msg  # both binaries are startup-guarded (F1 preflight needs ffprobe)


# ==========================================================================================
# (3) END-TO-END pixel probe -- skipped when ffmpeg is absent.
#     This is the test that catches a wrong colorspace / filter choice.
# ==========================================================================================

_E2E_SIZE = (64, 48)  # even dims so 4:2:0 is exact; small for speed.


def _make_color_video(path, color, size=_E2E_SIZE, frames=4, fps=16):
    w, h = size
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={w}x{h}:r={fps}",
        "-frames:v",
        str(frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        path,
    ]
    rv.subprocess.run(cmd, check=True, capture_output=True)


def _probe_stream(path):
    out = rv.subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,pix_fmt,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    d = {}
    for line in out.strip().splitlines():
        k, _, v = line.partition("=")
        d[k] = v
    return d


def _decoded_rgb_stats(path):
    """Decode the residual to raw rgb24 bytes and return (mean, mn, mx)."""
    raw = rv.subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-vf", "format=rgb24", "-f", "rawvideo", "-"],
        check=True,
        capture_output=True,
    ).stdout
    assert len(raw) > 0
    return sum(raw) / len(raw), min(raw), max(raw)


@requires_ffmpeg
def test_e2e_geometry_and_frames_preserved(tmp_path):
    step = tmp_path / "step_e2e"
    d = step / "sample_0000_clip0"
    d.mkdir(parents=True)
    _make_color_video(str(d / "ground_truth.mp4"), "0x646464")
    _make_color_video(str(d / "sample.mp4"), "0x282828")
    written = rv.process_step_dir(str(step), gain=3, overwrite=False)
    assert len(written) == 1 and os.path.exists(written[0])
    probe = _probe_stream(written[0])
    assert (int(probe["width"]), int(probe["height"])) == _E2E_SIZE  # input geometry preserved
    assert int(probe["nb_frames"]) == 4
    assert probe["pix_fmt"] == "yuv420p"  # browser-compatible
    assert probe["r_frame_rate"] == "16/1"  # same fps as the inputs (blend inherits input 0)
    assert list(d.glob("*.tmp.mp4")) == []  # F3: no temp remains after a successful encode


@requires_ffmpeg
def test_e2e_unequal_duration_rejected(tmp_path):
    # F1 with the REAL ffprobe: a 4-frame GT vs an 8-frame pred must be refused up front
    # (ffmpeg's framesync default would otherwise pad to 8 residual frames -- reproduced).
    step = tmp_path / "step_unequal"
    d = step / "sample_0000_clip0"
    d.mkdir(parents=True)
    _make_color_video(str(d / "ground_truth.mp4"), "0x646464", frames=4)
    _make_color_video(str(d / "sample.mp4"), "0x282828", frames=8)
    with pytest.raises(ValueError) as ei:
        rv.process_step_dir(str(step), gain=4, overwrite=False)
    msg = str(ei.value)
    assert "4" in msg and "8" in msg  # both real frame counts named
    assert not (d / "residual_gain4.mp4").exists()  # nothing written
    assert list(d.glob("*.tmp.mp4")) == []


@requires_ffmpeg
def test_e2e_identical_inputs_are_near_black(tmp_path):
    step = tmp_path / "step_black"
    d = step / "sample_0000_clip0"
    d.mkdir(parents=True)
    _make_color_video(str(d / "ground_truth.mp4"), "0x555555")
    # Identical content in both inputs -> |A - A| = 0 everywhere, regardless of gain.
    import shutil as _sh

    _sh.copyfile(str(d / "ground_truth.mp4"), str(d / "sample.mp4"))
    gain = 4
    written = rv.process_step_dir(str(step), gain=gain, overwrite=False)
    _mean, _mn, mx = _decoded_rgb_stats(written[0])
    assert mx <= 2 * gain  # near-black: max pixel <= small_epsilon * gain


@requires_ffmpeg
def test_e2e_two_grays_uniform_bright_equals_diff_times_gain(tmp_path):
    # GT=100, pred=40 -> |100-40|*gain must appear as a NEAR-UNIFORM bright field on decoded
    # RGB. This one assertion is red under: addition-not-difference (mean ~253), dropped gain
    # (mean ~60), or a difference taken on YUV luma/chroma (non-uniform, wrong mean).
    step = tmp_path / "step_grays"
    d = step / "sample_0000_clip0"
    d.mkdir(parents=True)
    _make_color_video(str(d / "ground_truth.mp4"), "0x646464")  # 100,100,100
    _make_color_video(str(d / "sample.mp4"), "0x282828")  # 40,40,40
    gain = 3
    expected = abs(100 - 40) * gain  # 180, below the 255 clip
    written = rv.process_step_dir(str(step), gain=gain, overwrite=False)
    mean, mn, mx = _decoded_rgb_stats(written[0])
    assert abs(mean - expected) <= 15, f"mean {mean} not ~= |A-B|*gain {expected}"
    assert (mx - mn) <= 8, f"residual of flat inputs must be near-uniform; spread {mx - mn}"


# ==========================================================================================
# (4) Gallery extension -- backward-compatible fourth video + definition line.
# ==========================================================================================

# Golden of the CURRENT (pre-cycle) renderer output for a no-residual sample set. The
# no-residual branch must reproduce these bytes exactly (byte-identity contract).
_GOLDEN_META = {
    "title": "golden",
    "run_name": "r",
    "checkpoint_step": 20000,
    "seed": 0,
    "dataset_path": "gs://x/val",
    "commits": {"generation_commit": "deadbee"},
    "mean_latent_mse": 0.25,
    "mean_pixel_mse": 0.019,
    "mean_ssim": 0.78,
    "num_samples": 2,
}
_GOLDEN_SHA = "ac44cd3bc5b9c1d7fc102134b6be66023fe119a9c9f70b7e45a438898a4b3111"


def _golden_sample(i, *, residual_gain=None):
    s = {
        "sample_index": i,
        "dataset_position": i * 100,
        "stored_ordinal": 9000 + i,
        "name": f"c{i}",
        "latent_mse": 0.20 + i * 0.01,
        "pixel_mse": 0.01,
        "ssim_avg": 0.70,
        "videos": {
            "ground_truth": f"sample_{i:04d}_c{i}/ground_truth.mp4",
            "sample": f"sample_{i:04d}_c{i}/sample.mp4",
            "comparison": f"sample_{i:04d}_c{i}/comparison_gt_top_pred_bottom.mp4",
        },
    }
    if residual_gain is not None:
        s["videos"]["residual"] = f"sample_{i:04d}_c{i}/residual_gain{residual_gain}.mp4"
        s["residual_gain"] = residual_gain
    return s


def test_gallery_no_residual_output_is_byte_identical():
    html = gallery.build_gallery_html(_GOLDEN_META, [_golden_sample(0), _golden_sample(1)])
    assert hashlib.sha256(html.encode()).hexdigest() == _GOLDEN_SHA
    assert html.count("<video") == 2 * 3  # still exactly three videos per card


def test_gallery_no_residual_has_no_new_markers():
    html = gallery.build_gallery_html(_GOLDEN_META, [_golden_sample(0), _golden_sample(1)])
    for marker in ("residual |pred", "Residual videos show", "residual_gain", "residual-note"):
        assert marker not in html


# The exact legacy card string (captured from the pre-cycle renderer) -- pins _card's
# no-residual output byte-for-byte.
_LEGACY_CARD_0 = (
    '<section class="card"><div class="ids"><div class="dpos"><span class="dpos-label">dataset position</span> '
    '<span class="dpos-value">0</span></div><div class="sub"><span class="ord-label">stored ordinal</span> '
    '<span class="ord-value">9000</span> · <span class="sname">c0</span> · '
    '<span class="sidx">sample_index 0</span></div></div><div class="videos"><figure class="vid">'
    '<figcaption>ground truth</figcaption><video controls preload="metadata" '
    'src="sample_0000_c0/ground_truth.mp4"></video></figure><figure class="vid">'
    '<figcaption>prediction</figcaption><video controls preload="metadata" '
    'src="sample_0000_c0/sample.mp4"></video></figure><figure class="vid">'
    "<figcaption>comparison (GT top / prediction bottom)</figcaption>"
    '<video controls preload="metadata" src="sample_0000_c0/comparison_gt_top_pred_bottom.mp4"></video>'
    '</figure></div><div class="metrics"><span class="metric">latent MSE <b>0.2000</b></span>'
    '<span class="metric">pixel MSE <b>0.0100</b></span><span class="metric">SSIM <b>0.7000</b></span>'
    "</div></section>"
)


def test_gallery_card_byte_identical_when_no_residual():
    assert gallery._card(_golden_sample(0)) == _LEGACY_CARD_0


def test_gallery_renders_residual_fourth_video_and_definition():
    samples = [_golden_sample(0, residual_gain=4), _golden_sample(1, residual_gain=4)]
    html = gallery.build_gallery_html(_GOLDEN_META, samples)
    # Fourth video per card, labeled with the gain parsed from the filename.
    assert html.count("<video") == 2 * 4
    assert html.count("residual |pred − GT| ×4") == 2
    assert "sample_0000_c0/residual_gain4.mp4" in html
    # One definition line, near (after) the provenance sentence.
    defline = (
        "Residual videos show the per-pixel absolute difference between prediction and "
        "ground truth, amplified ×4 and clipped; brighter = larger error."
    )
    assert html.count(defline) == 1
    assert html.index(gallery.PROVENANCE) < html.index(defline)
    # Single source of truth for the definition text (parametric in gain).
    assert gallery.RESIDUAL_DEFINITION_TEMPLATE.format(gain=4) == defline


def test_gallery_residual_gain_label_is_parsed_not_hardcoded():
    samples = [_golden_sample(0, residual_gain=8)]
    html = gallery.build_gallery_html(_GOLDEN_META, samples)
    assert "residual |pred − GT| ×8" in html
    assert "amplified ×8 and clipped" in html
    assert "×4" not in html  # nothing hardcoded to 4


# ---- gallery real-filesystem detection (write_gallery) ----


def _make_gallery_step_dir(tmp_path, *, residuals_per_sample):
    """A minimal FULL_FT val step dir (2 samples) with configurable residual files."""
    import json

    step = tmp_path / "step_020000"
    step.mkdir()
    config = {
        "run_name": "wan-full-ft-test",
        "checkpoint_step": 20000,
        "eval_data_dir": "gs://x/val",
        "seed": 0,
        "model_type": "FULL_FT_TI2V",
        "validation_ordinals": [0, 2927],
    }
    with open(step / "config.json", "w") as f:
        json.dump(config, f)
    with open(step / "summary.json", "w") as f:
        json.dump({"num_samples": 2, "mean_latent_mse": 0.2, "mean_pixel_mse": 0.01, "mean_ssim": 0.7}, f)
    for i in range(2):
        d = step / f"sample_{i:04d}_c{i}"
        d.mkdir()
        with open(d / "metrics.json", "w") as f:
            json.dump(
                {
                    "sample_index": i,
                    "ordinal": 9000 + i,
                    "name": f"c{i}",
                    "latent_mse": 0.2,
                    "pixel_mse": 0.01,
                    "ssim_avg": 0.7,
                },
                f,
            )
        for fn in ("ground_truth.mp4", "sample.mp4", "comparison_gt_top_pred_bottom.mp4"):
            _write_stub_mp4(str(d / fn))
        for name in residuals_per_sample.get(i, ()):
            _write_stub_mp4(str(d / name))
    return step


def test_write_gallery_detects_single_residual_and_renders_it(tmp_path):
    step = _make_gallery_step_dir(
        tmp_path, residuals_per_sample={0: ["residual_gain4.mp4"], 1: ["residual_gain4.mp4"]}
    )
    out = gallery.write_gallery(str(step))
    html = open(out).read()
    assert html.count("residual |pred − GT| ×4") == 2
    assert "amplified ×4 and clipped" in html
    # residual refs are relative like the other videos.
    import re

    for src in re.findall(r'src="([^"]+)"', html):
        assert not src.startswith("/") and "://" not in src
    assert any("residual_gain4.mp4" in src for src in re.findall(r'src="([^"]+)"', html))


def test_write_gallery_multiple_residuals_in_one_dir_is_actionable(tmp_path):
    step = _make_gallery_step_dir(
        tmp_path,
        residuals_per_sample={0: ["residual_gain4.mp4", "residual_gain8.mp4"], 1: ["residual_gain4.mp4"]},
    )
    with pytest.raises(ValueError) as ei:
        gallery.write_gallery(str(step))
    msg = str(ei.value)
    assert "residual" in msg
    assert "sample_0000" in msg  # names the offending dir
    # F2: the advice must match the residual tool's real contract -- explicit deletion.
    # (--overwrite only re-encodes the requested gain's filename; it can't heal mixed gains.)
    assert "Delete the stale file(s)" in msg
    assert "mixed-gain" in msg


def test_write_gallery_no_residual_is_unchanged(tmp_path):
    step = _make_gallery_step_dir(tmp_path, residuals_per_sample={})
    html = open(gallery.write_gallery(str(step))).read()
    for marker in ("residual |pred", "Residual videos show", "residual_gain"):
        assert marker not in html
    assert html.count("<video") == 2 * 3


# Sanity: the residual glob only matches the integer-gain pattern (a stray file is ignored,
# not miscounted as a second residual).
def test_write_gallery_ignores_non_integer_residual_name(tmp_path):
    step = _make_gallery_step_dir(
        tmp_path, residuals_per_sample={0: ["residual_gain4.mp4", "residual_gainX.mp4"], 1: ["residual_gain4.mp4"]}
    )
    out = gallery.write_gallery(str(step))  # residual_gainX.mp4 is not a valid residual -> no error
    html = open(out).read()
    assert html.count("residual |pred − GT| ×4") == 2
