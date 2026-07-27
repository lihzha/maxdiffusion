# Code review: exp_01 full_ft_overfit — round residual-videos
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-27

## Context loaded

- Query 10’s no-merge request covering six validation and sixteen training clips.
- Complete residual generator, gallery diff, new tests, and shared source-video writer.
- Empty diff for the pre-existing gallery test and exact no-residual equality with `HEAD` at SHA-256 `ac44cd…b3111`.
- All 44 inspected GT/pred streams matching at H.264 320×192 yuv420p, 16 fps, 33 frames, with identical unspecified color metadata.
- Pipe-only FFmpeg probes confirming identical→0 and gray 100 vs 40 at gain 3→179.
- Eighteen write-free tests passing; the requested full run blocked before collection solely by the read-only sandbox’s lack of a temporary directory.

## Verdict

APPROVE-WITH-CHANGES — Display-space gamma-RGB absolute difference with gain/clipping is an appropriate, clearly labeled residual visualization; the shared writer and inspected artifacts eliminate the matrix-mismatch concern here. The fourth-video layout and backward compatibility are acceptable, and nothing blocks application to the current Desktop data, but the edge cases below should be hardened.

## Findings

1. **MINOR — Mismatched durations silently produce false trailing residuals.** Evidence: the blend at [make_wan_residual_videos.py:95](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit/src/maxdiffusion/make_wan_residual_videos.py:95) leaves FFmpeg framesync defaults active (`shortest=false`, `repeatlast=true`); a 16-frame GT plus 8-frame prediction produced 16 residual frames by repeating the final prediction. Concrete fix: use `shortest=1:repeatlast=0`, or preflight and reject unequal frame counts/FPS; add an unequal-duration regression test.

2. **MINOR — The multiple-gain recovery instruction is ineffective.** Evidence: generation checks only the requested filename at [make_wan_residual_videos.py:164](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit/src/maxdiffusion/make_wan_residual_videos.py:164), so running gain 8 after gain 4 leaves both files, while the gallery advises `--overwrite` at [make_wan_val_gallery.py:182](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit/src/maxdiffusion/make_wan_val_gallery.py:182); that flag does not remove the gain-4 file. Concrete fix: fail-fast on residuals with another gain and instruct explicit removal, or add an explicit replace-all-gains option with a regression test.

3. **MINOR — Failed or interrupted encodes are not committed atomically.** Evidence: FFmpeg writes directly with `-y`, and the retry path skips any existing output at [make_wan_residual_videos.py:165](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit/src/maxdiffusion/make_wan_residual_videos.py:165); a partial file can therefore be mistaken for a completed residual. Concrete fix: encode to a sibling temporary file, atomically replace the final path only after success, and clean up the temporary file on failure.

---

# Strengthening record (Coder, same cycle — 2026-07-27; interrupted once by session-limit, resumed)

- **F1 (MINOR) — FIXED (strict + belt-and-suspenders).** ffprobe preflight on BOTH inputs of every sample dir (frame counts + fps by value via Fraction); unequal → ValueError naming both counts/rates; plus `shortest=1:repeatlast=0` in the blend so even direct callers can't get padded frames (reviewer's 4+8→8 repro reproduced, then eliminated). e2e unequal-duration test runs with real ffmpeg.
- **F2 (MINOR) — FIXED (strict fail-fast).** Mixed-gain dirs refuse with the stale files listed and explicit-deletion instruction, even under --overwrite; gallery advice string rewritten to match and test-pinned.
- **F3 (MINOR) — FIXED.** Encode to sibling `.tmp.mp4` → `os.replace` on rc==0 → temp unlinked on failure; failure leaves NO final and NO temp (fake subprocess models partial writes); argv/flow pin catches the direct-write mutant (the failure test alone would not — honestly noted).
- 3 mutants killed, sha-verified restores; suite 244→**253 passed + 2 skipped**; existing gallery tests byte-untouched.

**Cycle D closed:** write → review (APPROVE-WITH-CHANGES, 3 MINOR; colorspace decision verified by the reviewer against all 44 real streams) → strengthen (3 FIXED). 
