# exp_02 `overfit100` — command log

Every launch appended AT LAUNCH TIME (SOP artifact 7). Failed/superseded runs stay, marked. `_worklog.md` records why; this file records how to reproduce.

All cycle-A commands are **local CPU only** (no TPU, no remote job). The interpreter is the exp_01
worktree's venv driven with the exp_02 worktree as cwd + `PYTHONPATH=src` (the exp_02 worktree has
no `.venv` of its own yet; a dedicated venv is deferred to cycle C, when JAX/TPU deps matter):

```bash
export EXP02=/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100
export PY=/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_01_full_ft_overfit/.venv/bin/python
export SCRATCH=/private/tmp/claude-501/-Users-yixunhu-Home-maxdiffusion/c2031678-0562-461a-a938-1a894cf96cf2/scratchpad
```

## A0. Cycle-A unit suite (validation-ladder rung 1) — 2026-07-29T02:00Z

- **Status:** PASSED — 316 passed, 2 skipped (253+2 inherited from exp_01, 63 new in cycle A).
- **Commit:** working tree on `claude-exp_02_overfit100-20260728` (base `b88bac1`; cycle-A code uncommitted at run time — committed after the Codex review + strengthen phase).
- **Command:**
```bash
cd $EXP02 && PYTHONPATH=src $PY -m pytest src/maxdiffusion/tests/worklogs_yixun/ -q
```

## A1. V1-fixture extraction + GCS publish — 2026-07-29T02:14Z

- **Status:** SUCCEEDED. Published `gs://v6_east1d/datasets/exp02_overfit100/fixtures/v1_cache_windows.npz`, generation `1785291278937267`, md5 `Szm9uNUI2AtjyRNTtpm9SA==`, 693,246 bytes, windows `ep0_v0_s00000/s00004/s00008`.
- **Approval:** cycle A assigned by the Planner after Query-4 plan approval; local CPU + one small GCS write, no TPU (announcement 02 not engaged).
- **Command (real run, uploads):**
```bash
cd $EXP02 && PYTHONPATH=src $PY -m maxdiffusion.data_preprocessing.extract_v1_fixture \
  --out-fingerprint docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_fixture_fingerprint.json \
  --npz-path $SCRATCH/v1_cache_windows.npz
```
- **Dry pass first (no GCS write), same output bytes/md5:** add `--skip-upload` and point `--out-fingerprint`/`--npz-path` at `$SCRATCH`.
- **Artifacts:** `docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_fixture_fingerprint.json`.

## A2. overfit100 manifest build (seed 0, n=100) — 2026-07-29T02:22Z

- **Status:** SUCCEEDED (exit 0, ~25 min wall: 02:17:57Z → 02:42:2xZ). 100 episodes / 1,629 windows accepted in 129 draws; tally `{accepted: 100, not_success: 23, too_short: 6}`.
- **Approval:** as A1 (local CPU; read-only GCS access + MP4 downloads to a temp dir).
- **Command:**
```bash
cd $EXP02 && PYTHONPATH=src $PY -m maxdiffusion.data_preprocessing.build_overfit100_manifest \
  --seed 0 --n 100 \
  --out docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json \
  --fixture-fingerprint docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_fixture_fingerprint.json \
  --block-size 25 --tmp-dir $SCRATCH/manifest_full_tmp \
  > docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_2026-07-28_cycleA_manifest_build.log 2>&1
```
- **Artifacts:** `overfit100_manifest.json`, `overfit100_2026-07-28_cycleA_manifest_build.log`.
- **Preceding bounded probes (rungs 3–4, artifacts in `$SCRATCH`, not committed):**
  - `--n 5 --block-size 10 --dry-run` → 8 draws, 5 provisional accepts, 32 s (annotation path only).
  - `--n 3 --block-size 10` (full path incl. stat + MP4 download + ffprobe) → 6 draws, 3 accepts, 53 windows, 77 s.
  - `verify_manifest()` against the n=3 manifest → `[]` clean; with an injected md5/generation change → both drifts reported.

## A2b. Manifest REBUILD from the clean commit (seed 0, n=100) — 2026-07-29T04:0xZ

- **Status:** LAUNCHED. **Supersedes A2.** The Codex review (finding A1, BLOCKER) established that A2's artifact recorded `builder_commit b88bac1` — a commit containing none of the five cycle-A files. The strengthened builder now refuses to run from an uncommitted implementation, so this is the first manifest whose `builder_commit` names the code that produced it.
- **Commit:** `9a24518` (cycle A2, reviewed + strengthened) on `claude-exp_02_overfit100-20260728`; implementation verified clean at HEAD by `assert_implementation_committed()` before the walk starts.
- **Prerequisite verified:** the published fixture was re-downloaded from GCS and passed the strengthened `verify_fixture()` (`[]`); live generation `1785291278937267`, md5 match — so `overfit100_fixture_fingerprint.json` is reused unchanged.
- **Command:**
```bash
cd $EXP02 && PYTHONPATH=src $PY -m maxdiffusion.data_preprocessing.build_overfit100_manifest \
  --seed 0 --n 100 \
  --out docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json \
  --fixture-fingerprint docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_fixture_fingerprint.json \
  --block-size 25 --tmp-dir $SCRATCH/manifest_rebuild_tmp \
  > docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_2026-07-28_cycleA_manifest_rebuild.log 2>&1
```
- **Acceptance criteria (written before the launch):** exit 0; `builder_commit == 9a24518cbe82f35386e607169410f751fb1b7af7`; `tool_versions` carries python/gsutil/ffprobe/ffmpeg/numpy/jax; 100 episodes / 1,629 windows; `episodes`, `draw_log` and `totals` **identical** to the superseded A2 artifact (same seed, same corpus — any difference means the source drifted); the builder's own structural gate passes; a live `verify_manifest()` returns `[]`.
- **Artifacts:** `overfit100_manifest.json` (overwritten), `overfit100_2026-07-28_cycleA_manifest_rebuild.log`. The superseded A2 log stays in the folder as the record.

## A3. Post-build verification of the committed manifest (rung 3) — 2026-07-29T02:5xZ

- **Status:** PASSED (re-run after A2b — see below). 201 fingerprinted objects (1 fixture + 100 annotations + 100 MP4s) re-stat'ed against live GCS, `drift errors: []`, 100.7 s. This is exactly the call the cycle-B build job makes at preflight; after the strengthening it also runs `validate_manifest_structure()` first and fails closed without touching the network.
- **Command:**
```bash
cd $EXP02 && PYTHONPATH=src $PY -c "
import json
from maxdiffusion.data_preprocessing.build_overfit100_manifest import verify_manifest
m = json.load(open('docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json'))
print(verify_manifest(m))
"
```

## Job 3 — v6e-8 PROBE dataset build (rung 4) — launched 2026-07-29

Commit: `4783ed4817a0f26da2b73e61d92340ed87a5e6eb` (pushed). Pre-approved by Query 4 (dual sign-off recorded in `_worklog.md`).

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100
tpu create v6 -n 8 --name exp02-overfit100-probe-yixun \
  --worker0-only \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env PROBE=1 \
  --env MANIFEST_PATH="docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json" \
  --env OUT_ROOT="gs://v6_east1d/datasets/exp02_overfit100" \
  --env CONFIG_PATH="src/maxdiffusion/configs/base_wan_5b_full_ft.yml" \
  --env CONFIG_OVERRIDES="hardware=tpu" \
  --env COMMIT="4783ed4817a0f26da2b73e61d92340ed87a5e6eb" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  -- bash bash_scripts/build_overfit100_dataset.sh
```

Notes: `--worker0-only` because the builder is single-process and v6e-8 is two queue workers; the script prefetches ONLY `model_index.json vae/*` at the manifest's pinned revision. Job id + outcome appended below after submission.
