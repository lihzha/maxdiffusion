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
- **Job id:** `20260729-062523-1937c065-exp02-overfit100-probe-yixun` (submitted 2026-07-29T06:25Z; outcome pending)

## Job 4 — v6e-8 PROBE dataset build, attempt 2 (post tarball-guard fix) — launched 2026-07-29

Commit: `53d69f53185b6edc1867eb6be8b0540c18a501ad` (pushed). Same command as Job 3 except the SHA and `--env COMMIT`; Job 3 FAILED on the guard's git assumption (real bug, fixed in `49f4412`+`53d69f5`).

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
  --env COMMIT="53d69f53185b6edc1867eb6be8b0540c18a501ad" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  -- bash bash_scripts/build_overfit100_dataset.sh
```
- **Job id:** `20260729-172443-23bcb17a-exp02-overfit100-probe-yixun` (submitted 2026-07-29T17:24Z; outcome pending)

## Job 5 — v6e-8 PROBE dataset build, attempt 3 (post ffmpeg-ensure) — launched 2026-07-30

Commit: `934f80f` (pushed). Same as Job 4 except SHA/COMMIT. Job 4: queue attempt 1 spot-preempted (infra), attempt 2 APPLICATION_ERROR = no ffmpeg on TPU image (fixed in `934f80f`; V1 PASSED before the crash).

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
  --env COMMIT="934f80f60964af1d6b83635f15e0fb664970704c" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  -- bash bash_scripts/build_overfit100_dataset.sh
```
- **Job id:** `20260729-184156-72473301-exp02-overfit100-probe-yixun` (submitted 2026-07-29T18:41Z local-queue time; outcome pending)

## Job 6 — v6e-8 PROBE dataset build, attempt 4 (post deliberate probe2/ cleanup) — launched 2026-07-30

Commit: `934f80f60964af1d6b83635f15e0fb664970704c` (unchanged from Job 5 — no code change; stale `probe2/failed_gates.json` archived + deliberately deleted per the B3 guard's instruction). Same command as Job 5.
- **Job id:** `20260730-011201-16c5f6c8-exp02-overfit100-probe-yixun` (submitted; outcome pending)
- **Job 6 outcome:** attempt 1 SUCCEEDED in substance (probe2/ published with `_SUCCESS`, all gates passed) but the VM was suspended at completion; queue auto-retry hit the B3 guard (dataset present → deliberate refusal, exit 1) so the queue label is FAILED. `_SUCCESS` authoritative. PROBE PASSED.

## Job 7 — v6e-8 FULL dataset build (train100 + train10) — launched 2026-07-30

Commit: `934f80f60964af1d6b83635f15e0fb664970704c` (unchanged; probe-passed SHA). Pre-approved by Query 4 (dual sign-off; probe log-verified per `_worklog.md` 06:10Z entry). Same command as Job 5/6 with `PROBE=0`:

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100
tpu create v6 -n 8 --name exp02-overfit100-fullbuild-yixun \
  --worker0-only \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env PROBE=0 \
  --env MANIFEST_PATH="docs/worklogs_yixun/exp_02_overfit100_claude/overfit100_manifest.json" \
  --env OUT_ROOT="gs://v6_east1d/datasets/exp02_overfit100" \
  --env CONFIG_PATH="src/maxdiffusion/configs/base_wan_5b_full_ft.yml" \
  --env CONFIG_OVERRIDES="hardware=tpu" \
  --env COMMIT="934f80f60964af1d6b83635f15e0fb664970704c" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  -- bash bash_scripts/build_overfit100_dataset.sh
```
- **Job id:** `20260730-060037-ca6303aa-exp02-overfit100-fullbuild-yixun` (submitted 2026-07-30T06:00Z; outcome pending — `_SUCCESS` in train100/ + train10/ is the authoritative signal)

## Job 8 — v6e-8 FULL dataset build, attempt 2 (recalibrated V2 envelope) — launched 2026-07-30

Commit: `a08051e0005a20ba6bd58331eec1c349dc604430` (pushed; includes v2-envelope `bad4bff` — the only builder change vs the probe-passed SHA — plus cycle C trainer code, which the build does not import). Same command as Job 7 with the new COMMIT.
- **Job id:** `20260730-075213-732dec74-exp02-overfit100-fullbuild-yixun` (submitted 2026-07-30T07:52Z). Note: `COMMIT` env resolved to `319ed93` (= `a08051e` + one docs-only commit — code-identical builder); `_SUCCESS.build_commit` will read `319ed93…`.
- **Job 8 outcome:** attempt 1 spot-preempted (infra); attempt 2 SUCCEEDED in 642.9 s — train100 1,629/7 shards + train10 167/1 shard published with `_SUCCESS` (build_commit 319ed93). Dataset complete.

## Job 9 — v6e-8 S1 SMOKE (train10, 20 steps, storage-light) — launched 2026-07-30

Commit: `835085dbe2eb17b6d982eb6183e9d345cc655dac` (pushed). Pre-approved by Query 4 (dual sign-off in `_worklog.md` 21:30Z).

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100
tpu create v6 -n 8 --name exp02-overfit100-s1smoke-yixun \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env RUN_NAME="wan-overfit100-s1smoke-$(date -u +%Y%m%d-%H%M%S)" \
  --env MAX_TRAIN_STEPS=20 \
  --env DATA_DIR="gs://v6_east1d/datasets/exp02_overfit100/train10" \
  --env EXPECTED_WINDOWS=167 --env NUM_TEXT_SLOTS=10 \
  --env COMMIT="835085dbe2eb17b6d982eb6183e9d345cc655dac" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  -- bash bash_scripts/train_wan_overfit100.sh
```
- **Job id:** `20260730-142902-8eec4725-exp02-overfit100-s1smoke-yixun` (submitted; outcome pending)
- **Job 9 outcome:** SUCCEEDED attempt 1 (15.4 min) — S1 PASSED log-verified (see `_worklog.md` 23:40Z). 1.82 steps/s on v6e-8.

## Job 10 — v6e-8 S2 GATE RUN (train10, 2,500 steps, checkpoints [250,500,1000,2500]) — launched 2026-07-30

Commit: `e70062ef096ffb51c2a74862d41ec32c33922f16` (pushed). Pre-approved by Query 6 (S1 condition met, log-verified).

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100
tpu create v6 -n 8 --name exp02-overfit100-s2gate-yixun \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env RUN_NAME="wan-overfit100-s2gate-20260730" \
  --env MAX_TRAIN_STEPS=2500 \
  --env DATA_DIR="gs://v6_east1d/datasets/exp02_overfit100/train10" \
  --env EXPECTED_WINDOWS=167 --env NUM_TEXT_SLOTS=10 \
  --env COMMIT="e70062ef096ffb51c2a74862d41ec32c33922f16" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  -- bash bash_scripts/train_wan_overfit100.sh
```
- **Job id:** `20260730-173902-9b00e8e7-exp02-overfit100-s2gate-yixun` (submitted; outcome pending)
- **Job 10 outcome:** SUCCEEDED attempt 1 (36 min). Loss 0.533→0.061 (still falling), 4 checkpoints retained. S2 training PASSED log-verified.

## Jobs 11–14 — v6e-8 S2 GATE EVALS (role s2_gate, ckpts 250/500/1000/2500) — launched 2026-07-30

Commit: `a1d0fa84829e34b4871c551c1836ec277c138c0b`. Pre-approved by Query 6. One job per checkpoint; modes correct@all + ablations@2500; seeds 0,1,2; canonical windows; no videos.
- **Job ids:** `20260730-184240-…-s2eval-250`, `20260730-184307-…-s2eval-500`, `20260730-184333-…-s2eval-1000`, `20260730-184400-…-s2eval-2500` (submitted; outcomes pending)
## Job 15 — v6e-64 S3 TRAINING (train100, 2,500 steps, ckpts [250,500,1000,1750,2500]) — launched 2026-07-30

Commit: `d670809b208c9c68dbacf7563a6e7178eb1ecb5b` (pushed). Approved by Query 7 (option A).

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100
tpu create v6 -n 64 --name exp02-overfit100-s3train-yixun \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env RUN_NAME="wan-overfit100-s3-20260730" \
  --env MAX_TRAIN_STEPS=2500 \
  --env CHECKPOINT_STEPS="[250,500,1000,1750,2500]" \
  --env DATA_DIR="gs://v6_east1d/datasets/exp02_overfit100/train100" \
  --env EXPECTED_WINDOWS=1629 --env NUM_TEXT_SLOTS=100 \
  --env COMMIT="d670809b208c9c68dbacf7563a6e7178eb1ecb5b" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  -- bash bash_scripts/train_wan_overfit100.sh
```
- **Job id:** `20260731-005432-ce7e6955-exp02-overfit100-s3train-yixun` (submitted; outcome pending)
- **Job 15 outcome:** FAILED attempt 1 — worker 15 SIGABRT in jax.distributed.initialize (16-host barrier; infra, pre-code). Resubmitted below.

## Job 16 — v6e-64 S3 TRAINING resubmit — launched 2026-07-31

Commit: `b52dcc6430da83251ca2a061f88cef6758b718b3` (post aux-fix; trainer path code-identical to Job 15's SHA). Same command as Job 15 with the new COMMIT.
- **Job id:** `20260731-012023-3bbac3d0-exp02-overfit100-s3train-yixun` (submitted; outcome pending)
