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
- **Job 16 outcome:** SUCCEEDED attempt 1 (39 min); loss 0.586→0.145; 5 checkpoints. S3 training PASSED log-verified.

## Jobs 17–22 — v6e-8 S3 EVALS (4 intermediates + segment-final@2500 + full-set@2500) — launched 2026-07-31

Commit: `e27fdc37df3c9a10d6059833c4078160a955b8fc`. Approved by Query 7.
- **Job ids:** `…-160842-…-i250`, `…-160849-…-i500`, `…-160855-…-i1000`, `…-160901-…-i1750`, `…-160907-…-final2500`, `…-160912-…-fullset` (all 20260731; outcomes pending)

## Job 23 — v6e-64 S3 EXTENSION to 10k (resume from step 2500) — launched 2026-08-01

Commit: `ee10749bdd3d2224fe93f9834a828ed20d26aa79` (launch tip; code files byte-identical to review-APPROVED fc9ac52 — only docs commits on top; trainer path byte-identical to e27fdc3 lineage — resume series touched eval/launchers only, verified by empty `git diff e27fdc3..fc9ac52 -- trainers/ input_pipeline/ pyconfig.py`). **Approved by Yixun: "extend to 10k" (2026-08-01, in response to the complete two-tier step-2500 verdict).**

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100
tpu create v6 -n 64 --name exp02-overfit100-s3ext10k-yixun \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env RUN_NAME="wan-overfit100-s3-20260730" \
  --env MAX_TRAIN_STEPS=10000 \
  --env CHECKPOINT_STEPS="[250,500,1000,1750,2500,5000,7500,10000]" \
  --env DATA_DIR="gs://v6_east1d/datasets/exp02_overfit100/train100" \
  --env EXPECTED_WINDOWS=1629 --env NUM_TEXT_SLOTS=100 \
  --env COMMIT="ee10749bdd3d2224fe93f9834a828ed20d26aa79" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  -- bash bash_scripts/train_wan_overfit100.sh
```

- **Mechanics:** same RUN_NAME → Orbax restores step-2500 (params/opt_state/step; dataloader reseeded seed+2500 per the documented partial-resume convention); LR = absolute 250-step warmup then CONSTANT 1e-5 (D9 — segment-invariant, extension-safe by design); checkpoint_steps extends the retained list, planner saves only {5000, 7500, 10000}.
- **Cost estimate:** first segment did 2,500 steps in 39 min on v6e-64 → 7,500 steps ≈ **2 h** compute (plus queue/preemption weather).
- **Acceptance:** preflights green (pinned snapshot, dataset byte-verify, manifest-bound context table); restore reports start_step=2500; loss resumes ≈0.145 and declines; no NaN; 3 new checkpoints saved and retained; ~1.07 steps/s.
- **Follow-on evals (same approval umbrella, D11 structure):** s3_intermediate at 5000 and 7500 (seed-0 canonical), segment-final 3×3 + full-set at 10000 — launched at fc9ac52 after training lands (resume staging ON, ffmpeg fix live → ceilings populate).
- **Job id:** `20260801-032202-ee7d478b-exp02-overfit100-s3ext10k-yixun` (submitted 2026-08-01T03:22Z). Final `COMMIT=81ae5717cf631e654c6f2af918360a6e98787c3c` — the tip at submission (docs-only commits above ee10749; code byte-identical to review-APPROVED fc9ac52 throughout).

## Job 24 — v6e-8 SAMPLING-STEPS PROBE (H1/H2 discriminator) @ ckpt 2500 — launched 2026-08-01

Commit: `f4210037f795f605baf579c9e04243d200ed01a7` (tip at submission; probe code = review-APPROVED `a921917`, docs commits above). **Approved by Yixun: "approve sampling probe" (2026-08-01); design approval-pinned in code (arms {25,50,100}, ckpt 2500, baseline 25, 30 canonical windows).**

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100
tpu create v6 -n 8 --worker0-only --name exp02-o100-probe-steps-yixun \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env RUN_NAME="wan-overfit100-s3-20260730" \
  --env CHECKPOINT_STEP=2500 \
  --env COMMIT="f4210037f795f605baf579c9e04243d200ed01a7" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  -- bash bash_scripts/probe_wan_overfit100_sampling.sh
```

- **What:** 30 seeded canonical windows × sampling arms {25, 50, 100} at step-2500, seed 0, correct context; standalone diagnostic, verdict-isolated by construction (canonical output only: `validation_probe_sampling/probe_steps_ckpt2500.json`, confirmed unoccupied pre-launch).
- **Cost:** ~55 min on v6e-8 (37 min rollouts + decodes + setup); no resume — a preemption restarts it (single writer confirmed: no prior attempt exists). Predeclared fallback if weather turns: resubmit with PROBE_NUM_WINDOWS=15.
- **Acceptance:** (i) validity — probe 25-arm agrees row-level with the landed segment-final seed-0 correct rows for the same 30 windows (reference mean 0.8100125855 / median 0.8059329625; mismatch invalidates the PROBE, not the checkpoint); (ii) read-out — paired per-window deltas 50−25 and 100−25: material lift ⇒ H2 discretization component real; flat ⇒ sampling-side H2 excluded, residual gap is the training objective (H1 + objective mismatch).
- **Job id:** `20260801-042558-e41be63a-exp02-o100-probe-steps-yixun` (submitted 2026-08-01T04:26Z; COMMIT=c9224534 = tip at submission, probe code = APPROVED a921917).

## Jobs 25–28 — v6e-8 EXTENSION EVALS (i5000, i7500, segment-final@10000, full-set@10000) — launched 2026-08-01

Under the "extend to 10k" approval umbrella (Yixun 2026-08-01; D11 eval structure per Query 7 precedent). Training Job 23 SUCCEEDED attempt 1: resume verified (step 2501 loss 0.139), loss 0.145→0.132 (5000)→0.127 (7500)→≈0.12 (10000, noisy 0.111–0.137) — strong flattening; grad_norm healthy; 1.9 steps/s; checkpoints {5000,7500,10000} verified on GCS. Eval code = tip (eval-resume series APPROVED at fc9ac52 + probe round; staging/resume ON, ffmpeg-ensure live → ceilings populate; preemptions now cost only the incomplete tail).

Common: `RUN_NAME=wan-overfit100-s3-20260730`, COMMIT=<tip at submission, recorded below>, launcher `bash_scripts/validate_wan_overfit100.sh`, v6e-8.

| Job | CHECKPOINT_STEP | EVAL_PASS_ROLE | EVAL_WINDOWS | ROLLOUT_SEEDS | CONTEXT_MODES |
| --- | --- | --- | --- | --- | --- |
| 25 i5000 | 5000 | s3_intermediate | canonical | 0 | correct |
| 26 i7500 | 7500 | s3_intermediate | canonical | 0 | correct |
| 27 final10000 | 10000 | s3_segment_final | canonical | 0,1,2 | correct,null,shuffled |
| 28 fullset10000 | 10000 | s3_full_set | all | 0 | correct |

- **Acceptance:** role_validation ok per pass; immutable role-keyed artifacts (`step_005000_s3_intermediate/` etc.); aux_coverage 1.0 this time (ffmpeg fixed); verdict CLI then re-computes the two-tier claim over ALL admitted artifacts (c* by fraction tie-break between 2500 and 10000).
- **Launch note:** first submission attempt 2026-08-01T08:20Z failed on gcloud reauth (issue #6, 5th recurrence) creating NO jobs; relaunched cleanly after Yixun re-authed. `COMMIT=46e5f41…` — actual: `46c5f411738d4cfdcf7c3a16245b191c80d02e89` (tip at submission; eval code = APPROVED fc9ac52 lineage + probe round, docs commits above).
- **Job ids (submitted 2026-08-01T14:28–14:29Z):** `20260801-142820-66e1b5ce-…-i5000`, `20260801-142847-33331211-…-i7500`, `20260801-142914-9d6458a0-…-final10k`, `20260801-142941-95e2e0a1-…-fullset10k`

## Jobs 29–30 — v6e-8 VIDEO PASSES (write_videos=True) @ ckpt 2500 + 10000 — launched 2026-08-01

**Approved by Yixun** ("Please give me 5 ground truth vs. pred videos" + AskUserQuestion: representative
spread, both checkpoints). Commit `04223639032bdc4ee4236c65ccfdb6f3c7f2a725` (eval code = APPROVED fc9ac52 lineage; docs commits above).

Purpose: results visualization. No verdict impact — these write to **fresh** `step_00XXXX_s3_intermediate/`
role dirs (neither checkpoint had an s3_intermediate pass; no collision with any verdict artifact). The run
signature binds `write_videos`, so these are distinct runs and cannot admit prior passes' staged rows.

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100
for STEP in 2500 10000; do
  tpu create v6 -n 8 --worker0-only --name exp02-o100-vid-${STEP}-yixun \
    --code-dir . \
    --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
    --env RUN_NAME="wan-overfit100-s3-20260730" \
    --env CHECKPOINT_STEP="$STEP" --env EVAL_PASS_ROLE=s3_intermediate \
    --env EVAL_WINDOWS=canonical --env ROLLOUT_SEEDS=0 --env CONTEXT_MODES=correct \
    --env WRITE_VIDEOS=True \
    --env COMMIT="04223639032bdc4ee4236c65ccfdb6f3c7f2a725" \
    --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
    -- bash bash_scripts/validate_wan_overfit100.sh
done
```

- **Cost:** ~40–50 min each on v6e-8 (rollouts + 300 mp4 encodes per pass); resume staging protects both.
- **Acceptance:** role_validation ok; 100 windows; `comparison_gt_top_pred_bottom.mp4` present per window;
  the seed-0 SSIM column must reproduce the already-landed values for these checkpoints (0.8139 @ 2500,
  0.8416 @ 10000) — a mismatch would invalidate the video pass, not the verdict.
- **The 5 windows to deliver** (representative spread by 10k m_corr, same windows at both checkpoints):
  worst `ep30738_v0_s00132` (0.6756) · 25th `ep4358_v0_s00040` (0.8068) · median `ep4015_v0_s00000`
  (0.8446) · 75th `ep50125_v0_s00028` (0.8805) · best `ep36295_v0_s00020` (0.9484).
- **Job ids (submitted 2026-08-01T18:14Z):** ckpt 2500 → `20260801-181353-3e7419e4-exp02-o100-vid-2500-yixun`; ckpt 10000 → `20260801-181420-9e61cb62-exp02-o100-vid-10000-yixun`.

## Job 31 — v6e-8 DIAGNOSTIC D2: fixed-RNG one-step loss instrument, all 8 checkpoints — launched 2026-08-01

**Approved by Yixun** ("run the three cheap diagnostics"). Closes the instrument gap called out in
`_analysis.md` §3/§5-H5: the reported losses were noisy *training* logs, never the deterministic
all-window one-step instrument. Distinguishes "the objective saturated" from "this optimization run saturated".

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100
tpu create v6 -n 8 --worker0-only --name exp02-o100-d2-valloss-yixun \
  --code-dir . \
  --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
  --env RUN_NAME="wan-overfit100-s3-20260730" \
  --env CHECKPOINT_STEPS="250,500,1000,1750,2500,5000,7500,10000" \
  --env TRAIN_COMMIT="81ae5717cf631e654c6f2af918360a6e98787c3c" \
  --env COMMIT="09f65f6be078374108812403d3fb8dc83bee3843" \
  --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
  -- bash bash_scripts/eval_wan_overfit100_val_loss.sh
```

- **What it measures:** per-window one-step denoising loss at a **deterministic (t, ε) keyed by
  (episode_id, window_start)** — identical noise draw at every checkpoint, so the curve is comparable
  across checkpoints in a way training logs never were. All 1,629 windows, seed 0.
- **TRAIN_COMMIT nuance (documented, not hidden):** checkpoints ≤2500 were produced by Job 16
  (`b52dcc6`) and >2500 by the Job 23 extension (`81ae571`). One SHA is stamped for the whole sweep;
  this is sound because the trainer path was verified **byte-identical** between them
  (`git diff` over `trainers/`, `input_pipeline/`, `pyconfig.py` is empty). The extension SHA is used.
- **Cost:** ~30–50 min on v6e-8 (8 Orbax restores of the 5B + 1,629 one-step forwards each; no rollouts,
  no VAE decode — the loss arm frees the VAE at startup).
- **Acceptance:** 8 checkpoint rows; per-window count 1,629 each; loss finite and monotone-ish; the value at
  2500/10000 should sit in the neighbourhood of the training logs (~0.145 / ~0.12) but is the *authoritative*
  number since it is fixed-RNG.
- **Job id:** `20260803-025332-62de7eb5-exp02-o100-lr5e5ext-yixun` (submitted; COMMIT=bf912a574ae0a5f1ec4d70e72285c5741abbed9a).

## Jobs 32–34 — v6e-64 LR SWEEP: 3 arms × +2,500 steps from ckpt 10000 — launched 2026-08-02

**Approved by Yixun** ("run the LR sweep at 2e-5 and 5e-5" + AskUserQuestion: add the 1e-5 control arm).
Purpose: bound how much of the remaining 74% one-step-loss gap (D2) is reachable by optimization alone —
the one lever the diagnostics could not rule out; Wan 2.1's own pretraining used 1e-4 (tech report p.15),
so 2e-5/5e-5 are well inside the architecture's stable regime.

**Design.** Each arm = fresh RUN_NAME seeded by a server-side copy of the step-10,000 checkpoint (22.03 GiB)
into its own `checkpoints/` dir; Orbax restores it as latest and continues to 12,500. Warmup (absolute, 250
steps) is long past, so each arm jumps straight to its LR on restore — Adam moments carry over; an instability
would show as loss divergence and is an acceptable, informative outcome for the 5e-5 arm. No code changes:
`LEARNING_RATE` was already plumbed (train_wan_overfit100.sh:113/207).

Commit `2fcbeba1911a34c9db57ac6c8ebc718b688f2ce3`; trainer path unchanged (byte-identical since e27fdc3 lineage).

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_02_overfit100
for ARM_LR in "lr1e5c:1e-5" "lr2e5:2e-5" "lr5e5:5e-5"; do
  ARM=${ARM_LR%%:*}; LR=${ARM_LR##*:}
  tpu create v6 -n 64 --name "exp02-o100-${ARM}-yixun" \
    --code-dir . \
    --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
    --env RUN_NAME="wan-overfit100-s3ext-${ARM}-20260802" \
    --env LEARNING_RATE="$LR" \
    --env MAX_TRAIN_STEPS=12500 \
    --env CHECKPOINT_STEPS="[250,500,1000,1750,2500,5000,7500,10000,12500]" \
    --env DATA_DIR="gs://v6_east1d/datasets/exp02_overfit100/train100" \
    --env EXPECTED_WINDOWS=1629 --env NUM_TEXT_SLOTS=100 \
    --env COMMIT="2fcbeba1911a34c9db57ac6c8ebc718b688f2ce3" \
    --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
    -- bash bash_scripts/train_wan_overfit100.sh
done
```

- **Cost:** 3 × ~39 min v6e-64 (arms run 2,500 steps each; save scheduler plans only step 12,500 — the listed
  earlier steps are < start_step and skipped).
- **Acceptance (per arm):** preflights green; restore reports start_step=10000; `lr=` line shows the arm's
  LR; loss continues from ≈0.12 without NaN/divergence (5e-5 divergence = informative negative, not a bug);
  checkpoint 12,500 saved.
- **Measurement (follow-on, same approval):** one loss-instrument job per arm (`CHECKPOINT_STEPS="10000,12500"`,
  RUN_NAME per arm) — the 10,000 reading must reproduce **0.12227** exactly (identical bytes; validity anchor);
  the 12,500 readings are the A/B/C answer. Baseline for context: 1e-5's measured decelerating rate
  ≈0.0035/2,500 at this point; the control arm measures it directly.
- **Seed verification:** all three arm checkpoints byte-identical to source (23,654,557,930 bytes each). Note: first copy attempt hung — gsutil `-m` multiprocessing deadlock on macOS (21 orphaned workers, 0 objects); fixed with `-o "GSUtil:parallel_process_count=1"` (threads only). Recorded for the next 22 GiB copy.
- **Job ids (submitted 2026-08-02T17:22Z):** lr1e5c → `20260802-172153-c53c6a93-exp02-o100-lr1e5c-yixun`; lr2e5 → `20260802-172220-d3198513-exp02-o100-lr2e5-yixun`; lr5e5 → `20260802-172247-e5ce902d-exp02-o100-lr5e5-yixun`.

## Jobs 35–36 — v6e-8 LOSS-INSTRUMENT for completed LR-sweep arms — launched 2026-08-02T~22:10Z

Predeclared measurement from the Jobs 32–34 package (same approval). lr1e5c + lr2e5 SUCCEEDED attempt 1
(checkpoints 12500 verified); lr5e5 on attempt 2 after a maintenance kill (restart-from-10k per design —
its instrument job launches when it lands). CHECKPOINT_STEPS="10000,12500" per arm: the 10000 reading must
reproduce the anchor **0.12227** exactly (identical bytes = validity check), 12500 is the answer.
Preliminary from training logs (noisy, non-authoritative): lr2e5 ended at logged loss ≈0.093 — well below
the 1e-5 trend; instrument decides.

## Job 37 — v6e-8 LOSS-INSTRUMENT for lr5e5 — launched 2026-08-02T~23:20Z

Same predeclared measurement as Jobs 35–36. lr5e5 SUCCEEDED attempt 3 (two maintenance kills; each restart
from the 10k seed per the no-intermediate-saves design). Job id: `20260802-225142-dba871b9-exp02-o100-inst-lr5e5-yixun`.

## Jobs 38–39 — v6e-8 SSIM EVALS at lr2e5-12500 + lr5e5-12500 — launched 2026-08-03

**Approved by Yixun ("approve the SSIM evals").** Settled eval path (validate_wan_overfit100.sh,
s3_intermediate, seed 0, correct, canonical-100; resume staging + ffmpeg live). Purpose: does the exp_02
loss→SSIM line (SSIM ≈ 0.9885 − 1.201·loss, fit on the 1e-5 path) hold for checkpoints reached at higher
LR? Predictions ON the line: lr2e5 (loss 0.09793) → 0.8709; lr5e5 (loss 0.06061) → **0.9157**.
Line-break in either direction is decision-grade (see _results.md sweep section).

## Job 40 — v6e-64 lr5e5 EXTENSION 12,500 → 15,000 — launched 2026-08-03

**Approved by Yixun ("approve lr5e5 extension").** Same run (`wan-overfit100-s3ext-lr5e5-20260802`)
resumes from its own step-12,500 checkpoint; LEARNING_RATE=5e-5; checkpoint 15,000 added. Purpose: measure
the 5e-5 segment's own deceleration and chase the D11 bar (11/100 ≥ 0.95 at 12,500; the line puts
90%-at-0.95 near mean loss ≈0.03–0.04). Follow-on under the same approval (established pattern):
instrument job at {12500 anchor = 0.06061 exact, 15000} + SSIM eval at 15,000 when training lands.
- **Job id:** `20260803-025332-62de7eb5-exp02-o100-lr5e5ext-yixun` (submitted; COMMIT=bf912a574ae0a5f1ec4d70e72285c5741abbed9a).

## Jobs 41–43 — lr5e5-extension measurement pair — launched 2026-08-03T~05:30Z

Extension training (Job 40) SUCCEEDED attempt 2 (one maintenance kill; resumed from its own 12,500
checkpoint); checkpoint 15,000 verified. Follow-on measurements per the Job-40 approval:
- **Job 41 (CANCELLED, my error, disclosed):** `20260803-052906-4f0d7f08-exp02-o100-inst-l5x-yixun` was
  launched with a WRONG `TRAIN_COMMIT` — a chimera of the training SHA and the HF revision string
  (`…cf768468…`). It would have stamped bad provenance into every row. Cancelled within 3 minutes of
  submission; nothing consumed it.
- **Job 42 (instrument, corrected):** `20260803-054641-2e826802-exp02-o100-inst-l5x2-yixun` —
  CHECKPOINT_STEPS="12500,15000", output to `validation_loss_15000/` (fresh dir; the run's earlier
  instrument artifact at `validation_loss/` stays immutable), TRAIN_COMMIT=81ae5717cf631e… (correct).
  Anchor: 12,500 must reproduce **0.06061**.
- **Job 43 (SSIM @15000):** `20260803-052937-c63c0868-exp02-o100-ssim-l5x-yixun` — s3_intermediate,
  seed 0, correct, canonical-100 (takes no TRAIN_COMMIT; unaffected by the Job-41 error).

### Jobs 41–43 corrections (2026-08-03T~06:20Z)

- Job 42 (instrument) PUBLISHED: **12,500 anchor 0.06061 reproduced exactly; 15,000 = 0.03927** (n=1,629,
  train_commit correct). Line-predicted SSIM at 15,000 ≈ **0.9413**.
- Job 43's submission was killed by a local 2-min shell timeout mid-flight — only `code.tar.gz` uploaded,
  no job record; it will never run. Relaunched as **Job 43b**: `20260803-063541-649d9e59-exp02-o100-ssim-l5x2-yixun` (same spec, same approval).

## Job 44 — v6e-64 lr5e5 CONTINUATION 15,000 → 17,500 — launched 2026-08-03

**Approved by Yixun ("continue le5e5 to 17500").** Same run resumes from its own 15,000 checkpoint; LR
5e-5; checkpoint 17,500 added. Projection on the line: mean loss ≈0.025–0.030 → mean SSIM ≈0.953–0.959;
if seed-0 shows ≳90/100 ≥ 0.95, the formal 3-seed + full-set verdict passes get proposed. Follow-on under
this approval: instrument {15000 anchor = 0.03927, 17500} + SSIM at 17,500 on landing.
- **Job id:** `20260803-152023-05ca5770-exp02-o100-lr5e5x2-yixun` (COMMIT=0bb4dba37972efab37aed91312f4d98e56a8044c).

### Job 44 resubmit (2026-08-03T~17:25Z)

Job 44 FAILED attempt 2: worker-5 SIGABRT inside `maybe_initialize_jax_distributed_system` (16-host
barrier, pre-code) — same infra signature as exp_02 S3 attempt 1. Standing auto-resubmit policy (no
code/config change): **Job 44b** `20260803-171525-545493a4-exp02-o100-lr5e5x2b-yixun`.

## Jobs 45–46 — lr5e5@17,500 measurement pair — launched 2026-08-03T~20:10Z

Job 44b SUCCEEDED (17,500 reached; two prior infra kills classified + resubmitted on record). Per the
Job-44 approval: instrument `20260803-201009-74336dfa-…-inst-l17` ({15000 anchor = 0.03927 exact, 17500},
fresh output dir `validation_loss_17500/`) + SSIM `20260803-201058-94fb61a2-exp02-o100-ssim-l17-yixun` (s3_intermediate, canonical, seed 0). Projection
on the line: mean loss ~0.025–0.030 → mean SSIM ~0.953–0.959; the bar count (51/100 → ?) decides whether
the formal 3-seed + full-set verdict proposal goes to Yixun.

### Jobs 45–46 outcome (2026-08-04T~00:10Z)

- **Job 45 (instrument):** 15,000 anchor **0.03927 reproduced exactly**; 17,500 = **0.03476** (n=1,629).
  Segment gains −0.0617 → −0.0213 → −0.0045 per 2,500 steps: the 5e-5 trajectory itself flattened ~4.7×.
- **Job 46 (SSIM, attempt 5 after infra kills):** canonical seed-0 mean **0.9508** — the mean crossed the
  0.95 bar; **62/100 windows ≥ 0.95** (arc 0 → 11 → 51 → 62). Line held: predicted 0.9467, actual +0.0041.
  Decision menu (a)/(b)/(c) presented to Yixun.

## Job 47 — v6e-64 LR 1e-4 PROBE SEGMENT 17,500 → 20,000 — launched 2026-08-04T02:11Z

**Approved by Yixun ("you can use (a) for the exp_02 next step").** Option (a): ONE probe segment at
LR **1e-4** (Wan 2.1's own pretraining LR) from the lr5e5 run's step-17,500 checkpoint — does a
Wan-native LR restore the collapsed pace, or does 1e-4 destabilize this far into memorization?
- **Seed:** server-side copy of `wan-overfit100-s3ext-lr5e5-20260802/checkpoints/17500` →
  `wan-overfit100-s3ext-lr1e4-20260804/checkpoints/17500`, verified byte-identical (23,626,279,825 B;
  single-process gsutil per the Jobs 32–34 lesson).
- **Env:** RUN_NAME=wan-overfit100-s3ext-lr1e4-20260804, LEARNING_RATE=1e-4, MAX_TRAIN_STEPS=20000,
  CHECKPOINT_STEPS="[17500,20000]", train100 pins unchanged, COMMIT=a215d6096d8029bbbe301c2ff513627ced910e98
  (tip at submission; trainer path byte-identical lineage since e27fdc3).
- **Cost:** one ~39-min v6e-64 segment + the standard measurement pair after landing.
- **Acceptance:** restore reports start_step=17500; `lr=` line shows 1e-4; loss continues from ≈0.0348
  without NaN (divergence at 1e-4 = informative negative, not a bug); checkpoint 20,000 saved.
- **Follow-on under this approval (established pattern):** instrument {17500 anchor = **0.03476** exact,
  20000} to fresh `validation_loss_20000/` + SSIM eval at 20,000 (s3_intermediate, canonical, seed 0).
  Reading: segment gain ≫0.0045 ⇒ LR was still the binding lever; gain ≈0.0045 or divergence ⇒ the 5e-5
  echo plateau is not LR-curable at this depth.
- **Job id:** `20260804-021109-19080ff3-exp02-o100-lr1e4-yixun`.

### Job 47 failure + 47b (2026-08-04T~02:35Z)

Job 47 FAILED attempt 1: worker-9 SIGABRT (exit 134) inside `jax.distributed.initialize` — the
16-host barrier, pre-code; byte-for-byte the Job 44 / S3-attempt-1 infra signature (verified in
worker-9.log before classifying). Standing auto-resubmit policy applies (infra, no code/config
change). The session's permission layer intermittently blocked the resubmit; handed to Yixun.
**Job 47b:** `20260804-025418-b2e2884f-exp02-o100-lr1e4b-yixun` (submitted by Yixun in-session, 2026-08-04T02:54Z; identical spec, COMMIT=899cb5d tip — docs-only delta from a215d60, trainer bytes unchanged).

### Job 47b outcome (2026-08-04T04:01Z) — SUCCEEDED

Restored at 17,500 in the new run dir, ran the 1e-4 segment, saved checkpoint 20,000, exit 0
(single attempt). LR confirmed in config (`learning_rate: 0.0001`). Step-loss lines went to wandb
only (no process-0 stdout lines in any worker log) — divergence/health is read authoritatively by
the instrument below.

## Jobs 48–49 — lr1e4@20,000 measurement pair — launched 2026-08-04T04:08Z

Per the Job-47 (option-a) approval, established pattern (Jobs 42/43, 45/46):
- **Job 48 (instrument):** `20260804-040821-1d5cfe7d-exp02-o100-inst-l1e4-yixun` —
  CHECKPOINT_STEPS="17500,20000" on the lr1e4 run, fresh `validation_loss_20000/`,
  TRAIN_COMMIT=899cb5d27189ae8925c6c08cc8ea58e7dc3aec5a (47b), COMMIT=bd3cbef (eval tip, byte-identical
  eval lineage). **Validity anchor: 17,500 must reproduce 0.03476 exactly** (copied checkpoint,
  identical bytes). The 20,000 reading is the option-(a) answer: segment gain ≫0.0045 ⇒ LR was still
  the binding lever; ≈0.0045 or worse/NaN ⇒ the plateau is not LR-curable at this depth.
- **Job 49 (SSIM @20,000):** `20260804-040837-c6136dbd-exp02-o100-ssim-l1e4-yixun` — s3_intermediate,
  canonical, seed 0, correct (takes no TRAIN_COMMIT). Line check: predicted SSIM = 0.9885 − 1.201 ×
  (Job-48's 20,000 loss); sixth independent test of the law, first at 1e-4.

### Jobs 48–49 outcome (2026-08-04T~05:30Z) — the option-(a) answer

- **Job 48 (instrument):** 17,500 anchor **0.0347633288 reproduced exactly** (copied checkpoint
  validated). 20,000 = **0.03320** (n=1,629, stderr 0.00128). Segment gain **−0.00156** — about a
  THIRD of the last 5e-5 segment's −0.0045 and far from pace restoration. For reference, a ÷4.7
  extrapolation of 5e-5's own decay projected ≈−0.001 for its next segment: 1e-4 is marginally above
  that projection and firmly inside the flat regime. **No divergence** (loss still fell; stable
  at 1e-4 even 20k steps deep in memorization).
- **Job 49 (SSIM @20,000):** canonical seed-0 mean **0.9536** (from 0.9508), **67/100 ≥ 0.95** (from
  62), median 0.9584, max 0.9793, min 0.8721. Line check: predicted 0.9486, actual +0.0047 above —
  **the loss→SSIM law held for the SIXTH time**, first time at 1e-4.
- **Predeclared reading: LR is no longer the binding lever.** The option-(a) question is answered
  in the negative — the 5e-5 echo plateau is not LR-curable at this depth. The remaining gap to
  90/100 ≥ 0.95 shrinks at ~+5 windows/segment and decelerating; the law says the intercept (0.9885)
  clears the bar and the slope (compounding rollout price) is what stands in the way — which is
  exactly exp_03's target.

## Jobs 50–51 — FORMAL VERDICT PASSES @ lr1e4-20,000 — launched 2026-08-04T15:29Z

**Approved by Yixun ("use (b)")** — formalize the verdict at the best checkpoint. D11 structure
(Jobs 27–28 precedent), run `wan-overfit100-s3ext-lr1e4-20260804`, step 20,000, v6e-8, eval code
byte-identical lineage; COMMIT=632d44f5db216c6d8ed26f53c669328cb4e96aef (tip at submission).
- **Job 50 (segment-final 3×3):** `20260804-152857-5b1ec01e-exp02-o100-final20k-yixun` —
  canonical 100, ROLLOUT_SEEDS=0,1,2, CONTEXT_MODES=correct,null,shuffled (~2–3 h; resume staging ON).
- **Job 51 (full-set):** `20260804-152913-840b3465-exp02-o100-fullset20k-yixun` — all 1,629, seed 0,
  correct (~4–5 h; resume staging ON).
- **Then:** the verdict CLI recomputes the two-tier claim locally over the admitted artifacts at this
  eval commit (one-verdict-per-eval-commit; the seed-0 correct cells must reproduce Job 49's
  mean 0.9536 / 67 ≥ 0.95 as a validity check).
- **Read:** two-tier D11 at 20,000 — canonical bar count across 3 seeds + full-set ≥0.90 fraction;
  text-conditioning gap (correct vs shuffled/null) at the memorization-strong checkpoint.

Parallel-tracks note (Yixun, same message): exp_02 and exp_03 have no sequential dependency — both
run concurrently; exp_03's round-4 fix work continues independently.

### Jobs 50–51 outcome + FORMAL VERDICT at 20,000 (2026-08-04T~21:30Z)

Both SUCCEEDED through heavy spot weather (a5/a5; resume staging did its job). Validity held:
segment-final seed-0 correct reproduces Job 49 exactly (0.9536 / 67). Seeds are remarkably stable
(correct means 0.9536 / 0.9537 / 0.9536; ≥0.95 counts 67 / 69 / 69).

**Verdict CLI (`verdict_lr1e4_step20000_complete.json`, one-verdict-per-eval-commit, coverage
complete 1,629/1,629): `partial`.**
- Headline (canonical, median-of-3-seeds m_corr ≥ 0.95): fraction **0.69** < 0.90 — NOT established.
- Full-set gate at c*=20000: fraction **0.9932** ≥ 0.90 — numerically passes, but the two-tier claim
  requires the headline, so "full-set memorization" is correctly not claimed.
- Ablation (300 paired rows each): correct−null gap **+0.0974**, correct−shuffled **+0.1233** —
  the text-conditioning gap more than DOUBLED from 10k (0.053), and wrong context is now worse
  than no context: the model actively follows the instruction.

Arc of the formal statistic across the experiment: 10k verdict "none" (0/100 at 0.95; full-set
14.1%) → 20k verdict "partial" (69/100 at 0.95; full-set 99.3%).

