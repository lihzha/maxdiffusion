# exp_03 → next agent: handoff

Written 2026-08-08 at closure. Purpose: let a fresh agent design and run follow-up experiments on
exp_03's findings with zero context loss. Read this + `rollout_objective_analysis.md` (v1.1) +
`rollout_objective_results.md` (Results 1–9) + the tracker before anything else.

## 1. What exp_03 established (and its verdict)

Three trial losses vs one-step denoising on OVERFIT100 (100 DROID trajectories, frozen eval =
25-step rollout, canonical-100 seed-0 SSIM): **A** corrective scheduled sampling, **B** k=2
differentiable rollout, **C** λ·A+(1−λ)·B. Yixun's closing verdict (Query 8): **from scratch, C
(λ=0.5) works better than A and B**; the **+0.02 practical-effect hard gate was NOT met** by any
arm, so the experiment closed. Trained-state winner within-seed: **A** (+0.0079 @17,500, t=+30.8,
100/100, and it beats one-step loss simultaneously). λ-sweep: loss cost U-shaped (min 0.5),
off-law gain monotone in A-share, net peaks at 0.5 from init. Everything is ONE training seed.

## 2. The instruments you inherit (all reusable, all certified)

- **Control-by-identity**: `EXP03_OBJECTIVE=control` in the exp03 trainer reproduces the exp_02
  recipe at ~1e-11 (proven: ctrl0 AND-gate). Use it for ANY one-step baseline — including runs of
  exp_02-lineage RUN_NAMEs (the exp_02 script itself CANNOT run on this branch: it reads
  `exp03_snapshot_before_step`, absent from the overfit100 YAML — pyconfig raises).
- **The loss→SSIM law**: exact OLS `SSIM = 0.98895405 − 1.20004257·loss` (fit on exp_02's 8
  one-step means; use THESE coefficients for residuals, not the rounded 0.9885/1.201). 9 held-out
  holds within ±0.005 (a range, not an error bar; checkpoints share ancestry). "Off-law residual"
  = measured − predicted; one-step arms sit on the line, rollout-trained arms sit above it.
- **Fixed-RNG one-step instrument** (`eval_wan_overfit100_val_loss.sh`): deterministic per-window;
  a checkpoint's reading is bit-reproducible — USE AS SEED-VALIDITY ANCHOR at every handoff.
  Known anchors: 0.1222672 (s3@10,000 = every Tier-1 seed), 0.1200277 (control@12,500),
  0.1186258 (A@12,500), 0.1193259 (C@12,500), 0.1168342 (control@17,500), 0.1153891 (A@17,500).
- **SSIM eval** (`validate_wan_overfit100.sh`): EVAL_PASS_ROLE=s3_intermediate, EVAL_WINDOWS=canonical,
  ROLLOUT_SEEDS=0, CONTEXT_MODES=correct — the cells every number in Results uses.
- **S1.5 probe** (`probe_exp03_s1_5.py`): dual-state no-update gradient diagnostics; payloads for
  both states already on GCS (`exp03-s15-probe7-20260805/validation_probe_sampling/`).
- **Gradient accumulation**: `EXP03_GRAD_ACCUMULATION=N` (lax.scan, interleaved shard-local
  microbatches; N=1 bitwise-identical to no-accumulation). C at GBS 256 on v6e-64 NEEDS
  `EXP03_GRAD_ACCUMULATION=2 PER_DEVICE_BATCH_SIZE=4.0` (PDB=2 would silently halve GBS!).

## 3. Launch recipes + the rules that were paid for in failures

Branch/worktree: `claude-exp_03_rollout_objective-20260802` at
`/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective` (suite 1,584/2).
Queue: `tpu create v6 -n {8|64} [--worker0-only for evals] --code-dir . --setup-cmd
"EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" --env K=V ... -- bash
bash_scripts/<script>`. Trainers: `train_wan_exp03.sh` (objectives: control/corrective_ss/
rollout_loss/combined; EXP03_LAMBDA for C; EXP03_RAMP_ORIGIN=10000 for trained-state arms, 0 from
init).

**Hard rules (each one cost a failed run):**
1. Tier-1 (from-checkpoint) launches MUST pass `OUTPUT_DIR=gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-overfit100`
   AND `CHECKPOINT_STEPS="[<seed_step>,<target>]"` explicitly — the exp03 default root is
   `.../wan-ti2v-exp03` and the YAML's default save list omits late steps (a run once trained from
   scratch silently; another finished without saving its endpoint).
2. Seed checkpoints by server-side copy into `<OUTPUT_DIR>/<RUN_NAME>/checkpoints/<step>` and
   byte-verify (`gsutil du -s`, expect 23,654,557,930 for the 10k seed); gsutil on macOS needs
   `-o "GSUtil:parallel_process_count=1"`.
3. READ THE WHOLE SUBMISSION OUTPUT — an id is printed even when the code upload fails (stillborn
   job). Verify `status.json` exists after every submit.
4. Never build launch loops that split fields on ":" (gs:// URLs) and never leave `$extra` env args
   unquoted in zsh — spell every submission's env block explicitly.
5. v6e-64 startup SIGABRT (exit 134, in `jax.distributed.initialize`) is probabilistic pool
   weather: auto-resubmit identical spec (issue #10). Worker "exit 1 during setup" likewise.
   Anything with a Python traceback: diagnose first.
6. The fleet-watcher pattern (a watchlist file + a persistent monitor emitting terminal events;
   resubmits append to the file) is how 30+ jobs were run hands-free — reuse it.
7. gcloud reauth dies every ~4h (issue #6): Yixun runs `gcloud auth login`; treat "Reauth" in
   gsutil stderr as ALARM, silently-empty listings as suspicious. Git pushes use gh auth (immune).

## 4. Standing directives (bind every future session)

Announcement 01+amendment: Status block every reply, with an explicit `Blocked on you:` line.
Announcement 02: NO TPU launch without Yixun's explicit prior approval (conditional/blanket grants
count; record them verbatim in the query doc). Announcement 03: predeclared success/failure gates
are adjudicated by YIXUN, never by the agent. SOP: three roles (Planner / Coder subagent / Codex
`gpt-5.6-sol` xhigh reviewer via `codex exec -s read-only -m gpt-5.6-sol -c
model_reasoning_effort=xhigh "$(cat prompt)" < /dev/null`), review-until-approve before launches,
every launch recorded in `_command.md` at launch time, errors disclosed not hidden.

## 5. The parked ladder (pre-scoped next experiments, need Yixun's approval)

1. **Independent-training-seed replication** of Tier-1 A and C₀.₅-from-init, WITH matched controls
   (new seed → new control run too). Confirmatory (n=2). Cheapest decisive check of the headline.
2. **A saturation probe**: continue A + control 17,500 → ~25,000 (matched), eval at 20k/22.5k/25k.
   Framed as saturation measurement — the observed widening (+0.0005/5k) does not license gate
   extrapolation.
3. **λ-schedule arm** (start 0.5, anneal → 1 as loss flattens): run ONLY if the cancellation
   hypothesis first survives a cheap test — post-training D1 per-frame slopes + sigma-trace
   comparison of C0 vs A0/B0 (predeclared in the plan, never run), or gradient projections onto
   the control gradient from an S1.5-style probe at the trained arms' endpoints.
4. Smaller open items: S1.5 parity gradient-gap attribution (reviewer's canonicalized-target
   discriminator, spec in `rollout_objective_command.md` §S1.5-closure); trace_forward
   double-compile cause; per-window analysis of WHICH windows A helps (all 100 improved — is the
   gain uniform or structured by episode difficulty?).

## 6. Where everything lives

- Results/analysis/report: this folder (`rollout_objective_results.md`, `_analysis.md` v1.1,
  `rollout_objective_01_results_report.html`, review records, `_command.md` Jobs 1–34+).
- GCS: Tier-1 runs + control under `.../wan-ti2v-overfit100/` (runs `exp03-s2a-{corrss,rolloutl,combined}-20260806`,
  control `wan-overfit100-s3ext-lr1e5c-20260802` — now with checkpoints to 17,500); Tier-2 + λ
  under `.../wan-ti2v-exp03/exp03-s2b-{ctrl0,a0,b0,c0,lam25,lam75}-*`; every eval JSON beside its run.
- exp_02 inheritance (data pins, manifest, cohorts): `../exp_02_overfit100_claude/` and the exp_02
  sections of `CLAUDE.md`. DATA: `gs://v6_east1d/datasets/exp02_overfit100/train100`
  (EXPECTED_WINDOWS=1629, NUM_TEXT_SLOTS=100 — pass both, always).
