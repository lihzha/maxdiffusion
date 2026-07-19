# exp_01 `full_ft_overfit` — params & setup (written at full-run launch readiness, per SOP artifact 6 + plan §2.3)

## Primary run configuration (launcher `full_ft` arm defaults; yml `base_wan_5b_full_ft.yml`)
- Hardware: v6e-64, us-east1-d; pure FSDP (`ici_fsdp_parallelism=-1`); per-device batch **4** ⇒ **GBS 256** (amended per Query 7 after the probe-4 HBM finding: per-device 8 overflows 31.25G by ~37MB on any topology — FSDP collective buffers)
- Steps: **20000** (≈3.55 passes over 1,440,554 windows at GBS 256 — amended per Query 7; total compute unchanged vs 10k×512); LR **1e-5** AdamW (b1 .9, b2 .999, eps 1e-8, wd 1e-2, global-norm clip 1.0, warmup fraction 0.05)
- Objective: one-step flow-matching velocity MSE via shared helpers; frame-0 pinned; null-text context; **no adapter, no actions, no CFG** (guide 1.0 + fresh noise asserted at startup)
- Precision: weights/activations bf16 (mixed f32 norms per loader), remat FULL; Adam moments follow param dtype (smoke-verified per-dtype logs)
- Checkpoints: every **2500**, keep-period **2500** (retains 2500/5000/7500/10000; ≈30 GB each), async, GCS `gs://v6_east1d/checkpoints/maxdiffusion/wan-ti2v-full-ft/<run_name>/checkpoints`
- Eval-in-training: every 1000 on TRAIN shards (memorization signal); wandb project `maxdiffusion-wan-full-ft`

## Predeclared memorization cohort (plan §2.3 — fixed BEFORE the full run)
- **validation_ordinals (16, 0-based dataset positions, evaluated in this listed order):**
  `0,96037,192074,288111,384147,480184,576221,672258,768295,864332,960369,1056406,1152442,1248479,1344516,1440553`
  (deterministic near-even spread over [0, 1440553] ⇒ one clip from every ~48th shard; endpoints included)
- **validation_seed: 0** (per-sample rollout seeds derive sequentially in the listed order — round-4 contract)
- **Evaluation points:** step-0 pretrained baseline (`checkpoint_step=0` bypass) + checkpoints **5000/10000/15000/20000** (same samples-seen quarters as the original 2500/5000/7500/10000 at GBS 512; keep-period 2500 retains all), all via `bash_scripts/validate_wan_full_ft.sh` on v6e-8, metrics latent MSE / pixel MSE / SSIM + comparison videos; success = within-cohort delta vs step-0 baseline.

## Escalation controls (pre-specified §2.4; each needs its own approval)
1. Resume primary → 30000 steps (eval 20000/30000). 2. Fresh `-lr2e5` run (LR 2e-5, 10k). 3. Fresh `-fp32state` run (`weights_dtype: float32`, 10k; dtype log line = precondition).

## Aborted/superseded record
- Smoke attempt 1 (per-device 8 on v6e-8): FAILED CompileTimeHbmOom +44MB — config artifact, fixed by cycle-6 launcher override; kept in `_command.md` entry 1.
