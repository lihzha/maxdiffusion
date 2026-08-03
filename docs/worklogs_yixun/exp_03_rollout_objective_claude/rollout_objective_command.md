# exp_03 commands

Every launch recorded here at launch time per the SOP. (Bookkeeping note: this file was briefly committed
EMPTY at `0c08b70` — a 2-minute shell timeout killed the heredoc that was writing it, after the package text
had already been reviewed into the plan; the content below was restored minutes after the launches, from the
same text. The gap is disclosed rather than backdated.)

## Jobs 1–4 — v6e-8 S1 SMOKE: control / corrective_ss / rollout_loss / combined — launched 2026-08-03T02:45Z

**Approved by Yixun (Query 3, conditional grant: "approve S1 smoke when the package is ready"; conditions
met — round 3 CLOSED with APPROVE, package below).** Rounds 1–3 all CLOSED (extraction APPROVE; trainer
APPROVE; losses APPROVE). Suite 1,399 + 2. `COMMIT=0c08b70` (tip at submission).

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective
for obj in control corrective_ss rollout_loss combined; do
  short=$(echo $obj | tr -d '_' | cut -c1-8)
  tpu create v6 -n 8 --worker0-only --name "exp03-s1-${short}-yixun" \
    --code-dir . \
    --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
    --env RUN_NAME="exp03-s1-${short}-20260803" \
    --env EXP03_OBJECTIVE="$obj" \
    --env EXP03_RAMP_ORIGIN=0 --env EXP03_P_SS_RAMP_STEPS=10 \
    --env MAX_TRAIN_STEPS=30 --env SAVE_FINAL_CHECKPOINT=False \
    --env DATA_DIR="gs://v6_east1d/datasets/exp02_overfit100/train100" \
    --env EXPECTED_WINDOWS=1629 --env NUM_TEXT_SLOTS=100 \
    --env COMMIT="<tip>" \
    --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
    -- bash bash_scripts/train_wan_exp03.sh
done
```

**What:** 4 × v6e-8, from-init (Tier-2-style — no checkpoint seed), 30 steps each, tiny footprint,
`SAVE_FINAL_CHECKPOINT=False`. `EXP03_RAMP_ORIGIN=0` and `EXP03_P_SS_RAMP_STEPS=10` for the smoke ONLY, so
A/C's self-generation path is genuinely exercised within 30 steps (p_ss reaches 0.5 by step 10). Data =
train100 (unchanged pins).

**Gates (predeclared):**
1. All losses finite at every step, all four arms; grad norms same order as control.
2. Hook parity on hardware: control's first-step loss within fp tolerance of the exp_02 trainer's at the
   same seed (the smoke-scale proxy for the ctrl0 AND-gate).
3. **STOP budgets on measured step-time ratio vs the control smoke (same hardware):** A ≤ 1.6×, B ≤ 2.5×,
   C ≤ 3.2×. Exceeding a budget is a STOP for that arm (report to Yixun), not a silent accept.
   Reference (hardware-independent jaxpr census): fwd-call/eqn ratios 1.00 / 1.79 / 1.38 / 3.24.
4. B/C compile cleanly with scan+remat on-device (the real S1.6 mesh-fit at GBS 256 on v6e-64 is a
   separate gated launch).

- **Job ids (submitted 2026-08-03T02:45–02:46Z):** control → `20260803-024504-7175ecdb-exp03-s1-control-yixun`;
  corrective_ss → `20260803-024531-b4f93a1a-exp03-s1-correcti-yixun`; rollout_loss →
  `20260803-024558-3534905b-exp03-s1-rolloutl-yixun`; combined → `20260803-024622-0206cf9b-exp03-s1-combined-yixun`.
