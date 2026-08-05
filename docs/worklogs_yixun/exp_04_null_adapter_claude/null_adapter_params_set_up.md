# null_adapter_params_set_up.md — J1 configuration (SOP artifact 6)

## J1 — P1 capacity study + basin probe (v6e-8), launched 2026-08-06

- **Commit:** `f06dfc1836c585c3d0cd2fde4c1d1a8e1ea28a0b` (branch `claude-exp_04_null_adapter-20260803`, pushed; parity audit CLEAN at this SHA; suite 936 green)
- **Hardware:** v6e-8 (`tpu create v6 -n 8 --worker0-only`), zone us-east1-d, bucket `gs://v6_east1d`
- **Config:** `src/maxdiffusion/configs/base_wan_5b_null_inversion.yml` (208 keys) with per-phase env overrides below
- **Job runbook (three sequential invocations inside one queue job):**
  1. **SMOKE** — `NULL_MODE=capacity`, `NULL_SMOKE_EXAMPLES=2`, `NULL_SAMPLING_STEPS=4`, `NULL_INNER_ITERS=2`, arms A0/A1 only (`NULL_ARMS=a0,a1`), artifact root `gs://v6_east1d/datasets/droid_wan_null_adapter/j1/smoke` → one full publish + a `verify_replay` invocation over the smoke shard. Also asserts the R1 on-device noise golden (launcher preflight).
  2. **ADEQUACY** — `NULL_MODE=adequacy_probe` (first-8 DEV, the approved 6-cell grid J∈{10,25,50}×lr∈{1e-2,3e-2}), adoption artifact → `gs://v6_east1d/datasets/droid_wan_null_adapter/j1/adequacy/adoption.json`
  3. **FULL CAPACITY** — `NULL_MODE=capacity` at the production recipe (adoption consumed via `NULL_ADEQUACY_URI`), all six arms on DEV-64 + TRAINFIT-16, full-cohort decode, gates tables + selection, A3 single-update measurement stage (`NULL_A3_MEASURE=1`), videos for the 8-example subset, records for A1/A2 → `gs://v6_east1d/datasets/droid_wan_null_adapter/j1/capacity`
- **Fixed parameters (plan v5):** L_null=16; guide_scale 5.0; inversion w=1.0; σ grid 25 steps shift 5.0; Adam (0.9, 0.999, 1e-8, eps_root 0); noise seed 2026 (golden-pinned); manifests `gs://v6_east1d/datasets/droid_wan_null_adapter/manifests/j0/` (J0-published, listing checksum `5827f4da…0d14`); batch 8; latent_dtype fp16 (records); verify atol 1e-2.
- **Estimated wall:** ~3–6 h total (smoke ≲20 min; adequacy ≲1 h; capacity 2–4 h).
