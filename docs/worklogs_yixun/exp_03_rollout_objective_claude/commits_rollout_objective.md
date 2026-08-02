# exp_03 commits

| SHA | description |
| --- | --- |
| `8ccaf3a` | cycle A round 1 — sampler-step extraction into `models/wan/overfit100_sampling.py` (bitwise-inert: exact parity vs a verbatim pre-extraction copy + AST no-duplicate test), both eval rollouts rewired, and the two predeclared diagnostics `diagnostics_exp03/{d1_per_frame_slopes,sigma_trajectory_trace}.py`. Suite 1271/2 skipped. |
| `e4a11a4` | round-1 strengthening (Codex 2 HIGH + 3 MEDIUM) — D1 fail-closed contracts, trace design pin before the 5B load, measured bf16 oracle floor with RAW-metric predeclaration, fp32+bf16 chain parity and a verbatim adapter/CFG whole-rollout reference, structure-binding AST guard. Suite 1296/2 skipped. |
| `c0aaaa2` | round 2 — `_loss_and_step_fns` binding hook (parent byte-identical, late binding preserved), NEW `Exp03Trainer` (`EXP03_TI2V`) with control-by-identity and NotImplementedError trials, `exp03_aux_key` RNG discipline (derived, step-keyed, resume-stable), exp03 config + training launcher. Suite 1331/2 skipped. |
