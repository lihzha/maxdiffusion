"""exp_03 `rollout_objective` — the PREDECLARED diagnostics (plan v3.1, Mechanisms A and B).

Both modules are committed before any trial runs, so the reduction formula, the bootstrap
protocol, the trace definition and the acceptance directions are fixed while nobody yet knows which
arm wins. They live in the package (not under ``docs/``) because they are importable, suite-tested
code: ``sigma_trajectory_trace`` runs on the pod and needs the extracted sampler step, and keeping
its predeclared partner beside it means one place to look for "what exp_03 promised to measure".
"""
