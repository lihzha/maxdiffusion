# 02 — Every TPU run requires Yixun's explicit prior approval

**Directive (Yixun, 2026-07-17, verbatim):**

> Before tpu run, You need my permission (wait for my approval!)

**Interpretation / how to apply:**
- **No TPU/remote job is launched without Yixun's explicit approval in the conversation, per launch.** This covers EVERY job class: smoke runs, fit/batch-size probes, full training runs, escalation-control runs, validation/eval jobs, data builds — validation-ladder rungs 5–7 included. Local CPU work (tests, static checks) is exempt.
- Approval is requested with the pre-launch package: what will run, hardware/topology, expected cost/duration, the commit SHA, and the acceptance criteria (per the SOP's Running & failure discipline).
- **Interaction with announcement-level auto-resubmit policy:** after Yixun has approved a job, auto-resubmitting that SAME job on infrastructure failure (preemption etc.) remains allowed without re-asking — unless the code or config changed, which makes it a new launch requiring new approval.
- A launch made without approval is a process violation, regardless of outcome.
