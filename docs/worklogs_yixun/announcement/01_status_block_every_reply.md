# 01 — Every reply ends with a sub-process status block

**Directive (Yixun, 2026-07-16, verbatim):**

> Every time when you give me a reply, you need to reply with all the current sub process (experiment) and when they will finish and the earlist time for me to steer you.

**Interpretation / how to apply:** every assistant reply (both agents) ends with a **Status** block listing:
1. Every currently-running sub-process — experiments, TPU jobs, background reviews/agents, long builds — one line each: what it is, its state, and the **estimated finish time** (or "unknown — will notify").
2. The **earliest point Yixun can steer** — the next decision gate (approval, result read-out) and whether interrupting now is safe.

Applies to routine replies too ("no sub-processes running" is a valid block). Estimates are best-effort; update them as evidence arrives (smoke-run timings, queue state).
