# Issue Report

Open issues, recurring failures, and standing workarounds. Each entry: symptom, infra-vs-bug classification, workaround/fix, status. Updated at every handoff / wrap-up / pre-compact per the handoff protocol in `CLAUDE.md`.

Last updated: 2026-07-14

## OPEN / STANDING

### 1. `git push` over SSH fails (infra) — RESOLVED 2026-07-17 by HTTPS origin
- **Symptom:** `git@github.com: Permission denied (publickey)` — the SSH agent holds no identities on this machine.
- **Old workaround (RETIRED):** pushing via explicit HTTPS URL + hand-updating `refs/remotes/origin/yixun-dev`. This caused a false "diverged 15/9" `git status` on 2026-07-17: the hand-update once ran from the experiment-worktree cwd and recorded the exp-branch tip as `origin/yixun-dev` (no real divergence existed — remote was verified byte-identical to local).
- **Fix:** `git remote set-url origin https://github.com/lihzha/maxdiffusion.git` (gh credential helper authenticates). Plain `git push origin <branch>` now works and maintains tracking refs automatically. Never hand-edit `refs/remotes/*` again.

### 2. TPU queue spot preemptions are frequent (infra)
- **Symptom:** IROM TPU-queue jobs die with `failure_type: INFRASTRUCTURE_PREEMPTION`; the step-30000 validation job needed 6 attempts.
- **Policy (user directive):** auto-resubmit failed jobs without asking; report results by handing the user gsutil pull commands as text — never run the pulls.
- **Status:** standing; not a code bug — check `status.json` at `gs://v6_east1d/tpu-job-queue/jobs/<id>/` (authoritative) before diagnosing anything else.

### 3. `CLAUDE.md` is gitignored (`.gitignore:194`)
- **Symptom:** repo-level Claude instructions can't be committed/pushed; edits are local to this machine only.
- **Status:** accepted; durable, shareable process rules go into `docs/worklogs_yixun/experiment_SOP.md` (committed) instead — CLAUDE.md holds machine-local rules and pointers.

### 4. HF download 408 timeouts on distributed launches (infra)
- **Symptom:** multi-host JAX launches died when hosts hit HuggingFace `408` timeouts mid-download.
- **Workaround:** `bash_scripts/prefetch_hf_snapshot.sh` (robust retrying prefetch before JAX startup) — keep it in every launch path.
- **Status:** mitigated; re-check if a launch script bypasses the prefetch.

### 5. Background `codex exec` hangs on open stdin (infra, standing workaround)
- **Symptom:** `codex exec "prompt" > log 2>&1` launched as a background job blocks forever at `Reading additional input from stdin...` with ~0 CPU — codex waits for stdin EOF that never comes (the plan-v2 re-review sat stuck ~10 h; an earlier identical invocation happened to inherit a closed stdin and worked, which masked the bug).
- **Workaround:** ALWAYS append `< /dev/null` to non-interactive `codex exec` invocations. Detect recurrence by: log stuck at 1 line + process CPU ≈ 0 + no `OpenAI Codex vX.Y.Z` header line.
- **Status:** standing rule for every reviewer call; classified infrastructure (launch-env), no code change.

## RESOLVED (kept for the record)

### R1. `side_adapter_noise_mode=fixed` train/val mismatch (real bug, fixed)
- Good-looking train loss but broken generations. Must be `fresh`; the config defaults to `fresh`, but always pass it explicitly at launch (the wrapper historically defaulted to `fixed`).

### R2. Codex CLI too old for `gpt-5.6-sol` (infra, fixed 2026-07-12)
- 0.142.5 rejected the model ("requires a newer version"); upgraded via `brew upgrade --cask codex` to 0.144.1. Reviewer invocation verified: `codex exec -m gpt-5.6-sol -c model_reasoning_effort=xhigh`.
