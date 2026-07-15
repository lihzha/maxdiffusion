# Issue Report

Open issues, recurring failures, and standing workarounds. Each entry: symptom, infra-vs-bug classification, workaround/fix, status. Updated at every handoff / wrap-up / pre-compact per the handoff protocol in `CLAUDE.md`.

Last updated: 2026-07-14

## OPEN / STANDING

### 1. `git push` over SSH fails (infra, standing workaround)
- **Symptom:** `git@github.com: Permission denied (publickey)` — the SSH agent holds no identities on this machine.
- **Workaround:** `gh auth setup-git` is done; push with the explicit HTTPS URL, then sync the tracking ref:
  ```bash
  git push https://github.com/lihzha/maxdiffusion.git yixun-dev
  git update-ref refs/remotes/origin/yixun-dev $(git rev-parse HEAD)
  ```
- **Status:** standing pattern; every push in this repo uses it.

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

## RESOLVED (kept for the record)

### R1. `side_adapter_noise_mode=fixed` train/val mismatch (real bug, fixed)
- Good-looking train loss but broken generations. Must be `fresh`; the config defaults to `fresh`, but always pass it explicitly at launch (the wrapper historically defaulted to `fixed`).

### R2. Codex CLI too old for `gpt-5.6-sol` (infra, fixed 2026-07-12)
- 0.142.5 rejected the model ("requires a newer version"); upgraded via `brew upgrade --cask codex` to 0.144.1. Reviewer invocation verified: `codex exec -m gpt-5.6-sol -c model_reasoning_effort=xhigh`.
