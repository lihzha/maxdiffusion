# Issue Report

Open issues, recurring failures, and standing workarounds. Each entry: symptom, infra-vs-bug classification, workaround/fix, status. Updated at every handoff / wrap-up / pre-compact per the handoff protocol in `CLAUDE.md`.

Last updated: 2026-07-31

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

### 6. gcloud/gsutil reauth expires silently and recurs (infra, standing)
- **Symptom:** sixth recurrence 2026-08-02 (blocked the LR-sweep checkpoint copies; caught on first failed gsutil), fifth 2026-08-01 ~08:25Z (blocked the four 10k eval launches; caught immediately by empty-listing anomaly + stderr probe), fourth 2026-07-31 (previously 2026-07-29 ×2 + earlier) — `ReauthUnattendedError` / "Reauthentication required" from gsutil; the `tpu` CLI then reports jobs as "not found". A 2.5 h monitoring loop once read auth failures as "PENDING" because the poll's error fallback conflated them with a missing status file.
- **Workaround:** user runs `gcloud auth login` (account yh4742@princeton.edu). **Monitoring rule:** poll scripts must treat gsutil stderr containing "Reauth" as an ALARM state (stop and surface), never as pending/absent.
- **Status:** standing; recurs on a ~4-h cadence during long sessions. Rule reaffirmed: any silently-empty gsutil listing for objects known to exist = probe stderr immediately, never trust an empty result.

### 7. Shared side-adapter trainer: checkpoint retention works by accident with keep_period=-1 (real bug, latent)
- **Symptom:** `wan_ti2v_side_adapter_trainer` forwards `checkpoint_keep_period or None` to Orbax, so the repo-wide default `-1` reaches it verbatim; retention then rests on Python's `step % -1 == 0` being truthy — every checkpoint survives BY ACCIDENT. Flip the default or pass a positive keep_period and eviction semantics change silently (max_to_keep=3 then evicts early checkpoints, which would have deleted exp-critical baselines).
- **Found:** exp_02 cycle C (2026-07-30) while building the H2 checkpoint-schedule tests; both behaviors pinned by tests in `test_overfit100_checkpoint_schedule.py`. exp_02's own manager is immune (explicit max_to_keep=None, no keep_period). exp_01's runs relied on the accident but retained everything, so no data was lost.
- **Status:** open, LOW urgency; classified real bug in the shared trainer; fix belongs to a future shared-trainer round (not exp_02's scope — Codex reviewer concurred, cycle C judgment 10).

### 8. Eval launcher lacked ffmpeg-ensure → aux/ceiling rows silently degraded (real bug, fix committed, review pending)
- **Symptom:** exp_02 S3 intermediate evals (2026-07-31): all aux rows failed with `FileNotFoundError: 'ffmpeg'` — the ffmpeg-ensure block existed only in `build_overfit100_dataset.sh`, never in `validate_wan_overfit100.sh`; TPU images ship without ffmpeg. Primary SSIM metrics unaffected (latent path); only VAE-ceiling/aux columns degraded.
- **Fix:** commit `9c26070` on the exp_02 branch — ffmpeg-ensure block byte-identical across both launchers (enforced by test), executed-under-bash tests with PATH shims, loud one-line aux-degradation startup warning (names ffmpeg/gsutil), FATAL exit if install fails; +14 tests (suite 1,021). Focused Codex review pending as of this update.
- **Status:** fix committed, unreviewed; ceilings for already-run S3 intermediates recoverable via a cheap checkpoint-independent backfill job.

### 9. Codex reviewer account hit its usage limit (infra, BLOCKING the review discipline)
- **Symptom:** 2026-08-02 ~23:45Z — every `codex exec` returns "You've hit your usage limit … try again at Aug 7th, 2026 11:35 PM". Account-wide (verified with a minimal call), not model- or effort-specific. ~30 xhigh review passes across exp_02/exp_03 this week consumed the quota.
- **Impact:** the SOP's Reviewer role (Codex `gpt-5.6-sol` xhigh) is unavailable; per the SOP, no silent substitution. exp_03 round-3 strengthening (`371816c`) is committed but NOT re-reviewed; S1 cannot be packaged per the SOP until review capacity returns or Yixun explicitly directs a substitute/waiver.
- **Options (Yixun's call):** (a) purchase credits at chatgpt.com/codex/settings/usage; (b) wait for the Aug 7 reset; (c) explicitly approve a substitute reviewer for the interim (would be recorded in every affected review file per the SOP's no-silent-substitution rule).
- **Status:** OPEN — blocking exp_03 review cycles; non-review work (TPU measurement jobs on already-reviewed code, doc syntheses) continues.

## RESOLVED (kept for the record)

### R1. `side_adapter_noise_mode=fixed` train/val mismatch (real bug, fixed)
- Good-looking train loss but broken generations. Must be `fresh`; the config defaults to `fresh`, but always pass it explicitly at launch (the wrapper historically defaulted to `fixed`).

### R2. Codex CLI too old for `gpt-5.6-sol` (infra, fixed 2026-07-12)
- 0.142.5 rejected the model ("requires a newer version"); upgraded via `brew upgrade --cask codex` to 0.144.1. Reviewer invocation verified: `codex exec -m gpt-5.6-sol -c model_reasoning_effort=xhigh`.
