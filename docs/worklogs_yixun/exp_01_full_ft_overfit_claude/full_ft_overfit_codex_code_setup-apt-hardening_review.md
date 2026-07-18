# Code review: exp_01 full_ft_overfit — round setup-apt-hardening
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-18

## Context loaded
- Query 5 conditionally armed the full run after a passing fit probe
- Attempt 1 lost one worker to the unattended-upgrades lock and exposed the 8/16 queue-barrier under-count
- Attempt 2 reproduced the lock on two workers and isolated the unbounded root `setup.sh` wait
- `bash_scripts/setup.sh` delegates to the patched root `setup.sh`, so all three queue arms receive the change
- The heredoc bug was reproduced: the old fallback returned 0 after a consumed failing body, while the new invocation propagated the body’s nonzero status
- Focused write-free validation passed 5 tests with the expected Darwin `setup.sh` skip; the full suite could not acquire a temporary directory in the read-only review sandbox

## Verdict
REQUEST-REVISION. The bounded apt waits and heredoc rewrite are sound, but Jammy’s service semantics mean the asynchronous stop may leave the actual `unattended-upgrade` lock holder running, and the 600-second fallback is too close to the JAX deadline. The patch also permanently disables security-update machinery on every persistent Ubuntu host using this general-purpose setup script.

## Findings
- **F1 — MAJOR — The stop sequence does not reliably stop Jammy’s lock holder, and the fallback expires too late.** `setup.sh:94` asynchronously stops `apt-daily-upgrade.service`, then immediately enters apt. The essential unit set is correct—Ubuntu documents `apt-daily.timer` and `apt-daily-upgrade.timer` as the triggers—but `--no-block` only enqueues the stop. More importantly, `apt-daily-upgrade.service` uses `KillMode=process`, while `apt.systemd.daily` launches `unattended-upgrade` as a child; Ubuntu’s own investigation records that stopping the unit terminates only the controlling shell and leaves unattended-upgrades running. The following `disable --now` does not wait for either apt-daily service because they are not among its arguments. Thus the observed child may retain the lock until `DPkg::Lock::Timeout=600`, approximately the entire JAX coordination window, and potentially finish setup too late or emit its setup failure alongside the distributed timeout. [Ubuntu automatic-update chain](https://documentation.ubuntu.com/server/how-to/software/automatic-updates/), [Jammy systemctl semantics](https://manpages.ubuntu.com/manpages/jammy/man1/systemctl.1.html), [Ubuntu KillMode investigation](https://bugs.launchpad.net/bugs/1690980). **Fix:** stop the timers before apt, then use a substantially shorter total lock/setup deadline with coordination headroom and fail loudly; alternatively implement a Jammy-safe, bounded termination/wait for the actual unattended-upgrade process and verify lock release before invoking apt.

- **F2 — MAJOR — A queue-only mitigation permanently changes persistent GPU/dev hosts.** `setup.sh:95` uses persistent `systemctl disable`, although root `setup.sh` is the general TPU/GPU/local installer. This removes automatic security-update enablement across reboots and also disables the unattended-upgrades shutdown helper; a source comment describing ephemeral workers does not constrain execution. Containers remain guarded and the GPU package branch itself is untouched, but persistent Ubuntu GPU/dev boxes incur an unrelated security-policy change. Systemd documents `disable` as removing enablement symlinks persistently. [Jammy systemctl disable documentation](https://manpages.ubuntu.com/manpages/jammy/man1/systemctl.1.html). **Fix:** either stop the two timers only for the current boot, or gate persistent hardening behind an explicit ephemeral-worker argument/environment flag supplied by the queue launchers.

- **F3 — MINOR — The static tests do not pin the claimed four-unit command set.** The test only requires `apt-daily-upgrade.service` and `apt-daily.timer` somewhere in the complete file, so comments satisfy it; it omits command-level checks for `apt-daily.service` and `apt-daily-upgrade.timer`. Removing any of those names from the actual commands can therefore leave the test green. The Bash-3.2 skip rationale is valid—HEAD and the patched file both fail `bash -n` at the pre-existing `[[ ! -v MODE ]]`—but it skips syntax validation of the changed block entirely on Darwin. **Fix:** parse the two `systemctl` lines and assert all intended service/timer names there, and on Bash 3.2 syntax-check a copy with only the pre-existing `-v` expression neutralized.

---

# Follow-up review: exp_01 full_ft_overfit — setup-apt-hardening (strengthened)
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-18

## Focus-finding check

F1: **STILL-OPEN** — The timer stop and Bash-3.2-safe conservative-OR escalation are sound, but four independent 180s waits plus unbounded apt execution invalidate the claimed ≤335s/~10-minute bound, and lock-free after SIGKILL does not prove dpkg consistency.

F2: **CLOSED** — `EPHEMERAL_WORKER=1` flows correctly from `SETUP_CMD` through the wrapper and `$SUDO env ... bash`, with persistent disable confined to the explicit gate.

F3: **STILL-OPEN** — Systemctl and timeout checks now exclude comments, but the escalation, ordering, and sudo-environment assertions still search raw file text and remain partly comment-satisfiable.

## New issues

- **MAJOR — Unsafe continuation after SIGKILL.** Killing an updater during a dpkg transaction can leave inconsistent package state even after both locks disappear; an ephemeral worker should exit immediately and be discarded after the KILL path, not continue installing.
- **MINOR — Broad process matching.** `pkill -f unattended-upgrade` intentionally includes `unattended-upgrade-shutdown`, but also signals any unrelated command line containing that substring; matching captured updater/helper PIDs would reduce collateral risk.

## Verdict

**REQUEST-REVISION.** The persistent-host regression is fixed and the Jammy lock-holder handling is substantially improved, but the deadline is not globally bounded and the SIGKILL path can continue on damaged package state.

---

# Final follow-up: exp_01 full_ft_overfit — setup-apt-hardening (strengthen 2)
Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-18

## Check

1. **CLOSED** — The 420s deadline covers apt/curl execution, all four apt calls use 60s lock waits, and the arithmetic holds: ~241s worst-case non-apt work leaves ≥179s for apt, with clean completion by ~7.1 minutes and the KILL path terminal by ~245s.
2. **CLOSED** — The SIGKILL branch emits a loud error and exits 1 before any apt command, and the structural test pins both the exit and absence of apt fall-through.
3. **CLOSED** — Both signal sites capture PIDs with `pgrep` and pass those PIDs to `kill`; no command-text `pkill` remains, and the test pins this.
4. **CLOSED** — Every structural assertion operates on comment-stripped command text; only explicitly documentary checks inspect the full source text.

## Launch-blocking issues

None

## Verdict

**APPROVE** — All four required hardening items are closed and the frozen experiment is clear to launch.

---

# Strengthening record (Coder, cycle 7 — 2026-07-19)

Round 1 (F1–F3 of the initial review): jammy-safe unit set + heredoc failure propagation + EPHEMERAL_WORKER gate + command-line test parsing. Follow-up found F1/F3 still open + 2 new findings. Round 2 closed everything: **global 420s wall-clock budget** (`apt_deadline_run` → `timeout` on execution incl. curl; 60s per-apt lock waits; systemctl timeout-30), **exit-1-immediately after any SIGKILL** (dpkg state unverifiable → discard worker), **PID-exact kills**, **fully comment-insensitive structural tests** (zero darwin skips). 8 mutants caught across rounds; provable timing: clean ≤ ~7.1 min, KILL path terminal ≤ ~245s, inside the ~10-min JAX window. Final follow-up verdict: **APPROVE, no launch-blocking issues**.

**Cycle 7 closed** (write → review → strengthen → follow-up → strengthen-2 → final follow-up APPROVE).
