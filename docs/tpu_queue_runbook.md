# TPU Queue Runbook (new `tpu` queue tools)

Practical playbook for running and debugging jobs on the IROM TPU **queue**
(`tpu create` + central scheduler). Distilled from real runs of the Wan2.2
TI2V training/validation jobs. Covers only the queue CLI.

---

## 1. How the queue model works

- `tpu create` only **submits a job**: it uploads your code as a tarball and a
  job spec to a GCS queue, then returns. It does **not** create the TPU itself.
- A **central scheduler** (runs under a privileged service account) picks up the
  job, creates a **queued resource (QR)** for a TPU slice, waits for capacity,
  provisions the VM(s), runs `--setup-cmd` on **every worker**, then runs your
  command.
- **Code comes from `--code-dir`**, not a git branch. The tarball is built with
  `git ls-files` (tracked + untracked, `.gitignore`-respected, **excludes
  `.git/`**), and the working-tree contents are what run — so uncommitted edits
  are included, but there is **no `.git` on the worker**.
- **Logs upload to GCS only when an attempt ends** (the worker's cleanup trap).
  There is no live log tail during a run.

GCS layout (v6/us-east1 example):

```
gs://v6_east1d/tpu-job-queue/jobs/<job_id>/
  spec.json          # command, resources, env_vars, secret_refs (env_vars may hold plaintext secrets)
  status.json        # authoritative per-job state (poll this)
  code.tar.gz        # uploaded code
  logs/attempt-<N>/worker-<K>.log
gs://pi0-cot/tpu-job-queue/scheduler_state.json   # aggregated state (what `tpu status` reads; lags)
```

Worker runs the command from `$HOME/deployed_code/<job_id>/attempt-<N>/` in a
`bash -lc` **login shell** (default `ulimit -n` is **1024**).

---

## 2. Submitting a job

```bash
tpu create v6 -n 64 \
  --name <job-name> \
  --code-dir . \
  --setup-cmd "bash bash_scripts/setup.sh MODE=stable DEVICE=tpu && bash bash_scripts/prefetch_hf_snapshot.sh <MODEL>" \
  --env KEY=VALUE ... \
  --secret WANDB_API_KEY=<secret-manager-name> \
  -- <training command>
```

- `v6 -n 64` → resource `v6-64` (v6e-64). Requestable sizes: `v6-8/16/32/64`.
- `--name` is a **label only** (becomes the `job_id` suffix `<ts>-<uuid>-<name>`);
  it is **not** a TPU node name and need not be unique.
- Defaults: `--max-attempts 20`, `--priority 1`, `--code-dir .`.
- The training command goes after `--`.

In this repo, use the wrappers instead of raw `tpu create`:

```bash
# Full training (30000 steps, checkpoints, W&B):
bash bash_scripts/launch_wan_train.sh
# 1-step smoke (no checkpoint/eval/W&B):
SMOKE=1 bash bash_scripts/launch_wan_train.sh
# One-shot validation of an existing run (v6e-8):
RUN_NAME=<training-run-name> bash bash_scripts/launch_wan_validate.sh
```

---

## 3. Monitoring a job (the reliable way)

**Right after submit, `tpu status` may say "Job not found"** — the aggregated
`scheduler_state.json` lags. The job's own `status.json` is authoritative:

```bash
JOB=<job_id>
gsutil cat gs://v6_east1d/tpu-job-queue/jobs/$JOB/status.json
# key fields: status, current_attempt, attempts[].error, current_qr_state, provisioned_at
tpu status $JOB     # convenient once the aggregated state catches up
tpu logs   $JOB     # only meaningful after an attempt ends
```

Lifecycle: `PENDING → PROVISIONING` (`QR: WAITING_FOR_RESOURCES → PROVISIONING → ACTIVE`) `→ RUNNING → SUCCEEDED|FAILED`.

**Progress signal = the checkpoint directory** (logs aren't live). New step
directories appearing means it is genuinely training; the step numbers reveal
resume-vs-restart:

```bash
gsutil ls gs://.../<OUTPUT_DIR>/<RUN_NAME>/checkpoints/
# e.g. 700/ 800/ after a preemption ⇒ resumed from 600; 100/ 200/ ⇒ restarted
```

**Timing tell**: dead-slice aborts happen **~5–8 min after `provisioned_at`**
(JAX `RegisterTask` 5-min timeout). If a job is still `RUNNING` well past that
window, it got a **healthy slice**.

---

## 4. Failure triage

| Signal (status/exit) | Root cause | Auto-retried? | Action |
|---|---|---|---|
| `exit 128` | Worker has no `.git`; a `git` command in the run/setup script failed | no | Make git calls non-fatal in the script (see §6) |
| `exit 134` (SIGABRT) | A JAX process aborted → multi-host coordination `DEADLINE_EXCEEDED` (barrier/`RegisterTask`). **Symptom, not cause.** | no | Find the first dead task (see §5) |
| `TPU_VM_PREEMPTED` | Spot slice reclaimed | **yes** (up to `--max-attempts`) | Nothing; scheduler retries and resumes from checkpoint |
| `HEARTBEAT_TIMEOUT` | Worker stopped heartbeating | **yes** | Nothing; scheduler retries |
| `TPU_VM_HEALTH_UNHEALTHY_MAINTENANCE` / a worker log missing | Dead/unhealthy host in the slice | no (ends as `exit 134`) | Resubmit for a fresh slice; if repeated, escalate |

Key consequence: **preemption and heartbeat-timeout self-heal; `exit 128`/`exit
134`/dead-host do not** — those need a manual fix or resubmit.

---

## 5. Debugging `exit 134` (the important one)

`exit 134` almost always surfaces on worker-0 as
`"...another task died... DEADLINE_EXCEEDED ... RegisterTask/Shutdown barrier"`.
That is downstream noise. Do this:

1. **Count worker logs** — this separates infra from code in one step:
   ```bash
   gsutil ls gs://.../jobs/$JOB/logs/attempt-<N>/ | grep -oE 'worker-[0-9]+' | sort -t- -k2 -n
   ```
   - **Fewer than the slice's host count** (e.g. 13/16, 15/16) ⇒ a host never
     came up ⇒ **dead-host / infra** ⇒ resubmit for a fresh slice.
   - **All hosts present** ⇒ a real **code/config** error; go to step 2.
2. **Find the first task that died**: read worker-0's barrier line
   `"first task at the barrier: N"`, then read **worker-N**'s log — the real
   error (Traceback / `Errno 24` / `No API key` / OOM) is there, not on worker-0.

Root causes we actually hit behind `exit 134`:
- `OSError: [Errno 24] Too many open files` at `import wandb` → fd limit (see §6).
- `wandb ... No API key configured` → missing W&B key (see §6).
- A missing worker log → dead host (infra).

---

## 6. Worker-environment gotchas (fixes every run/setup script needs)

The worker is a fresh VM with **no `.git`**, a **1024 fd soft limit**, and none
of your local shell/rc/secrets. Scripts that run on the worker must handle this:

1. **Non-fatal git metadata** (avoid `exit 128`):
   ```bash
   echo "COMMIT=${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
   git status --short --branch 2>/dev/null || echo "(no git checkout; running from uploaded code)"
   ```
   Pass the real commit for provenance from the submitting machine via
   `--env COMMIT="$(git rev-parse HEAD)"`.
2. **Raise the open-file limit** right after `set -euo pipefail` (avoid `exit
   134` from fd exhaustion during model load / data pipeline / tensorstore):
   ```bash
   ulimit -n 1048576 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
   ```
3. **W&B credentials must reach the worker** (only if the job logs to W&B):
   - The worker has no `~/.zshrc`/`~/.bashrc` from your laptop, so a locally
     exported `WANDB_API_KEY` does **not** transfer by itself.
   - Options: `--secret WANDB_API_KEY=<secret-manager-name>` (worker pulls via
     `gcloud secrets versions access`; needs the **TPU service account** to have
     read access to that secret), **or** `--env WANDB_API_KEY="$WANDB_API_KEY"`
     (forwards your shell's key, but writes it **plaintext into the GCS job
     spec** — acceptable only for a shared/internal key).
   - If a job doesn't use W&B, don't pass the key (e.g. validation/generation).

> `--env` values (incl. any secret passed this way) are stored in `spec.json` on
> GCS and are readable by anyone with bucket access. Prefer Secret Manager for
> real secrets.

---

## 7. Spot / preemption & checkpoint durability

- v6e spot slices get **preempted**; the scheduler auto-retries and, because a
  retry reuses the same `RUN_NAME`/checkpoint dir, training **resumes from the
  latest checkpoint** (orbax reshards across topologies, so a v6e-64 checkpoint
  restores fine on a smaller slice).
- Progress is durable **only after the first checkpoint**. If preemptions arrive
  faster than `CHECKPOINT_EVERY`, the run never persists progress → lower
  `CHECKPOINT_EVERY` when the zone is preempt-heavy.
- **Bigger slices are more fragile**: 16 hosts = 16 chances for a dead host or a
  preemption. Small validation slices (v6e-8) are noticeably more robust.
- **Same code passing sometimes and failing other times ⇒ suspect the slice/
  zone, not your code.** Don't blindly resubmit expensive full runs; after a few
  consecutive infra failures, escalate (zone capacity/health, or ask whether the
  scheduler should health-check hosts / auto-retry coordination-startup failures).

---

## 8. Command quick reference

```bash
JOB=<job_id>

# submit (repo wrappers)
SMOKE=1 bash bash_scripts/launch_wan_train.sh
bash bash_scripts/launch_wan_train.sh
RUN_NAME=<run> bash bash_scripts/launch_wan_validate.sh

# monitor
gsutil cat gs://v6_east1d/tpu-job-queue/jobs/$JOB/status.json     # authoritative state
tpu status $JOB
tpu logs   $JOB                                                   # after an attempt ends
gsutil ls  gs://.../<OUTPUT_DIR>/<RUN_NAME>/checkpoints/          # live progress

# debug exit 134
gsutil ls  gs://v6_east1d/tpu-job-queue/jobs/$JOB/logs/attempt-<N>/   # count workers
gsutil cat gs://v6_east1d/tpu-job-queue/jobs/$JOB/logs/attempt-<N>/worker-<N>.log

# control
tpu delete $JOB     # cancel (scheduler tears down QR + VM)
tpu retry  $JOB     # retry a failed job
tpu list            # jobs / requestable resources
```
