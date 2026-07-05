# Experiment SOP (portable)

A generalizable standard operating procedure for AI-assisted research experiments. Drop this file into any project and reference it from that project's `CLAUDE.md` (e.g. "Follow `worklog/SOP.md` for all experiment work"). Written for Claude Code, but the roles are model-agnostic.

> **In this repository:** the portable `worklog/` root referenced throughout maps to `docs/worklogs_yixun/`. Read every path below as rooted there — e.g. `worklog/exp_01_foo_claude/` → `docs/worklogs_yixun/exp_01_foo_claude/`, and `worklog/announcement/` → `docs/worklogs_yixun/announcement/`.

## Roles (three-model separation of duties)

| Role | Who | Duty |
|---|---|---|
| **Planner / Analyst** | The main-session model (strongest reasoning tier; currently Claude Fable 5) | Writes plans, judges reliability, writes analyses. Does NOT write implementation code directly. |
| **Coder** | A subagent on the strongest coding tier at max effort (currently Claude Opus 4.8, max effort) | Implements exactly what the approved plan specifies, **test-first** (see Test-driven development). |
| **Reviewer** | **The opposite model family from the Coder** (mandatory cross-model review; see reciprocity note below). | Reviews **the plan** and **each round of new code** (small, focused per-round reviews — one per Coder round); each review is saved **under the reviewing model's name** (artifacts 3 and 5), not just read. If the reviewer is unavailable, say so — never silently substitute. |

> **Reviewer reciprocity — no model reviews its own plan or code.** The Reviewer reviews **both the plan and the code**. If the main session (Planner/Coder) is **Claude**, the Reviewer is **OpenAI Codex** at its strongest setting — model `gpt-5.5`, reasoning effort **Extra High** (`model_reasoning_effort = "xhigh"`) — via `codex mcp-server` (CLI fallback `codex exec -m gpt-5.5 -c model_reasoning_effort=xhigh`). If the main session is **Codex**, the Reviewer is **Claude Opus 4.8 at max effort**, invoked via the `claude` CLI. The Coder and Reviewer must always be different model families, so review is genuinely independent.

> **Reviewer briefing — load context before verdict.** The Reviewer starts cold; every review call (plan or code) must have it load context **before** it sees the thing under review: (1) this SOP, the experiment's `_yixun_query.md`, and the approved `plan_*.md`; (2) the experiment's `_worklog.md` so far; (3) the `_analysis.md` / results of prior related experiments, so it knows what has already been tried and learned; (4) a short statement of what the current Coder round is for — the function/TDD unit, its contract, and its `<marker>`. The Reviewer runs with repo access, so point it at file paths to read rather than pasting content. Each review file opens by listing the context it loaded; a verdict produced without the briefing is invalid — re-run it.

## Directory layout

All experiment bookkeeping lives in `worklog/` at the repo root:

- `worklog/announcement/<NN>_<topic>.md` — standing directives from the user. **Read every announcement before planning or running anything.** New standing instructions get the next number.
- `worklog/exp_<NN>_<exp name>_claude/` — one folder per experiment, `<NN>` zero-padded and sequential.

## Per-experiment artifacts (in lifecycle order)

Inside `worklog/exp_<NN>_<exp name>_claude/`:

1. `<exp name>_yixun_query.md` — the user's driving queries: each one verbatim, plus a summary, the user's assumption/hypothesis, and why the experiment needs to run. Started at scaffold time, appended as new queries arrive. (Rename the `yixun` part to the relevant user in other projects.)
2. `plan_<exp name>.md` — written by the Planner BEFORE any code: the English plan AND the planned code laid out per file (each existing file to edit, each new file to create). Surfaced for user approval before implementation.
3. `<exp name>_<reviewer>_plan_review.md` — the Reviewer's verdict on `plan_<exp name>.md` (soundness, parity with the reference, test coverage, risks/edge cases), produced **before user approval** so the user approves with the review in hand. Same `<reviewer>` naming as the code review (slug + first-line model name + version).
4. *(the implementation itself)* — the code, written by the Coder subagent per the approved plan. No dedicated markdown artifact; this step is the source-code changes.
5. `<exp name>_<reviewer>_code_<marker>_review.md` — the Reviewer's verdict on **one round of the Coder** (one function / TDD unit / small commit), not the whole experiment. **After every Opus-Coder round, call the Reviewer (Codex `gpt-5.5` xhigh) for a small, focused code review** and save it as its own file. `<marker>` identifies the round — the TDD function name, a code-snippet marker, or that round's short commit name (e.g. `sidewin_codex_code_test-rollout-sigmas_review.md`). The `<reviewer>` slug and the doc's first line name the reviewing model. Record "N/A — no code written" for code-free experiments.
6. `<exp name>_params_set_up.md` — full hyperparameters/configuration, written at launch.
7. `<exp name>_command.md` — exact reproduction command(s), written at launch.
8. `<exp name>_worklog.md` — an append-only, timestamped **lab notebook: one entry per action**, not per run. Started at scaffold, appended continuously through implementation, debugging, and every launch. This is where the validation ladder, parity audit, and failure triage (below) get recorded as they happen. Entry format in **Worklog entry template** below. Complements `_results.md` (final numbers) and `_analysis.md` (final judgment) by capturing the decision/debug trail in between.
9. `<exp name>_<YYYY-MM-DD_HH:MM:SS>.log` — ALL terminal output, one timestamped log per run (tee/redirect every training/eval command into the folder). Aborted runs keep their log, renamed with an `_ABORTED_<reason>` suffix.
10. `<exp name>_results.md` — results, appended as runs finish.
11. `<exp name>_analysis.md` — written by the Planner after results land: analyze all code + configuration + results; judge whether the result is reliable; state the outcome and the recommended next step.
12. `commits_<exp name>.md` — SHA + one-line description of every commit belonging to this experiment.

## Worklog entry template

Each `_worklog.md` entry is one action, headed by an ISO-8601 UTC timestamp and a short title, then these fields. Use the full set for substantive actions (code, launches, fixes); a lightweight **Goal / Result / Analysis / Next** is enough for routine monitoring checks.

`## <YYYY-MM-DDThh:mm:ssZ> — <short title>`

- **Goal** — what this action is for. *(every entry)*
- **Hypothesis** — the belief being tested, stated *before* the evidence. *(only when testing something)*
- **Change** — exact files/behavior touched. *(when code changed)*
- **Version Control** — branch, `base_commit`, `implementation_commit`, push/pull, changed_files. Every SHA inline.
- **Command / Validation** — exact commands (or static checks), job ids, run dirs, log + artifact paths.
- **Acceptance criteria** — the exact conditions that count as success, written *before* a launch (see **Running & failure discipline**).
- **Result** — status (`passed` / `partial` / `in_progress` / `launched` / `fix_ready`) + metrics/artifacts + key evidence.
- **Analysis** — interpretation; in particular classify any failure as *infrastructure vs. real bug*.
- **Next** — the immediate next step.

## Development discipline

- Develop **commit by commit, experiment by experiment** from a known-good base commit. No long-lived uncommitted state: an experiment concludes by committing its code and its worklog folder.
- **Each commit generally < 200 changed lines of code.** Several small commits per experiment are preferred over one large one. Log every SHA in `commits_<exp name>.md`.
- Superseded or exploratory code is archived (patch + files) under `worklog/archive_<reason>_<date>/` before being removed from the working tree — never destroyed.

## Test-driven development (TDD)

Write the test before the code, for every non-trivial function — red → green → refactor:

1. **Determine the test first.** For each small function, fix its contract and write `test_<function>` — concrete inputs → expected outputs plus edge cases — *before* the implementation exists. Run it; it must fail (red) for the right reason (missing/incorrect behavior, not an import typo).
2. **Implement to green.** Write the minimal function that makes its test pass, then run the test to confirm green.
3. **Refactor** with the test as a safety net, keeping it green.
4. **One small commit per function** — its test and implementation together (test written first in the working tree), so every commit ships with a passing test. Split a function that needs several tests into several commits; keep each within the < 200-LOC rule.

**Review every round.** After each Coder round (each function / small commit), call the Reviewer (Codex `gpt-5.5` xhigh, per reciprocity) — **briefed first** per the Reviewer briefing note (plan, worklog, prior experiments, what this round is for) — for a *small, focused* code review of just that change — never batch a whole experiment into one review. Save each as `<exp name>_<reviewer>_code_<marker>_review.md` (artifact 5), where `<marker>` is that round's TDD function name, code-snippet marker, or short commit name. Address the review before the next round.

**Location & naming.** Test files live in **`src/maxdiffusion/tests/worklogs_yixun/`**, one file per unit under test, named `test_<exp name>_<function>.py`. Run with `PYTHONPATH=src pytest src/maxdiffusion/tests/worklogs_yixun/ -v`.

The passing pytest suite is **rung 1 of the validation ladder** below — the cheapest executable gate, run before any smoke/probe/full run. Per reviewer reciprocity, the Reviewer reviews the tests too, not just the implementation.

## Validation ladder (cheapest-first)

Never jump straight to the expensive run. Climb this ladder; advance only when the current rung passes, and record each rung in `_worklog.md`.

1. **Static checks + unit tests** — `python -m py_compile <changed .py>`, config parse (`yaml.safe_load`), `bash -n <changed .sh>`, `git diff --check` (whitespace), and the TDD pytest suite for the changed functions (`PYTHONPATH=src pytest src/maxdiffusion/tests/worklogs_yixun/`). Seconds–minutes, no accelerator.
2. **Tiny synthetic forward** — smallest module instantiation + one forward/step on a small/cheap device with synthetic tensors. Catches graph/mesh/shape/dtype/timestep errors without loading full weights or data.
3. **Small real-data readback** — parse a few real records; assert shapes, byte lengths, and min/max/std match the schema. Catches data-pipeline mismatches.
4. **Bounded data build** — if the experiment produces a dataset, build the val split (or a bounded slice) first and read it back before the full build.
5. **Smoke run** — a few steps at the smallest batch on the target hardware, checkpointing/final-save **disabled** (storage-light), just to reach one completed optimizer step and produce logs.
6. **Fit / batch-size probe** — find the max batch that fits, still storage-light (no checkpoints), before committing to the full run.
7. **Full run** — only after 1–6 pass and the parity audit below is clean.

## Parity audit before scaling

Before spending real compute on a new method, audit the implementation **component by component against the reference** (paper code / upstream repo), and diff the *numbers*, not just the shapes:

- **Numeric recipe defaults** — LayerNorm eps, weight decay, betas, LR schedule, loss type, sigma/noise schedule, CFG handling. Silent mismatches (e.g. eps `1e-6` vs `1e-5`, weight decay `0` vs `1e-2`) don't crash; they quietly corrupt results.
- **Structural parity** — which params are trainable vs frozen, residual/injection points, stop-gradient boundaries, any pinning/masking.
- **Data parity** — take one concrete source example and its processed/cached counterpart; confirm identical dtype, byte counts, and min/max/std.

Record the audit in `_worklog.md`; launch the full-scale run only from the audited commit.

## Running & failure discipline

- **Pre-launch acceptance criteria.** Before every launch, write in `_worklog.md` the exact conditions that count as success: the commit SHA the worker must report, device/host count, per-device and global batch, parallelism axes, which params are trainable, and "reaches ≥1 optimizer step with no OOM / NaN / parse failure." Judge the run against these, not vibes.
- **Infrastructure vs. real bug.** In every failure Analysis, classify the cause: *infrastructure* (spot preemption, host maintenance, download/network stall, launch-env/PATH, quota) vs. *real bug* (wrong shape/sharding, wrong objective, bad schedule). Only real bugs get a code fix; infra failures get retry/resume. Prevents thrashing on non-bugs.
- **Commit + push before running remotely.** Any remote (TPU/cluster) run executes a *pushed* SHA; verify the SHA the worker actually checked out. Never run uncommitted code on a remote.
- **Never edit a script while it is running.** Wait for a safe boundary, then patch; fix orchestration bugs separately from the data/model path.
- **Resume from the last safe boundary.** On a mid-pipeline crash, find the last fully-committed unit (contiguous shard / checkpoint) and resume from the next; never duplicate or lose work, never reuse stale staging.
- **Storage guardrail.** Long data/checkpoint jobs hold only a bounded working set (one batch/segment resident), clean up after each unit, and stay above an explicit free-space floor; contiguous coverage is the correctness check.
- **Shared-resource etiquette.** Inspect a shared machine's processes before using it; don't interrupt others' jobs without approval; verify zero stale processes/watchers/queued-resources after a failed run before relaunching.

## Evaluation integrity

- Always compare against the baseline method on its **full published evaluation configuration** — the complete split files and dataset configs the baseline's paper used. Never invent new eval configurations (subsampled items, hand-picked examples, reduced splits) for comparisons; subsets are for debugging only and never appear in `_results.md`.
- Reproduce the baseline's reported numbers (within its reported variance) **before** evaluating any new method, to calibrate the pipeline and establish the noise floor.
- Match the baseline's aggregation convention (e.g. per-scene means) and its variance protocol (e.g. mean ± std over N generations/seeds).
- If a run is launched and the code state then changes (revert, edit), kill and relaunch rather than mixing code states across a sweep; document the abort in `_params_set_up.md`.

## Sequencing summary

scaffold folder + `_yixun_query.md` + `_worklog.md` → `plan_*.md` (Planner) → `_<reviewer>_plan_review.md` (Reviewer) → user approves → **TDD** code (Coder, test-first), with a small **Reviewer code review each round** (`_<reviewer>_code_<marker>_review.md`) → **validation ladder** (static → smoke → probe) → **parity audit** vs reference → `_params_set_up.md` + `_command.md` + **acceptance criteria** → launch with teed timestamped logs, triaging *infra-vs-bug* on failure → `_results.md` → `_analysis.md` (Planner) → commit(s) + `commits_*.md`. Log every action in `_worklog.md` as it happens.
