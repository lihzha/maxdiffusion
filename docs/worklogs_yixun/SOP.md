# Experiment SOP (portable)

A generalizable standard operating procedure for AI-assisted research experiments. Drop this file into any project and reference it from that project's `CLAUDE.md` (e.g. "Follow `worklog/SOP.md` for all experiment work"). Written for Claude Code, but the roles are model-agnostic.

> **In this repository:** the portable `worklog/` root referenced throughout maps to `docs/worklogs_yixun/`. Read every path below as rooted there — e.g. `worklog/exp_01_foo_claude/` → `docs/worklogs_yixun/exp_01_foo_claude/`, and `worklog/announcement/` → `docs/worklogs_yixun/announcement/`.

## Roles (three-model separation of duties)

| Role | Who | Duty |
|---|---|---|
| **Planner / Analyst** | The main-session model (strongest reasoning tier; currently Claude Fable 5) | Writes plans, judges reliability, writes analyses. Does NOT write implementation code directly. |
| **Coder** | A subagent on the strongest coding tier at max effort (currently Claude Opus 4.8, max effort) | Implements exactly what the approved plan specifies. |
| **Reviewer** | An independent model via MCP (currently OpenAI Codex, `codex mcp-server`; CLI fallback `codex exec`) | Reviews all newly written code; review is saved, not just read. If the reviewer is unavailable, say so — never silently substitute. |

## Directory layout

All experiment bookkeeping lives in `worklog/` at the repo root:

- `worklog/announcement/<NN>_<topic>.md` — standing directives from the user. **Read every announcement before planning or running anything.** New standing instructions get the next number.
- `worklog/exp_<NN>_<exp name>_claude/` — one folder per experiment, `<NN>` zero-padded and sequential.

## Per-experiment artifacts (in lifecycle order)

Inside `worklog/exp_<NN>_<exp name>_claude/`:

1. `<exp name>_yixun_query.md` — the user's driving queries: each one verbatim, plus a summary, the user's assumption/hypothesis, and why the experiment needs to run. Started at scaffold time, appended as new queries arrive. (Rename the `yixun` part to the relevant user in other projects.)
2. `plan_<exp name>.md` — written by the Planner BEFORE any code: the English plan AND the planned code laid out per file (each existing file to edit, each new file to create). Surfaced for user approval before implementation.
3. *(the implementation itself)* — the code, written by the Coder subagent per the approved plan. No dedicated markdown artifact; this step is the source-code changes.
4. `<exp name>_codex_code_review.md` — the Reviewer's verdict on the new code (record "N/A — no code written" for code-free experiments).
5. `<exp name>_params_set_up.md` — full hyperparameters/configuration, written at launch.
6. `<exp name>_command.md` — exact reproduction command(s), written at launch.
7. `<exp name>_<YYYY-MM-DD_HH:MM:SS>.log` — ALL terminal output, one timestamped log per run (tee/redirect every training/eval command into the folder). Aborted runs keep their log, renamed with an `_ABORTED_<reason>` suffix.
8. `<exp name>_results.md` — results, appended as runs finish.
9. `<exp name>_analysis.md` — written by the Planner after results land: analyze all code + configuration + results; judge whether the result is reliable; state the outcome and the recommended next step.
10. `commits_<exp name>.md` — SHA + one-line description of every commit belonging to this experiment.

## Development discipline

- Develop **commit by commit, experiment by experiment** from a known-good base commit. No long-lived uncommitted state: an experiment concludes by committing its code and its worklog folder.
- **Each commit generally < 200 changed lines of code.** Several small commits per experiment are preferred over one large one. Log every SHA in `commits_<exp name>.md`.
- Superseded or exploratory code is archived (patch + files) under `worklog/archive_<reason>_<date>/` before being removed from the working tree — never destroyed.

## Evaluation integrity

- Always compare against the baseline method on its **full published evaluation configuration** — the complete split files and dataset configs the baseline's paper used. Never invent new eval configurations (subsampled items, hand-picked examples, reduced splits) for comparisons; subsets are for debugging only and never appear in `_results.md`.
- Reproduce the baseline's reported numbers (within its reported variance) **before** evaluating any new method, to calibrate the pipeline and establish the noise floor.
- Match the baseline's aggregation convention (e.g. per-scene means) and its variance protocol (e.g. mean ± std over N generations/seeds).
- If a run is launched and the code state then changes (revert, edit), kill and relaunch rather than mixing code states across a sweep; document the abort in `_params_set_up.md`.

## Sequencing summary

scaffold folder + `_yixun_query.md` → `plan_*.md` (Planner) → user approves → code (Coder) → `_codex_code_review.md` (Reviewer) → `_params_set_up.md` + `_command.md` → launch with teed timestamped logs → `_results.md` → `_analysis.md` (Planner) → commit(s) + `commits_*.md`.
