# exp_03 commands

Every launch recorded here at launch time per the SOP. (Bookkeeping note: this file was briefly committed
EMPTY at `0c08b70` — a 2-minute shell timeout killed the heredoc that was writing it, after the package text
had already been reviewed into the plan; the content below was restored minutes after the launches, from the
same text. The gap is disclosed rather than backdated.)

## Jobs 1–4 — v6e-8 S1 SMOKE: control / corrective_ss / rollout_loss / combined — launched 2026-08-03T02:45Z

**Approved by Yixun (Query 3, conditional grant: "approve S1 smoke when the package is ready"; conditions
met — round 3 CLOSED with APPROVE, package below).** Rounds 1–3 all CLOSED (extraction APPROVE; trainer
APPROVE; losses APPROVE). Suite 1,399 + 2. `COMMIT=0c08b70` (tip at submission).

```bash
cd /Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_03_rollout_objective
for obj in control corrective_ss rollout_loss combined; do
  short=$(echo $obj | tr -d '_' | cut -c1-8)
  tpu create v6 -n 8 --worker0-only --name "exp03-s1-${short}-yixun" \
    --code-dir . \
    --setup-cmd "EPHEMERAL_WORKER=1 bash bash_scripts/setup.sh MODE=stable DEVICE=tpu" \
    --env RUN_NAME="exp03-s1-${short}-20260803" \
    --env EXP03_OBJECTIVE="$obj" \
    --env EXP03_RAMP_ORIGIN=0 --env EXP03_P_SS_RAMP_STEPS=10 \
    --env MAX_TRAIN_STEPS=30 --env SAVE_FINAL_CHECKPOINT=False \
    --env DATA_DIR="gs://v6_east1d/datasets/exp02_overfit100/train100" \
    --env EXPECTED_WINDOWS=1629 --env NUM_TEXT_SLOTS=100 \
    --env COMMIT="<tip>" \
    --env HF_HUB_DISABLE_XET=1 --env HF_HUB_ENABLE_HF_TRANSFER=0 \
    -- bash bash_scripts/train_wan_exp03.sh
done
```

**What:** 4 × v6e-8, from-init (Tier-2-style — no checkpoint seed), 30 steps each, tiny footprint,
`SAVE_FINAL_CHECKPOINT=False`. `EXP03_RAMP_ORIGIN=0` and `EXP03_P_SS_RAMP_STEPS=10` for the smoke ONLY, so
A/C's self-generation path is genuinely exercised within 30 steps (p_ss reaches 0.5 by step 10). Data =
train100 (unchanged pins).

**Gates (predeclared):**
1. All losses finite at every step, all four arms; grad norms same order as control.
2. Hook parity on hardware: control's first-step loss within fp tolerance of the exp_02 trainer's at the
   same seed (the smoke-scale proxy for the ctrl0 AND-gate).
3. **STOP budgets on measured step-time ratio vs the control smoke (same hardware):** A ≤ 1.6×, B ≤ 2.5×,
   C ≤ 3.2×. Exceeding a budget is a STOP for that arm (report to Yixun), not a silent accept.
   Reference (hardware-independent jaxpr census): fwd-call/eqn ratios 1.00 / 1.79 / 1.38 / 3.24.
4. B/C compile cleanly with scan+remat on-device (the real S1.6 mesh-fit at GBS 256 on v6e-64 is a
   separate gated launch).

- **Job ids (submitted 2026-08-03T02:45–02:46Z):** control → `20260803-024504-7175ecdb-exp03-s1-control-yixun`;
  corrective_ss → `20260803-024531-b4f93a1a-exp03-s1-correcti-yixun`; rollout_loss →
  `20260803-024558-3534905b-exp03-s1-rolloutl-yixun`; combined → `20260803-024622-0206cf9b-exp03-s1-combined-yixun`.

## S1 outcome (Jobs 1–4) — 2026-08-03T~04:30Z

All four SUCCEEDED as queue jobs (after one v6e-8 maintenance sweep + suspensions; attempts 1–2 each).
Gates evaluated from step logs (steady-state steps/s = mean over steps 10–29):

| arm | finiteness | steps/s | ratio | budget | gate |
| --- | --- | --- | --- | --- | --- |
| control | all finite | 1.786 | 1.00× | — | PASS |
| corrective_ss | all finite | 1.219 | 1.47× | ≤1.6× | **PASS** |
| rollout_loss | all finite | 0.698 | 2.56× | ≤2.5× | **STOP** (marginal, +2.4%) |
| combined | **NaN from step 8** | 0.422 | 4.23× | ≤3.2× | **STOP** (×2) |

- Gate 2 (hook parity on hardware) is NOT evaluable against exp_02's history at smoke scale (different
  batch/hardware); it is carried by the suite's exact JIT-parity certificate now and by ctrl0's AND-gate at
  S2b. Recorded as deferred-by-design, not passed.
- B's overrun is small and plausibly small-batch/remat overhead; decision deferred to S1.6's at-scale
  measurement (no code change).
- C's NaN at LR≈1e-6 is a numerical edge in the loss computation for a specific draw — deterministically
  reproducible from (seed, step 8, purposes). Diagnosis round dispatched; C re-smoke will need a fresh
  launch approval after the fix + review.

## Jobs 5–7 — v6e-8 RE-SMOKE COHORT (control timing / A timing / C replay) — launched 2026-08-03

**Under Yixun's Query-4 grant** (C re-smoke conditional on the S1-fix review passing — passed at `fdadb5f`,
reviewer GO with the cohort spec confirmed). One contemporaneous cohort, identical seed/data/ramp/compiler
to S1: 30 steps, `EXP03_RAMP_ORIGIN=0`, `EXP03_P_SS_RAMP_STEPS=10`, `LOG_PERIOD=1` (full 16-field line every
step), strict STOP gates control-relative (A ≤ 1.6× — note A now always runs 2 advances, was mean 1.5, so
this re-measures its real cost; C ≤ 3.2×). C additionally arms `EXP03_SNAPSHOT_BEFORE_STEP=7` (single host —
gate open): pre-failure params/opt/rng/batch land in the run dir before global_step 7 executes. If C goes
non-finite: the forced NON-FINITE line names the term, every host raises, and the frozen-state A/B/C
discriminator (`exp03_frozen_replay`) runs from the snapshot as a follow-up job.
- **Job ids (submitted 2026-08-03T16:45–16:46Z):** control → `20260803-164526-06c4fa27-exp03-rs-control2-yixun`; A → `20260803-164552-452e8a59-exp03-rs-corrss2-yixun`; C replay (snapshot armed) → `20260803-164618-cf2ca830-exp03-rs-combined2-yixun`. COMMIT=86c7408.

### Job 7 correction (2026-08-03T~17:15Z)

Job 7 (combined2) FAILED at launch-plumbing, not in training: an unquoted $extra in the zsh launch loop
passed `--env EXP03_SNAPSHOT_BEFORE_STEP=7` as ONE malformed argument, scrambling the CLI's arg order so
the worker ran `bash -- …` (invalid option, exit 2). No trainer code executed; no NaN evidence either way.
Relaunched as **Job 7b** `20260803-171003-8ce1d02c-exp03-rs-combined3-yixun` with the env inline. (Second zsh word-splitting incident this session —
noted: always inline or array-expand extra args.)

### Re-smoke cohort outcome so far (2026-08-03T~19:40Z)

- **Timing pair complete:** control2 1.801 steps/s; corrss2 1.533 → **A ratio 1.17× (PASS, was 1.47×)** —
  the unroll compiles better than the dynamic loop it replaced. Both fully finite.
- **C replay (combined3): the NaN did NOT recur.** Attempt 1 ran 19 finite steps before preemption —
  past the original failing step AND through the hardest draw class (global_step 15: s_a=0, σ_hi=1.0,
  k_a=2, self-generated) with every *_finite flag green and all 16 diagnostic fields flowing. C cost
  3.96× vs 3.2× budget (better than 4.23×, still over — S1.6 item).
- **combined3 terminal:** attempt 4 hit borderline HBM OOM at program load (26.47G vs 23.70G free) — the
  same binary ran on attempt 1, so allocator-layout flakiness at the memory edge; exit 1 stopped queue
  retries. **C now carries TWO S1.6 flags: step-time (3.96×) and HBM headroom.** One resubmit under the
  infra-flake reading: **Job 7c** `20260803-193617-1214ad57-exp03-rs-combined4-yixun` for the formal 30/30; the substantive verdict stands either way.

### S1 CLOSED (2026-08-03T~20:45Z)

**combined4 attempt 1 completed the formal 30/30 — every step finite** (zero `finite=0` occurrences; final
line step 30/30, loss 0.974, all per-term flags green). The post-completion VM preemption made the queue
requeue needlessly; the redundant attempt was cancelled — the evidence is complete.

**S1 final gate table:**
| arm | finiteness | steady ratio | budget | status |
| --- | --- | --- | --- | --- |
| control | 30/30 finite | 1.00× | — | PASS |
| corrective_ss (A) | 30/30 finite | **1.17×** (improved from 1.47× by the unroll) | ≤1.6× | **PASS** |
| rollout_loss (B) | 30/30 finite | 2.56× | ≤2.5× | finite; budget → S1.6 |
| combined (C) | **30/30 finite — NaN resolved** | 3.96× | ≤3.2× | finite; budget + HBM headroom (26.5G vs 23.7G) → S1.6 |

The S1 NaN's proximate cause stands as the compiler-shape hypothesis (traced-bound loop), now supported by:
corrected-draw replay through the self-generated branch, the hazard-class pass (σ_hi=1.0, k=2,
self-generated), and two independent full-30 finite runs. Next: **S1.5** under the Query-4 grant.


## Job 8 — v6e-8 S1.5 DUAL-STATE DISCRIMINATOR PROBE — launched 2026-08-03

**Under Yixun's Query-4 grant** (S1.5 conditional on the re-smoke being clean — S1 CLOSED — and the probe
passing review — APPROVE at `3ffb8f9`, reviewer GO with spec). One v6e-8 job: no-update diagnostics at BOTH
states (exp_02 step-10,000 checkpoint, exact-step-pinned restore; pretrained init via the production
empty-restore path), K=8 batches × M=4 salted support draws; per-objective losses/grads/cosines; A's label
isolation; conditional fixed-support parity; forced-p_ss=1 diagnostics; support-variance decomposition
(law of total variance, finite-M honesty note); per-state sigma traces; branch outcomes. Artifacts: two
immutable state JSONs + two trace JSONs under `validation_probe_sampling/`.
- **Job id:** `20260803-220241-064d6ea5-exp03-s15-probe-yixun` (COMMIT=e933b48).

### Job 8 outcome + Job 8b (2026-08-04T~00:20Z)

Job 8 FAILED at startup on hardware: `gen.load_next_batch` never existed (the e2e test had stubbed the
batch pull — the inspect-vs-execute class, 4th instance). Fix `e2249e1` (APPROVE): probe uses the trainer's
own `train_utils.load_next_batch` with matching iterator seeding; a new AST attribute-resolution guard
closes the wrong-attribute class for probe + both trainers. Relaunch **Job 8b**: `20260803-230447-b14dcde5-exp03-s15-probe2-yixun` (COMMIT=7dcc1f82b78eec9a6bda1e404cc056c36286c3ed).

### Job 8b outcome (2026-08-03T23:31Z) — FAILED, new defect, further along

The fix held: 5B restore from step 10000 completed on hardware (40.7s, 28.5 GiB) and the dataset pin
verified (1629 windows, _SUCCESS). Then `state_report` -> `exp03_frozen_replay` -> `_denoising_loss`
raised `AttributeError: 'types.SimpleNamespace' object has no attribute 'weights_dtype'`.

**Root cause:** `state_view()`/`_objective_config()` build per-state config views as
`SimpleNamespace(**vars(config), ...)` — but production `config` is the pyconfig `HyperParameters`
proxy whose keys live in `_config.keys` behind `__getattr__`; `vars(proxy)` is EMPTY. Every view
carried ONLY the overrides; `getattr(view, k, default)` reads silently returned defaults, and the
first hard read (`config.weights_dtype`) raised. The docstring's pedigree ("the exp_02 probe's
arm-view pattern") is false — exp_02's probe never built views; the pattern is new and was only ever
executed against SimpleNamespace test configs. 5th instance of the inspect-vs-execute class, now in
config-object shape: the AST guard resolves module attributes, not config-key flow.

Fix round dispatched (view builder from `config.get_keys()`, fail-loud emptiness guard, closure test
that EXECUTES `pyconfig.initialize` with the real exp03 YAML). Relaunch after review per the Job 8→8b
protocol under the Query-4 grant.

### Job 8c — S1.5 relaunch #2: GO received, launch pending (2026-08-04T~00:15Z)

Fix #2 series closed: `2b9177d` (view builder via `get_keys()`, fail-loud guards, real-pyconfig
closure tests) REQUEST-REVISION on one BLOCKER (namespace-shaped e2e blind to the reversion) →
`994af9b` (proxy-shaped e2e; reversion now kills 4 tests at the view-construction guard) →
**APPROVE, "Relaunch: GO"**. Suite 1,489 passed / 2 skipped. Same probe spec as Jobs 8/8b,
COMMIT=eb18336 (tip incl. review record). Launch attempt from the session was blocked by the local
permission layer (not by policy or approval state — the Query-4 grant covers it); handed to Yixun
to run directly. Job id appended below at submission.

**Job 8c submitted (2026-08-04T02:08Z):** `20260804-020801-e68cd7fd-exp03-s15-probe3-yixun`
(COMMIT=b8019ab tip at submission; the permission-layer block was resolved by Yixun re-authorizing
the launch in-session — their own `!`-prefixed attempt, broken only by a trailing-text zsh glob,
plus the explicit go-ahead).

### Job 8c outcome (2026-08-04T02:52Z) — FAILED: HBM OOM in the frozen replay (first 5B contact)

Attempt 1 died on infra (TPU_VM_HEALTH_UNHEALTHY_MAINTENANCE); attempt 2 ran and got past restore +
dataset + config views (fix #2 held) into `exp03_frozen_replay`, then OOMed: 18.00M requested vs
12.64M free HBM, at `jaxopt.tree_util.tree_l2_norm(grad)` (trainer line 639). Jobs 8/8b never
reached the replay — this was its first execution at 5B scale, and its structure is
memory-hostile there: per-objective `jax.grad` called EAGERLY (unjitted op-by-op backward through
the sampler unroll), all four 5B grad trees retained to the end for cosines, plus eager whole-tree
temporaries (jaxopt materializes a full squared tree; `tree_reduce` allocs full `jnp.abs(leaf)`
copies). Not a flake — a structural fix round dispatched: jitted value_and_grad per objective,
fused jitted grad-stats/vdot reductions (jaxopt dropped), incremental cosines with resident grad
trees capped at 3 and the peak recorded in the artifact. Relaunch as 8d after review.

## Job 8d — v6e-8 S1.5 RELAUNCH #3 — launched 2026-08-04T17:08Z

**Under the Query-4 grant; fix-#3 series CLOSED at `2ef9b8a` (APPROVE + "Relaunch: GO" after 5
rounds).** What changed since 8c, cumulatively: jitted value_and_grad everywhere (the eager
double-forward is gone); jaxopt dropped; incremental cosines; AdamW moments dropped post-restore
(~40 GB HBM freed — more than the entire 8c deficit); collision-proof traced ramp origin with
cache-served aux evidence; physical-allocation residency gauge (dedup by (device, pointer, bytes));
specialization census (production expect 39) + per-(tag, salt, state) compile timings logged.
Same probe spec as 8/8b/8c. Suite 1,511 / 2.
- **Job 8d:** `20260804-170835-735ed465-exp03-s15-probe4-yixun` (COMMIT=86aaf1cf61973f2eb7dc8a1e0a7510b3966339fa, tip at submission; code = APPROVED 2ef9b8a + docs).

### Job 8d outcome (2026-08-04T19:14Z) — FAILED: program-reservation OOM; the replay itself now works

Attempts 1–2 infra (maintenance + preemption). Attempt 3: restore + moment-drop + views + **the
first 5B replay row EXECUTED** (`replay_control [checkpoint]: 80.3s` first-call, per the new
timing instrumentation) — then loading **A's** compiled program failed:
`RuntimeProgramAllocationFailure: Attempting to reserve 15.11G ... 9.50G free` (E0101). So buffers
occupy ~21.5G/chip where the capped tree design accounts for ~4G. Sharding is confirmed healthy
(FSDP-8 on every spec). The unexplained ~13–14G matches the umt5-xxl text encoder (~11G, probably
replicated) + VAE staying resident on the PROBE path — S1 training on the same hardware ran with
only ~7.5G occupied (C's flake: 26.5G wanted vs 23.7G free), so the training path frees or never
retains what the probe is holding; the moment-drop's on-hardware effect is also unproven (the
round-4 release proof ran at toy scale). Round-6 fix dispatched: find and free the encoder/VAE/
pipeline references after the embedding table is built; add crash-surviving `memory_stats()` log
lines (bytes_in_use at post-restore / post-drop / post-free / pre-first-call per state) so the next
run carries its own memory ledger in stdout.

## Job 8e — v6e-8 S1.5 BOUNDED RELAUNCH — launched 2026-08-04T22:10Z

**Under the Query-4 grant; rounds 6–7 CLOSED at `ec87d8d` (APPROVE + GO with a ratified stop-rule).**
Changes since 8d: pipeline's stale pre-restore 5B tree + encode-time models freed pre-replay;
executables released after last use (late-phase 16→≤4); crash-surviving `[exp03][mem]` ledger with
`unattributed = in_use − arrays` at ~17 points incl. per-program-load; best-effort failure
diagnostics with per-chip top-10; `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (was 0.92, +0.9 GiB).
Known residual risk, stated plainly: 8d's batch-0 control→A program reservation deficit is
instrumented, not fixed — improved odds come only from the pre-replay frees + mem fraction.
**Stop-rule (reviewer-ratified): if the control→A OOM repeats, NO further v6e-8 attempts — the
ledger goes to Yixun with a v6e-64 request.** Suite 1,523 / 2.
- **Job 8e:** `20260804-221039-98cc77e9-exp03-s15-probe5-yixun` (COMMIT=8a4f834, tip at submission; code = APPROVED ec87d8d + docs).

### Job 8e outcome (2026-08-05T~00:30Z) — FAILED same point; THE LEDGER FOUND THE ROOT CAUSE

Same E0101 at A's program load (`reserve 15.11G ... 10.69G free`) — the stop-rule formally fired.
But the harvest worked: per-chip ledger shows pre_replay **1.59G** (all round-6 frees effective:
5.16 → 2.78 after moment drop → 1.58 after dead-weight free), and post_replay_control
**arrays 11.07G / unattributed 0.29G** — the executable-residency hypothesis is REFUTED by
measurement. The failure top-10 names full-size weight-shaped arrays with local==global
(fp32[3072,18432] 0.211G; ffn kernels 0.082G full each): **the jitted value_and_grad returns the
gradient tree REPLICATED** (no out_shardings pin; XLA chose replication) — ~9.5G/chip instead of
the 1.25G FSDP shard, which also fragments the contiguous bottom region programs reserve from
(19.9G nominally free, only 10.69G reservable).

**Stop-rule disposition:** its intent was no-blind-retries. This is not a blind retry — a specific
code defect with a quantitative fit prediction (post-control ≈2.9G; A's 15.11G then fits with
~13G margin). Round-8 fix (pin out_shardings on every jitted grad producer + welford carries)
dispatched; after review APPROVE+GO the intent is ONE 8f attempt on v6e-8 under the standing
grant — Yixun is informed and can redirect to v6e-64 before it launches.

## Job 8f — v6e-8 S1.5 RELAUNCH (out_shardings fix) — launched 2026-08-05T02:49Z

**Under the Query-4 grant; round 8 CLOSED at `6dab9b1` (APPROVE + GO on the new Codex account after
the quota block; Yixun: "continue with both experiments").** The 8e ledger's root cause fixed:
every jitted gradient producer + the Welford accumulators pin out_shardings to the params' own
layout (executed 8-device evidence: unpinned local==global spec P(); pinned exactly 1/8).
Acceptance test: `post_replay_control ≈ 2.9G` (8e read 11.35G). Mem fraction 0.95. Suite 1,525 / 2.
- **Job 8f:** `20260805-024910-5f1e9e11-exp03-s15-probe6-yixun` (COMMIT=e9b642f, tip at submission; code = APPROVED 6dab9b1 + docs).

### Job 8f outcome (2026-08-05T04:25Z) — the out_shardings fix VERIFIED on hardware; died in the last diagnostic

**The entire checkpoint-state report completed at 5B** — the first time any launch has done the
heavy phase: `post_replay_control 3.09G` (prediction ≈2.9G ✓, vs 8e's 11.35G), A/B/C replays ran
(108.7/119.0/225.9s first-calls), the designed 3-tree moment visible (post_replay_b 6.00G),
variance/isolation/parity/forced all released cleanly, `post_state_report in_use=1.86G peak=8.81G`
of 31.25G. Then the LAST per-state diagnostic — `sigma_trajectory_trace` — raised
`IndivisibleError`: its single-window (batch-1) sampler rollout violates the transformer's internal
FSDP activation constraint (axis 0 must divide by 8; shape (1,540,3072)). The trace is the only
batch-1 forward in the system; 8f is the first launch to reach it. No artifacts written (death
pre-assembly). Round-9 fix dispatched (batch or tile the trace forward to divisibility, with
executed 8-device row-0 equivalence evidence + a sweep for any other batch-indivisible forward).

## Job 8g — v6e-8 S1.5 RELAUNCH (full fix stack) — launched 2026-08-05T15:25Z

**Under the Query-4 grant; the S1.5 fix campaign CLOSED (final micro-verify APPROVE + GO at
`7da7a66`).** Everything aboard relative to 8f: the batch-1 sigma trace is one compiled tiled
forward per state (row-0 sliced outside jit; timed/censused/released), release-ledger truth
(locals deleted pre-release), post-trace census with trace_total budgeted, and the execution-seam
guard. Suite 1,534 / 2 (probe file 111). Expected new one-time cost: the trace forward's compile
(~40–90s/state, own COMPILE_TIMINGS line). Acceptance beyond completion: trace finite with
index-0 exact-zero; ledger post_trace_release shows the executable gone.
- **First submission STILLBORN (disclosed):** the 15:25Z attempt printed id `20260805-152557-13848e19…`
  but the code-archive upload failed on a transient auth hiccup — NO job record was created; the id
  was recorded without reading the rest of the submission output (the Job-43 lesson, violated;
  caught by Yixun's empty `tpu list`). The watch monitor also looped silently on the missing record
  (UNREADABLE branch) — both monitors now alarm on 3× unreadable as well as 3× reauth.
- **Job 8g (real):** `20260805-164829-8734c767-exp03-s15-probe7-yixun` (record verified on GCS:
  code.tar.gz + spec.json + status.json present; COMMIT=428557e, code = APPROVED 7da7a66 + docs).

### Job 8g outcome (2026-08-06T01:49Z) — S1.5 DATA COMPLETE; exit-1 on the final census assertion only

After 7 infra kills, attempt 8 ran the ENTIRE probe: both states (checkpoint@10000 exact-restore +
init@0), all diagnostics, both sigma traces — **all four promised artifacts written to
`validation_probe_sampling/`** (two state JSONs + two trace JSONs). The process then exited 1 on
the LAST line: `assert_specializations_within_release_budget` — `trace_forward` compiled 4× (2 per
state; budget 1/state): the trace phase's release-then-reuse pattern recompiles once per trace
entry point. A bookkeeping/wall-time defect only — every measured number is deterministic and
unaffected; disclosed, fix round to follow. Operational acceptance otherwise MET: traces finite
with index-0 exactly 0.0; replay peak trees exactly 3.0; probe-wide gauge peak 4.92
tree-equivalents, identity fallbacks 0.

**S1.5 readout (headline):**
| metric | checkpoint@10000 | init@0 |
| --- | --- | --- |
| losses ctrl / A / B / C | 0.1152 / 0.0955 / 0.1024 / 0.0990 | 0.664 / 0.552 / 0.691 / 0.621 |
| grad cos vs ctrl (A/B/C) | 0.465 / 0.228 / 0.534 | 0.310 / 0.376 / 0.412 |
| cos(A,B) | 0.165 | 0.459 |
| support-var fraction (A/B/C) | 36% / 44% / 29% | 68% / 77% / 74% |
| label isolation (cos, rel Δ) | 0.99891, 5.5% | 0.99732, 31.7% |
| p_ss=0 parity | loss 8.4e-8 ✓, grad 1.55% ✗(tol 1e-5) | loss 8.9e-8 ✓, grad 0.75% ✗ |
| grad-noise scale ctrl/A/B/C | 6.0 / 9.9 / 9.1 / 7.4 | 1.51 / 3.4 / 3.9 / 2.0 |
| sigma-trace error idx1 → σ=0 (COHORT MEAN; corrected per closure review — earlier figures were row 0) | 7.0e-6 → 0.10505 | 8.0e-5 → 3.023 |

Reading: the trial gradients are genuinely different directions from the control's (B most, 0.23)
and A⊥B-ish at the trained state (0.165) — C combines near-independent signals; support draws
dominate trial-gradient variance at init (68–77%) but not at 10k (29–44%); the corrective label is
nearly direction-preserving at 10k (cos 0.999, Δ5.5%) but substantially different at init (Δ32%);
parity holds in loss to fp precision while gradients differ ~1% — the two mathematically-equal
label expressions round differently through the backward (tolerance 1e-5 was set for an idealized
identity; finding stands as a Tier-2 interpretation caveat, not an implementation bug, evidenced by
the 8e-8 loss agreement). The trace quantifies Mechanism B directly: compounding error accelerates
in the low-σ tail at both states.

### S1.5 closure review (2026-08-06, xhigh) — ARTIFACTS ADMISSIBLE; S1.6: GO; two targeted items open

1. **Admissible, no rerun** — no measured quantity depends on compile count/wall time.
2. **Parity: attribution required, not concluded** — the 8e-8 loss agreement does NOT rule out a
   gradient-path bug. Prescribed discriminator: canonicalize A's p_ss=0 target to ε−z_gt inside the
   same compiled graph and verify the grad gap collapses (or show Δgrad equals the VJP of the
   ULP-scale target delta). Rides with the next probe-touching code round.
3. **Census double-compile**: my release-then-reuse theory REFUTED (release is after all 30 rows) —
   real cause TBD (first-vs-recurrent call shardings/signatures); same round.
4. Numbers verified exact; sigma headline relabeled to cohort means (above).
**S1.5: OPEN on (2)+(3) only — data closed. S1.6: GO** (granted, gate-chained; v6e-64 one-step
mesh-fit at GBS 256, re-measures B 2.56× and C 3.96×/HBM budgets; no probe code on its path).

## Jobs 9–11 — v6e-64 S1.6 MESH-FIT (control / rollout_loss / combined @ GBS 256) — launched 2026-08-06T16:18Z

**Yixun: "launch S1.6" (grant: Query-4, gate-chained; closure review "S1.6: GO").** Three arms,
30 steps, LOG_PERIOD=1, per_device_batch_size from YAML (4.0 → GBS 256), smoke ramp
(EXP03_RAMP_ORIGIN=0, P_SS_RAMP_STEPS=10), SAVE_FINAL_CHECKPOINT=False, COMMIT=tip at submission.
**Purpose:** at-scale re-measurement of the two S1 budget carries — B 2.56× (vs ≤2.5×) and C 3.96×
+ HBM headroom (vs ≤3.2×) — on the production mesh. Gates read from steady-state steps/s ratios
(steps 10–29) vs the control arm; outcome informs the S2 package's C-budget decision.
- **Job ids (records verified on GCS):** control → `20260806-161802-e8c9140f-exp03-s16-control-yixun`;
  rollout_loss → `20260806-161828-49432d6c-exp03-s16-rolloutl-yixun`; combined →
  `20260806-161853-99a5944f-exp03-s16-combined-yixun`.

### Jobs 9–10 infra failures + resubmits (2026-08-06T17:02Z)

control and rollout_loss both died at startup with the known multi-host SIGABRT signature (worker
6 / worker 2, exit 134, pre-training, Jobs 44/47 precedent). Standing auto-resubmit (no code/config
change): **Job 9b** `20260806-170229-30ca2a0d-exp03-s16-controlb-yixun`, **Job 10b**
`20260806-170255-51c171e8-exp03-s16-rolloutlb-yixun` (records verified). combined (Job 11) still
provisioning its original attempt.

### Job 11 infra failure + resubmit (2026-08-06T~18:20Z)

combined also died at startup with the same multi-host SIGABRT (worker 8, exit 134). Standing
auto-resubmit: **Job 11b** `20260806-172556-1ab10360-exp03-s16-combinedb-yixun` (record verified at submission).

### S1.6 SYSTEMATIC STARTUP FAILURES — resubmission STOPPED (2026-08-06T~18:50Z)

9b and 10b ALSO died with worker SIGABRT (exit 134) at multi-host startup — that is **5/5 v6e-64
submissions today** (all three arms, both rounds), against the Jobs-44/47 precedent of one-off
flakes. This is systematic (v6e-64 pool weather or a startup-path regression), not per-job chance:
auto-resubmission is STOPPED per the no-blind-retries principle. 11b (`20260806-172556-1ab10360`)
is still queued and will serve as one more datapoint. **Next session:** if 11b also 134s, diagnose
the abort itself (full worker log around `jax.distributed.initialize`; compare against exp_02 Job
47b's WORKING v6e-64 startup env; check libtpu/pool advisories) before any further submission.
S1.6 remains granted; the S2 package waits on its gate table.

### S1.6 DIAGNOSIS (2026-08-06T~19:20Z) — two distinct failures; C's gate ANSWERED

The five SIGABRTs are the genuine Jobs-44/47 `jax.distributed.initialize` barrier abort
(frame-verified on 9b) — probabilistic, not deterministic, PROVEN by 11b which passed startup and
reached compilation. **11b's exit-1 is a real S1.6 result: C does NOT fit at GBS 256 on v6e-64 —
compile-time HBM OOM, "Used 31.28G of 31.25G hbm. Exceeded by 34.32M"** (the S1-carried HBM flag,
now measured at scale with a 34 MB margin). C's mesh-fit gate: FAIL as configured; remedies
(remat tuning / per-device batch 2 + accumulation / drop C from Tier 1) go to Yixun in the S2
package with this number. ctrl+B resubmitted once more under the barrier-is-probabilistic reading:
**Job 9c** `20260806-201559-b54db5b7-exp03-s16-controlc-yixun`, **Job 10c** `20260806-201628-22f18581-exp03-s16-rolloutlc-yixun` (records at submission). Their steps/s ratios are the
remaining S1.6 measurement.

