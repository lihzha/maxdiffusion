# null_adapter — Codex code review: round R10 `launchers-config`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (read-only, exp_04 worktree). Working tree at HEAD `aecda3c`. Reviewer walked the J1 smoke path end-to-end, verified the decode trace independently, checked YAML key coverage and XLA-flag parity, and produced the J1-readiness breakage list.

## Reviewer output (verbatim)

Context loaded:

- `experiment_SOP.md`; announcements 01 and 02; `null_adapter_yixun_query.md`; related exp_02 analysis.
- `plan_null_adapter.md` v5, especially §4-P1/P2, §5 items 3/5/10, §9 J1.
- `null_adapter_worklog.md` through the R10 entry, including decode-range trace and Planner ratifications.
- All R1–R9 reviews: `sigma-embed-noise`, `invert-trajectory`, `optimize-nulls`, `replay-operator`, `record-schema-io`, `verify-replay`, `gates-module`, `runner-capacity-core`, `runner-decode-videos`, `runner-cache-resume`, and `manifests`.
- All six R10 deliverables in full: YAML, entrypoint, modes module, launcher, and both test files.
- Composed dependencies: runner core, gates, pixels, records/verifier, cache policy/shards, manifest loader, side-adapter trainer/launcher, WAN pipeline/TI2V subclass, video processor, and image processor.
- Validation: authorized suite passed `80 passed in 11.22s` with capture disabled because the read-only sandbox has no temp directory; `bash -n` and `git diff --check` passed. All direct `config.*` reads exist in the YAML, and the YAML retains every side-adapter base key. XLA flag blocks are exactly equal.

1. **BLOCKER — the production backend cannot load or construct `Backend`.** [run_wan_null_inversion.py:253](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:253), [run_wan_null_inversion.py:255](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:255), [run_wan_null_inversion.py:283](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:283), [run_wan_null_inversion.py:310](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:310)

   `_load_backend` imports base `WanPipeline`, which has no `from_pretrained`; that method exists on `WanPipelineTI2V_2_2`. Even after correcting the class, the returned dictionary has no `read_batch`, so `main` deterministically raises `KeyError("read_batch")`. There is also no manifest-bound TFRecord reader or generation-binding check.

   Concrete change: load `WanPipelineTI2V_2_2` under the trainer’s `axis_rules` context; implement `read_batch` from validated manifest rows, checking shard binding and returning `CapacityBatch` plus exact record fields; return the resolved HF snapshot path. Add a two-example `main` composition test.

2. **MAJOR — capacity does not implement the required decode→gate→record sequence.** [null_adapter_modes.py:161](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:161), [null_adapter_modes.py:171](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:171), [null_adapter_modes.py:180](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:180), [null_adapter_modes.py:183](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:183)

   Records are irreversibly published before VAE decode and pixel-table fill. A decode failure therefore leaves completed immutable shards that a retry cannot overwrite. No `gate_g1`, `gate_g2`, or `select_target` call exists; filled tables are written but never gated. Quarantined names are removed from `names`, so any future gate must receive the original manifest explicitly to preserve coverage honesty. Full-cohort decode is also one monolithic B=64 VAE call rather than bounded batches, risking OOM despite `null_batch_size`.

   There is no two-example smoke limiter: changing batch size to two still processes all 64 examples.

   Concrete change: add a smoke-only bounded-run setting; compute arms in batches, decode the entire declared cohort in bounded batches, fill tables, evaluate G1/G2/target selection against `plan["names"]`, then publish records/videos/report. Do not leave completed shards if decode/gating fails.

3. **MAJOR — adequacy is under-declared and discards its primary evidence.** [null_adapter_modes.py:208](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:208), [null_adapter_modes.py:217](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:217), [null_adapter_modes.py:228](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:228)

   Probe size is tied to `null_decode_subset`, not fixed to the first eight DEV examples; the tests deliberately approve three examples and an arbitrary grid. The JSON drops `per_example`, `[N,J,B]` tracking losses, grad-norm traces, final losses, and adoption numbers—the exact R6 evidence contract. Nothing consumes the adoption result to rerun full DEV before gating, applies the +2-hour stop, or performs the required `L_null` diagnostic.

   Concrete change: preflight `dev64`, exactly the first eight names, and the approved six-cell grid before reading data; persist all traces and adoption numbers; feed the adopted recipe into capacity with the projection stop and L-null diagnostic. A3 remains correctly deferred to R11.

4. **MAJOR — cache mode never performs the P2 fidelity decision and cannot cache A2.** [null_adapter_modes.py:243](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:243), [null_adapter_modes.py:252](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:252), [null_adapter_modes.py:276](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:276), [null_adapter_modes.py:298](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:298)

   `arm` defaults to A1 and `main` never passes another value; `null_noise_convention=global` therefore still writes A1/keyed. `fidelity_metrics` is optional, never supplied by `main`, and is evaluated only after records have already been written. Its returned fp32 fallback never changes the header or records. Each cache batch also executes all six capacity arms despite caching one selected arm.

   Concrete change: require the J1 target-selection artifact; run the first-eight-DEV fp32-versus-serialized-fp16 gate before any cohort caching; construct the final header only after its dtype verdict; execute only the selected A1 or A2 path.

5. **MAJOR — resume supplies the wrong fingerprint and cannot cleanly supersede quarantine history.** [null_adapter_modes.py:266](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:266), [null_adapter_modes.py:269](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:269), [null_adapter_modes.py:292](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:292), [null_adapter_shards.py:239](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_shards.py:239)

   `resume_plan` expects SHA-256 of serialized `ProvenanceHeader`; cache passes only `header.base_context_fingerprint`. My probe produced distinct digests (`95abaa…` versus `9c5489…`), so every valid prior shard is rejected instead of resumed.

   Additionally, an earlier quarantine remains in `resume.quarantined`; if that name later succeeds, the report still counts it quarantined, and the next resume sees the name both quarantined and covered across immutable shards and raises as a duplicate.

   Concrete change: expose/use the canonical full-header fingerprint; permit a later covered record to supersede prior quarantine history while still forbidding duplicate coverage; remove successful retries from current quarantine totals and derive collision-free shard identities from existing paths.

6. **MAJOR — verify mode can certify “nothing” and bypasses the validated-shard boundary.** [null_adapter_modes.py:311](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:311), [null_adapter_modes.py:325](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:325), [null_adapter_modes.py:345](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:345), [null_adapter_modes.py:383](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:383), [test_null_adapter_modes.py:366](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_modes.py:366)

   The test explicitly expects zero records to return exit code 0. Duplicate names overwrite earlier verdicts, coverage against the declared cohort is never checked, and `read_shard` ignores the marker’s record SHA-256 and full header fingerprint. A valid replacement record can therefore bypass R8’s validated safe boundary.

   Concrete change: validate every shard before reading, reject empty inputs, require exact unique cohort coverage, treat quarantines/missing records as failures, and derive the expected arm from J1 selection. Any artifact or replay failure must produce exit 1.

7. **MAJOR — published provenance is not bound to the actual launch.** [run_wan_null_inversion.py:304](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:304), [run_wan_null_inversion.py:318](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:318), [run_wan_null_inversion.py:319](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:319), [run_wan_null_inversion.sh:64](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/bash_scripts/run_wan_null_inversion.sh:64), [run_wan_null_inversion.sh:127](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/bash_scripts/run_wan_null_inversion.sh:127)

   `COMMIT` is printed but never assigned/exported, so records default to `code_sha="unknown"`. The backend never supplies `resolved`, making every model revision `@unresolved`. `manifest_hash` is only the source-shard listing checksum, not a digest of the selected immutable manifest rows/header. The launcher’s default manifest URI also disagrees with the worklog’s ratified mirror path.

   Concrete change: export and validate the 40-hex commit, capture the resolved HF commit, hash the complete canonical manifest set, and change the default to `gs://v6_east1d/datasets/droid_wan_null_adapter/manifests/j0/`.

8. **MAJOR — the production video sink cannot publish to the configured GCS artifact root.** [null_adapter_modes.py:192](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:192), [null_adapter_modes.py:408](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_modes.py:408), [null_adapter_pixels.py:329](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/null_adapter_pixels.py:329)

   `default_sinks` passes `gs://…/videos/*.mp4` to a local `os.makedirs/os.replace` writer. It will create a local `gs:` directory or fail; it will not publish GCS artifacts.

   Concrete change: encode to a checked local temporary path, then upload transactionally through `gfile`, including the PNG fallback, or require a separate explicit local video directory followed by a verified upload.

9. **MAJOR — the no-cover residue is much broader than ratified.** [run_wan_null_inversion.py:230](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:230), [run_wan_null_inversion.py:240](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:240), [run_wan_null_inversion.py:292](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:292)

   Function-level pragmas exclude the whole loader, whole shard discovery function, and whole `main`, not only literal pipeline calls plus one glob. That is precisely where the wrong pipeline class, missing reader, wrong fingerprint, and hard-coded arm escaped the 80 tests.

   Concrete change: remove coverage exclusion from `main` and decisionful loader code; inject loader/manifest/sinks into a tested entrypoint. Exclude only the literal model-loading/encoding/decoding calls.

10. **MINOR — decode citations are correct, but rejection is weaker than claimed.** [run_wan_null_inversion.py:80](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/run_wan_null_inversion.py:80)

   I independently confirmed the trace: inherited WAN decode at `wan_pipeline.py:663–671` → `VideoProcessor.postprocess_video:99–113` → default `do_normalize=True` → clamp in `VaeImageProcessor.denormalize:185–189` → float32 NumPy conversion. The default `unit` declaration is therefore correct.

   However, nested ranges cannot detect every wrong declaration: unit `[0,1]` declared as byte passes and becomes `[0,0.00392]`. The wrapper also accepts `1.0005` under unit and silently clips it, reopening R7’s strict-boundary exception.

   Concrete change: pin this production backend to exact `unit` with zero tolerance. If alternate backends are retained, document declaration as trusted rather than claiming all mismatches are rejected.

11. **MINOR — launcher preflight/logging is incomplete.** [run_wan_null_inversion.sh:78](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/bash_scripts/run_wan_null_inversion.sh:78), [run_wan_null_inversion.sh:98](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/bash_scripts/run_wan_null_inversion.sh:98), [run_wan_null_inversion.sh:130](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/bash_scripts/run_wan_null_inversion.sh:130)

   HF prefetch ordering and XLA parity are correct, and `pipefail` preserves Python failure. But prefetch, preflight, configuration echo, and git state occur before `tee`; the run log is not “all terminal output.” The ffmpeg check accepts an existing non-executable file, and the carried R1 on-device noise golden is absent.

   Concrete change: start tee before prefetch, use `os.access(path, os.X_OK)`, preflight the actual TI2V pipeline import, and assert the R1 golden on the TPU before arms execute.

### J1-readiness assessment

The two-example smoke cannot currently reach an arm:

- With defaults, manifest loading targets the wrong URI.
- With the URI corrected, `_load_backend` fails on `WanPipeline.from_pretrained`.
- With that import corrected, `main` fails on missing `backend["read_batch"]`.
- There is no two-example/max-batch smoke limit, so a “batch size 2” launch proceeds toward all 64 DEV examples.
- Beyond `_load_backend`, the first decode failure leaves immutable record shards; successful video generation targets local pseudo-`gs:` paths; no gates/selection run; cache cannot make the fp16/fp32 or A1/A2 decision; and empty verification exits successfully.

A3 is intentionally R11 scope and is not counted against this verdict.

### Final verdict

**REQUEST-REVISION — R10 is not J1-launchable because the real backend cannot construct, and the capacity, cache, and verification bodies do not yet preserve the approved execution, provenance, and exit contracts.**

**Status:** No subprocesses or jobs are running; review is complete, and Yixun can steer now.
