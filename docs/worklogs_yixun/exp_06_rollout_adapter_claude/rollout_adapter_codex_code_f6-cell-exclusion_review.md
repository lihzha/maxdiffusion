# exp_06 F6 `cell-exclusion` — Codex code review

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, 2026-08-12. Verdict: **REQUEST-REVISION — 1 BLOCKER + 2 MAJOR + 1 MINOR.** BLOCKER: the F6 delta changes manifest-covered files, so M1-5 cannot adopt the F5-era banked cells (manifests reproduced: 6eda654=4bbdbb28… vs working=64f92825…) — the Planner's "~30-min adoption" claim was WRONG; ruled resolved by policy (no migration rule — the manifest is doing its job; M1-5 re-measures, converging cross-attempt at the new SHA). MAJOR: loader accepts a doctored table with a cell in both authorized+excluded; the two headline serialization tests prove less than claimed. MINOR: empty parse tokens. Closed as round F6b.

**Verdict: REQUEST-REVISION — 1 BLOCKER, 2 MAJOR, 1 MINOR.**

1. **BLOCKER — M1-5 cannot adopt the 12 production-banked F5 cells.** Adoption compares the entire context digest, including `code_sha` and a manifest over every deployed non-test Python file ([pos_rollout_fit_probe.py:551](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:551), [pos_rollout_fit_probe.py:1827](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1827)). M1-4 ran at `6eda654` ([rollout_adapter_command.md:72](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/rollout_adapter_command.md:72)); F6 changes manifest-covered `pos_rollout_fit_probe.py` and must receive a new commit. I reproduced the manifests: `6eda654 = 4bbdbb28…`, working F6 = `64f92825…`; the code SHA will differ too. The claimed production adoption at [rollout_adapter_command.md:89](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/rollout_adapter_command.md:89) therefore cannot occur. The test at [test_pos_rollout_cell_publication.py:1318](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_cell_publication.py:1318) banks and adopts within one unchanged F6 process, so it does not model the F5→F6 production transition. Either introduce a narrowly reviewed compatibility/migration rule or explicitly budget M1-5 to remeasure all 12 surviving cells and remove the “adopts in ~30 min” claim.

2. **MAJOR — a v4 table can be both `AUTHORIZED` and `EXCLUDED`, and the quote gate accepts it.** The serializer emits the new lists independently ([pos_rollout_fit_probe.py:1151](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1151)). The loader only checks that they are lists, then reconstructs `ProbeEvidence` without enforcing uniqueness or disjointness ([pos_rollout_fit_probe.py:1380](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1380)). An edited-and-rehashed table can therefore append an authorized cell to `excluded_cells`, reproduce byte-identically, and load. `assert_cell_authorized` returns on `authorized_cells` before examining exclusions ([pos_rollout_fit_probe.py:1458](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1458)). Require authorized/refused, excluded, and skipped to be mutually disjoint and individually duplicate-free, with non-empty consistent exclusion reasons.

3. **MAJOR — the two headline serialization tests do not prove their claims.**

   - The run-digest test declares `one_step:8:2` excluded but requests only two rollout cells ([test_pos_rollout_cell_publication.py:1308](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_cell_publication.py:1308)). That exclusion is filtered out at [pos_rollout_fit_probe.py:2436](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2436); the digest changes because the two runs measured different cell sets. It proves neither the declaration nor its reason affects the digest. Production’s full-ladder path does serialize the semantic exclusion and reason, but this needs an isolated test varying the cell list and reason independently.
   - The “bit-for-bit” test removes `cell_provenance` and `sha256` through `_without_provenance` ([test_pos_rollout_cell_publication.py:217](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_cell_publication.py:217)) and compares two tables both produced by F6 ([test_pos_rollout_cell_publication.py:1288](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_cell_publication.py:1288)). It does not compare full serialized bytes against HEAD. Literal HEAD identity is impossible because v3 becomes v4 and three fields are added; narrow the claim to behavioral inertness or compare an explicitly defined stable projection.

4. **MINOR — malformed empty tokens are silently accepted.** `entries` discards every empty chunk before validation ([pos_rollout_fit_probe.py:488](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:488)). Thus `pos_fit_excluded_cells=","` silently parses as no exclusions and can reach the deterministic-fault cell, contrary to “malformed = loud.” Reject empty tokens in any non-empty declaration. Case, canonical duplicate detection, arm names, and ladder membership otherwise look strict.

The legitimate consumer path is sound: launcher and trainer both use the sole v4 loader, the trainer gates the exact `(arm, microbatch, k)` before expensive work, and no production v3/v4 dual acceptance remains. I also accept `skipped_cells` in principle: with GBS 256 it is inert, projections still iterate measured authorized cells, and training behavior does not change—subject to the disjointness validation above.

**Planner M2 intent:** Agreed. P3’s matched properties do not name microbatch, and running both arms at microbatch 16 is stricter anyway: both use the same microbatch and preserve the matched GBS/update stream. Both v6e-8 cells were measured and authorized. M3 still requires its separate v6e-64 M1′ v4 authorization because topology is context-bound.
**Verdict: REQUEST-REVISION — 1 BLOCKER, 2 MAJOR, 1 MINOR.**

1. **BLOCKER — M1-5 cannot adopt the 12 production-banked F5 cells.** Adoption compares the entire context digest, including `code_sha` and a manifest over every deployed non-test Python file ([pos_rollout_fit_probe.py:551](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:551), [pos_rollout_fit_probe.py:1827](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1827)). M1-4 ran at `6eda654` ([rollout_adapter_command.md:72](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/rollout_adapter_command.md:72)); F6 changes manifest-covered `pos_rollout_fit_probe.py` and must receive a new commit. I reproduced the manifests: `6eda654 = 4bbdbb28…`, working F6 = `64f92825…`; the code SHA will differ too. The claimed production adoption at [rollout_adapter_command.md:89](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/rollout_adapter_command.md:89) therefore cannot occur. The test at [test_pos_rollout_cell_publication.py:1318](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_cell_publication.py:1318) banks and adopts within one unchanged F6 process, so it does not model the F5→F6 production transition. Either introduce a narrowly reviewed compatibility/migration rule or explicitly budget M1-5 to remeasure all 12 surviving cells and remove the “adopts in ~30 min” claim.

2. **MAJOR — a v4 table can be both `AUTHORIZED` and `EXCLUDED`, and the quote gate accepts it.** The serializer emits the new lists independently ([pos_rollout_fit_probe.py:1151](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1151)). The loader only checks that they are lists, then reconstructs `ProbeEvidence` without enforcing uniqueness or disjointness ([pos_rollout_fit_probe.py:1380](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1380)). An edited-and-rehashed table can therefore append an authorized cell to `excluded_cells`, reproduce byte-identically, and load. `assert_cell_authorized` returns on `authorized_cells` before examining exclusions ([pos_rollout_fit_probe.py:1458](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:1458)). Require authorized/refused, excluded, and skipped to be mutually disjoint and individually duplicate-free, with non-empty consistent exclusion reasons.

3. **MAJOR — the two headline serialization tests do not prove their claims.**

   - The run-digest test declares `one_step:8:2` excluded but requests only two rollout cells ([test_pos_rollout_cell_publication.py:1308](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_cell_publication.py:1308)). That exclusion is filtered out at [pos_rollout_fit_probe.py:2436](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:2436); the digest changes because the two runs measured different cell sets. It proves neither the declaration nor its reason affects the digest. Production’s full-ladder path does serialize the semantic exclusion and reason, but this needs an isolated test varying the cell list and reason independently.
   - The “bit-for-bit” test removes `cell_provenance` and `sha256` through `_without_provenance` ([test_pos_rollout_cell_publication.py:217](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_cell_publication.py:217)) and compares two tables both produced by F6 ([test_pos_rollout_cell_publication.py:1288](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_cell_publication.py:1288)). It does not compare full serialized bytes against HEAD. Literal HEAD identity is impossible because v3 becomes v4 and three fields are added; narrow the claim to behavioral inertness or compare an explicitly defined stable projection.

4. **MINOR — malformed empty tokens are silently accepted.** `entries` discards every empty chunk before validation ([pos_rollout_fit_probe.py:488](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:488)). Thus `pos_fit_excluded_cells=","` silently parses as no exclusions and can reach the deterministic-fault cell, contrary to “malformed = loud.” Reject empty tokens in any non-empty declaration. Case, canonical duplicate detection, arm names, and ladder membership otherwise look strict.

The legitimate consumer path is sound: launcher and trainer both use the sole v4 loader, the trainer gates the exact `(arm, microbatch, k)` before expensive work, and no production v3/v4 dual acceptance remains. I also accept `skipped_cells` in principle: with GBS 256 it is inert, projections still iterate measured authorized cells, and training behavior does not change—subject to the disjointness validation above.

**Planner M2 intent:** Agreed. P3’s matched properties do not name microbatch, and running both arms at microbatch 16 is stricter anyway: both use the same microbatch and preserve the matched GBS/update stream. Both v6e-8 cells were measured and authorized. M3 still requires its separate v6e-64 M1′ v4 authorization because topology is context-bound.

---

# Strengthening record — round F6b (Coder, 2026-08-12)

All four findings resolved; none rejected. Narrative in `rollout_adapter_worklog.md` (2026-08-12T16:30Z).

## BLOCKER — M1-5 cannot adopt the 12 banked cells — **RESOLVED BY POLICY (Planner ruling)**

No compatibility or migration rule. Grandfathering cells measured by different code is the hole
F5b/F5c closed; the manifest refusing them is the mechanism working. M1-5 re-measures the 12 reachable
cells (~2–2.5 h for attempt 1, banking as it goes) and converges across attempts at the same new SHA.
The "~30-min adoption" claim in the command ledger is withdrawn (Planner is correcting it there).
Added `test_a_code_change_between_attempts_refuses_the_cells_banked_before_it`, which moves the
manifest BETWEEN attempts — the transition re-measures with `manifest_digest` named, and a third
attempt at the new manifest adopts the second's cells. The reviewer was right that the previous test
(bank + adopt inside one unchanged process) could not model this.

## MAJOR — a cell could hold two statuses — **FIXED**

`ProbeEvidence._assert_one_status_per_cell`, called from `as_payload`, so neither the publishing path
nor the loader's re-decide can produce one; plus explicit loader pre-checks for precise diagnosis.
Refuses duplicates within any list, overlaps across every pair of (authorized, refused, excluded,
skipped), and exclusions with an empty reason. Red-first with the reviewer's doctored-table
construction; probe `F6-2` refuses it **at load**, distinct from `F6-1`'s gate refusal.

## MAJOR — the two headline tests — **BOTH CORRECTED**

(a) The digest-isolation test declared an exclusion that was filtered out, so it observed a digest
change caused by measuring different cells. Replaced with a fixed-measured-set construction (X
measures {A,B}; Y measures {A,B} and declares C excluded; Z varies only the reason), asserting
identical `measured_cells`/`measurements`, an independent digest move for the declaration and for the
reason, and unmoved per-cell fingerprints. (b) "Bit-for-bit with today" was unprovable — v3→v4 plus
three new fields makes literal identity impossible. Renamed to
`test_an_empty_exclusion_declaration_is_behaviourally_inert` with the comparison projection declared
in `_UNSTABLE_FIELDS` and each stripped field justified; the claim is withdrawn from the F6 worklog.

## MINOR — empty tokens — **FIXED**

A blank declaration is no declaration; a declaration made of punctuation (`","`, trailing/doubled
commas) is malformed and refused by name.

## Verification

Canonical suite **2248 passed / 0 failed** (560 s). Battery **90 probes — 89 REFUSED, 1 DECLARED, 0
SUCCEEDED, 0 UNPARSED**, exit 0 (`harness/attacks_f6b_20260812.log`, sha256 `d31da372ab8344a0…`).
`black` / `ruff` / `git diff --check` clean. Three tests passed before their fix (the behaviour was
already right; the evidence was missing) and were mutation-checked rather than assumed — see the
worklog's red-side note, including the sha-verified restore of `pos_rollout_fit_probe.py`. Nothing
committed.
