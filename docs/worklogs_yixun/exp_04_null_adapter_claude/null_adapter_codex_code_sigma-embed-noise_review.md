# null_adapter — Codex code review: round R1 `sigma-embed-noise`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh (codex-cli 0.146.0, read-only, exp_04 worktree). Code under review: the four uncommitted R1 files at HEAD `defba0a` (working tree). Coder: Claude Opus 5 (max). Briefing per SOP (prompt archived in session scratchpad); reviewer independently re-derived all six golden noise heads from plan §3 and ran the suite (34 passed).

## Reviewer output (verbatim)

Context loaded:

- `docs/worklogs_yixun/experiment_SOP.md` — reviewer briefing, TDD, write→review→strengthen cycle, and validation ladder.
- `null_adapter_yixun_query.md` — both queries, including Query 2’s approval/grants and unchanged exp_04 scope.
- `plan_null_adapter.md` v5 — §§3, 5 items 1/12, 6, and 8.
- `null_adapter_worklog.md` through the final R1 write-phase entry and all five Planner acceptances.
- R1 implementation and all three test files.
- `side_adapter_wan.py:814-824` and trainer `_compute_null_context:313-325`.
- Repository state: HEAD `defba0a`; the four R1 files are untracked/uncommitted.

Validation: 34 tests passed in 2.75s; Ruff passed. An independent transcription of plan §3 reproduced all six golden noise heads and raw keys. The fixtures exercise SHA words above `2^31`, including both words for `GLOBAL`.

1. **MAJOR** — [null_inversion_wan.py:45](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_inversion_wan.py:45), [test_null_adapter_noise.py:104](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_noise.py:104)  
   `keyed_noise` and `global_noise` accept arbitrary shapes, although plan §3 defines the convention as exactly one float32 `(48,9,12,20)` draw. This makes the documented `(name, k) → noise` mapping additionally depend on an unrecorded argument. In particular, `global_noise(k, shape=(B,48,9,12,20))` produces sequential per-batch draws instead of one canonical tensor shared across examples, defeating batch invariance. Remove the public `shape` argument or reject anything other than `LATENT_SHAPE`; callers should stack keyed draws and broadcast the canonical global draw. Add explicit keyed batch-assembly and global-broadcast equality tests.

2. **MINOR** — [test_null_adapter_noise.py:29](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_noise.py:29)  
   Every golden name is ASCII, so replacing UTF-8 encoding with ASCII would leave the suite green despite violating the exact derivation. Add a hardcoded golden for a non-ASCII manifest name, preferably including both raw key words and the noise head.

3. **MINOR** — [test_null_adapter_embed_tokens.py:93](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/tests/worklogs_yixun/test_null_adapter_embed_tokens.py:93)  
   The claimed “bitwise” bf16 assertion uses numeric `==`; this does not distinguish `+0` from `-0` and handles NaNs unlike a bit comparison. Compare bf16 payloads via `jax.lax.bitcast_convert_type(..., jnp.uint16)` or an equivalent raw-byte assertion, using edge values that exercise signed zero.

4. **MINOR** — [null_inversion_wan.py:90](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_04_null_adapter/src/maxdiffusion/models/wan/null_inversion_wan.py:90)  
   The fingerprint documentation promises little-endian float32, but `dtype=np.float32` means native-endian. Canonicalize explicitly to `np.dtype("<f4")` after the host transfer and hash the contiguous C-order bytes, so artifacts remain stable across host architectures.

Final verdict: **REQUEST-REVISION** — the implementation is otherwise faithful and strongly tested, but the public arbitrary-shape noise API violates the fixed, batch-invariant convention on which downstream artifacts depend.

---

## Strengthening record (Coder, same round; all findings addressed, none rejected)

1. **MAJOR (shape argument) — FIXED.** Public `shape` removed from `keyed_noise`/`global_noise`; exactly one float32 `LATENT_SHAPE` draw per (name, k); assembly rules documented (stack keyed draws; broadcast the global draw; never batch-shaped draws). New tests: signature pin (`TypeError` on a shape arg), keyed batch = stacked per-name draws (permutation-equivariant), global batch = one broadcast tensor with an explicit assertion that a batch-shaped `jax.random.normal` draw would differ (the hazard the reviewer named). Mutants M9/M10a/M10b (reintroduced shape; batch-shaped global; batch-shaped keyed) all killed.
2. **MINOR (ASCII-only goldens) — FIXED.** Non-ASCII golden `"droid_ép_000七/w0"` added with raw key words (w0, w1) = (3608796426, 2981090583), post-fold threefry key data for k=0/1, and noise heads — all from the independent spec transcription. Mutant M11 (`encode("ascii","ignore")`) killed only by these tests, confirming the reviewer's gap; the six original goldens are byte-for-byte unchanged.
3. **MINOR (numeric == not bitwise) — FIXED.** Comparisons now via `jax.lax.bitcast_convert_type` (bf16→uint16, f32→uint32); edge-value fixture plants 0.0/−0.0/±inf/NaN/1e-45 in replaced and untouched rows; guard-the-guard test shows numeric == passes a `−0.0→+0.0` mutant that the bitwise form kills (M12), and the bitwise form correctly passes an identity copy containing NaN.
4. **MINOR (endianness) — FIXED.** Fingerprint hashes `astype("<f4")` contiguous C-order bytes; docstring updated; big-endian-input test added; mutant M13 (dropped canonicalization) killed. **Golden digest unchanged** (`e2c0a715…`) — the fix changes the cross-host guarantee, not the artifact value on this little-endian host.

Suite: **45 passed in 2.74s** (up from 34; `-W error::DeprecationWarning` clean); ruff/py_compile/`git diff --check` clean. Red evidence: the signature test failed against the reviewed code (`['name','k','shape'] != ['name','k']`); the other tests pin already-correct behavior and are justified by their killed mutants. Size note: 262 executable LOC (+67, all review-mandated tests) — Planner accepts the overage as review-driven. One forward-looking pin: the non-ASCII key-data test asserts `jax.random.key_data` (legacy threefry layout); a future JAX PRNG default change fails there first and loudly — desired, since it would invalidate every cached target.

Behavior changed beyond the findings: none (the shape-argument removal IS finding 1). Round R1 closed; committed with this record.
