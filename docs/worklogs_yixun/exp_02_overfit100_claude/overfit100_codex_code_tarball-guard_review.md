# Code review: exp_02 overfit100 — mini-round tarball-guard

Reviewer: OpenAI Codex gpt-5.6-sol (xhigh), 2026-07-29

## Verdict

REQUEST-REVISION. The Planner-side clean/pushed check is an acceptable trust boundary for a worker without `.git`, and the normal tarball path fails closed on an absent or malformed `COMMIT`. However, ambient Git variables can bypass the real-worktree dirty check, so launch sign-off is not yet re-established.

## Findings

1. **T1 — MAJOR — Ambient Git repository-selection variables can turn `COMMIT` into a bypass inside a real worktree.** Both `_git()` and `is_git_worktree()` inherit `GIT_DIR`, `GIT_WORK_TREE`, `GIT_COMMON_DIR`, and related variables (`build_overfit100_manifest.py:407-428`). For example, with `GIT_DIR=/definitely/not/a/repository`, the exact `git -C <real-root> rev-parse --is-inside-work-tree` probe exits 128; `is_git_worktree()` consequently returns false and `assert_implementation_committed()` accepts a valid-looking `COMMIT` without checking a dirty tree (`:481-486`). `GIT_WORK_TREE` can instead redirect the status check to another tree. The new tests exercise only ordinary repositories/plain directories and therefore miss this route. **Concrete change:** use a sanitized Git environment for discovery and every `_git()` call, removing repository/worktree/index selector variables; make an existing `.git` marker plus failed discovery fatal rather than deployed mode; apply equivalent isolation in the Bash probe. Add dirty-repository regressions under poisoned `GIT_DIR`/`GIT_WORK_TREE`, plus linked-worktree and submodule coverage.

2. **T2 — MODERATE — The missing-reference fallback does not establish that the accepted manifest came from the tarball.** When `root / DEFAULT_MANIFEST` is absent, `assert_manifest_matches_committed()` hashes and accepts any readable `path`, including one outside `root`, while logging it as the “shipped manifest” (`build_overfit100_dataset.py:1146-1159`). That weakens B6 beyond the stated tarball trust boundary. **Concrete change:** require the resolved consumed path to remain beneath the deployed-code root, or preferably compare it with a launcher-supplied expected manifest SHA-256; add a rejection test for an external path.

3. **T3 — MINOR — The Bash prefetch guard does not enforce exactly one 40-hex value.** `grep -Eq '^[0-9a-f]{40}$'` succeeds when any input line matches, so `COMMIT="<40 hex>\njunk"` passes the pre-prefetch check and is rejected only later by Python, after prefetch (`build_overfit100_dataset.sh:60-65`). **Concrete change:** validate both length and allowed characters with Bash string/pattern checks, and add a shell-level test covering export and fail-fast ordering.

---

## Strengthening record (Coder: Claude Opus 5, 2026-07-29)

Suite after strengthening: **638 passed, 2 skipped** (597+2 before this round). All three
findings implemented; none rejected. The T1 bypass was reproduced first on the real worktree
(`GIT_DIR=/definitely/not/a/repo` + `COMMIT=<40 hex>` → `is_git_worktree: False`, guard returned
the env sha with no dirty check) and re-checked after the fix (`is_git_worktree: True`, guard
refuses the dirty tree).

1. **T1 — FIXED (sanitized git environment + fatal marker).**
   `GIT_ENV_STRIP` names the repository/worktree/index/object selectors swept from
   `git help environment`: `GIT_DIR`, `GIT_WORK_TREE`, `GIT_COMMON_DIR`, `GIT_INDEX_FILE`,
   `GIT_INDEX_VERSION`, `GIT_OBJECT_DIRECTORY`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`,
   `GIT_CEILING_DIRECTORIES`, `GIT_DISCOVERY_ACROSS_FILESYSTEM`, `GIT_NAMESPACE`, `GIT_PREFIX`,
   `GIT_TOPLEVEL`. `sanitized_git_env()` strips them and is used by **both** `_git()` and the
   `is_git_worktree()` discovery probe, so neither discovery nor the dirty check can be
   redirected. Discovery hardening: if `<root>/.git` exists (directory **or** file — linked
   worktrees and submodules use a file) but sanitized discovery still fails, `is_git_worktree`
   now raises `DirtyImplementationError` instead of returning False; "we could not tell" can no
   longer resolve to the weaker deployed-code contract. The bash arm gained the same isolation
   via a `GIT_ISOLATED` `env -u …` prefix used for both its `rev-parse HEAD` fallback and its
   worktree probe.
   Regression tests (all with real repositories built in tmp dirs): dirty repo + poisoned
   `GIT_DIR` → still refuses; dirty repo + `GIT_WORK_TREE` pointing at a clean tree → still
   refuses; dirty repo + foreign `GIT_INDEX_FILE` → still refuses; clean repo under poison →
   returns the REAL HEAD, not the env sha; **every** variable in `GIT_ENV_STRIP` parametrized
   against `is_git_worktree`; linked `git worktree add` checkout detected (plus a test pinning
   that *this* repository's `.git` is a FILE — our actual setup); plain dir + `GIT_DIR` pointing
   at a real repo → deployed mode that still fails closed without `COMMIT` (never borrows the
   poisoned repo's HEAD); broken `.git` file and non-repository `.git` directory → fatal.
2. **T2 — FIXED (fallback is rooted in the deployed tree).**
   In deployed-code mode with no shipped reference, the consumed manifest path must now
   `Path.resolve()` beneath the resolved deployed-code root, else `BuildError("… resolves
   outside the deployed-code root …")`. `resolve()` also collapses symlinks, so a link planted
   inside the tarball that points outside is rejected too. Tests cover the external path, the
   inside-the-root acceptance, and the escaping symlink. (The stronger variant — a
   launcher-supplied expected manifest sha256 — is left to the Planner as a launch-protocol
   change; the rooting closes the hole this round without adding a new launch parameter.)
3. **T3 — FIXED (exactly one 40-hex token, in pure bash).**
   The `grep -Eq` line-oriented check is replaced by
   `[[ ${#COMMIT} -ne 40 || ! ${COMMIT} =~ ^[0-9a-f]{40}$ || ${COMMIT} == *[[:space:]]* ]]` —
   length, character class, and an explicit whitespace/newline rejection. The block is delimited
   by `# >>> launch-commit guard` / `# <<< launch-commit guard` sentinels and is executed
   **verbatim from the shipped script** by `test_overfit100_gates.py`: one clean sha is accepted
   and proven EXPORTED (a child `printenv` sees it), while `unknown`, empty, 39/41 chars,
   uppercase, non-hex, leading/trailing space, and three multi-line variants
   (`<40hex>\njunk`, `junk\n<40hex>`, two shas) all exit 1 with `FATAL` before any prefetch;
   inside a worktree the block stands down, and a poisoned `GIT_DIR` does not change that.
