# exp_06 F7+F7b `manifest-identity` — Codex code review

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, 2026-08-13. Verdict: **REQUEST-REVISION — 2 BLOCKER + 1 MINOR**; the manifest narrowing itself RATIFIED ("otherwise coherent; no stray executable code_sha authorization gate remains"). B1: compiler/runtime policy (LIBTPU_INIT_ARGS, XLA mem fraction, JAX_PLATFORMS) outside the binding — executed: materially different runtime args, binding_equal True ("falsifies 'a code change costs the ladder'"). B2: the harness's FOURTH false refusal — F7's publish_attempt signature change broke P3-5 (TypeError scored REFUSED; 91 REFUSED did not mean 91 executed attacks). MINOR: launcher resume report claims "at this SHA" post-narrowing. Closed as round F7c.

# REQUEST-REVISION

Two blockers and one minor prevent tonight’s ceremony.

1. **BLOCKER — compiler/runtime policy is outside the binding.**  
   The manifest hashes only Python under `src/maxdiffusion` ([pos_rollout_fit_probe.py:591](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:591)), while the launcher controls measurement-relevant TPU/XLA policy through `JAX_PLATFORMS`, `XLA_PYTHON_CLIENT_MEM_FRACTION`, and especially `LIBTPU_INIT_ARGS` ([train_wan_pos_rollout.sh:79](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:79), [line 97](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:97)). The dirty-tree check likewise considers only non-test `.py` files ([pos_rollout_fit_probe.py:633](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pos_rollout_fit_probe.py:633)).

   I executed two context derivations with identical source/config/devices but materially different `LIBTPU_INIT_ARGS` and memory fractions. Result:

   ```text
   full_equal True
   binding_equal True
   differences ()
   ```

   Consequently, a committed launcher change—or a same-label hand-edit/override—can change compilation or available memory while adoption, checkpoint resume, and M2 authorization all see the same binding. This falsifies “a code change costs the ladder.” Bind the effective compiler/runtime policy and relevant launcher content locally, then exercise the dangerous direction through adoption, publication/resume, and the gate. YAML dtype, seed, support policy, and other declared config values are otherwise covered by the denylist-based recipe fingerprint.

2. **BLOCKER — the 92-probe battery contains another false refusal.**  
   `P3-5` calls `publish_attempt` without the newly required `binding_digest` at [reviewer_attacks.py:691](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:691). It dies before reaching its local execution guard; its selector calls also still use the removed `context_digest` argument at [line 720](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:720). The generic reporter converts that exception to `REFUSED` at [line 902](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/docs/worklogs_yixun/exp_06_rollout_adapter_claude/harness/reviewer_attacks.py:902).

   The supplied log confirms:

   ```text
   P3-5 ... REFUSED (TypeError): publish_attempt() missing ... 'binding_digest'
   ```

   Thus 91 REFUSED does not mean 91 executed attacks. Repair P3-5, make publication/signature drift explicitly score `SUCCEEDED: THE PROBE DID NOT RUN`, rerun the battery, and update the evidence record. F5-1 and the new F7-1/F7-2 probes themselves do observe the real adoption/gate outcomes correctly.

3. **MINOR — resume preflight falsely says candidates are “at this SHA.”**  
   `describe_resume_candidates` no longer filters on `code_sha` ([wan_pos_rollout_trainer.py:275](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/trainers/wan_pos_rollout_trainer.py:275)), but the launcher still reports “for this arm at this SHA” ([train_wan_pos_rollout.sh:405](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/bash_scripts/train_wan_pos_rollout.sh:405)). It remains report-only and does not authorize, so this is not a safety failure, but the report is factually wrong. Remove the SHA claim or print each candidate’s recorded label.

The manifest-specific narrowing is otherwise coherent: same-manifest label drift proceeds, different manifests refuse in adoption/resume/gate, v4 is rejected before field validation, all production consumers use the sole v5 loader, and no stray executable `code_sha` authorization gate remains. `git diff --check`, `bash -n`, and AST parsing passed. Pytest could not be independently started because this read-only sandbox provides no writable temporary directory.
