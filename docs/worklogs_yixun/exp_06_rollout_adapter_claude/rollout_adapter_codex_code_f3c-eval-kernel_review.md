# exp_06 F3c — Codex re-review (of the eval-kernel round)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, 2026-08-10 (third focused pass; executed its own verification — production kernel bitwise-equal to the eager boundary at batch 1/2 x adapter on/off, 0 closure array bytes, cache once per (shape, adapter_enabled), 8-device oracle passed). Verdict: **REQUEST-REVISION with NO BLOCKER and NO MAJOR** — two test-only MINORs (fixture sharding contract synthetic vs loader's, 18/42 placement-equivalent; the BITWISE parity test deliberately eager so the new production JIT boundary lacks committed bitwise coverage). Closed as round F3d.

## Verdict: REQUEST-REVISION

No BLOCKER or MAJOR found. Two MINOR findings remain.

1. **MINOR — the sharding fixture still does not reproduce `wan_pipeline`’s leafwise placements.**

   [test_pos_rollout_trainer_wiring.py:1173](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1173) chooses `P("fsdp")` solely when a leaf’s leading dimension divides eight. Production instead derives specs from NNX annotations and logical-axis rules at [wan_pipeline.py:149](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pipelines/wan/wan_pipeline.py:149).

   My eight-device measurement confirms 39/42 fixture leaves are sharded, but only **18/42 placements are equivalent to the production logical mapper**. The fixture assigns `fsdp` to 39 leaves; production specs involve `fsdp` on 20. Biases and kernels are frequently sharded on the wrong dimension.

   `checks["frozen"]` is now asserted and will catch a missing/replicated index 1, but it is still checked against a synthetic contract rather than the loader’s contract. Commit the fake state using `nnx.get_partition_spec` plus `logical_to_mesh_sharding`, as production does.

2. **MINOR — the named bitwise parity test is genuinely bitwise, but deliberately not jitted.**

   `_kernel_from` explicitly returns an eager kernel at [test_pos_rollout_eval_end_to_end.py:152](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_eval_end_to_end.py:152), with its docstring stating it is “Deliberately NOT jitted.” `_backend` installs that kernel, and the BITWISE test at [test_pos_rollout_eval_end_to_end.py:668](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_eval_end_to_end.py:668) compares through it. `np.array_equal` is exact, but this is not regression coverage for the new production JIT boundary.

   I independently compared the production `build_rollout_kernel` against the old eager boundary: bf16 results were bitwise equal for batch sizes 1 and 2, adapter enabled and disabled. Thus I found no numerical defect, but the claimed committed test coverage is absent.

Verified:

- `DeviceBackend.score` always routes through `self.kernel`; no eager production branch survives.
- Both adapter variants trace without backbone constants.
- No bound-velocity seam remains on `DeviceBackend`.
- The guard and loader use the same production builder functions, although the guard constructs a separate one-step kernel instance.
- The real loader-built kernel’s underlying closure reaches 0 array bytes; donation is empty.
- Cache growth was exactly once per `(batch shape, adapter_enabled)` variant and unchanged on repeated calls.
- The eight-device oracle otherwise passed completely.
- Accepted supplied evidence: suite 2134/0, battery 80 REFUSED / 0 SUCCEEDED, restores 21/21.
