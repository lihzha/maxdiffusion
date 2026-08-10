# exp_06 F3b — Codex re-review (of the F3 fix round)

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh, 2026-08-10 (focused pass on the F3b delta; accepted the supplied suite/battery evidence — read-only env had no writable tmp for red re-execution). Verdict: **REQUEST-REVISION** — MAJOR-1/MAJOR-3 verified CLOSED; the evaluator-boundary MAJOR survives in refined form (guard proves a hypothetical kernel, not DeviceBackend.score's actual boundary; bind-then-jit spelling still exists; velocity_for's closure retains the registered FrozenBackbone via graphdef); one MINOR (step sharding check computed but never asserted; fixture forces replicated instead of production leafwise shardings). The prescribed remedy is quoted verbatim below.

## Verdict: REQUEST-REVISION

No BLOCKER in the trainer/M1 path, but one prior MAJOR is not structurally closed.

### MAJOR

1. The evaluator guard still crosses a safer JIT boundary than production.

Production obtains the backbone from `self.frozen_state` at [eval_wan_pos_rollout.py:1393](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1393), and `rollout_prediction` is not jitted. Moreover, `make_velocity(frozen_state)` executes while constructing the arguments to `cfg_rollout`, before its staged loop begins, at [eval_wan_pos_rollout.py:750](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:750).

The guard instead defines an outer-jitted function whose second explicit argument is `frozen_state` at [test_pos_rollout_captured_constants.py:570](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_captured_constants.py:570), then traces it with `frozen.state` at line 584. That proves a hypothetical kernel where the state is already a tracer; it does not prove `DeviceBackend.score`’s boundary.

A bind-then-jit spelling therefore remains:

```python
bound = backend.velocity_for(
    backend.params, backend.frozen_state, actions, adapter_enabled
)
jax.jit(lambda z: cfg_rollout(z, velocity_fn=bound, ...))(z)
```

The test’s own positive control demonstrates that form captures the marked backbone.

There is a second structural leak: production `velocity_for` references `frozen.graphdef` at [eval_wan_pos_rollout.py:1460](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/eval_wan_pos_rollout.py:1460), so its closure retains the entire now-pytree-registered `FrozenBackbone`, including state. A lightweight closure probe with a 4 MiB dummy state reported all 4 MiB reachable. This does not currently become a JAX literal because only `graphdef` is read, but it falsifies “no closure holds the backbone” and the guard never applies its closure detector here.

The robust closure is an actual compiled evaluator kernel taking `params` and `frozen_state` as explicit JIT arguments, invoked by `DeviceBackend.score`, plus extracting `frozen_graphdef = frozen.graphdef` before defining `velocity_for`.

### MINOR

1. The step sharding result is calculated but never asserted.

`_absolute` produces `checks["frozen"]` at [test_pos_rollout_trainer_wiring.py:1325](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1325), but the final assertion loop checks only `params`, `opt`, `batch`, and `draws` at [test_pos_rollout_trainer_wiring.py:1564](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1564). A false frozen result is silently ignored.

The fixture also forces every backbone leaf to replicated `P()` at [test_pos_rollout_trainer_wiring.py:1167](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_trainer_wiring.py:1167). Production instead derives leafwise shardings from parameter annotations and logical-axis rules at [wan_pipeline.py:148](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/pipelines/wan/wan_pipeline.py:148). Thus the final fixture still does not exercise the real sharded-backbone contract. The scorer assertion is wired correctly; the step assertion is not.

### Verified closed

- MAJOR-1: both builders freeze `frozen_state` leafwise. The narrowed rollout tripwire currently finds exactly one permitted construction, and the adversarial frozen-gradient plus unchanged branch-contrast oracles preserve clause (ii).
- MAJOR-3: `FrozenBackbone` registration exposes all state leaves. No production placement, donation, or optimizer logic relies on wrapper opacity.
- Current trainer/M1 placement still passes `frozen.state` untouched; I found no production per-call reshard introduced there.

I accepted the supplied 2134-pass and 80/80 refusal evidence. Full JAX red re-execution was unavailable because this read-only environment provides no writable temporary directory.
