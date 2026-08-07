# rollout_adapter — Codex code review: T2 `loss-kernels`

Reviewer: OpenAI Codex `gpt-5.6-sol` xhigh. Verdict: REQUEST-REVISION — **1 MAJOR, NO functional implementation defect**: the N13 imported-binding proof is still evadable by rebinding (`rollout_support = _draw_window` passes all three structural locks). Everything else independently CONFIRMED: the double equivalence on the real caller domain (incl. F=1 and sliced batches), all five extraction hashes vs the pin, the telescoping oracle against the actual kernel (max normalized rel. error **3.70e-6** while raw MSE varied **185.8×** ⇒ rtol=2e-3 is conservative, not fitted), the state-coupled FD (carry-cut ⇒ ≈48%/72% error at k=2/k=4), and every contract pin. Strengthen dispatched same cycle (N13 rebinding lock + a Planner-added float32-precondition pin, since the reviewer's own bf16-divergence observation becomes a live hazard when T3b becomes a new caller).

## Reviewer output (verbatim)

The implementation is functionally faithful; the focused suite passes `41/41`. One commit-blocking test-contract gap remains.

## Findings

- **MAJOR** — [test_pos_rollout_losses.py:553](/Users/yixunhu/Home/maxdiffusion-worktrees/claude-exp_06_rollout_adapter/src/maxdiffusion/tests/worklogs_yixun/test_pos_rollout_losses.py:553): N13 does not prove that `rollout_support` still resolves to the imported primitive. A later `rollout_support = _draw_window` rebinding passes all three structural locks: the import still exists, the `(start, end)` call still spells `rollout_support`, `_draw_window` lacks “support,” and it can avoid local randomness. Finite value checks can likewise be spoofed. Add:

  - Runtime identity: `losses.rollout_support is support.rollout_support`.
  - AST rejection of every `Store`/`Del`, argument, or second import binding named `rollout_support`.
  - Retain the existing call-site assertion.

No functional implementation defect was found.

## Verdicts

- **(a) Double equivalence:** Confirmed on the real caller domain. Re-homed bodies exactly match the pin. The inline mask tuple is identical because `f_lat/h_lat/w_lat` and `b` come from `z_video_f32`, while `v_target.shape == z_video_f32.shape`. `F=1` returns zero identically, and sliced batches remain bitwise equal. The noisy-latent paths diverge for bf16/fp16 inputs, but those are outside both branches’ actual contract: both prologues cast video/image to float32, create float32 noise, and use float32 sigmas before this math.

- **(b) Extraction fidelity:** Valid-input dataflow matches the pin. All five recorded source hashes independently match `2ef9b8a`. `ctx.b` was derived from the same `z_video.shape[0]`. Other harmless textual deltas are caller-supplied `velocity_fn`, `int(k_b)` where the pin already coerced `k_b` to `int`, and removal of the explicit `global_step is None` check; explicit `None` now fails downstream instead.

- **(c) Oracle strength:** The telescoping derivation is true of the actual Euler-and-pin kernel. Across every legal k=2 support, maximum normalized relative error was `3.70e-6`, raw-form error `3.75e-6`, and raw MSE varied `185.8×`. Thus `rtol=2e-3` is conservative rather than fitted, though much looser than observed error. The constant-offset oracle—not the zero-at-optimum oracle—proves normalization. The mask and per-example oracles discriminate the stated alternatives correctly.

- **(d) Contract pins:** Current source contains no stop-gradient, config access, baked-in forward, or duplicate inherited helpers. The state-coupled FD is genuine: a carry-cut mutation produced approximately `48%` and `72%` gradient errors at k=2 and k=4, far beyond the `2%` tolerance. k=1 is appropriately only a base case. CFG branch-internal paths remain T3a’s obligation.

- **(e) N13:** Still evadable through name rebinding as described above; strengthen before commit.

**REQUEST-REVISION — the kernels and equivalences are correct, but N13’s claimed imported-binding proof needs the small identity/rebinding lock before this extraction round is commit-ready.**
