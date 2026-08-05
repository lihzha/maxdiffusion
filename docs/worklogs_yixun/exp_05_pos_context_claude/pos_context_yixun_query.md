# exp_05 `pos_context` — Yixun's driving queries

## Query 1 — 2026-08-04T15:18Z (defining directive, given while approving exp_04)

**Verbatim:**

> Yes to all, but what I actually want is to get text token from inverse DDIM, and use pre_context structure to learn the text token (the positive text embedding and use loss function to constrain adapter to do that). You can treat what I am proposing here as exp_05, please design the plan for exp_05 and parallel run exp_04 and exp_05.

**Summary:** Extract per-sampling-step **positive text embeddings** ("text tokens") by DDIM inversion — the `positive_inversion` role-swap in `third_party/Wan2.2/scripts/embedding_search.py` (`optimize_positive_embeddings` / `regenerate_with_positive_embeds`): invert the target video latent at w=1, then per step optimize the **conditional** context C_t at deployment CFG while the null branch stays frozen at T5(""). Then train the existing **pre_context adapter structure** (`wan_pre_context_adapter_forward`: first-block features of z_t + action tokens → predicted context `[B, 8, 4096]` that replaces the T5 context) to **predict those inversion-derived text tokens, constrained by a regression loss** to the C*_t targets. Run exp_05 in parallel with exp_04 (the null-embedding analog).

**User's assumption / hypothesis:** The inversion-derived per-step positive embeddings are a supervised, constructive target in exactly the space the pre_context head already outputs — regressing the head onto them ("use loss function to constrain adapter to do that") should teach it the conditioning signal that provably reconstructs the video, instead of asking the one-step denoising loss to discover it indirectly.

**Why the experiment needs to run:** The pre_context adapter trained on one-step denoising reached only rollout SSIM 0.2946. exp_04 tests the null slot; exp_05 tests the positive slot with the *existing* pre_context architecture — a direct answer to whether the pre_context structure's capacity was the bottleneck or its training signal was. The PyTorch fork's positive line (TrajectoryAdaptor, L_pos=1, μ+rank-1) underfit and hit the noise-basin problem; exp_05 differs in: timestep-aware z_t-conditioned head (the pre_context structure sees the current latent), L_pos=8, teacher-forced regression on cached optimization-trajectory states, and the exp_04-shared de-risking arms (basin probes, fixed-noise variant, matched controls).

**Planner scope reading (to be confirmed by the plan):** "text token from inverse DDIM" = per-step C*_t from positive inversion (warm-started from T5("") — the dataset has no captions); "pre_context structure" = the existing `NNXPreContextFeatureContextHead` + action-encoder stack, reused as-is where possible; "loss function to constrain" = MSE regression of the head's predicted context onto C*_t (teacher-forced on cached trajectory states), with a combined regression+denoising variant as an optional ablation.

## Query 2 — 2026-08-04 (plan approval)

**Verbatim:**

> Yes to all for exp_05, continue

**Grants recorded (per plan v3 §11's decision points 1–4):**
1. **Plan v3 APPROVED.** L_pos = 8 approved ({1, 8} ablation diagnostic-only).
2. Primary training = pure teacher-forced regression approved; the combined regression+denoising arm stays deferred.
3. **K1 approved conditional on**: merge-1 (exp_04 R9 boundary) done + P0' tests green + exp_05 parity audit clean + J0 manifests published — conditional grant per announcement 02; no re-ask when the conditions are met. K2–K4 remain gated and will be asked at their gates.
4. Pilot scope acknowledged (TEST-64; DROID-wide claims deferred).

## Grant (2026-08-05T20:26Z) — K1 launch + continuation

Context: status report stating K1's gate set fully met (merge-1 ✓, P0' 1139-green ✓, J0 ✓, parity audit ✓, S10a committed `102ae84` ✓) with the four-phase runbook (smoke → adequacy → capacity dev64 → capacity trainfit16) and predeclared acceptance criteria in the worklog's K1-package entry, submission handed over per issue #10.

**Yixun, verbatim:** "Yes to all, continue with both experiments"

Planner reading: (1) K1 launch approved under the standing conditional grant, now unconditional — the session's own `tpu create` remains classifier-blocked (issue #10, re-confirmed 20:28Z), so the submission runs via Yixun's `!`; (2) the S6+ trainer rounds proceed in parallel with K1; (3) exp_04's J1-2b supplement is approved to launch AFTER J1-2 completes (it consumes J1-2's adequacy artifact) — same `!` handover.
