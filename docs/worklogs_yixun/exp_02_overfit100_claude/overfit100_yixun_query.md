# exp_02 `overfit100` — Driving queries

Source: Yixun, relaying/echoing Lihan's critique of exp_01. Append each new query verbatim.

## Query 1 — 2026-07-28 (experiment definition + critique of exp_01)

**Verbatim:**

> I need you to run the exp_02_overfit experiment, because previous your exp_01 is not the correct way to do overfit: I have some questions:
>
> Training on the full DROID is not the standard "overfitting" task - it is a big dataset and can be hard to fit. I was thinking about maybe fitting 100 trajectories first
> If you are not conditioned on text, nor on actions, then the model is essentially "unconditioned", so it has to recite what the future frames look like only conditioned on the initial frame without knowing why the initial frame will change to future frames
> But it seems that the model is learning something, even though the prediction is still a little bit off. I think this is a good sign that the code is correct, but we still need to find a case where the model can perfectly reconstruct the training samples, maybe by overfitting to 100 trajectories and evaluating on them. So WHAT I NEED YOU TO DO NOW IS THIS: use a radom selected, say, 100 successful trajectories from DROID dataset
> use their language instruction as the text prompt condition, and with the first frame latent and noise latent to be input into the wan2.2-5B DiT model
> fine tune the full weights of wan2.2. for the 3 language instructions from droid dataset, you should do it randomly selection, and filter empty instructions

**Critique of exp_01 accepted (recorded as the motivation):**
- exp_01 trained on the FULL DROID train split (1,440,554 windows, 3.55 passes) — that is a generalization-scale task, not a standard overfit/memorization task. A model failing to memorize there says little.
- exp_01 conditioned on NEITHER text NOR actions ⇒ effectively **unconditioned** generation from a single first frame: the model had no information about *why* the scene evolves, so exact reconstruction was under-determined (many futures are consistent with one frame). Its 0.79 train-clip SSIM is therefore a plausible ceiling for that setup, not evidence about pipeline capacity.
- Yixun's read: the model "is learning something… a good sign that the code is correct", but the decisive experiment is a case where the model can **perfectly reconstruct training samples**.

**exp_02 specification (as requested):**
1. **Data:** ~**100 randomly selected SUCCESSFUL DROID trajectories** (random selection; **filter out empty language instructions**).
2. **Conditioning:** the trajectory's **language instruction as the text prompt** (T5 context, per example — NOT the null prompt exp_01 used), plus the **first-frame latent**; noise latent as the diffusion input.
3. **Model/training:** Wan2.2 TI2V **5B DiT, FULL weight fine-tune**.
4. **Goal/success:** overfit these 100 trajectories and evaluate ON them — the target is (near-)perfect reconstruction of training samples, which would establish that the pipeline + conditioning can fit when the task is determined.
5. **"3 language instructions" note:** DROID episodes carry up to three natural-language annotations; select randomly among the non-empty ones (recorded as the interpretation below; to be confirmed against the data).

**Open questions to resolve in the plan phase (data-availability driven):**
- Where do the language instructions live for our cached windows? The existing TFRecords (`droid_wan_side_adapter`) store `z_i0`, `z_video`, `actions`, `ordinal`, `name`, `meta_json` — **no text**. Does `meta_json` carry the instruction, or must it be joined from the raw DROID source (Della cluster) / re-cached?
- Per-example T5 embeddings: precompute into a new cached dataset vs compute on the fly (100 trajectories is small enough that either may work).
- Trajectory→window mapping: 100 trajectories = how many windows? Do we train on ALL windows of those trajectories, or a fixed subset?
- Success criterion: "perfect reconstruction" needs a concrete metric threshold (rollout SSIM / latent MSE on the memorized set) + a step budget.

## Query 2 — 2026-07-28 (design decisions, from Yixun)

**Verbatim:** "use one view per episode (the first one (exterior view), which has the index 0), and lr 1e-5"

**Decisions locked:** (1) **one camera view per episode — view index 0** (the first exterior view; `latent_videos/train/<ep>/0.pt`); (2) **learning rate 1e-5** (same as exp_01's full-FT recipe — deliberate memorization is pursued via dose/epochs, not a hotter LR).

## Query 3 — 2026-07-28 (data-path decision, from Yixun)

**Verbatim:** "A'"

**Decision locked:** data path **A′ — re-encode the 100 selected episodes' view-0 MP4s with the Wan VAE** (from `gs://v6_east1d/datasets/droid_ctrl_world_aligned/videos/train/<ep>/0.mp4`), rather than mining the existing 334 GB Wan-latent cache. Chosen after the plan review + probes showed the aligned `latent_videos/*.pt` are Ctrl-World-space latents `(T, 4, 24, 40)` — unusable for Wan — while the MP4s turn out to be already at the exact cache geometry **320×192 @ 5 fps** with frame counts matching the annotations (ep0: 88, ep1: 128). Session model at this decision: Fable 5 (xhigh) — Planner tier restored per SOP.

## Query 4 — 2026-07-28 (plan approval + conditional launch grant, from Yixun)

**Verbatim:** "Approve plan + dataset build + smoke conditional on dual sign-off"

**Interpretation:** (1) **Plan v4 is approved** — the SOP plan gate is passed; cycles A–D may proceed. (2) The first two TPU jobs — the **v6e-8 dataset build** (cycle B, rung 4: 2-episode probe build then full build) and the **S1 smoke** (v6e-8, GBS 32, ~20 steps, storage-light) — are **pre-approved conditional on dual sign-off**, exp_01 Query-9 style: (a) Codex — every cycle review feeding that job is closed with all findings resolved; (b) Fable/Planner — final pre-launch verification (suite green, ladder rungs recorded, commit pushed, package written to `_command.md`/`_worklog.md` at launch time). Infra-failure resubmits covered by standing policy; any code/config change after sign-off voids the grant for the changed job. (3) **S2 (10-episode gate) and S3 (100-episode run) remain separate future approvals.**

## Query 5 — 2026-07-29 (dataset stewardship, from Yixun)

**Verbatim:** "I ask catherine to maintain the dataset, don't worry"

**Interpretation:** Catherine will maintain `droid_ctrl_world_aligned` (no deletion/reprocess-in-place of the objects exp_02 pins). Consequence: the planned re-point-manifest-to-snapshot mini-round is **cancelled**; the manifest keeps the original URIs. The verified snapshot (`exp02_overfit100/source_snapshot/`, 200/200 md5-verified) is retained as free insurance — if the originals ever drift, the fingerprint gates will catch it and the snapshot enables a fast re-point.

## Query 6 — 2026-07-30 (S2 pre-approval, from Yixun)

**Verbatim:** "Pre-approve S2 conditional on S1 pass"

**Interpretation:** the **S2 10-episode gate run** (v6e-8, `train10`, 2,500 steps, GBS 32, LR 1e-5, warmup 250, checkpoints [250,500,1000,2500] retained, per plan D10) **plus its D11 gate evaluations** (S2 eval passes: 3 seeds × correct mode at every checkpoint on the 10 canonical windows; null/shuffled ablations at step 2500; `eval_pass_role=s2_gate`) are **pre-approved conditional on the S1 smoke passing its acceptance criteria log-verified** (not merely queue-SUCCEEDED): deployed-code COMMIT relay, preflight order (dataset integrity → pinned snapshot → pipeline), context table [10,512,4096] built + audited, 20/20 steps finite loss no OOM/NaN, no checkpoints written, both workers exit 0. Infra-failure resubmits covered by standing policy; any code/config change voids the grant for the changed job. **S3 (100-episode run) remains a separate future approval**, to be requested with the S2 gate results + the S3 eval-cost extrapolation from S2 timings.
