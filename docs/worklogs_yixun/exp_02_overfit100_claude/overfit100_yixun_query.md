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
