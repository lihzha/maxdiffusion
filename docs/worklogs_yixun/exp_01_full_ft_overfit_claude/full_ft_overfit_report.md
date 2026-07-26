# Experiment 01: Full-Fine-Tuning Overfit Probe of Wan2.2 TI2V-5B on DROID

| Field | Value |
|---|---|
| Status | Final |
| Run | `wan-full-ft-v6e64-full-gbs256-fresh-20260719-165222` |
| Job | `20260719-165222-62b5c10e` |
| Code commit | `031228e` |
| Report date | 2026-07-26 |

## Executive Summary

This experiment asks whether the shared DROID latent, noising, loss, and rollout path can learn when the entire Wan2.2 TI2V-5B Transformer is trainable. It is a diagnostic experiment, not a proposed production model: it removes both action conditioning and the adapter, uses the first-frame latent plus a fixed empty-prompt T5 context, and fully fine-tunes approximately 5.00B Wan Transformer parameters.

The answer is positive for the tested training cohort. Training loss fell from `0.601` at step 10 to `0.194` at step 500 and was near its long-run plateau by roughly step 2,000. On the predeclared 16-clip training cohort, 25-step rollout SSIM improved from `0.1966` for the pretrained checkpoint to `0.7873` at the first measured trained checkpoint, step 5,000, and remained approximately flat through step 20,000.

Therefore, the strong hypothesis that the core pipeline cannot fit at all is disfavored. Under the predeclared decision rule, the result favors the explanation that restricted trainable leverage makes the frozen-backbone adapter problem harder. It does **not** causally prove that backbone freezing is the only or primary adapter bottleneck, nor does it distinguish DROID-domain adaptation from memorization of individual clips.

## 1. Background and Motivation

The two preceding action-conditioned approaches both kept the Wan backbone frozen:

1. **Side adapter / ControlNet-style path:** action features drive trainable residual streams injected after selected Transformer blocks.
2. **Pre-context path:** action tokens are translated into additional T5-space context tokens consumed by Wan cross-attention.

Because both approaches were difficult to fit, there were two broad explanations:

- the shared data, latent alignment, noising, loss, or rollout path might be fundamentally broken; or
- the path might be learnable, while a small action adapter has insufficient or difficult-to-optimize leverage through a frozen 5B backbone.

Experiment 01 separates these possibilities asymmetrically by giving the optimizer maximal leverage: remove the adapter and actions, unfreeze the whole Wan Transformer, and ask whether reconstruction on DROID training clips improves.

## 2. Hypotheses and Decision Rule

| Hypothesis | Prediction |
|---|---|
| **A — pipeline failure** | Even with the full Wan Transformer trainable, one-step loss does not fall substantially and fixed-cohort rollout reconstruction does not improve over the pretrained baseline. |
| **B — limited trainable leverage** | Full fine-tuning is numerically healthy, rapidly lowers the one-step loss, and substantially improves reconstruction of the fixed training cohort. |

The readout is intentionally asymmetric:

- A positive full-fine-tuning result establishes that the tested core path is learnable.
- A negative result at one compute budget would not, by itself, prove a pipeline bug; optimizer, precision, schedule, and capacity controls would still be required.

## 3. Data Configuration

Training used the cached DROID train split:

`gs://v6_east1d/datasets/droid_wan_side_adapter/train`

| Property | Value |
|---|---|
| Number of windows | 1,440,554 |
| Number of TFRecord shards | 704 |
| RGB geometry represented by each window | `height=192`, `width=320`, `num_frames=32` |
| First-frame latent `z_i0` | fp16, `[48, 1, 12, 20]` |
| Target video latent `z_video` | fp16, `[48, 9, 12, 20]` |
| Action sequence | fp32, `[32, 7]` |

The action field is present in each serialized example because this cache is shared with the adapter experiments. **Experiment 01 does not consume it in the model forward pass or loss.** The effective conditioning is only the first-frame latent and fixed null-text context.

## 4. Model and Conditioning

### 4.1 Trainable and frozen components

- Pretrained checkpoint: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- Trainable component: the complete Wan Transformer, approximately 5.00B parameters
- Adapter: none
- Action encoder or action tokens: none
- T5 text encoder: used once to construct the empty-prompt embedding, then not trained
- VAE: not trained; used for decoding rollout latents during evaluation
- Effective classifier-free guidance scale: `1.0`, so no conditional/unconditional CFG pair is used

The Transformer architecture is instantiated from the Hugging Face checkpoint configuration rather than the architecture-like placeholder values retained in the base YAML. The resolved checkpoint configuration has 30 Transformer blocks, 24 attention heads, head dimension 128, hidden size 3,072, FFN size 14,336, and latent patch size `(1, 2, 2)`.

### 4.2 What actually enters Wan

The user-facing shorthand “first image + noisy video + null text” is correct semantically, with one implementation detail: `z_i0` is not passed as a separate Transformer argument. It overwrites latent frame 0 inside the noised video tensor.

For each batch:

1. Construct a noisy target-video latent:

   $$
   z_t=(1-\sigma_t)z_{\mathrm{video}}+\sigma_t\epsilon,
   \qquad \epsilon\sim\mathcal{N}(0,I).
   $$

2. Replace its first latent frame:

   $$
   z_t[:,:,0,:,:]\leftarrow z_{i0}.
   $$

3. Call the Wan Transformer once:

   ```python
   transformer(
       hidden_states=z_t,
       timestep=timestep_2d,
       encoder_hidden_states=null_context,
   )
   ```

The effective tensors are:

| Transformer argument | Meaning |
|---|---|
| `hidden_states` | `[B, 48, 9, 12, 20]` noised video latent with frame 0 replaced by `z_i0` |
| `timestep` | per-token timestep; first-frame tokens use time 0 and future-frame tokens use sampled time `t` |
| `encoder_hidden_states` | the same fixed empty-prompt T5 embedding, broadcast to every example |
| output | `[B, 48, 9, 12, 20]` latent velocity prediction |

With patch size `(1,2,2)`, the latent becomes 540 Transformer tokens: 60 first-frame tokens and 480 future-frame tokens.

### 4.3 Is the null-text embedding fixed?

Yes. The code encodes the empty string `""`—not a one-space string `" "`—once at startup:

```python
prompt_embeds, _ = pipeline.encode_prompt(
    prompt=[""],
    negative_prompt=[""],
    ...
)
```

Within each training or validation job, this produces one `[1, 512, 4096]` T5 context tensor, which is broadcast across the batch and reused for every model call. The T5 encoder and tokenizer are then released.

There is one important nuance: the **input embedding tensor is fixed**, but the fully trainable Wan Transformer can change how it interprets that tensor because Wan's text projection and cross-attention weights are updated. Thus every generated clip receives the same text input, while generation can still vary through the first frame, initial Gaussian noise, and the learned DROID video prior.

### 4.4 Wan output versus final video

A single Wan call outputs a **latent velocity**, not an RGB video. The training target is

$$
v_{\mathrm{target}}=\epsilon-z_{\mathrm{video}}.
$$

At generation time, the sampler starts from Gaussian video latents, pins frame 0 to `z_i0`, performs 25 Euler velocity updates while re-pinning frame 0 after each step, and finally decodes the resulting latent with the VAE into an RGB video.

## 5. Training Objective

Experiment 01 uses one-step flow-matching denoising:

1. Sample one timestep per example uniformly from 25 shifted sigma levels (`shift=5`).
2. Sample fresh Gaussian noise for every example and step.
3. Form `z_t` and pin its first frame to `z_i0`.
4. Predict `v_pred` with one native Wan Transformer forward pass.
5. Minimize the masked mean-squared error:

   $$
   \mathcal{L}
   =
   \operatorname{MSE}
   \left(
     v_{\mathrm{pred}},
     \epsilon-z_{\mathrm{video}}
   \right)
   \quad \text{over latent frames }1\ldots8.
   $$

Frame 0 is excluded from the loss because it is a supplied condition rather than a prediction target. There is no action loss, adapter residual, pre-context token generation, or CFG combination in this path.

## 6. Final Training Configuration

| Parameter | Final value |
|---|---|
| Hardware | TPU v6e-64 |
| Parallelism | pure FSDP |
| Per-device batch | 4 |
| Global batch size | 256 |
| Training steps | 20,000 |
| Sample exposures | 5.12M, approximately 3.55 train-set passes |
| Optimizer | AdamW |
| Learning rate | `1e-5` |
| Adam betas / epsilon | `0.9`, `0.999`, `1e-8` |
| Weight decay | `1e-2` |
| Schedule | 5% warmup (1,000 steps), then constant |
| Global gradient-norm clipping | 1.0 |
| Weights / activations | bf16 |
| Rematerialization | `FULL` |
| Noise mode | fresh |
| Effective guide scale | 1.0 |
| Online eval | every 1,000 steps, four train-split batches of 256 |
| Checkpoint interval | every 2,500 steps; every checkpoint retained |

The generic YAML keys `guidance_scale: 5.0` and `do_classifier_free_guidance: true` are inherited but inert for this custom full-fine-tuning path. The effective value is `side_adapter_guide_scale: 1.0`, which the trainer asserts before loading the model.

## 7. Evaluation Protocol

Two complementary measurements were used.

### 7.1 One-step eval-on-train loss

Every 1,000 steps, the training objective was evaluated on four batches from the train split, using fresh timestep and noise samples. This measures the learned velocity field under the one-step objective, not end-to-end decoded video quality.

### 7.2 Fixed-cohort rollout

The rollout cohort contained 16 predeclared, evenly spaced training-set ordinals. Checkpoints `0`, `5,000`, `10,000`, `15,000`, and `20,000` were evaluated with:

- the same clips at every checkpoint;
- validation seed 0 and paired initial rollout noise;
- 25 Euler sampling steps;
- latent MSE, pixel MSE, and SSIM;
- step 0 as the untouched pretrained baseline.

The reported “ground truth” RGB videos are VAE decodes of cached `z_video`, not the original DROID RGB files. This keeps prediction and reference on the same cached-latent/decode path.

## 8. Results

### 8.1 Training dynamics

![Training loss falls rapidly and reaches a long plateau](full_ft_overfit_report_assets/training-loss.svg)

Most of the one-step loss reduction occurred in the first 500–1,000 steps:

| Step | Training loss |
|---:|---:|
| 10 | 0.6010 |
| 500 | 0.1940 |
| 1,000 | 0.1867 |
| 2,000 | 0.1829 |
| 20,000 | 0.1763 |

From approximately step 1,000 onward, logged training loss stayed mostly within `0.176–0.183`; the minimum logged window was `0.1722`. It is therefore accurate to say that the one-step objective was near its long-run plateau by roughly step 2,000. It is **not** possible to locate rollout-quality saturation at step 2,000 because no rollout was measured there.

Gradient norm remained stable at approximately `0.09–0.10`, with no NaNs, loss spikes, or preemptions. The run completed in about 4 h 40 min at 1.90 steps/s, or approximately 487 samples/s.

### 8.2 Eval-on-train loss

![Eval-on-train loss declines gradually after the early training plateau](full_ft_overfit_report_assets/eval-loss.svg)

Eval-on-train loss decreased from `0.1905` at step 1,000 to `0.1784` at step 20,000. This is a further reduction of about 6% after the early plateau, indicating slow continued improvement despite the visually flat training curve.

### 8.3 Fixed training-cohort rollout

| Checkpoint | Latent MSE ↓ | Pixel MSE ↓ | SSIM ↑ | Provenance |
|---:|---:|---:|---:|---|
| 0, pretrained | 3.4794 | 0.19922 | 0.1966 | official 16/16 |
| 5,000 | 0.2536 | 0.01912 | 0.7873 | official 16/16 |
| 10,000 | 0.2573 | 0.01946 | 0.7851 | official 16/16 |
| 15,000 | 0.2537 | 0.01926 | 0.7876 | official 16/16 |
| 20,000 | 0.2495 | 0.01896 | 0.7875 | official 16/16 |

Relative to the pretrained baseline, the first measured trained checkpoint at step 5,000 achieved:

- 92.7% lower latent MSE;
- 90.4% lower pixel MSE;
- an absolute SSIM increase of 0.5907, from 0.1966 to 0.7873.

All three metrics remained in a narrow band from step 5,000 through step 20,000. The defensible conclusion is therefore: **the earliest measured trained checkpoint already contained essentially all observed rollout reconstruction gain.** The experiment does not determine whether this saturation happened at step 1,000, 2,000, or later within the unmeasured interval before step 5,000.

## 9. Interpretation

### 9.1 What the experiment establishes

- The no-action, guide-1, full-fine-tuning path is trainable.
- The shared cached-latent, frame-pin, flow-matching, and rollout machinery can produce a large reconstruction improvement on a fixed DROID training cohort.
- The strong form of Hypothesis A—“the pipeline cannot fit even with all Wan parameters trainable”—is disfavored.
- Under the predeclared decision rule, the evidence favors continuing to investigate methods with greater trainable leverage rather than prioritizing catastrophic pipeline debugging.

### 9.2 What the experiment does not establish

- It does not prove that the entire pipeline is bug-free.
- It does not causally isolate a frozen backbone as the sole or dominant adapter bottleneck.
- It does not show held-out DROID generalization.
- It does not prove that the model memorized the full 1.44M-window dataset.
- It does not show that video quality converged at step 2,000.
- It does not support a direct numeric quality ranking against the prior adapter runs.

The adapter and full-fine-tuning experiments differ simultaneously in trainable parameterization, action conditioning, guide objective, and evaluation split. Their metrics are therefore useful as context, not as an apples-to-apples causal comparison.

## 10. Limitations

- One training run and one training seed
- One paired rollout-noise protocol
- Only 16 deterministic training clips in the rollout cohort
- No uncertainty intervals or cohort-representativeness analysis
- No matched held-out rollout cohort
- No separation of DROID-domain adaptation from per-clip memorization
- No action conditioning
- Guide-1 full-fine-tuning objective versus guide-5 adapter objective
- bf16 Adam moments following parameter dtype
- Approximately 3.55 train-set passes
- No completed human qualitative review of the generated comparison videos

## 11. Conclusion and Next Step

Experiment 01 passes its intended diagnostic: when the full Wan Transformer is trainable, the tested DROID pipeline rapidly lowers its denoising loss and substantially improves fixed training-cohort reconstruction. This makes a total inability of the core path to learn unlikely.

The clean next identifying experiment should vary **trainable leverage only**—for example, LoRA or last-\(N\)-block unfreezing—while holding no-action conditioning, guide scale, training cohort, rollout noise, and evaluation protocol fixed. Action conditioning should then be restored in a separate or factorial arm. A matched 16-clip held-out cohort at step 20,000 would additionally help distinguish train-domain adaptation from individual-clip memorization.

## 12. Reproducibility and Evidence

| Artifact | Purpose |
|---|---|
| [`full_ft_overfit_command.md`](full_ft_overfit_command.md) | exact launch command, run identity, and final overrides |
| [`full_ft_overfit_params_set_up.md`](full_ft_overfit_params_set_up.md) | finalized training and validation protocol |
| [`full_ft_overfit_results.md`](full_ft_overfit_results.md) | authoritative scalar and rollout results |
| [`full_ft_overfit_analysis.md`](full_ft_overfit_analysis.md) | final hypothesis adjudication and caveats |
| [`full_ft_overfit_01-overview_results.html`](full_ft_overfit_01-overview_results.html) | earlier interactive overview with source chart data |

Key implementation paths at code commit `031228e`:

- `src/maxdiffusion/trainers/wan_ti2v_full_ft_trainer.py` — full-Transformer optimizer, one-step objective, and loop
- `src/maxdiffusion/models/wan/side_adapter_wan.py` — shared latent noising, first-frame pin, and masked velocity loss helpers
- `src/maxdiffusion/generate_wan_side_adapter.py` — full-fine-tuning rollout branch and VAE decode
- `src/maxdiffusion/configs/base_wan_5b_full_ft.yml` — experiment configuration
- `bash_scripts/launch_wan_train.sh` — launch routing and final runtime overrides
