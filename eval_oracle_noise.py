"""Evaluate oracle_noise_offset across test videos.

Loads the WAN 5B model once, then for each video in test_videos/ runs inference
under each oracle_noise_offset configuration, captures MSE/MAE, and plots results.

Usage:
    python eval_oracle_noise.py src/maxdiffusion/configs/base_wan_5b.yml \
        pretrained_model_name_or_path=... [other overrides]
"""

import contextlib
import io
import json
import os
import re
import sys
import glob
from collections import defaultdict

import flax
import jax
import jax.numpy as jnp
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

jax.config.update("jax_use_shardy_partitioner", True)

from maxdiffusion import pyconfig, max_logging, max_utils
from maxdiffusion.checkpointing.wan_checkpointer_ti2v_2p2 import WanCheckpointerTI2V_2_2
from maxdiffusion.utils.loading_utils import load_video
from maxdiffusion.utils import export_to_video

# ── Configuration matrix ───────────────────────────────────────────────────────
# Each entry: (label, num_privileged_frames, oracle_noise_offset, privileged)
#   label                  : display name for plots
#   num_privileged_frames  : written to pipeline.config before call
#   oracle_noise_offset    : written to pipeline.config before call (-1 = clean)
#   privileged             : passed as kwarg to pipeline()
CONFIGS = [
    ("no_gt",    0,  -1, False),
    ("clean",   -1,  -1, True),
    ("offset=0", -1,  0,  True),
    ("offset=1", -1,  1,  True),
    ("offset=10",-1,  10, True),
    ("offset=30",-1,  30, True),
    ("offset=50",-1,  50, True),
]

MSE_RE = re.compile(r"MSE:\s*([\d.eE+\-]+)")
MAE_RE = re.compile(r"MAE:\s*([\d.eE+\-]+)")


@contextlib.contextmanager
def capture_stdout():
    """Redirect stdout to a string buffer and yield it."""
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


def load_video_as_pipeline_input(video_path, height, width):
    """Returns (image_pil, conditioning_video_jnp)."""
    frames = load_video(video_path)
    frames = [f.convert("RGB").resize((width, height), PIL.Image.LANCZOS) for f in frames]
    image = frames[0]
    arr = np.stack([np.array(f) for f in frames], axis=0).astype(np.float32)
    arr = arr / 127.5 - 1.0           # [T, H, W, C]
    arr = arr.transpose(3, 0, 1, 2)   # [C, T, H, W]
    conditioning_video = jnp.array(arr)[None]  # [1, C, T, H, W]
    return image, conditioning_video


def run_inference(pipeline, config, image, conditioning_video, privileged, label, video_stem, out_dir):
    """Run one pipeline call; return (mse, mae) parsed from printed output, or (None, None)."""
    prompt = [config.prompt] * config.global_batch_size_to_train_on
    negative_prompt = [config.negative_prompt] * config.global_batch_size_to_train_on

    with capture_stdout() as buf:
        videos = pipeline(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            height=config.height,
            width=config.width,
            num_frames=config.num_frames,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            use_cfg_cache=getattr(config, "use_cfg_cache", False),
            use_sen_cache=getattr(config, "use_sen_cache", False),
            conditioning_video=conditioning_video,
            privileged=privileged,
        )

    output = buf.getvalue()
    # Also print captured output so the slurm log has it
    print(output, end="")

    # Save output video
    safe_label = label.replace("=", "").replace(" ", "_")
    video_path = os.path.join(out_dir, f"{video_stem}__{safe_label}.mp4")
    if videos is not None and len(videos) > 0:
        export_to_video(videos[0], video_path, fps=getattr(config, "fps", 16))

    mse_match = MSE_RE.search(output)
    mae_match = MAE_RE.search(output)
    mse = float(mse_match.group(1)) if mse_match else None
    mae = float(mae_match.group(1)) if mae_match else None
    return mse, mae


def plot_results(results, out_dir):
    """
    results: dict  label -> list of (mse, mae) across videos
    Plots average MSE and MAE vs oracle_noise_offset configuration.
    """
    labels = [c[0] for c in CONFIGS]
    avg_mse, avg_mae = [], []
    err_mse, err_mae = [], []

    for label in labels:
        vals = [(m, a) for m, a in results[label] if m is not None and a is not None]
        if vals:
            mses, maes = zip(*vals)
            avg_mse.append(np.mean(mses))
            avg_mae.append(np.mean(maes))
            err_mse.append(np.std(mses))
            err_mae.append(np.std(maes))
        else:
            avg_mse.append(float("nan"))
            avg_mae.append(float("nan"))
            err_mse.append(0)
            err_mae.append(0)

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(x, avg_mse, width, yerr=err_mse, capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Average MSE vs Oracle Noise Offset")

    axes[1].bar(x, avg_mae, width, yerr=err_mae, capsize=4, color="orange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Average MAE vs Oracle Noise Offset")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "oracle_noise_eval.png")
    plt.savefig(plot_path, dpi=150)
    print(f"[eval] Plot saved to {plot_path}")
    plt.close()


def main():
    try:
        flax.config.update("flax_always_shard_variable", False)
    except LookupError:
        pass

    pyconfig.initialize(sys.argv)
    config = pyconfig.config

    max_utils.ensure_machinelearning_job_runs(config)

    # Load model once
    print("[eval] Loading model...")
    checkpoint_loader = WanCheckpointerTI2V_2_2(config=config)
    pipeline, _, _ = checkpoint_loader.load_checkpoint()
    print("[eval] Model loaded.")

    video_paths = sorted(glob.glob("test_videos_resized/*.mp4"))
    if not video_paths:
        raise FileNotFoundError("No .mp4 files found in test_videos/")
    print(f"[eval] Found {len(video_paths)} videos: {[os.path.basename(p) for p in video_paths]}")

    out_dir = os.path.join(getattr(config, "output_dir", "./outputs"), "oracle_noise_eval")
    os.makedirs(out_dir, exist_ok=True)

    # results[label] = list of (mse, mae)
    results = defaultdict(list)

    for video_path in video_paths:
        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[eval] ── Video: {video_stem} ──")

        image, conditioning_video = load_video_as_pipeline_input(video_path, config.height, config.width)

        for label, num_priv, offset, privileged in CONFIGS:
            print(f"[eval]   config={label}")
            pyconfig._config.keys["num_privileged_frames"] = num_priv
            pyconfig._config.keys["oracle_noise_offset"] = offset

            mse, mae = run_inference(
                pipeline, config, image, conditioning_video, privileged, label, video_stem, out_dir
            )
            results[label].append((mse, mae))
            print(f"[eval]   → MSE={mse}, MAE={mae}")

    # Save raw results
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)
    print(f"[eval] Results saved to {results_path}")

    plot_results(results, out_dir)


if __name__ == "__main__":
    main()
