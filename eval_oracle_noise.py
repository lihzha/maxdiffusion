"""Evaluate oracle_noise_offset across test videos and/or TFRecord val samples.

Loads the WAN 5B model once, then for each data source runs inference under
each oracle_noise_offset configuration, captures MSE/MAE, and plots results.

Usage:
    # Video eval only (reads test_videos_resized/*.mp4):
    python eval_oracle_noise.py src/maxdiffusion/configs/base_wan_5b.yml \
        pretrained_model_name_or_path=...

    # TFRecord val eval only (or in addition to video eval):
    python eval_oracle_noise.py src/maxdiffusion/configs/base_wan_5b.yml \
        pretrained_model_name_or_path=... \
        val_data_dir=/path/to/val/tfrecords
"""

import contextlib
import glob
import io
import json
import os
import re
import sys
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
#   num_privileged_frames  : written to config before call (-1 = all frames)
#   oracle_noise_offset    : written to config before call (-1 = clean)
#   privileged             : passed to pipeline() in video mode only;
#                            ignored in TFRecord mode (always forced True by preencoded_oracle_latents)
CONFIGS = [
    ("no_gt",     0,  -1, False),
    # ("clean",    -1,  -1, True),
    # ("offset=0",  -1,  0,  True),
    # ("offset=1",  -1,  1,  True),
    # ("offset=10", -1,  10, True),
    # ("offset=30", -1,  30, True),
    # ("offset=50", -1,  50, True),
]

MSE_RE = re.compile(r"MSE:\s*([\d.eE+\-]+)")
MAE_RE = re.compile(r"MAE:\s*([\d.eE+\-]+)")


@contextlib.contextmanager
def capture_stdout():
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


def _parse_mse_mae(output: str):
    mse_match = MSE_RE.search(output)
    mae_match = MAE_RE.search(output)
    mse = float(mse_match.group(1)) if mse_match else None
    mae = float(mae_match.group(1)) if mae_match else None
    return mse, mae


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_video_as_pipeline_input(video_path, height, width):
    """Returns (image_pil, conditioning_video_jnp)."""
    frames = load_video(video_path)
    frames = [f.convert("RGB").resize((width, height), PIL.Image.LANCZOS) for f in frames]
    image = frames[0]
    arr = np.stack([np.array(f) for f in frames], axis=0).astype(np.float32)
    arr = arr / 127.5 - 1.0         # [T, H, W, C]
    arr = arr.transpose(3, 0, 1, 2) # [C, T, H, W]
    conditioning_video = jnp.array(arr)[None]  # [1, C, T, H, W]
    return image, conditioning_video


def iter_val_latents(val_data_dir: str, window_size: int):
    """Yield (oracle_latents, text_embed, sample_idx) for every TFRecord in val_data_dir.

    Stacks cameras on H exactly as training does (cam0/cam1/cam2 on axis 2).
    Takes the first `window_size` latent frames, matching training eval.

    oracle_latents: (1, window_size, H_lat*3, W_lat, C_z) channels-last, float32, normalized.
    text_embed:     (1, 512, 4096) float32.
    """
    import tensorflow as tf

    pattern = os.path.join(val_data_dir, "*.tfrecord*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TFRecord files found at {pattern}")

    feature_spec = {
        "latent_cam0": tf.io.FixedLenFeature([], tf.string),
        "latent_cam1": tf.io.FixedLenFeature([], tf.string),
        "latent_cam2": tf.io.FixedLenFeature([], tf.string),
        "text_embed":  tf.io.FixedLenFeature([], tf.string),
        "traj_len":    tf.io.FixedLenFeature([], tf.int64),
    }
    ds = tf.data.TFRecordDataset(files)
    ds = ds.map(lambda x: tf.io.parse_single_example(x, feature_spec))

    for idx, raw in enumerate(ds):
        cam0 = tf.cast(tf.io.parse_tensor(raw["latent_cam0"], out_type=tf.float16), tf.float32).numpy()
        cam1 = tf.cast(tf.io.parse_tensor(raw["latent_cam1"], out_type=tf.float16), tf.float32).numpy()
        cam2 = tf.cast(tf.io.parse_tensor(raw["latent_cam2"], out_type=tf.float16), tf.float32).numpy()
        text = tf.cast(tf.io.parse_tensor(raw["text_embed"],  out_type=tf.float16), tf.float32).numpy()

        latent = np.concatenate([cam0, cam1, cam2], axis=2)  # (F_lat, C, H*3, W)
        latent = latent[:window_size]                         # first window, matches training eval
        latent = latent.transpose(0, 2, 3, 1)                # (F_lat, H*3, W, C)
        oracle_latents = jnp.array(latent[None])              # (1, F_lat, H*3, W, C)
        text_embed = jnp.array(text[None])                    # (1, 512, 4096)

        yield oracle_latents, text_embed, idx


# ── Inference runners ─────────────────────────────────────────────────────────

def run_inference_video(pipeline, config, image, conditioning_video, privileged, label, video_stem, out_dir):
    """Run pipeline with pixel-space conditioning video; return (mse, mae)."""
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
    print(output, end="")

    safe_label = label.replace("=", "").replace(" ", "_")
    video_path = os.path.join(out_dir, f"{video_stem}__{safe_label}.mp4")
    if videos is not None and len(videos) > 0:
        export_to_video(videos[0], video_path, fps=getattr(config, "fps", 16))

    return _parse_mse_mae(output)


def run_inference_tfrecord(pipeline, config, oracle_latents, text_embed, label, sample_idx, out_dir):
    """Run pipeline with preencoded oracle latents from a TFRecord; return (mse, mae).

    In TFRecord mode `privileged` is always forced True by the pipeline (because
    preencoded_oracle_latents is set). The effective conditioning is controlled
    by num_privileged_frames: 0 → no_gt behavior, -1 → all frames privileged.
    """
    negative_prompt = [getattr(config, "negative_prompt", "")]

    lat_h = int(oracle_latents.shape[2])
    lat_w = int(oracle_latents.shape[3])
    height = lat_h * pipeline.vae_scale_factor_spatial
    width  = lat_w * pipeline.vae_scale_factor_spatial

    with capture_stdout() as buf:
        latents = pipeline(
            prompt=[""],
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=config.num_frames,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            use_cfg_cache=getattr(config, "use_cfg_cache", False),
            use_sen_cache=getattr(config, "use_sen_cache", False),
            prompt_embeds=text_embed,
            preencoded_oracle_latents=oracle_latents,
            output_type="latent",
        )

    output = buf.getvalue()
    print(output, end="")

    # latents: (B, C, T, H_lat*3, W_lat) denormalized channels-first.
    # Decode each camera separately then stack horizontally → (T, H_pix, W_pix*3, C).
    h = latents.shape[3] // 3
    cam_videos = [
        pipeline._decode_latents_to_video(latents[:, :, :, i * h:(i + 1) * h, :])
        for i in range(3)
    ]  # each: (B, T, H_pix, W_pix, C)
    stacked = np.concatenate([v[0] for v in cam_videos], axis=2)  # (T, H_pix, W_pix*3, C)

    safe_label = label.replace("=", "").replace(" ", "_")
    video_path = os.path.join(out_dir, f"sample{sample_idx:03d}__{safe_label}.mp4")
    export_to_video(stacked, video_path, fps=getattr(config, "fps", 16))

    return _parse_mse_mae(output)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results, out_dir, filename="oracle_noise_eval.png", title_prefix=""):
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
    bar_width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(x, avg_mse, bar_width, yerr=err_mse, capsize=4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("MSE")
    axes[0].set_title(f"{title_prefix}Average MSE vs Oracle Noise Offset")

    axes[1].bar(x, avg_mae, bar_width, yerr=err_mae, capsize=4, color="orange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("MAE")
    axes[1].set_title(f"{title_prefix}Average MAE vs Oracle Noise Offset")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, filename)
    plt.savefig(plot_path, dpi=150)
    print(f"[eval] Plot saved to {plot_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        flax.config.update("flax_always_shard_variable", False)
    except LookupError:
        pass

    pyconfig.initialize(sys.argv)
    config = pyconfig.config

    max_utils.ensure_machinelearning_job_runs(config)

    print("[eval] Loading model...")
    checkpoint_loader = WanCheckpointerTI2V_2_2(config=config)
    checkpoint_step = getattr(config, "checkpoint_step", -1)
    pipeline, _, _, _ = checkpoint_loader.load_checkpoint(
        step=checkpoint_step if checkpoint_step > 0 else None
    )
    print("[eval] Model loaded.")

    out_dir = os.path.join(getattr(config, "output_dir", "./outputs"), getattr(config, "run_name", "oracle_noise_eval"))
    os.makedirs(out_dir, exist_ok=True)

    # ── Video eval ────────────────────────────────────────────────────────────
    video_paths = sorted(glob.glob("test_videos_resized/*.mp4"))
    video_paths = None
    if video_paths:
        print(f"[eval] Found {len(video_paths)} videos: {[os.path.basename(p) for p in video_paths]}")
        video_results = defaultdict(list)

        for video_path in video_paths:
            video_stem = os.path.splitext(os.path.basename(video_path))[0]
            print(f"\n[eval] ── Video: {video_stem} ──")
            image, conditioning_video = load_video_as_pipeline_input(
                video_path, config.height, config.width
            )

            for label, num_priv, offset, privileged in CONFIGS:
                print(f"[eval]   config={label}")
                pyconfig._config.keys["num_privileged_frames"] = num_priv
                pyconfig._config.keys["oracle_noise_offset"] = offset

                mse, mae = run_inference_video(
                    pipeline, config, image, conditioning_video, privileged, label, video_stem, out_dir
                )
                video_results[label].append((mse, mae))
                print(f"[eval]   → MSE={mse}, MAE={mae}")

        results_path = os.path.join(out_dir, "video_results.json")
        with open(results_path, "w") as f:
            json.dump(dict(video_results), f, indent=2)
        print(f"[eval] Video results saved to {results_path}")
        plot_results(video_results, out_dir, filename="oracle_noise_eval_video.png", title_prefix="Video: ")

    # ── TFRecord val eval ─────────────────────────────────────────────────────
    val_data_dir = getattr(config, "val_data_dir", "")
    if val_data_dir:
        window_size = 1 + config.num_frames // pipeline.vae_scale_factor_temporal
        print(f"\n[eval] TFRecord val eval: {val_data_dir}  window_size={window_size}")
        tfrecord_out_dir = os.path.join(out_dir, "tfrecord")
        os.makedirs(tfrecord_out_dir, exist_ok=True)
        tfrecord_results = defaultdict(list)

        for oracle_latents, text_embed, idx in iter_val_latents(val_data_dir, window_size):
            print(f"\n[eval] ── TFRecord sample {idx} ──")
            for label, num_priv, offset, _ in CONFIGS:
                print(f"[eval]   config={label}")
                pyconfig._config.keys["num_privileged_frames"] = num_priv
                pyconfig._config.keys["oracle_noise_offset"] = offset

                mse, mae = run_inference_tfrecord(
                    pipeline, config, oracle_latents, text_embed, label, idx, tfrecord_out_dir
                )
                tfrecord_results[label].append((mse, mae))
                print(f"[eval]   → MSE={mse}, MAE={mae}")

        results_path = os.path.join(out_dir, "tfrecord_results.json")
        with open(results_path, "w") as f:
            json.dump(dict(tfrecord_results), f, indent=2)
        print(f"[eval] TFRecord results saved to {results_path}")
        plot_results(
            tfrecord_results, out_dir,
            filename="oracle_noise_eval_tfrecord.png",
            title_prefix="TFRecord: ",
        )

    if not video_paths and not val_data_dir:
        raise RuntimeError(
            "No data found: place .mp4 files in test_videos_resized/ or pass val_data_dir=..."
        )


if __name__ == "__main__":
    main()
