"""Split the torch Ctrl-World ``checkpoint-10000.pt`` into a JAX-loadable layout.

The torch checkpoint contains one flat ``state_dict`` with dotted keys of the form::

    unet.<...>           # 1428 tensors
    vae.<...>            #  374 tensors
    image_encoder.<...>  #  520 tensors  (unused by the JAX rollout — CLIP vision
                         #                cross-attn is replaced by the action MLP)
    text_encoder.<...>   #  197 tensors  (unused — we reload stock
                         #                openai/clip-vit-base-patch32 at runtime,
                         #                which is bit-identical since ctrl-world
                         #                froze the text encoder)
    action_encoder.action_encode.{0,2,4}.{weight,bias}

``FlaxVideoUNet.from_pretrained(..., from_pt=True)`` and
``FlaxSVDAutoencoderKL.from_pretrained(..., from_pt=True)`` both expect the HF
Diffusers repo layout (``<dir>/{unet,vae}/diffusion_pytorch_model.safetensors``
+ a matching ``config.json``). This script produces that layout by:

  1. Stripping the ``unet.``/``vae.`` prefix, dumping each subset as
     ``diffusion_pytorch_model.safetensors`` under ``out_dir/{unet,vae}/``.
  2. Copying ``unet/config.json`` and ``vae/config.json`` verbatim from a base
     SVD template (ctrl-world uses the exact same UNet / VAE architecture as
     stabilityai/stable-video-diffusion-img2vid — verified with
     ``UNetSpatioTemporalConditionModel().config`` on the torch side).
  3. Writing the 6 action-encoder tensors to
     ``out_dir/action_encoder.safetensors`` for the driver to load directly.

Usage::

    python scripts/convert_ctrl_world_ckpt.py \\
        --in_pt     /path/to/ctrl-world/checkpoint-10000.pt \\
        --svd_template_dir /path/to/stable-video-diffusion-img2vid \\
        --out_dir   /path/to/ctrl-world-jax
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


_BUCKET_PREFIXES = ("unet", "vae", "action_encoder")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in_pt", required=True, help="Path to checkpoint-10000.pt")
    ap.add_argument(
        "--svd_template_dir",
        required=True,
        help="Base SVD HF-Diffusers directory (source of unet/config.json + vae/config.json)",
    )
    ap.add_argument("--out_dir", required=True, help="Destination directory for the JAX-loadable layout")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing out_dir instead of skipping when outputs already exist.",
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    unet_st = out / "unet" / "diffusion_pytorch_model.safetensors"
    vae_st = out / "vae" / "diffusion_pytorch_model.safetensors"
    ae_st = out / "action_encoder.safetensors"

    if not args.force and unet_st.exists() and vae_st.exists() and ae_st.exists():
        print(f"[convert] outputs already exist under {out}; pass --force to overwrite. Skipping.")
        return

    template = Path(args.svd_template_dir)
    for sub in ("unet", "vae"):
        if not (template / sub / "config.json").is_file():
            raise FileNotFoundError(
                f"Expected {template}/{sub}/config.json — point --svd_template_dir at an "
                "HF-Diffusers-style SVD directory (e.g. the one downloaded by Ctrl-World)."
            )

    print(f"[convert] reading {args.in_pt} ...")
    ckpt = torch.load(args.in_pt, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"{args.in_pt} is not a state_dict (got {type(ckpt).__name__})")

    buckets: dict[str, dict[str, torch.Tensor]] = {p: {} for p in _BUCKET_PREFIXES}
    skipped = 0
    for k, v in ckpt.items():
        top, _, rest = k.partition(".")
        if top in buckets and rest:
            # float() also upcasts bf16 → fp32 so safetensors can serialize; the JAX
            # loader downcasts back to weights_dtype during convert_pytorch_state_dict_to_flax.
            buckets[top][rest] = v.detach().contiguous().float()
        else:
            skipped += 1

    print(
        f"[convert] state_dict buckets: "
        + ", ".join(f"{k}={len(v)}" for k, v in buckets.items())
        + f" (skipped={skipped})"
    )

    for sub in ("unet", "vae"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template / sub / "config.json", out / sub / "config.json")
        save_file(buckets[sub], str(out / sub / "diffusion_pytorch_model.safetensors"))
        print(
            f"[convert] wrote {sub}/diffusion_pytorch_model.safetensors "
            f"({len(buckets[sub])} tensors) + config.json"
        )

    if not buckets["action_encoder"]:
        raise RuntimeError(
            "No action_encoder.* keys found in the checkpoint — expected "
            "action_encoder.action_encode.{0,2,4}.{weight,bias}."
        )
    save_file(buckets["action_encoder"], str(out / "action_encoder.safetensors"))
    print(
        f"[convert] wrote action_encoder.safetensors ({len(buckets['action_encoder'])} tensors)"
    )

    # Copy auxiliary subfolders so out_dir is a drop-in HF-Diffusers repo — this lets
    # SVDCheckpointer / FlaxVideoUNet.from_pretrained resolve scheduler / feature
    # extractor / image_encoder configs without a second --svd_template_dir flag at
    # inference time. The CLIP vision + feature extractor + scheduler configs are
    # frozen during ctrl-world training, so the base-SVD copies are bit-identical.
    for sub in ("image_encoder", "feature_extractor", "scheduler"):
        src = template / sub
        dst = out / sub
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"[convert] copied {sub}/ from template")
    for f in ("model_index.json",):
        src = template / f
        if src.is_file():
            shutil.copyfile(src, out / f)
            print(f"[convert] copied {f} from template")

    print(f"[convert] done → {out}")


if __name__ == "__main__":
    main()
