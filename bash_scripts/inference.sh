#!/usr/bin/env bash

# TODO: use pre-converetd checkpoint
MAXDIFF_ROOT="/path/to/maxdiffusion"
CTRL_ROOT="/path/to/Ctrl-World"
SAVE_DIR="/path/to/Ctrl-World/outputs"

TORCH_CKPT="$CTRL_ROOT/pretrained_models/ctrl-world/checkpoint-10000.pt"
SVD_TEMPLATE="$CTRL_ROOT/pretrained_models/stable-video-diffusion-img2vid"
CLIP_DIR="$CTRL_ROOT/pretrained_models/clip-vit-base-patch32"
JAX_CKPT_DIR="$CTRL_ROOT/pretrained_models/ctrl-world-jax"

cd "$MAXDIFF_ROOT"
source .venv/bin/activate

# Prepend wheel-bundled NVIDIA libs so JAX 0.9's strict cuSPARSE version check
# finds the matching .so before any system CUDA on LD_LIBRARY_PATH.
NV_LIB_ROOT="$MAXDIFF_ROOT/.venv/lib/python3.12/site-packages/nvidia"
NV_LIBS="$(python -c "import glob, os; print(':'.join(p for p in glob.glob('$NV_LIB_ROOT/*/lib') if os.path.isdir(p)))")"
export LD_LIBRARY_PATH="$NV_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export HF_HOME="${HF_HOME:-/home/allen/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95   # let JAX pre-grab more memory upfront
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Let XLA pattern-match the SDPA in dot-product attention and lower it to
# cuDNN's fused flash kernel. No code change required; if XLA can't match the
# pattern (e.g. dtype mismatch, unusual head_dim), it silently falls back.
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_cudnn_fmha=true"

for p in "$TORCH_CKPT" "$SVD_TEMPLATE" "$CLIP_DIR"; do
  [[ -e "$p" ]] || { echo "Missing input: $p" >&2; exit 1; }
done

# One-time conversion. The converter skips when outputs already exist so this
# stays a no-op on subsequent runs; pass --force to the converter manually if
# the torch checkpoint is updated.
if [[ ! -f "$JAX_CKPT_DIR/unet/diffusion_pytorch_model.safetensors" ]] \
   || [[ ! -f "$JAX_CKPT_DIR/action_encoder.safetensors" ]]; then
  echo "[run_maxdiff_ctrlworld] converting torch checkpoint → $JAX_CKPT_DIR"
  python scripts/convert_ctrl_world_ckpt.py \
    --in_pt "$TORCH_CKPT" \
    --svd_template_dir "$SVD_TEMPLATE" \
    --out_dir "$JAX_CKPT_DIR"
fi

mkdir -p "$SAVE_DIR"

python -m maxdiffusion.generate_ctrl_world_replay \
  --ctrl_world_dir   "$JAX_CKPT_DIR" \
  --clip_path        "$CLIP_DIR" \
  --val_dataset_dir  "$CTRL_ROOT/dataset_example/droid_subset" \
  --data_stat_path   "$CTRL_ROOT/dataset_meta_info/droid/stat.json" \
  --save_dir         "$SAVE_DIR" \
  "$@"
