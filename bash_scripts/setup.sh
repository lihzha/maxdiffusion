curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc
uv venv --python 3.12 ~/maxdiffusion_venv --seed
source ~/maxdiffusion_venv/bin/activate
bash setup.sh MODE=stable DEVICE=tpu

### command

# --- 1. Bucket mount (paths and bucket name) ---
export GCS_BUCKET=v6_east1d
export GCS_MOUNT=/home/irom-lab/gcs-mount

# --- 2. Install gcsfuse (first time only; re-running is a no-op) ---
if ! command -v gcsfuse >/dev/null; then
  export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
  echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  sudo apt-get update
  sudo apt-get install -y gcsfuse
fi

# --- 3. Mount the bucket (cache on /dev/shm because / is full) ---
mkdir -p "$GCS_MOUNT" /dev/shm/gcsfuse-cache
if ! mountpoint -q "$GCS_MOUNT"; then
  gcsfuse \
    --implicit-dirs \
    --file-cache-max-size-mb=-1 \
    --cache-dir=/dev/shm/gcsfuse-cache \
    "$GCS_BUCKET" "$GCS_MOUNT"
fi

# --- 4. Resolve the actual snapshot dir containing config.json ---
export WAN_MODEL_DIR="$(ls -d $GCS_MOUNT/wan/wan-diffusers/snapshots/*/* | head -1)"
echo "Using WAN_MODEL_DIR=$WAN_MODEL_DIR"