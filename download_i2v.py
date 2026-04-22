from huggingface_hub import snapshot_download


snapshot_download(
    repo_id="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    local_dir="/home/irom-lab/gcs-mount/wan/wan-i2v-diffusers",
    ignore_patterns=["*.pth"],
)
