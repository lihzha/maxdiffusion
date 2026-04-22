#!/usr/bin/env bash
# Stream Wan2.1-I2V-14B-720P-Diffusers directly from HuggingFace to GCS.
# Zero local disk usage: curl pipes each file straight into gsutil.
#
# Usage:
#   bash bash_scripts/download_wan_i2v.sh
#   bash bash_scripts/download_wan_i2v.sh gs://my-other-bucket/wan/wan-i2v-diffusers

set -euo pipefail

REPO_ID="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
GCS_DEST="${1:-gs://v6_east1d/wan/wan-i2v-diffusers}"
HF_BASE="https://huggingface.co/${REPO_ID}/resolve/main"

echo "Downloading ${REPO_ID} → ${GCS_DEST}"

# List all files in the repo (uses huggingface_hub, which is tiny in memory).
mapfile -t FILES < <(python3 - <<'EOF'
from huggingface_hub import list_repo_files
for f in list_repo_files("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"):
    # Skip legacy PyTorch weights (redundant with safetensors, saves ~15 GB).
    if not f.endswith(".pth"):
        print(f)
EOF
)

echo "Found ${#FILES[@]} files to download."

for filepath in "${FILES[@]}"; do
    gcs_path="${GCS_DEST}/${filepath}"
    hf_url="${HF_BASE}/${filepath}"

    # Skip if already present (allows resuming interrupted downloads).
    if gsutil -q stat "${gcs_path}" 2>/dev/null; then
        echo "  [skip] ${filepath}"
        continue
    fi

    echo "  [fetch] ${filepath}"
    # -L follows redirects; -f fails fast on HTTP errors; -S shows errors.
    if [ -n "${HF_TOKEN:-}" ]; then
        curl -fsSL -H "Authorization: Bearer ${HF_TOKEN}" "${hf_url}" \
            | gsutil cp - "${gcs_path}"
    else
        curl -fsSL "${hf_url}" \
            | gsutil cp - "${gcs_path}"
    fi
done

echo "Done. Model at ${GCS_DEST}"
