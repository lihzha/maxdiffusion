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

# List all files with their expected sizes from the HF repo.
# Format per line: "<size_bytes> <path>"
mapfile -t FILE_ENTRIES < <(python3 - <<'EOF'
from huggingface_hub import list_repo_tree
for item in list_repo_tree("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", recursive=True):
    if not hasattr(item, 'size'):
        continue  # skip directory entries
    if item.path.endswith(".pth"):
        continue  # skip legacy PyTorch weights (~15 GB saved)
    print(item.size, item.path)
EOF
)

echo "Found ${#FILE_ENTRIES[@]} files to download."

for entry in "${FILE_ENTRIES[@]}"; do
    expected_size="${entry%% *}"
    filepath="${entry#* }"
    gcs_path="${GCS_DEST}/${filepath}"
    hf_url="${HF_BASE}/${filepath}"

    # Check if already present with the correct size; re-download if truncated.
    if gsutil -q stat "${gcs_path}" 2>/dev/null; then
        actual_size=$(gsutil ls -l "${gcs_path}" 2>/dev/null | awk 'NR==1{print $1}')
        if [ "${actual_size}" = "${expected_size}" ]; then
            echo "  [skip] ${filepath}"
            continue
        fi
        echo "  [redownload — size mismatch: got ${actual_size}, want ${expected_size}] ${filepath}"
        gsutil rm "${gcs_path}"
    fi

    echo "  [fetch] ${filepath}"
    if [ -n "${HF_TOKEN:-}" ]; then
        curl -fsSL -H "Authorization: Bearer ${HF_TOKEN}" "${hf_url}" \
            | gsutil cp - "${gcs_path}"
    else
        curl -fsSL "${hf_url}" \
            | gsutil cp - "${gcs_path}"
    fi
done

echo "Done. Model at ${GCS_DEST}"
