#!/usr/bin/env bash
set -euo pipefail

# Run locally from a host that can SSH to both Della and a1001 and has
# authenticated gcloud/gsutil. This avoids writing more data to Della GPFS when
# it is full: source files are streamed from Della to a1001, conversion runs on
# a1001 Slurm, and finished TFRecords are uploaded directly from a1001 to GCS
# with a short-lived OAuth token.

LOCAL_MANIFEST=${LOCAL_MANIFEST:-/tmp/wan_train_full.jsonl}
DELLA_HOST=${DELLA_HOST:-della-gpu}
DELLA_CACHE_ROOT=${DELLA_CACHE_ROOT:-/scratch/gpfs/AM43/lz3952/Wan2.2/data/droid_cache_windows_v0/train}
A1001_HOST=${A1001_HOST:-a1001}
A1001_BASE=${A1001_BASE:-/lustre/fsw/portfolios/nvr/users/lzha/wan_side_adapter_a1001}
A1001_REPO=${A1001_REPO:-/lustre/fsw/portfolios/nvr/users/lzha/maxdiffusion-side-adapter}
A1001_ACCOUNT=${A1001_ACCOUNT:-nvr_lpr_rvp}
A1001_PARTITION=${A1001_PARTITION:-cpu_short}
DEST=${DEST:-gs://v6_east1d/datasets/droid_wan_side_adapter/train}

SHARD_SIZE=${SHARD_SIZE:-2048}
TOTAL_SHARDS=${TOTAL_SHARDS:-704}
START_SHARD=${START_SHARD:-97}
END_SHARD=${END_SHARD:-703}
BATCH_SHARDS=${BATCH_SHARDS:-16}
CONCURRENCY=${CONCURRENCY:-8}
UPLOAD_CONCURRENCY=${UPLOAD_CONCURRENCY:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
MEM=${MEM:-32G}
TIME_LIMIT=${TIME_LIMIT:-02:00:00}
ESTIMATE_SAMPLES=${ESTIMATE_SAMPLES:-8}
MAX_OUTPUT_GB=${MAX_OUTPUT_GB:-1}

if [[ ! -f "${LOCAL_MANIFEST}" ]]; then
  echo "Missing LOCAL_MANIFEST=${LOCAL_MANIFEST}" >&2
  exit 2
fi
if (( START_SHARD < 0 || END_SHARD >= TOTAL_SHARDS || START_SHARD > END_SHARD )); then
  echo "Invalid shard range ${START_SHARD}-${END_SHARD}; total shards=${TOTAL_SHARDS}" >&2
  exit 2
fi

total_examples=$(wc -l < "${LOCAL_MANIFEST}" | tr -d ' ')
if (( total_examples <= 0 )); then
  echo "Manifest is empty: ${LOCAL_MANIFEST}" >&2
  exit 2
fi

remote() {
  ssh "${A1001_HOST}" "$@"
}

gcs_object_url() {
  local gcs_path=$1
  local without_scheme=${gcs_path#gs://}
  local bucket=${without_scheme%%/*}
  local object=${without_scheme#*/}
  printf 'https://storage.googleapis.com/%s/%s' "${bucket}" "${object}"
}

make_file_list() {
  local start_shard=$1
  local end_shard=$2
  local output=$3
  python3 - "${LOCAL_MANIFEST}" "${start_shard}" "${end_shard}" "${SHARD_SIZE}" "${output}" <<'PY'
import json
import sys

manifest, start_shard, end_shard, shard_size, output = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
start = start_shard * shard_size
end = (end_shard + 1) * shard_size
with open(manifest, "r", encoding="utf-8") as f, open(output, "w", encoding="utf-8") as out:
    for idx, line in enumerate(f):
        if idx < start:
            continue
        if idx >= end:
            break
        name = json.loads(line)["name"]
        out.write(f"{name}/z_I0.pt\n")
        out.write(f"{name}/z_video.pt\n")
        out.write(f"{name}/actions.npy\n")
PY
}

stage_source_batch() {
  local batch_start=$1
  local batch_end=$2
  local file_list=$3
  make_file_list "${batch_start}" "${batch_end}" "${file_list}"
  remote "set -euo pipefail; rm -rf '${A1001_BASE}/cache_chunk' '${A1001_BASE}/tfrecord_stage'; mkdir -p '${A1001_BASE}/cache_chunk' '${A1001_BASE}/tfrecord_stage' '${A1001_BASE}/logs'"
  ssh "${DELLA_HOST}" "cd '${DELLA_CACHE_ROOT}' && tar -cf - -T -" < "${file_list}" \
    | remote "tar -xf - -C '${A1001_BASE}/cache_chunk'"
  remote "set -euo pipefail; find '${A1001_BASE}/cache_chunk' -name z_video.pt | wc -l; du -sh '${A1001_BASE}/cache_chunk'; df -h '${A1001_BASE}'"
}

submit_convert_batch() {
  local batch_start=$1
  local batch_end=$2
  remote "set -euo pipefail; sbatch --parsable --account='${A1001_ACCOUNT}' --partition='${A1001_PARTITION}' --cpus-per-task='${CPUS_PER_TASK}' --mem='${MEM}' --time='${TIME_LIMIT}' --job-name='wan-tfr-a1-${batch_start}-${batch_end}' --output='${A1001_BASE}/logs/wan_tfr_${batch_start}_${batch_end}_%j.out' --export=ALL,A1001_BASE='${A1001_BASE}',A1001_REPO='${A1001_REPO}',BATCH_START='${batch_start}',BATCH_END='${batch_end}',SHARD_SIZE='${SHARD_SIZE}',TOTAL_SHARDS='${TOTAL_SHARDS}',TOTAL_EXAMPLES='${total_examples}',CONCURRENCY='${CONCURRENCY}',ESTIMATE_SAMPLES='${ESTIMATE_SAMPLES}',MAX_OUTPUT_GB='${MAX_OUTPUT_GB}' <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

cd \"${A1001_REPO}\"
source .venv/bin/activate
mkdir -p \"${A1001_BASE}/tfrecord_stage\"
export PYTHONUNBUFFERED=1

launch_one() {
  local shard=\$1
  local start=\$((shard * SHARD_SIZE))
  local end=\$((start + SHARD_SIZE))
  local shard_tag
  if (( end > TOTAL_EXAMPLES )); then
    end=\${TOTAL_EXAMPLES}
  fi
  if (( start >= TOTAL_EXAMPLES )); then
    return 0
  fi
  shard_tag=\$(printf \"%05d\" \"\${shard}\")
  echo \"[start] shard=\${shard} indices=\${start}:\${end}\"
  python src/maxdiffusion/data_preprocessing/wan_side_adapter_droid_cache_to_tfrecord.py \\
    --cache-root \"${A1001_BASE}/cache_chunk\" \\
    --manifest-jsonl \"${A1001_BASE}/train_full.jsonl\" \\
    --require-manifest \\
    --output-dir \"${A1001_BASE}/tfrecord_stage\" \\
    --shard-prefix train \\
    --shard-offset \"\${shard}\" \\
    --total-shards \"${TOTAL_SHARDS}\" \\
    --shard-size \"${SHARD_SIZE}\" \\
    --start-index \"\${start}\" \\
    --end-index \"\${end}\" \\
    --estimate-samples \"${ESTIMATE_SAMPLES}\" \\
    --max-output-gb \"${MAX_OUTPUT_GB}\" \\
    --fail-fast \\
    --summary-path \"${A1001_BASE}/tfrecord_stage/summary-train-\${shard_tag}.json\"
  echo \"[done] shard=\${shard}\"
}

for shard in \$(seq \"\${BATCH_START}\" \"\${BATCH_END}\"); do
  launch_one \"\${shard}\" &
  while (( \$(jobs -pr | wc -l) >= CONCURRENCY )); do
    wait -n
  done
done

while (( \$(jobs -pr | wc -l) > 0 )); do
  wait -n
done
SBATCH"
}

wait_for_job() {
  local job_id=$1
  while remote "squeue -h -j '${job_id}' | grep -q ."; do
    remote "squeue -j '${job_id}' -o '%.18i %.9P %.30j %.8T %.10M %.6D %R'"
    sleep 60
  done
  local states
  states=$(remote "sacct -j '${job_id}' --format=State -P -n | sed '/^$/d' | sort -u | tr '\n' ' '")
  echo "job ${job_id} states: ${states}"
  if [[ "${states}" != *COMPLETED* ]] || [[ "${states}" == *FAILED* ]] || [[ "${states}" == *CANCELLED* ]] || [[ "${states}" == *TIMEOUT* ]] || [[ "${states}" == *OUT_OF_MEMORY* ]]; then
    remote "sacct -j '${job_id}' --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -P"
    return 4
  fi
}

upload_one() {
  local shard=$1
  local shard_name
  shard_name=$(printf 'train-%05d-of-%05d.tfrecord' "${shard}" "${TOTAL_SHARDS}")
  local gcs_path="${DEST}/${shard_name}"
  local url
  url=$(gcs_object_url "${gcs_path}")
  if gsutil -q stat "${gcs_path}"; then
    echo "[skip-existing] ${gcs_path}"
    remote "rm -f '${A1001_BASE}/tfrecord_stage/${shard_name}' '${A1001_BASE}/tfrecord_stage/summary-train-$(printf "%05d" "${shard}").json'"
    return 0
  fi
  local token
  token=$(gcloud auth print-access-token)
  printf '%s\n' "${token}" | remote "set -euo pipefail; read -r TOKEN; test -s '${A1001_BASE}/tfrecord_stage/${shard_name}'; curl -fS --retry 5 --retry-delay 10 -X PUT -H \"Authorization: Bearer \${TOKEN}\" -H 'Content-Type: application/octet-stream' --upload-file '${A1001_BASE}/tfrecord_stage/${shard_name}' '${url}'"
  gsutil -q stat "${gcs_path}"
  remote "rm -f '${A1001_BASE}/tfrecord_stage/${shard_name}' '${A1001_BASE}/tfrecord_stage/summary-train-$(printf "%05d" "${shard}").json'"
  echo "[uploaded] ${gcs_path}"
}

cleanup_batch() {
  remote "set -euo pipefail; rm -rf '${A1001_BASE}/cache_chunk' '${A1001_BASE}/tfrecord_stage'; mkdir -p '${A1001_BASE}/cache_chunk' '${A1001_BASE}/tfrecord_stage'; du -sh '${A1001_BASE}'; df -h '${A1001_BASE}'"
}

batch_start=${START_SHARD}
while (( batch_start <= END_SHARD )); do
  batch_end=$((batch_start + BATCH_SHARDS - 1))
  if (( batch_end > END_SHARD )); then
    batch_end=${END_SHARD}
  fi
  file_list=$(mktemp "/tmp/wan_a1001_${batch_start}_${batch_end}_XXXXXX.files")
  echo "=== batch ${batch_start}-${batch_end} ==="
  stage_source_batch "${batch_start}" "${batch_end}" "${file_list}"
  rm -f "${file_list}"
  job_id=$(submit_convert_batch "${batch_start}" "${batch_end}")
  echo "submitted ${job_id}"
  wait_for_job "${job_id}"
  for shard in $(seq "${batch_start}" "${batch_end}"); do
    upload_one "${shard}" &
    while (( $(jobs -pr | wc -l) >= UPLOAD_CONCURRENCY )); do
      wait -n
    done
  done
  while (( $(jobs -pr | wc -l) > 0 )); do
    wait -n
  done
  cleanup_batch
  current_count=$( (gsutil ls "${DEST}/train-*.tfrecord" 2>/dev/null || true) | wc -l | tr -d ' ')
  echo "[progress] gcs_train_shards=${current_count}/${TOTAL_SHARDS}"
  batch_start=$((batch_end + 1))
done

echo "completed requested range ${START_SHARD}-${END_SHARD}"
