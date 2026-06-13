#!/usr/bin/env bash
set -euo pipefail

# Run this on a Della login node. It stages only one bounded batch of TFRecords
# at a time, uploads the batch to GCS, then deletes local shards before
# submitting the next batch.

REPO_DIR=${REPO_DIR:-/scratch/gpfs/AM43/lz3952/maxdiffusion-side-adapter}
VENV_DIR=${VENV_DIR:-/scratch/gpfs/AM43/lz3952/Wan2.2/.venv}
CACHE_ROOT=${CACHE_ROOT:-/scratch/gpfs/AM43/lz3952/Wan2.2/data/droid_cache_windows_v0/train}
MANIFEST_JSONL=${MANIFEST_JSONL:-/scratch/gpfs/AM43/lz3952/maxdiffusion_tfrecord_manifests/train_full.jsonl}
STAGE_DIR=${STAGE_DIR:-/scratch/gpfs/AM43/lz3952/maxdiffusion_tfrecord_stage/droid_wan_side_adapter/train}
DEST=${DEST:-gs://v6_east1d/datasets/droid_wan_side_adapter/train}
SPLIT=${SPLIT:-train}

SHARD_SIZE=${SHARD_SIZE:-2048}
BATCH_SHARDS=${BATCH_SHARDS:-48}
CONCURRENCY=${CONCURRENCY:-8}
START_SHARD=${START_SHARD:-0}
END_SHARD=${END_SHARD:-}
ESTIMATE_SAMPLES=${ESTIMATE_SAMPLES:-8}
MAX_OUTPUT_GB=${MAX_OUTPUT_GB:-1}
MIN_FREE_GB=${MIN_FREE_GB:-35}
PARTITION=${PARTITION:-cpu}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
MEM=${MEM:-64G}
TIME_LIMIT=${TIME_LIMIT:-02:00:00}

LOG_DIR=${LOG_DIR:-${REPO_DIR}/slurm_outputs}
mkdir -p "${LOG_DIR}" "${STAGE_DIR}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Missing REPO_DIR: ${REPO_DIR}" >&2
  exit 2
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Missing Python venv: ${VENV_DIR}" >&2
  exit 2
fi
if [[ ! -f "${MANIFEST_JSONL}" ]]; then
  echo "Missing manifest: ${MANIFEST_JSONL}" >&2
  exit 2
fi

TOTAL_EXAMPLES=$(wc -l < "${MANIFEST_JSONL}" | tr -d ' ')
TOTAL_SHARDS=$(( (TOTAL_EXAMPLES + SHARD_SIZE - 1) / SHARD_SIZE ))
if [[ -z "${END_SHARD}" ]]; then
  END_SHARD=$((TOTAL_SHARDS - 1))
fi
if (( START_SHARD < 0 || END_SHARD >= TOTAL_SHARDS || START_SHARD > END_SHARD )); then
  echo "Invalid shard range ${START_SHARD}-${END_SHARD}; total shards=${TOTAL_SHARDS}" >&2
  exit 2
fi

printf 'split=%s examples=%d shard_size=%d total_shards=%d range=%d-%d batch_shards=%d concurrency=%d\n' \
  "${SPLIT}" "${TOTAL_EXAMPLES}" "${SHARD_SIZE}" "${TOTAL_SHARDS}" "${START_SHARD}" "${END_SHARD}" \
  "${BATCH_SHARDS}" "${CONCURRENCY}"

check_free_space() {
  local batch_count=$1
  local free_kb free_gb estimated_gb free_after_gb
  free_kb=$(df -Pk "${STAGE_DIR}" | awk 'NR == 2 {print $4}')
  free_gb=$((free_kb / 1024 / 1024))
  # A shard is ~0.45 GiB for this Wan latent/action format. Use 1 GiB here to
  # leave room for logs, summaries, and filesystem accounting variance.
  estimated_gb=$((batch_count * 1))
  free_after_gb=$((free_gb - estimated_gb))
  if (( free_after_gb < MIN_FREE_GB )); then
    echo "Refusing batch: free=${free_gb}GiB estimated_batch=${estimated_gb}GiB would leave ${free_after_gb}GiB < MIN_FREE_GB=${MIN_FREE_GB}GiB" >&2
    exit 3
  fi
}

wait_for_job() {
  local job_id=$1
  while squeue -h -j "${job_id}" | grep -q .; do
    squeue -j "${job_id}" -o "%.18i %.9P %.30j %.8T %.10M %.6D %R"
    sleep 60
  done
  local states
  states=$(sacct -j "${job_id}" --format=State -P -n | sed '/^$/d' | sort -u | tr '\n' ' ')
  echo "job ${job_id} states: ${states}"
  if [[ "${states}" != *COMPLETED* ]] || [[ "${states}" == *FAILED* ]] || [[ "${states}" == *CANCELLED* ]] || [[ "${states}" == *TIMEOUT* ]] || [[ "${states}" == *OUT_OF_MEMORY* ]]; then
    sacct -j "${job_id}" --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS -P
    exit 4
  fi
}

submit_batch() {
  local batch_start=$1
  local batch_end=$2
  sbatch --parsable \
    --job-name="wan-tfr-${SPLIT}-${batch_start}-${batch_end}" \
    --partition="${PARTITION}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --time="${TIME_LIMIT}" \
    --output="${LOG_DIR}/wan_tfr_${SPLIT}_${batch_start}_${batch_end}_%j.out" \
    --export=ALL,REPO_DIR="${REPO_DIR}",VENV_DIR="${VENV_DIR}",CACHE_ROOT="${CACHE_ROOT}",MANIFEST_JSONL="${MANIFEST_JSONL}",STAGE_DIR="${STAGE_DIR}",SPLIT="${SPLIT}",SHARD_SIZE="${SHARD_SIZE}",TOTAL_SHARDS="${TOTAL_SHARDS}",TOTAL_EXAMPLES="${TOTAL_EXAMPLES}",BATCH_START="${batch_start}",BATCH_END="${batch_end}",CONCURRENCY="${CONCURRENCY}",ESTIMATE_SAMPLES="${ESTIMATE_SAMPLES}",MAX_OUTPUT_GB="${MAX_OUTPUT_GB}" <<'SBATCH'
#!/usr/bin/env bash
set -euo pipefail

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"
mkdir -p "${STAGE_DIR}"

export PYTHONUNBUFFERED=1
git rev-parse HEAD

launch_one() {
  local shard=$1
  local start=$((shard * SHARD_SIZE))
  local end=$((start + SHARD_SIZE))
  local shard_tag
  if (( end > TOTAL_EXAMPLES )); then
    end=${TOTAL_EXAMPLES}
  fi
  if (( start >= TOTAL_EXAMPLES )); then
    return 0
  fi
  shard_tag=$(printf "%05d" "${shard}")
  echo "[start] shard=${shard} indices=${start}:${end}"
  python src/maxdiffusion/data_preprocessing/wan_side_adapter_droid_cache_to_tfrecord.py \
    --cache-root "${CACHE_ROOT}" \
    --manifest-jsonl "${MANIFEST_JSONL}" \
    --require-manifest \
    --output-dir "${STAGE_DIR}" \
    --shard-prefix "${SPLIT}" \
    --shard-offset "${shard}" \
    --total-shards "${TOTAL_SHARDS}" \
    --shard-size "${SHARD_SIZE}" \
    --start-index "${start}" \
    --end-index "${end}" \
    --estimate-samples "${ESTIMATE_SAMPLES}" \
    --max-output-gb "${MAX_OUTPUT_GB}" \
    --fail-fast \
    --summary-path "${STAGE_DIR}/summary-${SPLIT}-${shard_tag}.json"
  echo "[done] shard=${shard}"
}

for shard in $(seq "${BATCH_START}" "${BATCH_END}"); do
  launch_one "${shard}" &
  while (( $(jobs -pr | wc -l) >= CONCURRENCY )); do
    wait -n
  done
done

while (( $(jobs -pr | wc -l) > 0 )); do
  wait -n
done
SBATCH
}

upload_and_delete_batch() {
  local batch_start=$1
  local batch_end=$2
  local shard shard_name local_path dest_path missing=0
  for shard in $(seq "${batch_start}" "${batch_end}"); do
    if (( shard >= TOTAL_SHARDS )); then
      continue
    fi
    shard_name=$(printf "%s-%05d-of-%05d.tfrecord" "${SPLIT}" "${shard}" "${TOTAL_SHARDS}")
    local_path="${STAGE_DIR}/${shard_name}"
    dest_path="${DEST}/${shard_name}"
    if [[ ! -s "${local_path}" ]]; then
      echo "Missing staged shard: ${local_path}" >&2
      missing=1
      continue
    fi
    gsutil cp "${local_path}" "${dest_path}"
    rm -f "${local_path}" "${STAGE_DIR}/summary-${SPLIT}-$(printf "%05d" "${shard}").json"
  done
  if (( missing != 0 )); then
    exit 5
  fi
}

batch_start=${START_SHARD}
while (( batch_start <= END_SHARD )); do
  batch_end=$((batch_start + BATCH_SHARDS - 1))
  if (( batch_end > END_SHARD )); then
    batch_end=${END_SHARD}
  fi
  batch_count=$((batch_end - batch_start + 1))
  check_free_space "${batch_count}"
  job_id=$(submit_batch "${batch_start}" "${batch_end}")
  echo "submitted batch ${batch_start}-${batch_end} as job ${job_id}"
  wait_for_job "${job_id}"
  upload_and_delete_batch "${batch_start}" "${batch_end}"
  df -h "${STAGE_DIR}"
  batch_start=$((batch_end + 1))
done

summary=$(mktemp)
python - "${summary}" <<PY
import json, sys
value = {
    "dataset": "droid_wan_side_adapter",
    "split": "${SPLIT}",
    "source_cache_root": "${CACHE_ROOT}",
    "manifest_jsonl": "${MANIFEST_JSONL}",
    "converter_commit": "$(cd "${REPO_DIR}" && git rev-parse HEAD)",
    "selected_examples": ${TOTAL_EXAMPLES},
    "shard_size": ${SHARD_SIZE},
    "total_shards": ${TOTAL_SHARDS},
    "format": "tf.train.Example serialized in standard TFRecord",
    "features": {
        "z_i0": {"dtype": "float16", "shape": [48, 1, 12, 20], "encoding": "raw_bytes"},
        "z_video": {"dtype": "float16", "shape": [48, 9, 12, 20], "encoding": "raw_bytes"},
        "actions": {"dtype": "float32", "shape": [32, 7], "encoding": "raw_bytes"},
        "name": {"dtype": "bytes"},
        "ordinal": {"dtype": "int64"},
    },
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(value, f, indent=2, sort_keys=True)
    f.write("\\n")
PY
gsutil cp "${summary}" "${DEST}/summary.json"
rm -f "${summary}"

actual_count=$(gsutil ls "${DEST}/${SPLIT}-*.tfrecord" | wc -l | tr -d ' ')
if (( START_SHARD == 0 && END_SHARD == TOTAL_SHARDS - 1 )); then
  if (( actual_count != TOTAL_SHARDS )); then
    echo "Expected ${TOTAL_SHARDS} GCS shards, found ${actual_count}" >&2
    exit 6
  fi
else
  expected_range_count=$((END_SHARD - START_SHARD + 1))
  if (( actual_count < expected_range_count )); then
    echo "Expected at least ${expected_range_count} GCS shards for partial range, found ${actual_count}" >&2
    exit 6
  fi
fi

echo "completed ${SPLIT}: ${actual_count} shards at ${DEST}"
