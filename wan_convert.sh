#!/bin/bash
# Two-step submission:
#   1. sbatch wan_setup.sh          — builds index, split_metadata, action stats
#   2. sbatch --dependency=afterok:<setup_job_id> wan_convert.sh

#SBATCH --job-name=wan_convert
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=60:00:00
#SBATCH --output=slurm_outputs/%x/out_%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu
#SBATCH --exclude=neu301,neu306,neu309,neu312
#SBATCH --array=0-7   # 8 parallel jobs; adjust N_JOBS below to match

cd /n/fs/cat10301/projects/maxdiffusion
source .venv/bin/activate

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# --- array job config ---
TOTAL_EPISODES=73393
N_JOBS=8
NO_RECORDS_PER_SHARD=10
# Round CHUNK up to a multiple of NO_RECORDS_PER_SHARD so that START values are
# always shard-aligned — the Python script snaps start_episode to the shard
# boundary, so non-aligned CHUNKs would cause jobs to overlap.
_RAW_CHUNK=$(( (TOTAL_EPISODES + N_JOBS - 1) / N_JOBS ))
CHUNK=$(( ( (_RAW_CHUNK + NO_RECORDS_PER_SHARD - 1) / NO_RECORDS_PER_SHARD ) * NO_RECORDS_PER_SHARD ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

# Last job encodes everything from its start to end of dataset.
MAX_EP=${CHUNK}

if [ "${SLURM_ARRAY_TASK_ID}" = "$(( N_JOBS - 1 ))" ]; then
    MAX_EP=-1
fi

python src/maxdiffusion/data_preprocessing/wan2.2_txt2vid_data_preprocessing.py \
    src/maxdiffusion/configs/base_wan_ctrl_world.yml \
    pretrained_model_name_or_path=model/Wan2.2-TI2V-5B-Diffusers \
    raw_data_root=/n/fs/iromdata/droid_raw/1.0.1 \
    data_root=/n/fs/iromdata/droid_wan2.2_lowres \
    video_index_path=/n/fs/iromdata/droid_wan2.2_lowres/video_index.json \
    action_stats_path=/n/fs/iromdata/droid_wan2.2_lowres/stats.json \
    no_records_per_shard=${NO_RECORDS_PER_SHARD} \
    max_frames=300 \
    max_episodes=${MAX_EP} \
    start_episode=${START} \
    height=480 \
    width=832 \
    hardware=gpu \
    val_fraction=0.05

# --- single-job test run (comment out array lines above and uncomment below) ---
# python src/maxdiffusion/data_preprocessing/wan2.2_txt2vid_data_preprocessing.py \
#     src/maxdiffusion/configs/base_wan_ctrl_world.yml \
#     pretrained_model_name_or_path=model/Wan2.2-TI2V-5B-Diffusers \
#     raw_data_root=/n/fs/iromdata/droid_raw/1.0.1 \
#     data_root=droid_wan_tfrecords_test \
#     video_index_path=/n/fs/iromdata/droid_wan2.2/video_index.json \
#     action_stats_path=/n/fs/iromdata/droid_wan2.2/stats.json \
#     no_records_per_shard=10 \
#     max_frames=300 \
#     max_episodes=210 \
#     start_episode=0 \
#     height=480 \
#     width=832 \
#     hardware=gpu \
#     val_fraction=0.05
