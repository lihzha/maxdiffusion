#!/bin/bash
# Builds video_index.json, split_metadata.json, and action_stats.json.
# Run this once before launching wan_convert.sh:
#   SETUP_ID=$(sbatch --parsable wan_setup.sh)
#   sbatch --dependency=afterok:${SETUP_ID} wan_convert.sh

#SBATCH --job-name=wan_setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=slurm_outputs/%x/out_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu
#SBATCH --exclude=neu301,neu306,neu309,neu312

cd /n/fs/cat10301/projects/maxdiffusion
source .venv/bin/activate
export JAX_PLATFORMS=cpu

python src/maxdiffusion/data_preprocessing/wan2.2_txt2vid_data_preprocessing.py \
    src/maxdiffusion/configs/base_wan_ctrl_world.yml \
    pretrained_model_name_or_path=model/Wan2.2-TI2V-5B-Diffusers \
    raw_data_root=/n/fs/iromdata/droid_raw/1.0.1 \
    data_root=/n/fs/iromdata/droid_wan2.2 \
    video_index_path=/n/fs/iromdata/droid_wan2.2/video_index.json \
    action_stats_path=/n/fs/iromdata/droid_wan2.2/stats.json \
    val_fraction=0.05 \
    setup_only=True \
    hardware=cpu \
    skip_jax_distributed_system=True