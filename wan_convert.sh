#!/bin/bash
#SBATCH --job-name=wan_data_convert  # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --gres=gpu:1                 # Number of GPUs
#SBATCH --ntasks-per-node=4          # Number of tasks (processes)
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=40G                    # Memory per node
#SBATCH --time=24:00:00              # Time limit (hh:mm:ss)
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu
#SBATCH --exclude=neu301,neu306,neu309,neu312

cd /n/fs/cat10301/projects/maxdiffusion

source .venv/bin/activate

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

python src/maxdiffusion/data_preprocessing/wan2.2_txt2vid_data_preprocessing.py \
    src/maxdiffusion/configs/base_wan_ctrl_world.yml \
    pretrained_model_name_or_path=model/Wan2.2-TI2V-5B-Diffusers \
    data_root=/n/fs/iromdata/droid_ctrl_world \
    tfrecords_dir=droid_wan_tfrecords_test \
    train_split=train \
    no_records_per_shard=50 \
    action_stats_path=action_stats_test.json \
    max_frames=300 \
    max_episodes=200