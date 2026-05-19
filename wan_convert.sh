#!/bin/bash
#SBATCH --job-name=wan_data_convert  # Job name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --gres=gpu:1                 # Number of GPUs
#SBATCH --ntasks-per-node=4          # Number of tasks (processes)
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=80G                    # Memory per node
#SBATCH --time=30:00:00              # Time limit (hh:mm:ss)
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Output File
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=zz8976@princeton.edu
#SBATCH --exclude=neu301,neu306,neu309,neu312

cd /n/fs/cat10301/projects/maxdiffusion

source .venv/bin/activate

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

python src/maxdiffusion/data_preprocessing/wan2.2_txt2vid_data_preprocessing.py \
    src/maxdiffusion/configs/base_wan_ctrl_world.yml \
    pretrained_model_name_or_path=model/Wan2.2-TI2V-5B-Diffusers \
    raw_data_root=/n/fs/iromdata/droid_raw/1.0.1 \
    data_root=droid_wan_tfrecords_test_full \
    video_index_path=droid_wan_tfrecords_test/video_index.json \
    no_records_per_shard=10 \
    action_stats_path=action_stats_test.json \
    max_frames=300 \
    max_episodes=200 \
    start_episode=0 \
    hardware=gpu

# python src/maxdiffusion/data_preprocessing/wan2.2_txt2vid_data_preprocessing.py \
#     src/maxdiffusion/configs/base_wan_ctrl_world.yml \
#     pretrained_model_name_or_path=model/Wan2.2-TI2V-5B-Diffusers \
#     raw_data_root=/n/fs/iromdata/droid_raw/1.0.1 \
#     data_root=/n/fs/iromdata/droid_wan \
#     video_index_path=/n/fs/iromdata/droid_wan/video_index.json \
#     train_split=val \
#     no_records_per_shard=200 \
#     action_stats_path=/n/fs/iromdata/droid_wan/action_stats.json \
#     max_frames=300 \
#     max_episodes=-1