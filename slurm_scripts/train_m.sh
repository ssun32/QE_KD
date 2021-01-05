#!/bin/bash
## SLURM scripts have a specific format. 
#SBATCH --job-name=xlmr_layer_pruning
#SBATCH --partition=learnfair
#SBATCH --time=2880
#SBATCH --nodes=1
#SBATCH --signal=USR1@120
#SBATCH --cpus-per-task=20
#SBATCH --constraint=volta32gb
#SBATCH --gpus-per-node=8
#SBATCH --output=/checkpoint/%u/QE_KD/logs/%j.out
#SBATCH --error=/checkpoint/%u/QE_KD/logs/%j.err

/private/home/ssfei81/miniconda3/bin/activate
cd /checkpoint/ssfei81/QE_KD
config_file=$(ls configs/models_linformer/*/run4/config.json | sed -n ${SLURM_ARRAY_TASK_ID}p)
#config_file=$(ls configs/models_kd/*/run4/config.json | sed -n ${SLURM_ARRAY_TASK_ID}p)

#config_file=$(ls configs/models/*/*/kd_config_*.json | sed -n ${SLURM_ARRAY_TASK_ID}p)
python train_mini.py --config_file $config_file --num_gpus=8

