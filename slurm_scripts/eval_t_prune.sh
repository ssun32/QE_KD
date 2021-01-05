#!/bin/bash
## SLURM scripts have a specific format. 
#SBATCH --job-name=xlmr_layer_pruning
#SBATCH --partition=learnfair
#SBATCH --time=20
#SBATCH --nodes=1
#SBATCH --signal=USR1@120
#SBATCH --cpus-per-task=10
#SBATCH --constraint=volta32gb
#SBATCH --gpus-per-node=1
#SBATCH --output=/checkpoint/%u/QE_KD/logs/%j.out
#SBATCH --error=/checkpoint/%u/QE_KD/logs/%j.err

/private/home/ssfei81/miniconda3/bin/activate
cd /checkpoint/ssfei81/QE_KD
#config_file=$(ls configs/models_cp_drop*/*/*/config.json | sed -n ${SLURM_ARRAY_TASK_ID}p)
#config_file=$(ls configs/models/*/*/config.json | sed -n ${SLURM_ARRAY_TASK_ID}p)
config_file=configs/models_linformer/all/run0/config.json

prune_layer=$1
prune_head=$2

python evaluate_mini.py --config_file $config_file --num_gpus=1 --prune_layer $prune_layer --prune_head $prune_head > /checkpoint/ssfei81/QE_KD/prune_experiments/${prune_layer}_${prune_head}.txt
