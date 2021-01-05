#!/bin/bash
## SLURM scripts have a specific format. 
#SBATCH --job-name=xlmr_layer_pruning
#SBATCH --partition=learnfair
#SBATCH --time=4320
#SBATCH --nodes=1
#SBATCH --signal=USR1@120
#SBATCH --cpus-per-task=10
#SBATCH --constraint=volta32gb
#SBATCH --gpus-per-node=1
#SBATCH --output=/checkpoint/%u/QE_KD/logs/%A_%a.out
#SBATCH --error=/checkpoint/%u/QE_KD/logs/%A_%a.err

/private/home/ssfei81/miniconda3/bin/activate
cd /checkpoint/ssfei81/QE_KD
#config_file=$(ls configs/models_alternate/*/*/config.json | sed -n ${SLURM_ARRAY_TASK_ID}p)
config_file=$(ls new_configs2/xlmr_*/ordinal_regression/*/run*/config.json | sed -n ${SLURM_ARRAY_TASK_ID}p)

python train_teacher.py --config_file $config_file --num_gpus=1
#python evaluate_mini.py --config_file $config_file --num_gpus=1 > $(echo $config_file | sed -e "s/.json/.eval.out/g")
