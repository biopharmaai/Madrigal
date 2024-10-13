#!/bin/bash
#SBATCH -J finetune
#SBATCH -o /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.out
#SBATCH -e /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.err
#SBATCH -c 2
#SBATCH -t 16:00:00
#SBATCH -p gpu_quad,gpu_requeue
#SBATCH --qos=gpuquad_qos
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1
#SBATCH -x compute-g-17-147,compute-g-17-148,compute-g-17-149,compute-g-17-150,compute-g-17-151,compute-g-17-152
#SBATCH --mem=16G

base="/home/yeh803/workspace/DDI/NovelDDI"
split_method="split_by_pairs"
repeat_num=None

config_file="configs/ddi_finetune/TWOSIDES/sweep_config_hardy_sweep_321.yaml"
checkpoint="2024-02-06_18:12_helpful-field-81/checkpoint_1000.pt"  # hardy-sweep-321 (corresponding new run)
full_finetune_mode="str_random_sample"  # hardy-sweep-321

seed=99

source activate primekg
cd $base

# Full
python train_ddi_batch.py --checkpoint=$checkpoint --finetune_mode=$full_finetune_mode --split_method=$split_method --repeat=$repeat_num --seed=$seed --from_yaml=$config_file

# Full without finetuning
python train_ddi_batch.py --finetune_mode=$full_finetune_mode --split_method=$split_method --repeat=$repeat_num --seed=$seed --from_yaml=$config_file

# Structure ablation without finetuning
python train_ddi_batch.py --finetune_mode="ablation_str_str" --split_method=$split_method --repeat=$repeat_num --seed=$seed --from_yaml=$config_file

# Structure ablation with finetuning
python train_ddi_batch.py --checkpoint=$checkpoint --finetune_mode="ablation_str_str" --split_method=$split_method --repeat $repeat_num --seed=$seed --from_yaml $config_file


# -p gpu_quad,gpu_requeue
# --qos=gpuquad_qos
# --gres=gpu:a100:1
# --requeue

