#!/bin/bash
#SBATCH -J pretrain
#SBATCH -o /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.out
#SBATCH -e /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.err
#SBATCH -c 2
#SBATCH -t 2-00:00
#SBATCH -p gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

base="/home/yeh803/workspace/DDI/NovelDDI"
source activate primekg
cd $base

python pretrain.py --from_yaml configs/cl_pretrain/pretrain_twosides_tx_basal.yaml

# -p gpu_quad,gpu_requeue
# --qos=gpuquad_qos
# --gres=gpu:a100:1
