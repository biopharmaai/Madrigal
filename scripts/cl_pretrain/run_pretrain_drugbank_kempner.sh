#!/bin/bash
#SBATCH -J pretrain
#SBATCH -o /n/home09/yeh803/workspace/NovelDDI/out/%x_%j.out
#SBATCH -e /n/home09/yeh803/workspace/NovelDDI/out/%x_%j.err
#SBATCH -c 2
#SBATCH -t 2-00:00
#SBATCH --account=kempner_mzitnik_lab -p kempner,kempner_requeue
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

base="/n/home09/yeh803/workspace/NovelDDI"
source activate primekg
cd $base

python pretrain.py --from_yaml configs/cl_pretrain/pretrain_drugbank.yaml

