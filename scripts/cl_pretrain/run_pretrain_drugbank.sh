#!/bin/bash
#SBATCH -J pretrain
#SBATCH -o /path/to/Madrigal/out/%x_%j.out
#SBATCH -e /path/to/Madrigal/out/%x_%j.err
#SBATCH -c 2
#SBATCH -t 2-00:00
#SBATCH -p gpu_quad
#SBATCH --qos=gpuquad_qos
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

base="/path/to/Madrigal"
source activate primekg
cd $base

python pretrain.py --from_yaml configs/cl_pretrain/pretrain_drugbank_basal.yaml
