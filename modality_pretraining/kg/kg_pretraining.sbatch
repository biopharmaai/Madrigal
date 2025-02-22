#!/bin/bash
#SBATCH -J test
#SBATCH -o kg.out
#SBATCH -e kg.err
#SBATCH -c 1
#SBATCH -t 3:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

cd /path/to/Madrigal/modality_pretraining/kg
source activate primekg

python kg_pretraining.py
