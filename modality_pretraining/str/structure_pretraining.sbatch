#!/bin/bash
#SBATCH -J test
#SBATCH -o str.out
#SBATCH -e str.err
#SBATCH -c 1
#SBATCH -t 3:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=15G

source activate primekg

cd /path/to/Madrigal/modality_pretraining/str/

python structure_pretraining_muv.py 
