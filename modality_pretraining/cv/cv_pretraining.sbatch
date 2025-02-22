#!/bin/bash
#SBATCH -J test
#SBATCH -o cv.out
#SBATCH -e cv.err
#SBATCH -c 1
#SBATCH -t 5:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=6G

cd /path/to/Madrigal/modality_pretraining/cv
source activate primekg

python cv_pretraining.py
