#!/bin/bash
#SBATCH -J normalize_scores
#SBATCH -o /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.out
#SBATCH -e /home/yeh803/workspace/DDI/NovelDDI/out/%x_%j.err
#SBATCH -c 2
#SBATCH -t 2-00:00
#SBATCH -p priority
#SBATCH --mem=160G

base="/home/yeh803/workspace/DDI/NovelDDI/notebooks"

source activate primekg
cd $base

python normalize_scores.py
