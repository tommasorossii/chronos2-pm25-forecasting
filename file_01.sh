#!/bin/bash
#SBATCH -p gpuh100
#SBATCH -A laurapolab
#SBATCH --qos=lph100
#SBATCH --gres=gpu:1
#SBATCH --time 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=100G
#SBATCH --job-name=Multi_Span_QA

python 01_prepare_data.py