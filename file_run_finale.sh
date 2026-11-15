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

export HF_HUB_OFFLINE=1
cd /unimore_home/trossi/chronos2_pm25
python 03_run_scenario_y.py --target-mode cal --steps 1000 --lr 1e-5