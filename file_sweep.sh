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

cd /unimore_home/trossi/chronos2_pm25

python 03_run_scenario_y.py --sensors ari-2433 "sensor delhi" --configs 24:12 --target-mode lcs --steps 1000 --lr 1e-5 --tag sweep_lcs_1000_1e-5
python 03_run_scenario_y.py --sensors ari-2433 "sensor delhi" --configs 24:12 --target-mode lcs --steps 1000 --lr 5e-5 --tag sweep_lcs_1000_5e-5
python 03_run_scenario_y.py --sensors ari-2433 "sensor delhi" --configs 24:12 --target-mode nan --steps 1000 --lr 1e-5 --tag sweep_nan_1000_1e-5
python 03_run_scenario_y.py --sensors ari-2433 "sensor delhi" --configs 24:12 --target-mode nan --steps 1000 --lr 5e-5 --tag sweep_nan_1000_5e-5