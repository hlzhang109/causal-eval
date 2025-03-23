#!/bin/bash

# Optionally put minimal Slurm directives here, or none at all
#SBATCH -J eval
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH -t 3-00:00
#SBATCH -p kempner #seas_compute
#SBATCH --mem=128GB # 256GB
#SBATCH --account=kempner_sham_lab # barak_lab
#SBATCH -o logs/eval_%j.out
#SBATCH -e logs/eval_%j.err

bash run_eval.sh