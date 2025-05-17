#!/bin/bash

# Optionally put minimal Slurm directives here, or none at all
#SBATCH -J eval
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH -t 3-00:00
#SBATCH -p kempner_requeue # seas_gpu #
#SBATCH --mem=128GB # 256GB
#SBATCH --account=kempner_barak_lab # kempner_sham_lab #
#SBATCH -o logs/eval/eval_%j.out
#SBATCH -e logs/eval/eval_%j.err

bash run_eval.sh