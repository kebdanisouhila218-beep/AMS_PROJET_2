#!/bin/bash
#SBATCH --job-name=lstm_generate
#SBATCH --output=generate_2%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

echo "=== GPU disponible ==="
nvidia-smi

echo "=== Debut generation ==="
cd $SLURM_SUBMIT_DIR

python3 generate_lstm.py

echo "=== Fin generation ==="