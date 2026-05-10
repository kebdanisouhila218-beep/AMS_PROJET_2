#!/bin/bash
#SBATCH --job-name=transformer_train
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --output=./logs/last_%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err

echo "=========================================="
echo "Job SLURM demarre"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Date     : $(date)"
echo "=========================================="

export CUDA_HOME=/usr/local/cuda-10.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

nvidia-smi

mkdir -p logs

cd $SLURM_SUBMIT_DIR

echo "Lancement train_transformer.py..."
echo "Heure debut : $(date)"

srun python3 train_transformer.py 2>&1 | tee logs/training_transformer_last.log

EXIT_CODE=$?

echo "Heure fin : $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Entrainement termine avec succes !"
else
    echo "Erreur lors de l entrainement (code: $EXIT_CODE)"
fi