#!/bin/bash
#SBATCH --job-name=transformer_generate
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --output=./logs/%x_%A.out
#SBATCH --error=./logs/%x_%A.err

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
mkdir -p logs runs_2

cd $SLURM_SUBMIT_DIR

echo "Lancement generate_passwords.py..."
echo "Heure debut : $(date)"

srun python3 -u generate_passwords.py \
    --model  transformer_best_model.h5 \
    --vocab  vocabulary_transformer.json \
    --train  ../../Data/train.txt \
    --eval   ../../Data/eval.txt \
    --outdir runs_2 \
    --batch  2048 \
    2>&1 | tee logs/generation_2.log

EXIT_CODE=$?
echo "Heure fin : $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Generation terminee avec succes !"
    echo "Fichiers generes :"
    ls -lh runs_2/*.txt
else
    echo "Erreur lors de la generation (code: $EXIT_CODE)"
fi