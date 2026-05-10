#!/bin/bash
#SBATCH --job-name=lstm_train
#SBATCH --output=lstm_%j.log
#SBATCH --error=lstm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu

echo "=========================================="
echo "Job SLURM démarré"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "=========================================="

echo ""
echo "=== Informations GPU ==="
nvidia-smi

echo ""
echo "=== Variables d'environnement ==="
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

echo ""
echo "=== Début de l'entraînement ==="
cd $SLURM_SUBMIT_DIR

# Afficher le contenu du dossier
echo "Contenu du dossier:"
ls -lh

echo ""
echo "Lancement de train.py..."
echo "Heure de début: $(date)"

python3 train.py 2>&1 | tee training_output_2.log

EXIT_CODE=$?

echo ""
echo "Heure de fin: $(date)"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Entraînement terminé avec succès!"
else
    echo "✗ Erreur lors de l'entraînement (code: $EXIT_CODE)"
fi

echo "=========================================="