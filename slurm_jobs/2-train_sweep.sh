#!/bin/bash
#SBATCH --job-name=echo_vild_train_sweep
#SBATCH --partition=a30_4.6gb
#SBATCH --gres=nvidia_a30_1g.6gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_sweep/job_%j/slurm_main.out
#SBATCH --error=logs/train_sweep/job_%j/slurm_main.err

# ================= CONFIGURATION =================

RUN_ID="job_${SLURM_JOB_ID}"
PROJECT_DIR="$HOME/work/Echo-ViLD"

SCRATCH_LOGS="$HOME/scratch/train_sweep/$RUN_ID/logs"
mkdir -p "$SCRATCH_LOGS"
mkdir -p "$PROJECT_DIR/logs/train_sweep/$RUN_ID"

# ================= EXECUTION =================

module load miniconda
source ~/.bashrc
conda activate echo-vild

cd $PROJECT_DIR
echo "Working Directory: $(pwd)"

echo "--- Starting Echo-ViLD Training Sweep ---"
for cfg in train/configs/*.yaml; do
    echo "============================="
    echo "Training config: $cfg"
    echo "============================="
    python train/train_echo_vild.py --config "$cfg"
done

echo "Sweep complete. Checkpoints saved to weights/"
