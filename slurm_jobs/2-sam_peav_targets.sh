#!/bin/bash
#SBATCH --job-name=sam_peav_targets
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1     
#SBATCH --mem=24GB
#SBATCH --time=8:00:00
#SBATCH --output=logs/sam_peav_targets/job_%j/slurm_main.out
#SBATCH --error=logs/sam_peav_targets/job_%j/slurm_main.err
#SBATCH --mail-type=END,FAIL      
#SBATCH --mail-user=dal696598@utdallas.edu

# ================= CONFIGURATION =================

# 1. Project Paths (Source)
RUN_ID="job_${SLURM_JOB_ID}"
PROJECT_DIR="$HOME/work/Echo-ViLD"
IMAGES_DIR="/home/dal696598/scratch/echo-vild/coco_subset_100/train2017"
SAM_CHECKPOINT="/home/dal696598/scratch/echo-vild/sam_vit_l_0b3195.pth"
FINAL_LOGS_BASE="$PROJECT_DIR/logs/sam_peav_targets/$RUN_ID"

# 2. Scratch Setup
JOB_SCRATCH="$HOME/scratch/echo-vild/sam_peav_targets/$RUN_ID"
SCRATCH_LOGS="$JOB_SCRATCH/logs/$RUN_ID"
SCRATCH_OUTPUT_DIR="$JOB_SCRATCH/sam_peav_outputs"

mkdir -p "$SCRATCH_LOGS"
mkdir -p "$SCRATCH_OUTPUT_DIR"

# ================= EXECUTION =================

module load miniconda
source ~/.bashrc
conda activate sam-peav

cd $PROJECT_DIR
echo "Working Directory: $(pwd)"

echo "--- Starting SAM PEAV Targets Generation ---"

python $PROJECT_DIR/offline_prep/generate_sam_peav_targets.py \
    --images_dir "$IMAGES_DIR" \
    --sam_checkpoint "$SAM_CHECKPOINT" \
    --output_dir "$SCRATCH_OUTPUT_DIR" \
    --min_pred_score=0.3 \
    --peav_batch_size=8 \
    --sam_batch_size=4

echo "Syncing logs..."
# rsync -avz "$SCRATCH_LOGS/" "$FINAL_LOGS_BASE/"

echo "Job Done!"