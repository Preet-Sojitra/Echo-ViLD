#!/bin/bash
#SBATCH --job-name=extract_maskrnn_features
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1     
#SBATCH --mem=16GB
#SBATCH --time=8:00:00
#SBATCH --output=logs/mask_rnn_features/job_%j/slurm_main.out
#SBATCH --error=logs/mask_rnn_features/job_%j/slurm_main.err
#SBATCH --mail-type=END,FAIL      # Send email on job end or failure
#SBATCH --mail-user=dal696598@utdallas.edu

# ================= CONFIGURATION =================

# 1. Project Paths (Source)
RUN_ID="job_${SLURM_JOB_ID}"
PROJECT_DIR="$HOME/work/Echo-ViLD"
COCO_IMG_DIR="/home/dal696598/scratch/coco_subset_100/train2017"
ANNOTATION_FILE="/home/dal696598/scratch/coco_subset_100/annotations/instances_train2017.json"
FINAL_LOGS_BASE="$PROJECT_DIR/logs/mask_rnn_features/$RUN_ID"

# 2. Scratch Setup
JOB_SCRATCH="$HOME/scratch/mask_rnn_features/$RUN_ID"
SCRATCH_LOGS="$JOB_SCRATCH/logs/$RUN_ID"
SCRATCH_OUTPUT_DIR="$JOB_SCRATCH/Bboxes_and_256D_features"

mkdir -p "$SCRATCH_LOGS"
mkdir -p "$SCRATCH_OUTPUT_DIR"

# ================= EXECUTION =================

module load miniconda
source ~/.bashrc
conda activate echo-vild

cd $PROJECT_DIR
echo "Working Directory: $(pwd)"

echo "--- Starting Mask R-CNN Feature Extraction ---"
python $PROJECT_DIR/offline_prep/extract_maskrnn_features.py \
    --coco_img_dir "$COCO_IMG_DIR" \
    --ann_file "$ANNOTATION_FILE" \
    --output_dir "$SCRATCH_OUTPUT_DIR" \
    --max_proposals 300 \
    --max_images 10

echo "Syncing logs..."
rsync -avz "$SCRATCH_LOGS/" "$FINAL_LOGS_BASE/"

echo "Job Done!"