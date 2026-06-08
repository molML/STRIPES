#!/bin/bash
#SBATCH --job-name=stripes2smiles-pretrain
#SBATCH --output=/home/%u/STRIPES/Code_stripes/STRIPES2SMILES/logs/pretrain-%j.out
#SBATCH --error=/home/%u/STRIPES/Code_stripes/STRIPES2SMILES/logs/pretrain-%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH -t 1-00:00:00

# ── Paths ──────────────────────────────────────────────────────────────────────
CONDA_ENV=/home/$USER/miniconda3/envs/stripes
CODE_DIR=/home/$USER/STRIPES/Code_stripes/STRIPES2SMILES
DATA_PATH=/home/$USER/STRIPES/Code_stripes/data/MISATO
OUTPUT_DIR=/home/$USER/STRIPES/Code_stripes/STRIPES2SMILES/results_pretrain

mkdir -p $OUTPUT_DIR

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU(s):    $CUDA_VISIBLE_DEVICES"
echo "Start:     $(date)"

$CONDA_ENV/bin/python -u $CODE_DIR/run_pretraining.py \
    --data_path  $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size 32 \
    --num_epochs 100 \
    --d_model    512 \
    --n_heads    8 \
    --n_layers   8 \
    --dim_ff     2048 \
    --dropout    0.1 \
    --max_len    1200 \
    --lr         1e-4 \
    --seed       42

echo "End: $(date)"
