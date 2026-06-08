#!/bin/bash
#SBATCH --job-name=stripes2smiles-generation
#SBATCH --output=/home/%u/STRIPES/Code_stripes/STRIPES2SMILES/logs/generation-%j.out
#SBATCH --error=/home/%u/STRIPES/Code_stripes/STRIPES2SMILES/logs/generation-%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH -t 2-00:00:00

# ── Paths ──────────────────────────────────────────────────────────────────────
CONDA_ENV=/home/$USER/miniconda3/envs/stripes
CODE_DIR=/home/$USER/STRIPES/Code_stripes/STRIPES2SMILES
PRETRAIN_DIR=/home/$USER/STRIPES/Code_stripes/STRIPES2SMILES/results_pretrain
RESULTS_DIR=/home/$USER/STRIPES/Code_stripes/STRIPES2SMILES/results_finetune

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU(s):    $CUDA_VISIBLE_DEVICES"
echo "Start:     $(date)"

$CONDA_ENV/bin/python -u $CODE_DIR/extract_test_and_generate.py \
    --pretrained_model $PRETRAIN_DIR/pretrained_stripes_encoder.pth \
    --pretrained_vocab $PRETRAIN_DIR/stripes_vocab.pkl \
    --results_dir      $RESULTS_DIR \
    --datasets         PPAR \
    --beam_sizes       5 10 15 \
    --n_molecules      5 10 \
    --temperatures     1.2 1.4 \
    --increments       -0.2 -0.3 -0.4 -0.5 \
    --max_attempts     2

echo "End: $(date)"
