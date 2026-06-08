#!/bin/bash
#SBATCH --job-name=stripes2smiles-finetune
#SBATCH --output=/home/%u/STRIPES/Code_stripes/STRIPES2SMILES/logs/finetune-%j.out
#SBATCH --error=/home/%u/STRIPES/Code_stripes/STRIPES2SMILES/logs/finetune-%j.err
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH -t 4-00:00:00

# ── Paths ──────────────────────────────────────────────────────────────────────
CONDA_ENV=/home/$USER/miniconda3/envs/stripes
CODE_DIR=/home/$USER/STRIPES/Code_stripes/STRIPES2SMILES
DATA_PATH=/home/$USER/STRIPES/Code_stripes/data
PRETRAIN_DIR=/home/$USER/STRIPES/Code_stripes/STRIPES2SMILES/results_pretrain
OUTPUT_DIR=/home/$USER/STRIPES/Code_stripes/STRIPES2SMILES/results_finetune

mkdir -p $OUTPUT_DIR

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU(s):    $CUDA_VISIBLE_DEVICES"
echo "Start:     $(date)"

$CONDA_ENV/bin/python -u $CODE_DIR/run_finetuning.py \
    --data_path        $DATA_PATH \
    --pretrained_model $PRETRAIN_DIR/pretrained_stripes_encoder.pth \
    --pretrained_vocab $PRETRAIN_DIR/stripes_vocab.pkl \
    --output_dir       $OUTPUT_DIR \
    --datasets         PPAR \
    --n_trials         200 \
    --seed             42

echo "End: $(date)"
