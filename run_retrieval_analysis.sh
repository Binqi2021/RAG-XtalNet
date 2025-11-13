#!/bin/bash
#SBATCH --job-name=retrieval_analysis
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/retrieval_analysis_%j.log
#SBATCH --error=logs/retrieval_analysis_%j.err

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "========================================"

# Activate environment (proper way for SLURM)
source ~/.bashrc
source $(conda info --base)/etc/profile.d/conda.sh
conda activate xtalnet

# Check environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Run analysis
cd /public/home/huangtianqi/XtalNet
python retrieval_experiments/analyze_retrieval.py

echo "========================================"
echo "Finished at: $(date)"
echo "========================================"
