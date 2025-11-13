#!/bin/bash
#SBATCH --job-name=xtalnet_demo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gradio_%j.log
#SBATCH --error=logs/gradio_%j.err

# Job info
echo "========================================"
echo "Starting XtalNet Gradio Demo"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "========================================"

# Activate environment
source ~/.bashrc
conda activate xtalnet

# Go to project
cd /public/home/huangtianqi/XtalNet
mkdir -p logs gradio_outputs

# Get node IP for remote access
NODE_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "========================================"
echo "Gradio will be accessible at:"
echo "  http://${NODE_IP}:7860"
echo "========================================"
echo ""

# Run Gradio demo
python scripts/gradio_demo.py \
    --ccsg_ckpt_path ckpt/hmof_100/CCSG/hmof_100_ccsg.ckpt \
    --cpcp_ckpt_path ckpt/hmof_100/CPCP/hmof_100_cpcp.ckpt \
    --save_path gradio_outputs

echo ""
echo "Demo stopped at: $(date)"

