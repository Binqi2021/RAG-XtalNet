#!/bin/bash
#SBATCH --job-name=build_rag_index
#SBATCH --partition=gpu              # GPU partition name (modify if needed)
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --cpus-per-task=8            # 8 CPU cores
#SBATCH --mem=32G                    # 32GB memory
#SBATCH --time=02:00:00              # Max 2 hours
#SBATCH --output=logs/build_index_%j.log
#SBATCH --error=logs/build_index_%j.err

# Print job info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started at: $(date)"
echo "========================================"

# Check GPU
nvidia-smi

# Activate environment
source ~/.bashrc
conda activate xtalnet

# Go to project directory
cd /public/home/huangtianqi/XtalNet

# Create logs directory
mkdir -p logs

# Test with small sample first
echo "Testing with 100 samples..."
python scripts/build_retrieval_index.py \
    --max_samples 100 \
    --output_dir retrieval_indices/test \
    --device cuda \
    --batch_size 32

# If successful, build full index
if [ $? -eq 0 ]; then
    echo "Test passed! Building full index..."
    python scripts/build_retrieval_index.py \
        --output_dir retrieval_indices/hmof_100 \
        --device cuda \
        --batch_size 64
    
    # Test retrieval quality
    if [ $? -eq 0 ]; then
        echo "Testing retrieval quality..."
        python scripts/test_retrieval.py \
            --index_path retrieval_indices/hmof_100 \
            --n_queries 100
    fi
else
    echo "Test failed! Check errors above."
    exit 1
fi

echo "========================================"
echo "Job finished at: $(date)"
echo "========================================"

