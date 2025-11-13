#!/bin/bash
#SBATCH --job-name=eval_xtalnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_ccsg_%j.log
#SBATCH --error=logs/eval_ccsg_%j.err

# Job info
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

# Go to project
cd /public/home/huangtianqi/XtalNet
mkdir -p logs results

# ============================================================
# Part 1: Generate crystal structures from PXRD
# ============================================================
echo ""
echo "Generating crystal structures..."
echo "This will:"
echo "  - Load CCSG model (structure generator)"
echo "  - Load CPCP model (PXRD encoder)"
echo "  - Generate 10 structure samples for test set indices 0-9"
echo "  - Each sample takes ~20 seconds on V100"
echo ""

python scripts/evaluate_ccsg.py \
    --ccsg_ckpt_path /public/home/huangtianqi/XtalNet/ckpt/hmof_100/CCSG/hmof_100_ccsg.ckpt \
    --cpcp_ckpt_path /public/home/huangtianqi/XtalNet/ckpt/hmof_100/CPCP/hmof_100_cpcp.ckpt \
    --save_path results/test_eval \
    --label test_run \
    --num_evals 10 \
    --begin_idx 0 \
    --end_idx 10

# Check if generation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "Generation completed! Results saved to: results/test_eval"
    echo ""
    
    # ============================================================
    # Part 2: Compute metrics
    # ============================================================
    echo "Computing evaluation metrics..."
    echo "This will calculate:"
    echo "  - Match Rate: % of correctly predicted structures"
    echo "  - RMSD: Root mean square deviation"
    echo "  - Coverage: Property prediction accuracy"
    echo ""
    
    python scripts/compute_ccsg_metrics.py \
        --root_path results/test_eval \
        --save_path results/test_eval/metrics \
        --multi_eval \
        --label test_run
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================"
        echo "Evaluation completed successfully!"
        echo "Check results at: results/test_eval"
        echo "========================================"
    else
        echo "Metrics computation failed!"
    fi
else
    echo "Generation failed! Check logs above."
    exit 1
fi

echo ""
echo "Job finished at: $(date)"

