#!/bin/bash
#SBATCH --job-name=eval_ccsg_with_rag
#SBATCH --output=logs/slurm/eval_ccsg_with_rag_%j.out
#SBATCH --error=logs/slurm/eval_ccsg_with_rag_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu          # 请根据你的分区修改
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1            # 评估通常只需要1个GPU
#SBATCH --qos=default           # 请根据你的QoS设置修改
#SBATCH --dependency=afterok:<build_db_job_id>  # 等待数据库构建完成

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PROJECT_ROOT=/path/to/your/RAG-XtalNet  # 请修改为你的路径

# 创建必要的目录
mkdir -p logs/slurm
mkdir -p outputs/evaluation

# 加载模块（根据你的集群修改）
# module load cuda/11.8
# module load python/3.9
# module load conda
# conda activate xtalnet

# 进入项目目录
cd ${PROJECT_ROOT}

# 检查点路径（请修改为实际路径）
export cpcp_ckpt_path="${PROJECT_ROOT}/outputs/slurm/cpcp_hmof100_train/checkpoints/last.ckpt"
export ccsg_ckpt_path="${PROJECT_ROOT}/outputs/slurm/ccsg_hmof100_train/checkpoints/last.ckpt"

# 数据库路径
export retrieval_db_npz="${PROJECT_ROOT}/outputs/retrieval/hmof_100_train_db.npz"
export retrieval_index="${PROJECT_ROOT}/outputs/retrieval/hmof_100_train_pxrd.index"

# 检查文件是否存在
if [ ! -f "$cpcp_ckpt_path" ]; then
    echo "错误: CPCP检查点文件不存在: $cpcp_ckpt_path"
    exit 1
fi

if [ ! -f "$ccsg_ckpt_path" ]; then
    echo "错误: CCSG检查点文件不存在: $ccsg_ckpt_path"
    exit 1
fi

if [ ! -f "$retrieval_db_npz" ]; then
    echo "错误: 检索数据库文件不存在: $retrieval_db_npz"
    exit 1
fi

if [ ! -f "$retrieval_index" ]; then
    echo "错误: 检索索引文件不存在: $retrieval_index"
    exit 1
fi

# RAG参数
export rag_top_m=4
export rag_strength=1.0

# 评估参数
export save_path="${PROJECT_ROOT}/outputs/evaluation/ccsg_with_rag"
export label="with_rag"
export num_evals=10
export begin_idx=0
export end_idx=10

echo "开始CCSG RAG增强评估..."
echo "CPCP检查点: ${cpcp_ckpt_path}"
echo "CCSG检查点: ${ccsg_ckpt_path}"
echo "检索数据库: ${retrieval_db_npz}"
echo "RAG模板数: ${rag_top_m}"
echo "RAG强度: ${rag_strength}"
echo "保存路径: ${save_path}"
echo "评估样本数: ${num_evals}"
echo "开始时间: $(date)"

# 运行RAG增强评估
python scripts/evaluate_ccsg.py \
    --ccsg_ckpt_path "${ccsg_ckpt_path}" \
    --cpcp_ckpt_path "${cpcp_ckpt_path}" \
    --save_path "${save_path}" \
    --label "${label}" \
    --num_evals "${num_evals}" \
    --begin_idx "${begin_idx}" \
    --end_idx "${end_idx}" \
    --retrieval_db_npz "${retrieval_db_npz}" \
    --retrieval_index "${retrieval_index}" \
    --rag_top_m "${rag_top_m}" \
    --rag_strength "${rag_strength}"

# 计算指标
echo "计算评估指标..."
python scripts/compute_ccsg_metrics.py \
    --root_path "${save_path}" \
    --save_path "${save_path}" \
    --multi_eval \
    --label "${label}"

echo "RAG增强评估完成时间: $(date)"