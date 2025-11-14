#!/bin/bash
#SBATCH --job-name=build_retrieval_db
#SBATCH --output=logs/slurm/build_retrieval_%j.out
#SBATCH --error=logs/slurm/build_retrieval_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu          # 请根据你的分区修改
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1            # 构建数据库通常只需要1个GPU
#SBATCH --qos=default           # 请根据你的QoS设置修改

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PROJECT_ROOT=/path/to/your/RAG-XtalNet  # 请修改为你的路径

# 创建必要的目录
mkdir -p logs/slurm
mkdir -p outputs/retrieval

# 加载模块（根据你的集群修改）
# module load cuda/11.8
# module load python/3.9
# module load conda
# conda activate xtalnet

# 进入项目目录
cd ${PROJECT_ROOT}

# CPCP检查点路径（请修改为实际路径）
export cpcp_ckpt_path="${PROJECT_ROOT}/outputs/slurm/cpcp_hmof100_train/checkpoints/last.ckpt"

# 检查CPCP检查点是否存在
if [ ! -f "$cpcp_ckpt_path" ]; then
    echo "错误: CPCP检查点文件不存在: $cpcp_ckpt_path"
    echo "请先运行CPCP训练作业"
    exit 1
fi

echo "开始构建检索数据库..."
echo "CPCP检查点: ${cpcp_ckpt_path}"
echo "数据集: hmof_100"
echo "GPU数量: 1"
echo "开始时间: $(date)"

# 构建hmof_100训练集数据库
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path "${cpcp_ckpt_path}" \
    --data_name hmof_100 \
    --split train \
    --device cuda

# 构建hmof_100验证集数据库
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path "${cpcp_ckpt_path}" \
    --data_name hmof_100 \
    --split val \
    --device cuda

# 构建hmof_100测试集数据库
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path "${cpcp_ckpt_path}" \
    --data_name hmof_100 \
    --split test \
    --device cuda

# 检查输出文件
db_files=(
    "outputs/retrieval/hmof_100_train_db.npz"
    "outputs/retrieval/hmof_100_train_pxrd.index"
    "outputs/retrieval/hmof_100_val_db.npz"
    "outputs/retrieval/hmof_100_val_pxrd.index"
    "outputs/retrieval/hmof_100_test_db.npz"
    "outputs/retrieval/hmof_100_test_pxrd.index"
)

echo "检查输出文件..."
for file in "${db_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ $file (文件不存在)"
    fi
done

echo "数据库构建完成时间: $(date)"