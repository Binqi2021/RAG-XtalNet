#!/bin/bash
#SBATCH --job-name=rag_xtalnet_pipeline
#SBATCH --output=logs/slurm/pipeline_%j.out
#SBATCH --error=logs/slurm/pipeline_%j.err
#SBATCH --time=120:00:00      # 5天时间完成整个流程
#SBATCH --partition=gpu          # 请根据你的分区修改
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2            # 使用2个GPU
#SBATCH --qos=default           # 请根据你的QoS设置修改

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export PROJECT_ROOT=/path/to/your/RAG-XtalNet  # 请修改为你的路径
export HYDRA_JOBS=${PROJECT_ROOT}/outputs/slurm
export WANDB_DIR=${PROJECT_ROOT}/wandb

# 创建必要的目录
mkdir -p logs/slurm
mkdir -p outputs/slurm
mkdir -p outputs/retrieval
mkdir -p outputs/evaluation
mkdir -p wandb

# 加载模块（根据你的集群修改）
# module load cuda/11.8
# module load python/3.9
# module load conda
# conda activate xtalnet

# 进入项目目录
cd ${PROJECT_ROOT}

echo "=========================================="
echo "RAG-XtalNet 完整流程开始"
echo "开始时间: $(date)"
echo "=========================================="

# 步骤1: 训练CPCP模型
echo ""
echo "=========================================="
echo "步骤 1/5: 训练CPCP模型"
echo "=========================================="

export expname=cpcp_hmof100_train
export model=cpcp
export data_name='hmof_100'
export freeze=false
export bsz=16
export lr=5e-4

echo "训练参数:"
echo "  实验名称: ${expname}"
echo "  数据集: ${data_name}"
echo "  批次大小: ${bsz}"
echo "  学习率: ${lr}"

bash train.sh

# 检查CPCP训练结果
cpcp_ckpt_path="${PROJECT_ROOT}/outputs/slurm/cpcp_hmof100_train/checkpoints/last.ckpt"
if [ ! -f "$cpcp_ckpt_path" ]; then
    echo "错误: CPCP训练失败，检查点不存在"
    exit 1
fi
echo "✅ CPCP训练完成: ${cpcp_ckpt_path}"

# 步骤2: 训练CCSG模型
echo ""
echo "=========================================="
echo "步骤 2/5: 训练CCSG模型"
echo "=========================================="

export expname=ccsg_hmof100_train
export model=ccsg
export data_name='hmof_100'
export pretrained=$cpcp_ckpt_path
export freeze=true
export bsz=16
export lr=1e-3

echo "训练参数:"
echo "  实验名称: ${expname}"
echo "  预训练模型: ${pretrained}"
echo "  批次大小: ${bsz}"
echo "  学习率: ${lr}"

bash train.sh

# 检查CCSG训练结果
ccsg_ckpt_path="${PROJECT_ROOT}/outputs/slurm/ccsg_hmof100_train/checkpoints/last.ckpt"
if [ ! -f "$ccsg_ckpt_path" ]; then
    echo "错误: CCSG训练失败，检查点不存在"
    exit 1
fi
echo "✅ CCSG训练完成: ${ccsg_ckpt_path}"

# 步骤3: 构建检索数据库
echo ""
echo "=========================================="
echo "步骤 3/5: 构建检索数据库"
echo "=========================================="

echo "构建hmof_100数据库..."

# 构建训练集数据库
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path "${cpcp_ckpt_path}" \
    --data_name hmof_100 \
    --split train \
    --device cuda

# 构建验证集数据库
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path "${cpcp_ckpt_path}" \
    --data_name hmof_100 \
    --split val \
    --device cuda

# 构建测试集数据库
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path "${cpcp_ckpt_path}" \
    --data_name hmof_100 \
    --split test \
    --device cuda

# 检查数据库文件
db_files=(
    "outputs/retrieval/hmof_100_train_db.npz"
    "outputs/retrieval/hmof_100_train_pxrd.index"
    "outputs/retrieval/hmof_100_val_db.npz"
    "outputs/retrieval/hmof_100_val_pxrd.index"
    "outputs/retrieval/hmof_100_test_db.npz"
    "outputs/retrieval/hmof_100_test_pxrd.index"
)

echo "检查数据库文件..."
all_db_exist=true
for file in "${db_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (文件不存在)"
        all_db_exist=false
    fi
done

if [ "$all_db_exist" = false ]; then
    echo "错误: 数据库构建失败"
    exit 1
fi
echo "✅ 检索数据库构建完成"

# 步骤4: 无RAG评估
echo ""
echo "=========================================="
echo "步骤 4/5: 无RAG评估"
echo "=========================================="

export save_path="${PROJECT_ROOT}/outputs/evaluation/ccsg_no_rag"
export label="no_rag"
export num_evals=10
export begin_idx=0
export end_idx=10

python scripts/evaluate_ccsg.py \
    --ccsg_ckpt_path "${ccsg_ckpt_path}" \
    --cpcp_ckpt_path "${cpcp_ckpt_path}" \
    --save_path "${save_path}" \
    --label "${label}" \
    --num_evals "${num_evals}" \
    --begin_idx "${begin_idx}" \
    --end_idx "${end_idx}"

# 计算指标
python scripts/compute_ccsg_metrics.py \
    --root_path "${save_path}" \
    --save_path "${save_path}" \
    --multi_eval \
    --label "${label}"

echo "✅ 无RAG评估完成"

# 步骤5: RAG增强评估
echo ""
echo "=========================================="
echo "步骤 5/5: RAG增强评估"
echo "=========================================="

export retrieval_db_npz="${PROJECT_ROOT}/outputs/retrieval/hmof_100_train_db.npz"
export retrieval_index="${PROJECT_ROOT}/outputs/retrieval/hmof_100_train_pxrd.index"
export rag_top_m=4
export rag_strength=1.0

export save_path="${PROJECT_ROOT}/outputs/evaluation/ccsg_with_rag"
export label="with_rag"
export num_evals=10
export begin_idx=0
export end_idx=10

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
python scripts/compute_ccsg_metrics.py \
    --root_path "${save_path}" \
    --save_path "${save_path}" \
    --multi_eval \
    --label "${label}"

echo "✅ RAG增强评估完成"

# 总结
echo ""
echo "=========================================="
echo "完整流程完成!"
echo "=========================================="
echo "结束时间: $(date)"
echo ""
echo "输出文件位置:"
echo "  CPCP检查点: ${cpcp_ckpt_path}"
echo "  CCSG检查点: ${ccsg_ckpt_path}"
echo "  检索数据库: outputs/retrieval/"
echo "  无RAG评估: outputs/evaluation/ccsg_no_rag/"
echo "  RAG增强评估: outputs/evaluation/ccsg_with_rag/"
echo ""
echo "运行以下命令查看评估结果对比:"
echo "  ls outputs/evaluation/ccsg_no_rag/"
echo "  ls outputs/evaluation/ccsg_with_rag/"