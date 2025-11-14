#!/bin/bash
#SBATCH --job-name=ccsg_hmof100_train
#SBATCH --output=logs/slurm/ccsg_hmof100_%j.out
#SBATCH --error=logs/slurm/ccsg_hmof100_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu          # 请根据你的分区修改
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2            # 请根据你的GPU配置修改
#SBATCH --qos=default           # 请根据你的QoS设置修改
#SBATCH --dependency=afterok:<cpcp_job_id>  # 等待CPCP训练完成

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1
export PROJECT_ROOT=/path/to/your/RAG-XtalNet  # 请修改为你的路径
export HYDRA_JOBS=${PROJECT_ROOT}/outputs/slurm
export WANDB_DIR=${PROJECT_ROOT}/wandb

# 创建必要的目录
mkdir -p logs/slurm
mkdir -p outputs/slurm
mkdir -p wandb

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

# 训练参数
export expname=ccsg_hmof100_train
export model=ccsg
export data_name='hmof_100'
export pretrained=$cpcp_ckpt_path
export freeze=true
export bsz=16
export lr=1e-3

echo "开始训练 CCSG 模型..."
echo "实验名称: ${expname}"
echo "数据集: ${data_name}"
echo "预训练模型: ${pretrained}"
echo "批次大小: ${bsz}"
echo "学习率: ${lr}"
echo "GPU数量: 2"
echo "开始时间: $(date)"

# 运行训练
bash train.sh

echo "训练完成时间: $(date)"