#!/bin/bash
#SBATCH --job-name=cpcp_hmof100_train
#SBATCH --output=logs/slurm/cpcp_hmof100_%j.out
#SBATCH --error=logs/slurm/cpcp_hmof100_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu          # 请根据你的分区修改
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2            # 请根据你的GPU配置修改
#SBATCH --qos=default           # 请根据你的QoS设置修改

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

# 训练参数
export expname=cpcp_hmof100_train
export model=cpcp
export data_name='hmof_100'
export freeze=false
export bsz=16
export lr=5e-4

echo "开始训练 CPCP 模型..."
echo "实验名称: ${expname}"
echo "数据集: ${data_name}"
echo "批次大小: ${bsz}"
echo "学习率: ${lr}"
echo "GPU数量: 2"
echo "开始时间: $(date)"

# 运行训练
bash train.sh

echo "训练完成时间: $(date)"