# SLURM 作业脚本使用指南

本目录包含了在SLURM集群上运行RAG-XtalNet完整流程的所有作业脚本。

## 📁 脚本列表

### 🚀 **单个步骤脚本**

1. **`train_cpcp_hmof100.slurm`** - CPCP模型训练
2. **`train_ccsg_hmof100.slurm`** - CCSG模型训练
3. **`build_retrieval_db.slurm`** - 构建检索数据库
4. **`evaluate_ccsg_no_rag.slurm`** - 无RAG评估
5. **`evaluate_ccsg_with_rag.slurm`** - RAG增强评估

### 🔄 **完整流程脚本**

6. **`complete_pipeline.slurm`** - 从训练到评估的完整流程（推荐）

## ⚙️ **使用前准备**

### 1. 修改路径配置

在每个脚本中修改以下变量：

```bash
export PROJECT_ROOT=/path/to/your/RAG-XtalNet  # 修改为你的实际路径
```

### 2. 调整集群资源配置

根据你的集群情况修改SLURM参数：

```bash
#SBATCH --partition=gpu          # 修改为你的分区名称
#SBATCH --gres=gpu:2            # 修改为可用的GPU数量
#SBATCH --mem=64G                # 修改为可用内存
#SBATCH --qos=default           # 修改为你的QoS设置
```

### 3. 配置模块加载

如果你的集群使用模块系统，取消注释并修改：

```bash
# module load cuda/11.8
# module load python/3.9
# module load conda
# conda activate xtalnet
```

## 🚀 **运行方式**

### 方式1: 完整流程（推荐）

```bash
# 提交完整流程作业
sbatch complete_pipeline.slurm

# 监控作业状态
squeue -u $USER

# 查看作业日志
tail -f logs/slurm/pipeline_*.out
```

### 方式2: 分步运行

```bash
# 1. 提交CPCP训练
cpcp_job=$(sbatch --parsable train_cpcp_hmof100.slurm)

# 2. 提交CCSG训练（依赖CPCP完成）
ccsg_job=$(sbatch --parsable --dependency=afterok:$cpcp_job train_ccsg_hmof100.slurm)

# 3. 构建数据库（依赖CCSG完成）
db_job=$(sbatch --parsable --dependency=afterok:$ccsg_job build_retrieval_db.slurm)

# 4. 无RAG评估
eval_no_rag_job=$(sbatch --parsable --dependency=afterok:$ccsg_job evaluate_ccsg_no_rag.slurm)

# 5. RAG评估（依赖数据库构建完成）
eval_rag_job=$(sbatch --parsable --dependency=afterok:$db_job evaluate_ccsg_with_rag.slurm)

echo "所有作业已提交!"
echo "CPCP作业ID: $cpcp_job"
echo "CCSG作业ID: $ccsg_job"
echo "数据库构建作业ID: $db_job"
echo "无RAG评估作业ID: $eval_no_rag_job"
echo "RAG评估作业ID: $eval_rag_job"
```

## 📊 **作业依赖关系**

```
CPCP训练 (train_cpcp_hmof100.slurm)
    ↓
CCSG训练 (train_ccsg_hmof100.slurm)
    ↓
┌─────────────────┬─────────────────┐
│   构建数据库      │    无RAG评估      │
│ (build_retrieval) │ (eval_no_rag)   │
└─────────────────┴─────────────────┘
    ↓                           ↓
    └───────── RAG评估 (eval_with_rag)
```

## 📁 **输出文件结构**

运行完成后，你会得到以下输出：

```
RAG-XtalNet/
├── logs/slurm/
│   ├── cpcp_hmof100_*.out        # CPCP训练日志
│   ├── ccsg_hmof100_*.out        # CCSG训练日志
│   ├── build_retrieval_*.out     # 数据库构建日志
│   ├── eval_ccsg_no_rag_*.out    # 无RAG评估日志
│   ├── eval_ccsg_with_rag_*.out  # RAG评估日志
│   └── pipeline_*.out            # 完整流程日志
├── outputs/slurm/
│   ├── cpcp_hmof100_train/       # CPCP训练输出
│   └── ccsg_hmof100_train/       # CCSG训练输出
├── outputs/retrieval/
│   ├── hmof_100_train_db.npz     # 训练集数据库
│   ├── hmof_100_train_pxrd.index # 训练集索引
│   ├── hmof_100_val_db.npz       # 验证集数据库
│   ├── hmof_100_val_pxrd.index   # 验证集索引
│   ├── hmof_100_test_db.npz      # 测试集数据库
│   └── hmof_100_test_pxrd.index  # 测试集索引
└── outputs/evaluation/
    ├── ccsg_no_rag/              # 无RAG评估结果
    └── ccsg_with_rag/            # RAG增强评估结果
```

## 🔧 **常用SLURM命令**

```bash
# 查看作业状态
squeue -u $USER

# 查看作业详情
scontrol show job <job_id>

# 取消作业
scancel <job_id>

# 取消所有作业
scancel -u $USER

# 查看作业日志
cat logs/slurm/<job_name>_<job_id>.out

# 查看作业错误日志
cat logs/slurm/<job_name>_<job_id>.err

# 查看分区信息
sinfo

# 查看GPU使用情况
nvidia-smi
```

## ⚠️ **注意事项**

1. **路径配置**: 确保所有路径都正确设置
2. **资源限制**: 根据集群限制调整资源配置
3. **作业依赖**: 使用`--dependency`确保作业按正确顺序执行
4. **监控日志**: 定期检查作业日志确保正常运行
5. **存储空间**: 确保有足够的存储空间用于输出文件

## 🆘 **故障排除**

### 作业无法提交
- 检查SLURM语法是否正确
- 确认分区和资源请求合理
- 检查路径权限

### 作业运行失败
- 查看`.err`文件了解错误信息
- 检查环境变量和模块加载
- 确认数据文件和检查点文件存在

### 内存不足
- 减少`bsz`（批次大小）
- 增加`--mem`内存限制
- 使用梯度累积

### GPU不足
- 检查GPU可用性
- 调整`--gres=gpu:N`参数
- 等待GPU资源释放

如有其他问题，请查看相关日志文件或联系集群管理员。