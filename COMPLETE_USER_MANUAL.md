# RAG-XtalNet 完整使用手册

> 📚 **一份手册，搞定一切！** 从环境配置到RAG增强的完整流程指南

## 📋 目录

1. [项目概述](#1-项目概述)
2. [环境配置](#2-环境配置)
3. [数据准备](#3-数据准备)
4. [模型训练](#4-模型训练)
5. [RAG系统使用](#5-rag系统使用)
6. [模型评估](#6-模型评估)
7. [完整流程示例](#7-完整流程示例)
8. [故障排除](#8-故障排除)

---

## 1. 项目概述

### 1.1 什么是RAG-XtalNet？

RAG-XtalNet是一个基于检索增强生成（RAG）的晶体结构预测系统，结合了：
- **CPCP模块**: PXRD-晶体对比学习
- **CCSG模块**: 晶体结构扩散生成
- **RAG系统**: 基于相似结构的模板检索增强

### 1.2 核心架构

```
RAG-XtalNet/
├── xtalnet/
│   ├── pl_modules/          # 核心模型
│   │   ├── cpcp_module.py   # CPCP对比学习模块
│   │   ├── ccsg_module.py   # CCSG生成模块
│   │   └── cspnet_ccsg.py   # CSPNet架构
│   ├── pl_data/            # 数据加载
│   ├── retrieval/          # RAG检索系统
│   │   └── pxrd_retriever.py
│   └── common/             # 通用工具
├── scripts/                # 脚本工具
├── conf/                   # 配置文件
└── outputs/               # 输出文件
```

---

## 2. 环境配置

### 2.1 创建Conda环境

```bash
# 创建环境
conda env create -f xtalnet.yaml

# 激活环境
conda activate xtalnet
```

### 2.2 配置环境变量

创建 `.env` 文件：

```bash
# 项目根目录
PROJECT_ROOT=/path/to/RAG-XtalNet

# Hydra输出目录
HYDRA_JOBS=/path/to/RAG-XtalNet/outputs

# WandB目录（可选）
WANDB_DIR=/path/to/RAG-XtalNet/wandb
```

### 2.3 验证安装

```bash
# 检查Python模块
python -c "import xtalnet; print('✅ xtalnet导入成功')"

# 检查依赖
python -c "import torch, faiss, numpy; print('✅ 核心依赖正常')"
```

---

## 3. 数据准备

### 3.1 支持的数据集

- **hmof_100**: 100个原子的氢有机框架
- **hmof_400**: 400个原子的氢有机框架

### 3.2 数据路径配置

编辑 `conf/data/hmof_100.yaml` 和 `conf/data/hmof_400.yaml`：

```yaml
# 示例配置
data:
  root_path: /path/to/your/data  # 更新为你的数据路径
```

---

## 4. 模型训练

### 4.1 CPCP模块训练

```bash
# 设置参数
export expname=cpcp_training
export model=cpcp
export data_name='hmof_100'  # 或 'hmof_400'
export freeze=false
export bsz=16               # 4 gpus, hmof_400用8
export lr=5e-4              # hmof_400用2e-4

# 开始训练
bash train.sh
```

### 4.2 CCSG模块训练

```bash
# 设置参数
export expname=ccsg_training
export model=ccsg
export data_name='hmof_100'
export pretrained=<cpcp_ckpt_path>  # CPCP检查点路径
export freeze=true
export bsz=16               # 4 gpus, hmof_400用4
export lr=1e-3

# 开始训练
bash train.sh
```

---

## 5. RAG系统使用

### 5.1 构建检索数据库

#### 方法1：自动路径生成（推荐）

```bash
# 构建hmof_100训练数据库
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path logs/cpcp_hmof100/checkpoints/last.ckpt \
    --data_name hmof_100 \
    --split train \
    --device cuda

# 构建hmof_400训练数据库
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path logs/cpcp_hmof400/checkpoints/last.ckpt \
    --data_name hmof_400 \
    --split train \
    --device cuda
```

#### 方法2：自定义路径

```bash
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path logs/cpcp_hmof100/checkpoints/last.ckpt \
    --data_name hmof_100 \
    --split train \
    --save_prefix outputs/my_custom_db \
    --device cuda
```

**输出文件**：
- `outputs/retrieval/hmof_100_train_db.npz` - 数据库文件
- `outputs/retrieval/hmof_100_train_pxrd.index` - FAISS索引文件

### 5.2 测试检索系统

```bash
# 基础数据库测试
python scripts/test_db_loading.py \
    --db_path outputs/retrieval/hmof_100_train_db.npz \
    --index_path outputs/retrieval/hmof_100_train_pxrd.index

# 完整检索器测试
python scripts/test_pxrd_retriever.py \
    --db_path outputs/retrieval/hmof_100_train_db.npz \
    --index_path outputs/retrieval/hmof_100_train_pxrd.index \
    --num_queries 10 \
    --top_m 10
```

### 5.3 使用检索器（Python代码）

```python
from xtalnet.retrieval import PXRDTemplateRetriever

# 初始化检索器
retriever = PXRDTemplateRetriever(
    db_npz_path='outputs/retrieval/hmof_100_train_db.npz',
    faiss_index_path='outputs/retrieval/hmof_100_train_pxrd.index'
)

# 单次查询
query_embedding = your_pxrd_embedding  # [512] 维向量
results = retriever.query(query_embedding, top_m=5)

# 查看结果
print(f"Top-5 formulas: {results['formula']}")
print(f"Similarity scores: {results['scores']}")
```

---

## 6. 模型评估

### 6.1 CPCP评估

```bash
# 生成预测
python scripts/evaluate_cpcp.py \
    --model_path <ckpt_dir_path> \
    --ckpt_path <ckpt_path> \
    --save_path <save_path> \
    --label <label>

# 计算指标
python scripts/compute_cpcp_metrics.py --root_path <results_path>
```

### 6.2 CCSG评估（无RAG）

```bash
# 生成样本
python scripts/evaluate_ccsg.py \
    --ccsg_ckpt_path <ccsg_ckpt_path> \
    --cpcp_ckpt_path <cpcp_ckpt_path> \
    --save_path <save_path> \
    --label <label> \
    --num_evals <num_evals> \
    --begin_idx <begin_idx> \
    --end_idx <end_idx>

# 计算指标
python scripts/compute_ccsg_metrics.py \
    --root_path <results_path> \
    --save_path <save_path> \
    --multi_eval \
    --label <label>
```

### 6.3 CCSG评估（RAG增强）

```bash
# 使用hmof_100数据库的RAG评估
python scripts/evaluate_ccsg.py \
    --ccsg_ckpt_path <ccsg_ckpt_path> \
    --cpcp_ckpt_path <cpcp_ckpt_path> \
    --save_path <save_path> \
    --label <label> \
    --num_evals <num_evals> \
    --begin_idx <begin_idx> \
    --end_idx <end_idx> \
    --retrieval_db_npz outputs/retrieval/hmof_100_train_db.npz \
    --retrieval_index outputs/retrieval/hmof_100_train_pxrd.index \
    --rag_top_m 4 \
    --rag_strength 1.0

# 使用hmof_400数据库的RAG评估
python scripts/evaluate_ccsg.py \
    --ccsg_ckpt_path <ccsg_ckpt_path> \
    --cpcp_ckpt_path <cpcp_ckpt_path> \
    --save_path <save_path> \
    --label <label> \
    --num_evals <num_evals> \
    --begin_idx <begin_idx> \
    --end_idx <end_idx> \
    --retrieval_db_npz outputs/retrieval/hmof_400_train_db.npz \
    --retrieval_index outputs/retrieval/hmof_400_train_pxrd.index \
    --rag_top_m 4 \
    --rag_strength 1.0
```

---

## 7. 完整流程示例

### 7.1 从零开始的完整流程

```bash
# ======================
# 1. 训练CPCP模型
# ======================
export expname=cpcp_hmof100
export model=cpcp
export data_name='hmof_100'
export freeze=false
export bsz=16
export lr=5e-4
bash train.sh

# ======================
# 2. 训练CCSG模型
# ======================
export expname=ccsg_hmof100
export model=ccsg
export data_name='hmof_100'
export pretrained=logs/cpcp_hmof100/checkpoints/last.ckpt
export freeze=true
export bsz=16
export lr=1e-3
bash train.sh

# ======================
# 3. 构建RAG数据库
# ======================
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path logs/cpcp_hmof100/checkpoints/last.ckpt \
    --data_name hmof_100 \
    --split train \
    --device cuda

# ======================
# 4. 测试RAG系统
# ======================
python scripts/test_db_loading.py \
    --db_path outputs/retrieval/hmof_100_train_db.npz \
    --index_path outputs/retrieval/hmof_100_train_pxrd.index

# ======================
# 5. 评估无RAG性能
# ======================
python scripts/evaluate_ccsg.py \
    --ccsg_ckpt_path logs/ccsg_hmof100/checkpoints/last.ckpt \
    --cpcp_ckpt_path logs/cpcp_hmof100/checkpoints/last.ckpt \
    --save_path outputs/eval_no_rag \
    --label no_rag \
    --num_evals 10 \
    --begin_idx 0 \
    --end_idx 10

# ======================
# 6. 评估RAG增强性能
# ======================
python scripts/evaluate_ccsg.py \
    --ccsg_ckpt_path logs/ccsg_hmof100/checkpoints/last.ckpt \
    --cpcp_ckpt_path logs/cpcp_hmof100/checkpoints/last.ckpt \
    --save_path outputs/eval_with_rag \
    --label with_rag \
    --num_evals 10 \
    --begin_idx 0 \
    --end_idx 10 \
    --retrieval_db_npz outputs/retrieval/hmof_100_train_db.npz \
    --retrieval_index outputs/retrieval/hmof_100_train_pxrd.index \
    --rag_top_m 4 \
    --rag_strength 1.0

# ======================
# 7. 计算并比较指标
# ======================
python scripts/compute_ccsg_metrics.py \
    --root_path outputs/eval_no_rag \
    --save_path outputs/eval_no_rag \
    --multi_eval \
    --label no_rag

python scripts/compute_ccsg_metrics.py \
    --root_path outputs/eval_with_rag \
    --save_path outputs/eval_with_rag \
    --multi_eval \
    --label with_rag
```

### 7.2 使用示例脚本

```bash
# 运行检索使用示例
python scripts/example_retrieval_usage.py \
    --data_name hmof_100 \
    --split train \
    --cpcp_ckpt_path logs/cpcp_hmof100/checkpoints/last.ckpt
```

---

## 8. 故障排除

### 8.1 常见问题

#### Q1: CUDA out of memory
```bash
# 解决方案：使用CPU构建数据库
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path <path> \
    --data_name hmof_100 \
    --split train \
    --device cpu
```

#### Q2: 找不到模块 'xtalnet.retrieval'
```bash
# 解决方案：设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# 或在代码中添加
import sys
sys.path.insert(0, '/path/to/RAG-XtalNet')
```

#### Q3: 数据库文件不存在
```bash
# 检查文件是否存在
ls outputs/retrieval/
# 如果不存在，先构建数据库
python scripts/build_pxrd_crystal_db.py ...
```

#### Q4: 检查点路径错误
```bash
# 检查检查点文件
ls logs/cpcp_*/checkpoints/
ls logs/ccsg_*/checkpoints/
```

### 8.2 性能调优

#### 数据库构建优化
```bash
# 增加批处理大小（如果内存允许）
# 在脚本中修改 batch_size 参数
```

#### RAG参数调优
```bash
# 调整检索模板数量
--rag_top_m 2    # 更少模板，更快但可能效果差
--rag_top_m 8    # 更多模板，更慢但可能效果好

# 调整RAG强度
--rag_strength 0.5  # 较弱的RAG影响
--rag_strength 2.0  # 较强的RAG影响
```

### 8.3 验证安装

```bash
# 运行完整验证脚本
python -c "
import sys
sys.path.insert(0, '.')
from xtalnet.retrieval import PXRDTemplateRetriever
from xtalnet.pl_modules.cpcp_module import CPCP
from xtalnet.pl_modules.ccsg_module import CCSG
print('✅ 所有核心模块导入成功')
"
```

---

## 🎯 快速参考

### 核心命令速查

```bash
# 训练CPCP
export expname=cpcp_hmof100; export model=cpcp; export data_name='hmof_100'; bash train.sh

# 训练CCSG
export expname=ccsg_hmof100; export model=ccsg; export data_name='hmof_100'; export pretrained=<cpcp_path>; bash train.sh

# 构建数据库
python scripts/build_pxrd_crystal_db.py --cpcp_ckpt_path <path> --data_name hmof_100 --split train

# 测试数据库
python scripts/test_db_loading.py --db_path outputs/retrieval/hmof_100_train_db.npz --index_path outputs/retrieval/hmof_100_train_pxrd.index

# RAG评估
python scripts/evaluate_ccsg.py --ccsg_ckpt_path <path> --cpcp_ckpt_path <path> --retrieval_db_npz outputs/retrieval/hmof_100_train_db.npz --retrieval_index outputs/retrieval/hmof_100_train_pxrd.index
```

### 文件路径规范

```
数据库: outputs/retrieval/{dataset}_{split}_db.npz
索引:  outputs/retrieval/{dataset}_{split}_pxrd.index
检查点: logs/{model}_{dataset}/checkpoints/last.ckpt
```

---

## 🎉 完成！

现在你已经拥有了完整的RAG-XtalNet使用知识。按照这个手册，你可以：

✅ 从零开始配置环境
✅ 训练CPCP和CCSG模型
✅ 构建和使用RAG检索系统
✅ 评估模型性能（有无RAG）
✅ 解决常见问题

祝你使用愉快！如有问题，参考故障排除部分或查看具体脚本的帮助信息。