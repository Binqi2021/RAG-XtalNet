# 📤 GitHub上传指南

## ✅ 已完成的配置

我已经为您配置好了 `.gitignore`，以下文件**不会**被上传：

### ❌ 被忽略的大文件（~5.7GB）
- `data/` (4.4GB) - 数据集
- `ckpt/` (594MB) - 模型checkpoint
- `retrieval_indices/` (372MB) - FAISS索引
- `results/` (596KB) - 评估结果
- `logs/` (932KB) - 日志文件
- `retrieval_experiments/visualizations/*.png` - 可视化图表
- `retrieval_experiments/metrics.json` - 指标文件

### ✅ 会被上传的代码（~10-20MB）
- `xtalnet/` - 核心代码
- `scripts/` - 所有脚本
- `conf/` - 配置文件
- `*.py`, `*.sh`, `*.yaml` - 代码文件
- `*.md` - 文档
- `retrieval_experiments/analyze_retrieval.py` - 分析脚本
- `retrieval_experiments/README.md` - 说明文档

---

## 🚀 上传步骤

### 1. 检查状态
```bash
cd /public/home/huangtianqi/XtalNet
git status
```

**应该看到**：
- ✅ 代码文件显示为 `??` (未跟踪)
- ✅ `data/`, `ckpt/`, `retrieval_indices/` 等**不出现**（已被忽略）

### 2. 添加文件
```bash
git add .
```

### 3. 验证大小（重要！）
```bash
# 查看总大小
git ls-files | xargs du -ch | tail -1

# 应该显示 < 50MB
# 如果 > 100MB，检查是否有大文件被误添加
```

### 4. 提交
```bash
git commit -m "Add RAG-XtalNet: Retrieval-Augmented Generation for Crystal Structure Prediction

Features:
- Stage 1: FAISS-based retrieval system
- Retrieval quality: 94% Recall@1, 0.7ms query time
- Integration with XtalNet diffusion model
- Comprehensive evaluation tools"
```

### 5. 创建GitHub仓库并推送

**在GitHub上**：
1. 创建新仓库（**不要**初始化README）
2. 复制仓库URL

**在本地**：
```bash
# 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 推送
git branch -M main
git push -u origin main
```

---

## ⚠️ 重要提示

### 对于使用您代码的人

他们需要：
1. **下载数据**：从[原始XtalNet](https://zenodo.org/records/13629658)下载
2. **下载checkpoint**：从[原始XtalNet](https://zenodo.org/records/13629658)下载
3. **构建索引**：运行 `python scripts/build_retrieval_index.py`

**建议在README.md中添加**：
```markdown
## Data and Checkpoints

This repository contains code only. To reproduce results:

1. Download data and checkpoints from [XtalNet Zenodo](https://zenodo.org/records/13629658)
2. Extract to `data/` and `ckpt/` directories
3. Build retrieval index:
   ```bash
   python scripts/build_retrieval_index.py
   ```
```

---

## 🔍 验证清单

上传前确认：

- [ ] `git status` 不显示 `data/`, `ckpt/`, `retrieval_indices/`
- [ ] `git ls-files | xargs du -ch` 显示 < 50MB
- [ ] 所有 `.py`, `.sh`, `.yaml`, `.md` 文件都被跟踪
- [ ] `.gitignore` 已更新
- [ ] README.md 包含数据下载说明

---

## 📊 预期上传大小

```
代码文件:     ~10-20MB
配置文件:     ~1MB
文档:         ~100KB
总计:         ~15-25MB ✅ (远小于GitHub限制)
```

---

## 🎯 快速命令

```bash
# 一键检查
cd /public/home/huangtianqi/XtalNet
git status --short | head -20
git check-ignore -v data/ ckpt/ retrieval_indices/

# 如果都正确，直接上传
git add .
git commit -m "Add RAG-XtalNet retrieval system"
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

---

**现在可以安全上传了！** 🎉

