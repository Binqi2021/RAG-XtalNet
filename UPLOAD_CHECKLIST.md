# GitHub Upload Checklist

## ✅ Pre-Upload Checklist

### 1. Large Files (Will be ignored)
- [x] `data/` (4.4GB) - ✅ Ignored
- [x] `ckpt/` (594MB) - ✅ Ignored  
- [x] `retrieval_indices/` (372MB) - ✅ Ignored
- [x] `results/` (596KB) - ✅ Ignored
- [x] `logs/` (932KB) - ✅ Ignored
- [x] `*.ckpt`, `*.pkl`, `*.bin` - ✅ Ignored

### 2. Code Files (Will be uploaded)
- [x] `xtalnet/` - Core code ✅
- [x] `scripts/` - All scripts ✅
- [x] `conf/` - Configurations ✅
- [x] `*.py` - Python files ✅
- [x] `*.sh` - Shell scripts ✅
- [x] `*.yaml` - Config files ✅
- [x] `*.md` - Documentation ✅
- [x] `retrieval_experiments/analyze_retrieval.py` - ✅
- [x] `retrieval_experiments/README.md` - ✅

### 3. Excluded from Upload
- [x] `retrieval_experiments/visualizations/*.png` - ✅ Ignored
- [x] `retrieval_experiments/metrics.json` - ✅ Ignored

## 📋 Upload Steps

### Step 1: Initialize Git (if not done)
```bash
cd /public/home/huangtianqi/XtalNet
git init
```

### Step 2: Check what will be tracked
```bash
git status
```

### Step 3: Verify .gitignore is working
```bash
# Should show "ignored" for large dirs
git status --ignored | grep -E "data|ckpt|retrieval_indices|results|logs"
```

### Step 4: Add files
```bash
git add .
```

### Step 5: Check total size
```bash
git ls-files | xargs du -ch | tail -1
# Should be < 50MB (GitHub limit is 100MB per file)
```

### Step 6: Commit
```bash
git commit -m "Initial commit: RAG-XtalNet with retrieval system

- Stage 1: Retrieval infrastructure
  - FAISS index builder
  - Crystal retriever
  - Quality analysis tools
- Integration with XtalNet
- Comprehensive documentation"
```

### Step 7: Create GitHub repo and push
```bash
# 1. Create new repo on GitHub (don't initialize with README)

# 2. Add remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 3. Push
git branch -M main
git push -u origin main
```

## ⚠️ Important Notes

### For Repository Users
Add to README.md:
```markdown
## Setup

1. Clone this repository
2. Download data from [original XtalNet](https://zenodo.org/records/13629658)
3. Download checkpoints from [original XtalNet](https://zenodo.org/records/13629658)
4. Build retrieval index:
   ```bash
   python scripts/build_retrieval_index.py
   ```
```

### File Size Limits
- GitHub file limit: **100MB**
- Recommended repo size: **< 1GB**
- Your code-only repo: **~10-20MB** ✅

## 🔍 Final Verification

Before pushing, run:
```bash
# 1. Check ignored files
git check-ignore -v data/ ckpt/ retrieval_indices/

# 2. List all tracked files
git ls-files

# 3. Check for any large files
git ls-files | xargs ls -lh | awk '$5 > 10000000' | sort -k5 -h
# (Files > 10MB will be listed)
```

## 📝 Recommended README Addition

Add this section to your README.md:

```markdown
## Data and Checkpoints

This repository contains code only. To run experiments:

1. **Download data**: See [original XtalNet repository](https://github.com/dptech-corp/XtalNet)
2. **Download checkpoints**: Pre-trained models from [Zenodo](https://zenodo.org/records/13629658)
3. **Build retrieval index**: 
   ```bash
   python scripts/build_retrieval_index.py
   ```
```

