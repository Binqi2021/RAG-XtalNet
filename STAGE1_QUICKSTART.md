# Stage 1: Quick Start Guide

## What Was Done
✅ Created retrieval module (`xtalnet/retrieval/`)  
✅ Updated data path configuration  
✅ Built index construction pipeline  
✅ Added testing utilities  

**No existing code was modified** - this is pure addition.

---

## Step 1: Install Dependencies

```bash
# Activate your environment
conda activate xtalnet

# Install FAISS (choose one)
pip install faiss-cpu      # Safe choice, works everywhere
pip install faiss-gpu      # If you have GPU access
```

---

## Step 2: Test with Small Sample (5 minutes)

```bash
cd /public/home/huangtianqi/XtalNet

# Build tiny test index (100 samples only)
python scripts/build_retrieval_index.py \
    --max_samples 100 \
    --output_dir retrieval_indices/test

# If it works, you'll see: "SUCCESS! Index saved to: ..."
```

---

## Step 3: Build Full Index (15-30 min on CPU)

```bash
# Full training set
python scripts/build_retrieval_index.py

# Or submit as SLURM job (if on GPU node)
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=build_index
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

cd /public/home/huangtianqi/XtalNet
conda activate xtalnet
python scripts/build_retrieval_index.py
EOF
```

---

## Step 4: Verify Quality

```bash
# Test retrieval recall
python scripts/test_retrieval.py --n_queries 50

# Expected output:
# Recall@1:  ~95%  (finds itself)
# Recall@5:  ~99%
# Recall@10: ~100%
```

---

## Expected Output Structure

```
retrieval_indices/
└── hmof_100/
    ├── faiss_index.bin    (~500MB)
    ├── metadata.pkl       (~1GB)
    └── config.pkl         (few KB)
```

---

## Troubleshooting

**"No module named faiss"**  
→ Run: `pip install faiss-cpu`

**"CUDA out of memory"**  
→ Add: `--batch_size 16` or `--device cpu`

**Takes too long**  
→ Normal on CPU. Use `--max_samples 1000` for testing.

---

## Next Steps (Stage 2)

Once index is built and tested, you can:
1. Integrate into CCSG sampling
2. Train RAG-enhanced model
3. Evaluate improvements

**Ready to proceed?** Run Step 2 above and report any errors.

