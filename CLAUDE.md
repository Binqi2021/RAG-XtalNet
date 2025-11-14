# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

XtalNet is an equivariant deep generative model for end-to-end crystal structure prediction from Powder X-Ray Diffraction (PXRD) patterns. The repository includes both the original XtalNet implementation and a Retrieval-Augmented Generation (RAG) extension.

## Architecture

The codebase follows a modular architecture with two main components:

### Core XtalNet System
- **CPCP Module** (`xtalnet/pl_modules/cpcp_module.py`): Crystal-PXRD Contrastive Prediction module that learns joint embeddings of PXRD patterns and crystal structures
- **CCSG Module** (`xtalnet/pl_modules/ccsg_module.py`): Crystal Structure Generation module that generates crystal structures from PXRD embeddings
- **Data Processing** (`xtalnet/pl_data/`): LMDB-based data loading for crystal structures and PXRD patterns
- **Configuration** (`conf/`): Hydra-based configuration system for models, training, and data

### RAG Extension
- **PXRD Retrieval System** (`xtalnet/retrieval/pxrd_retriever.py`): Fast similarity search using FAISS indices
- **Database Builder** (`scripts/build_pxrd_crystal_db.py`): Creates searchable indices from training data
- **Template Integration**: Retrieved templates guide the crystal generation process

## Common Development Commands

### Training Models

#### CPCP Module Training
```bash
export expname=cpcp_training
export model=cpcp
export data_name='hmof_100'  # or 'hmof_400'
export freeze=false
export bsz=16  # 4 gpus, 8 for hmof_400
export lr=5e-4  # 2e-4 for hmof_400
bash train.sh
```

#### CCSG Module Training
```bash
export expname=ccsg_training
export model=ccsg
export data_name='hmof_100'
export pretrained=<cpcp_ckpt_path>
export freeze=true
export bsz=16  # 4 gpus, 4 for hmof_400
export lr=1e-3
bash train.sh
```

### Evaluation

#### CPCP Evaluation
```bash
# Generate predictions
python scripts/evaluate_cpcp.py --model_path <ckpt_dir_path> --ckpt_path <ckpt_path> --save_path <save_path> --label <label>

# Compute metrics
python scripts/compute_cpcp_metrics.py --root_path <results_path>
```

#### CCSG Evaluation
```bash
# Generate samples
python scripts/evaluate_ccsg.py \
    --ccsg_ckpt_path <ccsg_ckpt_path> \
    --cpcp_ckpt_path <clip_ckpt_path> \
    --save_path <save_path> \
    --label <label> --num_evals <num_evals> \
    --begin_idx <begin_idx> --end_idx <end_idx>

# Compute metrics
python scripts/compute_ccsg_metrics.py --root_path <results_path> \
    --save_path <save_path> --multi_eval --label <label>
```

### RAG System Commands

#### Build Retrieval Database
```bash
# For hmof_100 dataset
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path logs/cpcp/checkpoints/last.ckpt \
    --data_name hmof_100 \
    --split train \
    --save_prefix outputs/retrieval/hmof100_train \
    --device cuda

# For hmof_400 dataset
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path logs/cpcp/checkpoints/last.ckpt \
    --data_name hmof_400 \
    --split train \
    --save_prefix outputs/retrieval/hmof400_train \
    --device cuda
```

#### Test Retrieval System
```bash
# For hmof_100 dataset
python scripts/test_db_loading.py \
    --db_path outputs/retrieval/hmof100_train_db.npz \
    --index_path outputs/retrieval/hmof100_train_pxrd.index

python scripts/test_pxrd_retriever.py \
    --db_path outputs/retrieval/hmof100_train_db.npz \
    --index_path outputs/retrieval/hmof100_train_pxrd.index \
    --num_queries 10 \
    --top_m 10

# For hmof_400 dataset
python scripts/test_db_loading.py \
    --db_path outputs/retrieval/hmof400_train_db.npz \
    --index_path outputs/retrieval/hmof400_train_pxrd.index

python scripts/test_pxrd_retriever.py \
    --db_path outputs/retrieval/hmof400_train_db.npz \
    --index_path outputs/retrieval/hmof400_train_pxrd.index \
    --num_queries 10 \
    --top_m 10
```

#### Interactive Demo
```bash
python scripts/gradio_demo.py --ccsg_ckpt_path <ccsg_ckpt_path> --cpcp_ckpt_path <cpcp_ckpt_path> --save_path <save_path>
```

## Data Configuration

### Environment Setup
- Use `xtalnet.yaml` to create conda environment
- Copy `.env.template` to `.env` and set:
  - `PROJECT_ROOT`: Absolute path to repository
  - `HYDRA_JOBS`: Absolute path for hydra outputs
  - `WABDB_DIR`: Absolute path for wandb outputs

### Data Paths
Update `root_path` in:
- `conf/data/hmof_100.yaml`
- `conf/data/hmof_400.yaml`

## Key Implementation Details

### Model Architecture
- CPCP uses contrastive learning between PXRD BERT encoder and crystal graph encoder
- CCSG uses diffusion-based generation with CSPNet architecture
- Both models support equivariant operations respecting crystal symmetries

### Retrieval Integration
- FAISS index enables O(log n) similarity search in CPCP latent space
- Retrieved templates provide conditioning for crystal generation
- Supports both single and batch querying via `PXRDTemplateRetriever`

### Configuration System
- Hydra-based configuration with composable YAML files
- Model configs in `conf/model/` (cpcp.yaml, ccsg.yaml, ccsg_with_alignment.yaml)
- Training configs in `conf/train/` and optimization in `conf/optim/`

## File Structure

```
xtalnet/
├── pl_modules/          # Model definitions (CPCP, CCSG, CSPNet)
├── pl_data/            # Data loading and preprocessing
├── common/             # Utilities and constants
└── retrieval/          # RAG retrieval system
scripts/
├── evaluate_*.py       # Model evaluation scripts
├── build_*.py          # Database building for RAG
└── test_*.py           # Testing utilities
conf/                   # Hydra configurations
```

## Performance Notes

- RAG retrieval achieves 94% Recall@1 and 97% Recall@10
- Query speed: 0.7ms (GPU), 8.5ms (CPU)
- Index size: 372MB for 73K structures
- V100 GPU generation time: ~20 seconds per sample