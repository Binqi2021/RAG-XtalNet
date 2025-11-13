# XtalNet Evaluation Guide

## Command Parameters Explained

### 1. evaluate_ccsg.py - Generate Crystal Structures

```bash
python scripts/evaluate_ccsg.py \
    --ccsg_ckpt_path <path>  \  # CCSG model checkpoint (structure generator)
    --cpcp_ckpt_path <path>  \  # CPCP model checkpoint (PXRD encoder)
    --save_path <path>       \  # Where to save generated structures
    --label <string>         \  # Experiment label for organizing results
    --num_evals <int>        \  # Number of samples to generate per structure
    --begin_idx <int>        \  # Start index in test set
    --end_idx <int>             # End index in test set (exclusive)
```

#### Parameter Details

| Parameter | Description | Example | Notes |
|:----------|:------------|:--------|:------|
| `--ccsg_ckpt_path` | Path to CCSG checkpoint | `ckpt/hmof_100/CCSG/hmof_100_ccsg.ckpt` | Diffusion model that generates structures |
| `--cpcp_ckpt_path` | Path to CPCP checkpoint | `ckpt/hmof_100/CPCP/hmof_100_cpcp.ckpt` | Encodes PXRD patterns as conditions |
| `--save_path` | Output directory | `results/my_experiment` | Will be created if doesn't exist |
| `--label` | Experiment name | `baseline_run` | Used in saved file names |
| `--num_evals` | Samples per structure | `10` | More = better statistics but slower |
| `--begin_idx` | Test set start | `0` | First structure to evaluate |
| `--end_idx` | Test set end | `10` | Process structures [begin_idx, end_idx) |

#### What It Does

For each test structure in range [begin_idx, end_idx):
1. Load ground truth PXRD pattern
2. Encode PXRD with CPCP → condition vector
3. Generate `num_evals` candidate structures using CCSG
4. Save all candidates as CIF files

#### Time Estimate

- V100 GPU: ~20 seconds per sample
- Example: 10 structures × 10 samples = 100 samples → 35 minutes

#### Output Structure

```
<save_path>/
├── pt/
│   └── <label>/
│       ├── eval_0.pt    # Predictions for test idx 0
│       ├── eval_1.pt
│       └── ...
└── cif/
    └── <label>/
        ├── 0_0.cif      # Structure 0, sample 0
        ├── 0_1.cif      # Structure 0, sample 1
        └── ...
```

---

### 2. compute_ccsg_metrics.py - Evaluate Quality

```bash
python scripts/compute_ccsg_metrics.py \
    --root_path <path>       \  # Directory containing generated structures
    --save_path <path>       \  # Where to save metrics
    --multi_eval             \  # Multiple samples per structure flag
    --label <string>            # Same label used in generation
```

#### Parameter Details

| Parameter | Description | Example | Notes |
|:----------|:------------|:--------|:------|
| `--root_path` | Generated results dir | `results/my_experiment` | Same as `--save_path` in generation |
| `--save_path` | Metrics output dir | `results/my_experiment/metrics` | Summary statistics saved here |
| `--multi_eval` | Flag for multi-sample mode | (no value) | Use when `num_evals > 1` |
| `--label` | Experiment label | `baseline_run` | Must match generation label |

#### What It Does

1. Load all generated structures
2. Compare with ground truth structures
3. Compute metrics:
   - **Match Rate**: % structures matched within RMSD threshold
   - **RMSD**: Root mean square deviation of atomic positions
   - **Coverage**: Property prediction accuracy
4. Save summary statistics

#### Output Files

```
<save_path>/
├── metrics_summary.json       # Overall statistics
├── per_structure_metrics.csv  # Detailed per-structure results
└── best_structures/           # Best match for each test case
    ├── 0_best.cif
    └── ...
```

---

### 3. gradio_demo.py - Interactive Web Interface

```bash
python scripts/gradio_demo.py \
    --ccsg_ckpt_path <path>  \  # CCSG checkpoint
    --cpcp_ckpt_path <path>  \  # CPCP checkpoint
    --save_path <path>          # Where to save user-generated structures
```

#### Parameter Details

| Parameter | Description | Example |
|:----------|:------------|:--------|
| `--ccsg_ckpt_path` | CCSG model | `ckpt/hmof_100/CCSG/hmof_100_ccsg.ckpt` |
| `--cpcp_ckpt_path` | CPCP model | `ckpt/hmof_100/CPCP/hmof_100_cpcp.ckpt` |
| `--save_path` | Output directory | `gradio_outputs` |

#### What It Does

1. Starts a web interface on port 7860
2. Users can:
   - Input chemical formula (e.g., `Cu2H8C28N6O8`)
   - Upload PXRD pattern file
   - Generate crystal structure
   - Download result as CIF file

#### Example Input

```
Formula: Cu2H8C28N6O8
PXRD file: example/case.txt
```

Ground truth available at: `example/case_gt.cif`

---

## Quick Start Examples

### Minimal Test (5 minutes)

```bash
# Generate 5 structures with 1 sample each
python scripts/evaluate_ccsg.py \
    --ccsg_ckpt_path ckpt/hmof_100/CCSG/hmof_100_ccsg.ckpt \
    --cpcp_ckpt_path ckpt/hmof_100/CPCP/hmof_100_cpcp.ckpt \
    --save_path results/quick_test \
    --label quick \
    --num_evals 1 \
    --begin_idx 0 \
    --end_idx 5

# Compute metrics
python scripts/compute_ccsg_metrics.py \
    --root_path results/quick_test \
    --save_path results/quick_test/metrics \
    --label quick
```

### Full Evaluation (several hours)

```bash
# Use the provided script
sbatch eval_ccsg.sh

# Monitor progress
tail -f logs/eval_ccsg_*.log
```

### Launch Gradio Demo

```bash
# Interactive mode (if on GPU node)
python scripts/gradio_demo.py \
    --ccsg_ckpt_path ckpt/hmof_100/CCSG/hmof_100_ccsg.ckpt \
    --cpcp_ckpt_path ckpt/hmof_100/CPCP/hmof_100_cpcp.ckpt \
    --save_path gradio_outputs

# Or submit as job
sbatch run_gradio.sh
```

---

## Understanding Results

### Match Rate
- **Good**: >30% for hmof_100
- **Excellent**: >50%
- Measures: % of test structures where best generated sample matches ground truth

### RMSD
- **Good**: <0.5 Å
- **Excellent**: <0.3 Å
- Measures: Average atomic position error

### Typical Performance (hmof_100)
- Match Rate: ~40%
- RMSD: ~0.4 Å
- Generation time: ~20s per sample on V100

