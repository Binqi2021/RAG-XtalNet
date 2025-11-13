# Retrieval System Analysis

## Quick Start

```bash
cd /public/home/huangtianqi/XtalNet
conda activate xtalnet

# Run complete analysis (5-10 minutes)
python retrieval_experiments/analyze_retrieval.py
```

## Output

```
retrieval_experiments/
├── metrics.json                    # All metrics in JSON format
└── visualizations/
    ├── recall_curve.png           # Recall@k performance
    ├── similarity_dist.png        # Similarity distribution
    ├── speed_benchmark.png        # Query speed benchmark
    └── embedding_tsne.png         # t-SNE visualization
```

## Metrics Collected

- **Recall@k**: k=1,5,10,20
- **Similarity**: mean, std, min, max
- **Speed**: single query, batch query
- **Embeddings**: t-SNE visualization

## Requirements

- Already installed in xtalnet environment
- Optional: `scikit-learn` for t-SNE (auto-skips if not available)

## Customization

Edit `analyze_retrieval.py`:
- `n_samples=100` - number of test queries
- `n_queries=100` - number of speed benchmark queries
- `tsne_samples=500` - samples for t-SNE visualization

