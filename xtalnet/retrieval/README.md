# PXRD Template Retriever

基于PXRD嵌入的晶体结构模板检索器。

## 快速使用

```python
from xtalnet.retriever import PXRDTemplateRetriever

# 初始化
retriever = PXRDTemplateRetriever(
    db_npz_path='outputs/retrieval/hmof_100_train_db.npz',
    faiss_index_path='outputs/retrieval/hmof_100_train_pxrd.index'
)

# 单次查询
results = retriever.query(query_embedding, top_m=5)

# 批量查询
results = retriever.batch_query(query_embeddings, top_m=5)
```

## API参考

### PXRDTemplateRetriever

```python
class PXRDTemplateRetriever:
    def __init__(self, db_npz_path: str, faiss_index_path: str, normalize: bool = True)

    def query(self, query_pxrd_feat: np.ndarray, top_m: int = 4) -> dict

    def batch_query(self, query_pxrd_feats: np.ndarray, top_m: int = 4) -> dict

    def get_by_indices(self, indices: np.ndarray) -> dict
```

### 返回结果

```python
results = {
    'indices': np.ndarray,    # [top_m] 数据库索引
    'scores': np.ndarray,     # [top_m] 相似度分数
    'h_crystal': np.ndarray,  # [top_m, 512] 晶体嵌入
    'lattice': np.ndarray,    # [top_m, 6] 晶格参数
    'num_atoms': np.ndarray,  # [top_m] 原子数
    'formula': list           # [top_m] 化学式
}
```

## 构建数据库

使用 `scripts/build_pxrd_crystal_db.py` 构建数据库：

```bash
python scripts/build_pxrd_crystal_db.py \
    --cpcp_ckpt_path <checkpoint_path> \
    --data_name hmof_100 \
    --split train
```

详细用法参见 `COMPLETE_USER_MANUAL.md`。