"""Crystal structure retrieval using FAISS index."""

import torch
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import faiss
except ImportError:
    raise ImportError("Please install faiss: pip install faiss-cpu or faiss-gpu")


class CrystalRetriever:
    """Efficient crystal structure retrieval using learned embeddings."""
    
    def __init__(self, index_path: str, device: str = 'auto'):
        """
        Args:
            index_path: Path to the saved index directory
            device: 'auto', 'cpu', or 'cuda'
        """
        self.index_path = Path(index_path)
        self.device = self._setup_device(device)
        self._load_index()
    
    def _setup_device(self, device: str) -> str:
        """Auto-detect device if needed."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_index(self):
        """Load FAISS index and metadata."""
        # Load FAISS index
        index_file = self.index_path / 'faiss_index.bin'
        if not index_file.exists():
            raise FileNotFoundError(f"Index not found: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        
        # Move to GPU if available
        if self.device == 'cuda' and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Load metadata
        with open(self.index_path / 'metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"Loaded index with {self.index.ntotal} structures on {self.device}")
    
    def search(self, query_embedding: torch.Tensor, k: int = 5) -> Dict:
        """
        Retrieve k most similar structures.
        
        Args:
            query_embedding: PXRD embedding tensor [B, D] or [D]
            k: Number of neighbors to retrieve
        
        Returns:
            Dictionary with 'indices', 'distances', and 'metadata'
        """
        # Handle single query
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        # Convert to numpy and normalize
        query_np = query_embedding.detach().cpu().numpy().astype('float32')
        faiss.normalize_L2(query_np)
        
        # Search
        distances, indices = self.index.search(query_np, k)
        
        # Gather metadata
        batch_metadata = []
        for idx_list in indices:
            batch_metadata.append([self.metadata[i] for i in idx_list])
        
        return {
            'indices': indices,
            'distances': distances,
            'metadata': batch_metadata
        }
    
    def get_template_embeddings(self, search_result: Dict) -> torch.Tensor:
        """Extract crystal embeddings from search results."""
        batch_embeddings = []
        for meta_list in search_result['metadata']:
            embs = torch.stack([m['crystal_emb'] for m in meta_list])
            batch_embeddings.append(embs)
        return torch.stack(batch_embeddings)  # [B, K, D]

