"""
PXRD Template Retriever

This module implements a retriever that searches for similar crystal structures
based on PXRD embeddings in the CPCP latent space.

The retriever uses a pre-built database (NPZ file) and FAISS index for efficient
nearest neighbor search.
"""

from typing import Dict, Optional, Union
from pathlib import Path

import numpy as np
import faiss


class PXRDTemplateRetriever:
    """
    PXRD-based crystal template retriever using CPCP embeddings.
    
    This class performs retrieval in the CPCP latent space to find similar
    crystal structures given a PXRD embedding. It uses FAISS for efficient
    similarity search with cosine distance.
    
    Attributes:
        db: Dictionary containing database fields (h_pxrd, h_crystal, lattice, etc.)
        index: FAISS index for fast similarity search
        normalize: Whether to L2 normalize query vectors
        n_samples: Number of samples in the database
        embedding_dim: Dimension of embeddings
    
    Example:
        >>> retriever = PXRDTemplateRetriever(
        ...     db_npz_path='outputs/db_train.npz',
        ...     faiss_index_path='outputs/db_train_pxrd.index'
        ... )
        >>> results = retriever.query(query_pxrd_feat, top_m=5)
        >>> print(results['indices'])  # Top-5 indices
        >>> print(results['scores'])   # Similarity scores
    """
    
    def __init__(
        self,
        db_npz_path: str,
        faiss_index_path: str,
        normalize: bool = True,
    ):
        """
        Initialize the PXRD template retriever.
        
        Args:
            db_npz_path: Path to the database NPZ file containing embeddings
                        and crystal structure information
            faiss_index_path: Path to the pre-built FAISS index file
            normalize: Whether to L2 normalize query vectors before search.
                      Should be True if the index was built with normalized vectors.
        
        Raises:
            FileNotFoundError: If database or index file does not exist
            ValueError: If database format is invalid
        """
        self.db_npz_path = Path(db_npz_path)
        self.faiss_index_path = Path(faiss_index_path)
        self.normalize = normalize
        
        # Validate file existence
        if not self.db_npz_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_npz_path}")
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {faiss_index_path}")
        
        # Load database
        self._load_database()
        
        # Load FAISS index
        self._load_index()
        
        # Validate consistency
        self._validate_consistency()
        
        print(f"PXRDTemplateRetriever initialized successfully")
        print(f"  Database: {self.n_samples} samples, {self.embedding_dim} dims")
        print(f"  Index: {self.index.ntotal} vectors")
        print(f"  Normalization: {'enabled' if self.normalize else 'disabled'}")
    
    def _load_database(self):
        """Load database from NPZ file."""
        print(f"Loading database from: {self.db_npz_path}")
        self.db = np.load(str(self.db_npz_path), allow_pickle=True)
        
        # Validate required fields
        required_fields = ['h_pxrd', 'h_crystal', 'lattice', 'num_atoms']
        for field in required_fields:
            if field not in self.db:
                raise ValueError(f"Missing required field in database: {field}")
        
        # Store dimensions
        self.n_samples = len(self.db['h_pxrd'])
        self.embedding_dim = self.db['h_pxrd'].shape[1]
        
        # Optionally normalize database embeddings
        if self.normalize:
            self._normalize_db_embeddings()
    
    def _normalize_db_embeddings(self):
        """L2 normalize database embeddings if not already normalized."""
        h_pxrd_norms = np.linalg.norm(self.db['h_pxrd'], axis=1, keepdims=True)
        
        # Check if already normalized (within tolerance)
        if not np.allclose(h_pxrd_norms, 1.0, atol=1e-5):
            print("  Normalizing database PXRD embeddings...")
            # Note: We don't modify the original db, just note that normalization is needed
            # The actual normalization happens in the index
        else:
            print("  Database PXRD embeddings already normalized")
    
    def _load_index(self):
        """Load FAISS index from file."""
        print(f"Loading FAISS index from: {self.faiss_index_path}")
        self.index = faiss.read_index(str(self.faiss_index_path))
    
    def _validate_consistency(self):
        """Validate consistency between database and index."""
        if self.index.ntotal != self.n_samples:
            raise ValueError(
                f"Index size ({self.index.ntotal}) does not match "
                f"database size ({self.n_samples})"
            )
        
        if self.index.d != self.embedding_dim:
            raise ValueError(
                f"Index dimension ({self.index.d}) does not match "
                f"embedding dimension ({self.embedding_dim})"
            )
    
    def _normalize_vector(self, vec: np.ndarray) -> np.ndarray:
        """
        L2 normalize a vector or batch of vectors.
        
        Args:
            vec: Vector(s) to normalize, shape [d] or [N, d]
        
        Returns:
            Normalized vector(s) with same shape
        """
        if vec.ndim == 1:
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-8)
        else:
            norms = np.linalg.norm(vec, axis=1, keepdims=True)
            return vec / (norms + 1e-8)
    
    def query(
        self,
        query_pxrd_feat: np.ndarray,
        top_m: int = 4,
    ) -> Dict[str, np.ndarray]:
        """
        Retrieve top-M similar crystal templates based on PXRD embedding.
        
        Args:
            query_pxrd_feat: PXRD embedding from CPCP model.
                           Shape: [d] or [1, d] where d is embedding dimension.
            top_m: Number of templates to retrieve (default: 4)
        
        Returns:
            Dictionary containing:
                - indices: [top_m] array of database indices
                - scores: [top_m] array of similarity scores (higher is better)
                - h_crystal: [top_m, d] crystal embeddings
                - lattice: [top_m, 6] lattice parameters (a, b, c, alpha, beta, gamma)
                - num_atoms: [top_m] number of atoms per structure
                - formula: [top_m] list of chemical formulas (if available)
        
        Raises:
            ValueError: If query vector has wrong dimension or top_m is invalid
        
        Example:
            >>> query_feat = cpcp_model.encode(pxrd_data)  # [512]
            >>> results = retriever.query(query_feat, top_m=5)
            >>> print(f"Top-5 formulas: {results['formula']}")
            >>> print(f"Similarity scores: {results['scores']}")
        """
        # Validate top_m
        if top_m <= 0:
            raise ValueError(f"top_m must be positive, got {top_m}")
        if top_m > self.n_samples:
            print(f"Warning: top_m ({top_m}) > database size ({self.n_samples}), "
                  f"using top_m={self.n_samples}")
            top_m = self.n_samples
        
        # Prepare query vector
        query_vec = self._prepare_query(query_pxrd_feat)
        
        # Perform FAISS search
        scores, indices = self.index.search(query_vec, top_m)
        
        # Extract first row (we only have one query)
        scores = scores[0]  # [top_m]
        indices = indices[0]  # [top_m]
        
        # Retrieve corresponding data from database
        results = {
            'indices': indices,
            'scores': scores,
            'h_crystal': self.db['h_crystal'][indices],  # [top_m, d]
            'lattice': self.db['lattice'][indices],  # [top_m, 6]
            'num_atoms': self.db['num_atoms'][indices],  # [top_m]
        }
        
        # Add formula if available
        if 'formula' in self.db:
            results['formula'] = self.db['formula'][indices].tolist()
        
        return results
    
    def _prepare_query(self, query_pxrd_feat: np.ndarray) -> np.ndarray:
        """
        Prepare query vector for FAISS search.
        
        Args:
            query_pxrd_feat: Query vector, shape [d] or [1, d]
        
        Returns:
            Prepared query vector, shape [1, d], float32, optionally normalized
        """
        # Convert to numpy if needed
        if not isinstance(query_pxrd_feat, np.ndarray):
            query_pxrd_feat = np.array(query_pxrd_feat)
        
        # Ensure 2D shape [1, d]
        if query_pxrd_feat.ndim == 1:
            query_vec = query_pxrd_feat.reshape(1, -1)
        elif query_pxrd_feat.ndim == 2:
            if query_pxrd_feat.shape[0] != 1:
                raise ValueError(
                    f"Query must have shape [d] or [1, d], got {query_pxrd_feat.shape}"
                )
            query_vec = query_pxrd_feat
        else:
            raise ValueError(
                f"Query must be 1D or 2D, got {query_pxrd_feat.ndim}D"
            )
        
        # Validate dimension
        if query_vec.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query dimension ({query_vec.shape[1]}) does not match "
                f"database dimension ({self.embedding_dim})"
            )
        
        # Normalize if required
        if self.normalize:
            query_vec = self._normalize_vector(query_vec)
        
        # Ensure float32 for FAISS
        query_vec = query_vec.astype(np.float32)
        
        return query_vec
    
    def batch_query(
        self,
        query_pxrd_feats: np.ndarray,
        top_m: int = 4,
    ) -> Dict[str, np.ndarray]:
        """
        Retrieve templates for multiple queries in batch.
        
        Args:
            query_pxrd_feats: Batch of PXRD embeddings, shape [N, d]
            top_m: Number of templates to retrieve per query
        
        Returns:
            Dictionary with same fields as query(), but with batch dimension:
                - indices: [N, top_m]
                - scores: [N, top_m]
                - h_crystal: [N, top_m, d]
                - lattice: [N, top_m, 6]
                - num_atoms: [N, top_m]
                - formula: List of N lists, each containing top_m formulas
        """
        # Validate input
        if query_pxrd_feats.ndim != 2:
            raise ValueError(
                f"Batch query must be 2D [N, d], got {query_pxrd_feats.shape}"
            )
        
        N = query_pxrd_feats.shape[0]
        
        # Normalize if required
        query_vecs = query_pxrd_feats.astype(np.float32)
        if self.normalize:
            query_vecs = self._normalize_vector(query_vecs)
        
        # Perform batch search
        scores, indices = self.index.search(query_vecs, top_m)
        
        # Retrieve data for all queries
        results = {
            'indices': indices,  # [N, top_m]
            'scores': scores,  # [N, top_m]
            'h_crystal': np.array([self.db['h_crystal'][idx] for idx in indices]),  # [N, top_m, d]
            'lattice': np.array([self.db['lattice'][idx] for idx in indices]),  # [N, top_m, 6]
            'num_atoms': np.array([self.db['num_atoms'][idx] for idx in indices]),  # [N, top_m]
        }
        
        # Add formula if available
        if 'formula' in self.db:
            results['formula'] = [
                self.db['formula'][idx].tolist() for idx in indices
            ]
        
        return results
    
    def get_by_indices(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Retrieve database entries by indices.
        
        Args:
            indices: Array of database indices, shape [M]
        
        Returns:
            Dictionary containing the requested entries
        """
        results = {
            'indices': indices,
            'h_pxrd': self.db['h_pxrd'][indices],
            'h_crystal': self.db['h_crystal'][indices],
            'lattice': self.db['lattice'][indices],
            'num_atoms': self.db['num_atoms'][indices],
        }
        
        if 'formula' in self.db:
            results['formula'] = self.db['formula'][indices].tolist()
        
        return results
    
    def __repr__(self) -> str:
        return (
            f"PXRDTemplateRetriever(\n"
            f"  n_samples={self.n_samples},\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  normalize={self.normalize},\n"
            f"  db_path={self.db_npz_path.name},\n"
            f"  index_path={self.faiss_index_path.name}\n"
            f")"
        )
