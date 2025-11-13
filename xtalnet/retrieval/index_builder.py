"""Build FAISS index from CPCP embeddings."""

import torch
import numpy as np
import pickle
import faiss
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional

from torch_geometric.loader import DataLoader
from xtalnet.pl_data.dataset import CrystMOFLMDBDataset
from xtalnet.pl_modules.cpcp_module import CPCPModule


class IndexBuilder:
    """Build and save retrieval index from pretrained CPCP model."""
    
    def __init__(self, 
                 cpcp_ckpt: str,
                 data_path: str,
                 output_dir: str,
                 device: str = 'auto',
                 batch_size: int = 32):
        """
        Args:
            cpcp_ckpt: Path to CPCP checkpoint
            data_path: Path to LMDB dataset (e.g., data/hmof_100/train.lmdb)
            output_dir: Where to save the index
            device: 'auto', 'cpu', or 'cuda'
            batch_size: Batch size for embedding extraction
        """
        self.device = self._setup_device(device)
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        print(f"Loading CPCP model from {cpcp_ckpt}")
        self.model = CPCPModule.load_from_checkpoint(cpcp_ckpt, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)
        
        # Load dataset
        print(f"Loading dataset from {data_path}")
        self.dataset = CrystMOFLMDBDataset(
            name='train', path=data_path, use_pxrd=True, is_training=False
        )
        print(f"Dataset size: {len(self.dataset)}")
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        return device
    
    @torch.no_grad()
    def extract_embeddings(self, max_samples: Optional[int] = None) -> Dict:
        """Extract PXRD and crystal embeddings from all samples."""
        loader = DataLoader(self.dataset, batch_size=self.batch_size, 
                          shuffle=False, num_workers=4)
        
        pxrd_embs, crystal_embs = [], []
        metadata = []
        
        total = min(len(self.dataset), max_samples) if max_samples else len(self.dataset)
        
        for i, batch in enumerate(tqdm(loader, desc="Extracting embeddings")):
            if max_samples and i * self.batch_size >= max_samples:
                break
            
            batch = batch.to(self.device)
            results = self.model.inference(batch)
            
            pxrd_embs.append(results['pxrd_feat'].cpu())
            crystal_embs.append(results['atom_feat'].cpu())
            
            # Store metadata
            for j in range(len(batch)):
                metadata.append({
                    'idx': batch.idx[j].item(),
                    'num_atoms': batch.num_atoms[j].item(),
                    'crystal_emb': results['atom_feat'][j].cpu(),
                    'lattice': results['lattices'][j].cpu(),
                    'lengths': batch.lengths[j].cpu(),
                    'angles': batch.angles[j].cpu(),
                })
        
        pxrd_embs = torch.cat(pxrd_embs, dim=0).numpy().astype('float32')
        
        print(f"Extracted {len(pxrd_embs)} embeddings with shape {pxrd_embs.shape}")
        return {'pxrd': pxrd_embs, 'metadata': metadata}
    
    def build_index(self, embeddings: Dict, index_type: str = 'IVF'):
        """Build FAISS index from embeddings."""
        pxrd_embs = embeddings['pxrd']
        dim = pxrd_embs.shape[1]
        n_samples = len(pxrd_embs)
        
        # Normalize embeddings
        faiss.normalize_L2(pxrd_embs)
        
        # Build index
        if index_type == 'Flat':
            index = faiss.IndexFlatIP(dim)  # Inner product = cosine after normalization
        elif index_type == 'IVF':
            n_clusters = min(int(np.sqrt(n_samples)), 256)
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            index.train(pxrd_embs)
            index.nprobe = 10  # Search more clusters for better recall
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        print(f"Building {index_type} index...")
        index.add(pxrd_embs)
        print(f"Index built with {index.ntotal} vectors")
        
        return index
    
    def save(self, index, metadata: list):
        """Save index and metadata."""
        # Save FAISS index
        index_file = self.output_dir / 'faiss_index.bin'
        faiss.write_index(faiss.index_gpu_to_cpu(index) if hasattr(index, 'getDevice') else index, 
                         str(index_file))
        
        # Save metadata
        with open(self.output_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save config
        config = {
            'n_samples': len(metadata),
            'embedding_dim': metadata[0]['crystal_emb'].shape[0],
            'device': self.device,
        }
        with open(self.output_dir / 'config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Index saved to {self.output_dir}")
    
    def build_and_save(self, max_samples: Optional[int] = None):
        """Complete pipeline: extract, build, save."""
        embeddings = self.extract_embeddings(max_samples)
        index = self.build_index(embeddings, index_type='Flat' if len(embeddings['metadata']) < 10000 else 'IVF')
        self.save(index, embeddings['metadata'])
        return self.output_dir

