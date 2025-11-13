"""Test retrieval quality with various metrics."""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_geometric.loader import DataLoader
from xtalnet.retrieval.retriever import CrystalRetriever
from xtalnet.pl_data.dataset import CrystMOFLMDBDataset
from xtalnet.pl_modules.cpcp_module import CPCPModule


def compute_recall(retriever, model, dataset, k_values=[1, 5, 10], n_queries=100, device='cpu'):
    """Compute retrieval recall@k metrics."""
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    recalls = {k: [] for k in k_values}
    
    print(f"\nTesting retrieval on {n_queries} random queries...")
    
    for i, batch in enumerate(loader):
        if i >= n_queries:
            break
        
        batch = batch.to(device)
        
        # Get query embedding
        with torch.no_grad():
            pxrd_emb = model.pxrd_encoder(batch)['cls_token']
        
        # Retrieve
        results = retriever.search(pxrd_emb, k=max(k_values))
        
        # Check if query's own index is in top-k
        query_idx = batch.idx[0].item()
        retrieved_indices = results['indices'][0]
        
        for k in k_values:
            recalls[k].append(1.0 if query_idx in retrieved_indices[:k] else 0.0)
    
    # Print results
    print("\n" + "="*60)
    print("RETRIEVAL QUALITY METRICS")
    print("="*60)
    for k in k_values:
        recall = np.mean(recalls[k]) * 100
        print(f"Recall@{k:2d}: {recall:6.2f}%")
    print("="*60 + "\n")


def visualize_retrieval(retriever, model, dataset, device='cpu'):
    """Show example retrieval results."""
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    batch = next(iter(loader)).to(device)
    
    # Get query
    with torch.no_grad():
        pxrd_emb = model.pxrd_encoder(batch)['cls_token']
    
    # Retrieve
    results = retriever.search(pxrd_emb, k=5)
    
    print("\n" + "="*60)
    print("EXAMPLE RETRIEVAL")
    print("="*60)
    print(f"Query structure: idx={batch.idx[0].item()}, n_atoms={batch.num_atoms[0].item()}")
    print("\nTop-5 retrieved structures:")
    for i, (idx, dist, meta) in enumerate(zip(results['indices'][0], 
                                               results['distances'][0],
                                               results['metadata'][0])):
        print(f"  {i+1}. idx={idx:5d}, n_atoms={meta['num_atoms']:3d}, similarity={dist:.4f}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Test retrieval index quality')
    parser.add_argument('--index_path', type=str, 
                       default='retrieval_indices/hmof_100',
                       help='Path to retrieval index')
    parser.add_argument('--cpcp_ckpt', type=str,
                       default='ckpt/hmof_100/CPCP/hmof_100_cpcp.ckpt',
                       help='Path to CPCP checkpoint')
    parser.add_argument('--data_path', type=str,
                       default='data/hmof_100/val.lmdb',
                       help='Path to validation LMDB')
    parser.add_argument('--n_queries', type=int, default=100,
                       help='Number of test queries')
    parser.add_argument('--device', type=str, default='auto')
    
    args = parser.parse_args()
    
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    print(f"Using device: {device}")
    
    # Load components
    print("Loading retriever...")
    retriever = CrystalRetriever(args.index_path, device=device)
    
    print("Loading CPCP model...")
    model = CPCPModule.load_from_checkpoint(args.cpcp_ckpt, map_location=device)
    model.eval()
    
    print("Loading validation dataset...")
    dataset = CrystMOFLMDBDataset(
        name='val', path=args.data_path, use_pxrd=True, is_training=False
    )
    
    # Run tests
    visualize_retrieval(retriever, model, dataset, device)
    compute_recall(retriever, model, dataset, k_values=[1, 5, 10, 20], 
                  n_queries=args.n_queries, device=device)


if __name__ == '__main__':
    main()

