#!/usr/bin/env python
"""
Example: Using PXRDTemplateRetriever in practice

This script demonstrates practical usage of the retriever in different scenarios:
1. Standalone retrieval
2. Integration with CPCP model
3. Retrieval for RAG-enhanced generation

Usage:
    # With dataset/split (recommended)
    python scripts/example_retrieval_usage.py --data_name hmof_100 --split train --cpcp_ckpt_path logs/cpcp_hmof100/checkpoints/last.ckpt

    # With custom paths
    python scripts/example_retrieval_usage.py \
        --db_path outputs/retrieval/hmof100_train_db.npz \
        --index_path outputs/retrieval/hmof100_train_pxrd.index \
        --cpcp_ckpt_path logs/cpcp/checkpoints/last.ckpt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xtalnet.retrieval import PXRDTemplateRetriever


def example_1_standalone_retrieval(db_path, index_path):
    """Example 1: Standalone retrieval without model."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Standalone Retrieval")
    print("="*70)
    
    # Initialize retriever
    print("\n1. Initializing retriever...")
    retriever = PXRDTemplateRetriever(
        db_npz_path=db_path,
        faiss_index_path=index_path,
        normalize=True
    )
    
    # Use a sample from the database as query
    print("\n2. Selecting a query sample...")
    query_idx = 42
    query_pxrd = retriever.db['h_pxrd'][query_idx]
    query_info = {
        'formula': retriever.db['formula'][query_idx],
        'num_atoms': retriever.db['num_atoms'][query_idx],
        'lattice': retriever.db['lattice'][query_idx]
    }
    
    print(f"   Query sample {query_idx}:")
    print(f"     Formula: {query_info['formula']}")
    print(f"     Atoms: {query_info['num_atoms']}")
    print(f"     Lattice: a={query_info['lattice'][0]:.3f}, "
          f"b={query_info['lattice'][1]:.3f}, "
          f"c={query_info['lattice'][2]:.3f}")
    
    # Retrieve similar structures
    print("\n3. Retrieving top-5 similar structures...")
    results = retriever.query(query_pxrd, top_m=5)
    
    print("\n   Results:")
    print(f"   {'Rank':<6} {'Index':<8} {'Score':<10} {'Formula':<20} {'Atoms':<8}")
    print("   " + "-"*60)
    for rank, (idx, score, formula, n_atoms) in enumerate(zip(
        results['indices'],
        results['scores'],
        results['formula'],
        results['num_atoms']
    ), 1):
        print(f"   {rank:<6} {idx:<8} {score:<10.6f} {formula:<20} {n_atoms:<8}")
    
    return retriever, results


def example_2_cpcp_integration(retriever, cpcp_ckpt_path):
    """Example 2: Integration with CPCP model."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Integration with CPCP Model")
    print("="*70)
    
    try:
        from eval_utils import load_model_ckpt
        
        # Load CPCP model
        print("\n1. Loading CPCP model...")
        model_path = Path(cpcp_ckpt_path).parent
        cpcp_model, test_loader, cfg = load_model_ckpt(
            model_path,
            cpcp_ckpt_path,
            load_data=True,
            testing=True
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cpcp_model = cpcp_model.to(device)
        cpcp_model.eval()
        
        print(f"   Model loaded on {device}")
        
        # Get a test sample
        print("\n2. Processing a test sample...")
        batch = next(iter(test_loader))
        batch = batch.to(device)
        
        # Extract PXRD embedding
        with torch.no_grad():
            outputs = cpcp_model.inference(batch)
            pxrd_feat = outputs['pxrd_feat'][0].cpu().numpy()  # [512]
        
        print(f"   PXRD embedding shape: {pxrd_feat.shape}")
        print(f"   PXRD embedding norm: {np.linalg.norm(pxrd_feat):.6f}")
        
        # Retrieve templates
        print("\n3. Retrieving templates...")
        results = retriever.query(pxrd_feat, top_m=5)
        
        print("\n   Retrieved templates:")
        for i, (idx, score, formula) in enumerate(zip(
            results['indices'],
            results['scores'],
            results['formula']
        ), 1):
            print(f"   {i}. Index={idx}, Score={score:.4f}, Formula={formula}")
        
        return results
        
    except Exception as e:
        print(f"\n   ⚠ Could not load CPCP model: {e}")
        print("   Skipping this example...")
        return None


def example_3_rag_preparation(retriever):
    """Example 3: Prepare templates for RAG-enhanced generation."""
    print("\n" + "="*70)
    print("EXAMPLE 3: RAG Template Preparation")
    print("="*70)
    
    # Simulate a query
    print("\n1. Simulating a query PXRD embedding...")
    query_idx = 100
    query_pxrd = retriever.db['h_pxrd'][query_idx]
    
    # Retrieve templates
    print("\n2. Retrieving templates for RAG...")
    top_m = 4  # Typical number for RAG
    results = retriever.query(query_pxrd, top_m=top_m)
    
    print(f"\n   Retrieved {top_m} templates:")
    
    # Prepare template information for RAG
    templates = []
    for i in range(top_m):
        template = {
            'index': results['indices'][i],
            'score': results['scores'][i],
            'h_crystal': results['h_crystal'][i],  # [512]
            'lattice': results['lattice'][i],  # [6]
            'num_atoms': results['num_atoms'][i],
            'formula': results['formula'][i]
        }
        templates.append(template)
        
        print(f"\n   Template {i+1}:")
        print(f"     Score: {template['score']:.6f}")
        print(f"     Formula: {template['formula']}")
        print(f"     Atoms: {template['num_atoms']}")
        print(f"     Lattice: a={template['lattice'][0]:.3f}, "
              f"b={template['lattice'][1]:.3f}, "
              f"c={template['lattice'][2]:.3f}")
    
    # Aggregate template information
    print("\n3. Aggregating template information...")
    
    # Average crystal embeddings
    avg_crystal_emb = np.mean(results['h_crystal'], axis=0)
    print(f"   Average crystal embedding shape: {avg_crystal_emb.shape}")
    
    # Average lattice parameters
    avg_lattice = np.mean(results['lattice'], axis=0)
    print(f"   Average lattice: a={avg_lattice[0]:.3f}, "
          f"b={avg_lattice[1]:.3f}, "
          f"c={avg_lattice[2]:.3f}")
    
    # Weighted average by scores
    weights = results['scores'] / results['scores'].sum()
    weighted_crystal_emb = np.sum(
        results['h_crystal'] * weights[:, np.newaxis],
        axis=0
    )
    print(f"   Weighted crystal embedding shape: {weighted_crystal_emb.shape}")
    
    print("\n   These aggregated features can be used to condition the CCSG model!")
    
    return templates


def example_4_batch_retrieval(retriever):
    """Example 4: Batch retrieval for multiple queries."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Retrieval")
    print("="*70)
    
    # Prepare batch of queries
    print("\n1. Preparing batch of queries...")
    batch_size = 3
    query_indices = [10, 50, 100]
    query_batch = retriever.db['h_pxrd'][query_indices]
    
    print(f"   Batch size: {batch_size}")
    print(f"   Query indices: {query_indices}")
    print(f"   Batch shape: {query_batch.shape}")
    
    # Batch retrieval
    print("\n2. Performing batch retrieval...")
    results = retriever.batch_query(query_batch, top_m=3)
    
    print("\n   Batch results:")
    for i, query_idx in enumerate(query_indices):
        print(f"\n   Query {i+1} (index={query_idx}):")
        print(f"     Top-3 indices: {results['indices'][i]}")
        print(f"     Top-3 scores: {results['scores'][i]}")
        print(f"     Top-3 formulas: {results['formula'][i]}")


def get_database_path(data_name, split):
    """Generate standardized database path for given dataset and split."""
    return f"outputs/retrieval/{data_name}_{split}_db.npz"


def get_index_path(data_name, split):
    """Generate standardized index path for given dataset and split."""
    return f"outputs/retrieval/{data_name}_{split}_pxrd.index"


def main():
    parser = argparse.ArgumentParser(description="Example retrieval usage")

    # Dataset selection (alternative to custom paths)
    parser.add_argument('--data_name', type=str, choices=['hmof_100', 'hmof_400'],
                       help='Dataset name (alternative to custom paths)')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'],
                       help='Data split (alternative to custom paths)')

    # Custom paths
    parser.add_argument('--db_path', type=str,
                       help='Path to database NPZ file (overrides dataset/split)')
    parser.add_argument('--index_path', type=str,
                       help='Path to FAISS index file (overrides dataset/split)')

    # CPCP model
    parser.add_argument('--cpcp_ckpt_path', type=str, default=None,
                       help='Path to CPCP checkpoint (optional)')

    args = parser.parse_args()

    # Determine database paths
    if args.data_name and args.split:
        # Use dataset/split to generate paths
        db_path = get_database_path(args.data_name, args.split)
        index_path = get_index_path(args.data_name, args.split)
        print(f"Using database: {db_path}")
        print(f"Using index: {index_path}")
    elif args.db_path and args.index_path:
        # Use custom paths
        db_path = args.db_path
        index_path = args.index_path
        print(f"Using custom database: {db_path}")
        print(f"Using custom index: {index_path}")
    else:
        print("Error: Either provide --data_name and --split, or --db_path and --index_path")
        sys.exit(1)

    # Check if files exist
    if not Path(db_path).exists():
        print(f"Error: Database file not found: {db_path}")
        print("Build the database first using: python scripts/build_unified_db.py")
        sys.exit(1)

    if not Path(index_path).exists():
        print(f"Error: Index file not found: {index_path}")
        print("Build the database first using: python scripts/build_unified_db.py")
        sys.exit(1)

    print("="*70)
    print("PXRD TEMPLATE RETRIEVER - USAGE EXAMPLES")
    print("="*70)

    # Example 1: Standalone retrieval
    retriever, _ = example_1_standalone_retrieval(db_path, index_path)

    # Example 2: CPCP integration (if checkpoint provided)
    if args.cpcp_ckpt_path:
        example_2_cpcp_integration(retriever, args.cpcp_ckpt_path)
    else:
        print("\n⚠ Skipping Example 2 (no CPCP checkpoint provided)")
        print("   Provide --cpcp_ckpt_path to run CPCP integration example")

    # Example 3: RAG preparation
    example_3_rag_preparation(retriever)

    # Example 4: Batch retrieval
    example_4_batch_retrieval(retriever)

    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nNext steps:")
    print("  1. Integrate retriever into CCSG sampling")
    print("  2. Design feature fusion strategies")
    print("  3. Evaluate RAG-enhanced generation quality")

    if args.data_name:
        print(f"\nDataset info:")
        print(f"  Dataset: {args.data_name}")
        print(f"  Split: {args.split}")
        print(f"  Database size: {len(retriever.db['h_pxrd'])} samples")


if __name__ == "__main__":
    main()
