#!/usr/bin/env python
"""
Test script for PXRDTemplateRetriever

This script demonstrates how to use the PXRDTemplateRetriever class
to perform PXRD-based crystal structure retrieval.

Usage:
    python scripts/test_pxrd_retriever.py \
        --db_path outputs/retrieval/hmof100_train_db.npz \
        --index_path outputs/retrieval/hmof100_train_pxrd.index \
        --num_queries 5 \
        --top_m 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from xtalnet.retrieval import PXRDTemplateRetriever


def test_single_query(retriever, query_idx, top_m=10):
    """Test single query retrieval."""
    print("\n" + "="*70)
    print(f"TEST 1: Single Query (query_idx={query_idx}, top_m={top_m})")
    print("="*70)
    
    # Get a query vector from the database
    query_pxrd = retriever.db['h_pxrd'][query_idx]
    query_formula = retriever.db['formula'][query_idx] if 'formula' in retriever.db else 'N/A'
    query_num_atoms = retriever.db['num_atoms'][query_idx]
    
    print(f"\nQuery sample:")
    print(f"  Index: {query_idx}")
    print(f"  Formula: {query_formula}")
    print(f"  Num atoms: {query_num_atoms}")
    print(f"  Embedding shape: {query_pxrd.shape}")
    print(f"  Embedding norm: {np.linalg.norm(query_pxrd):.6f}")
    
    # Perform retrieval
    print(f"\nPerforming retrieval (top-{top_m})...")
    results = retriever.query(query_pxrd, top_m=top_m)
    
    # Display results
    print(f"\nRetrieval results:")
    print(f"{'Rank':<6} {'Index':<8} {'Score':<10} {'Formula':<20} {'Atoms':<8} {'Self?':<6}")
    print("-" * 70)
    
    for rank, (idx, score, formula, n_atoms) in enumerate(zip(
        results['indices'],
        results['scores'],
        results.get('formula', ['N/A'] * top_m),
        results['num_atoms']
    ), 1):
        is_self = "✓" if idx == query_idx else ""
        print(f"{rank:<6} {idx:<8} {score:<10.6f} {formula:<20} {n_atoms:<8} {is_self:<6}")
    
    # Verify self-retrieval
    if results['indices'][0] == query_idx:
        print(f"\n✓ Self-retrieval successful (top-1 is query itself)")
        print(f"  Self-similarity score: {results['scores'][0]:.6f}")
    else:
        print(f"\n⚠ Self-retrieval failed (top-1 is index {results['indices'][0]}, not {query_idx})")
    
    # Check result shapes
    print(f"\nResult shapes:")
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
    
    return results


def test_batch_query(retriever, query_indices, top_m=5):
    """Test batch query retrieval."""
    print("\n" + "="*70)
    print(f"TEST 2: Batch Query (num_queries={len(query_indices)}, top_m={top_m})")
    print("="*70)
    
    # Get query vectors
    query_pxrds = retriever.db['h_pxrd'][query_indices]
    
    print(f"\nQuery batch:")
    print(f"  Indices: {query_indices}")
    print(f"  Batch shape: {query_pxrds.shape}")
    
    # Perform batch retrieval
    print(f"\nPerforming batch retrieval...")
    results = retriever.batch_query(query_pxrds, top_m=top_m)
    
    # Display results for each query
    for i, query_idx in enumerate(query_indices):
        query_formula = retriever.db['formula'][query_idx] if 'formula' in retriever.db else 'N/A'
        print(f"\nQuery {i+1} (idx={query_idx}, formula={query_formula}):")
        print(f"  Top-{top_m} indices: {results['indices'][i]}")
        print(f"  Top-{top_m} scores: {results['scores'][i]}")
        
        # Check self-retrieval
        if results['indices'][i][0] == query_idx:
            print(f"  ✓ Self-retrieval successful")
        else:
            print(f"  ⚠ Self-retrieval failed")
    
    # Check result shapes
    print(f"\nBatch result shapes:")
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
    
    return results


def test_edge_cases(retriever):
    """Test edge cases and error handling."""
    print("\n" + "="*70)
    print("TEST 3: Edge Cases")
    print("="*70)
    
    # Test 1: Query with 1D vector
    print("\n1. Testing 1D query vector...")
    query_1d = retriever.db['h_pxrd'][0]
    try:
        results = retriever.query(query_1d, top_m=3)
        print(f"   ✓ 1D query successful, got {len(results['indices'])} results")
    except Exception as e:
        print(f"   ✗ 1D query failed: {e}")
    
    # Test 2: Query with 2D vector [1, d]
    print("\n2. Testing 2D query vector [1, d]...")
    query_2d = retriever.db['h_pxrd'][0:1]
    try:
        results = retriever.query(query_2d, top_m=3)
        print(f"   ✓ 2D query successful, got {len(results['indices'])} results")
    except Exception as e:
        print(f"   ✗ 2D query failed: {e}")
    
    # Test 3: top_m = 1
    print("\n3. Testing top_m=1...")
    try:
        results = retriever.query(retriever.db['h_pxrd'][0], top_m=1)
        print(f"   ✓ top_m=1 successful, got {len(results['indices'])} results")
    except Exception as e:
        print(f"   ✗ top_m=1 failed: {e}")
    
    # Test 4: top_m > database size
    print(f"\n4. Testing top_m > database size (top_m={retriever.n_samples + 10})...")
    try:
        results = retriever.query(retriever.db['h_pxrd'][0], top_m=retriever.n_samples + 10)
        print(f"   ✓ Large top_m handled, got {len(results['indices'])} results")
    except Exception as e:
        print(f"   ✗ Large top_m failed: {e}")
    
    # Test 5: get_by_indices
    print("\n5. Testing get_by_indices...")
    try:
        indices = np.array([0, 10, 20, 30])
        results = retriever.get_by_indices(indices)
        print(f"   ✓ get_by_indices successful, got {len(results['indices'])} entries")
    except Exception as e:
        print(f"   ✗ get_by_indices failed: {e}")


def test_retrieval_quality(retriever, num_samples=10):
    """Test retrieval quality metrics."""
    print("\n" + "="*70)
    print(f"TEST 4: Retrieval Quality (num_samples={num_samples})")
    print("="*70)
    
    # Random sample queries
    np.random.seed(42)
    query_indices = np.random.choice(retriever.n_samples, size=num_samples, replace=False)
    
    # Metrics
    self_retrieval_success = 0
    avg_self_score = 0.0
    avg_top2_score = 0.0
    
    print("\nTesting self-retrieval accuracy...")
    for query_idx in query_indices:
        query_pxrd = retriever.db['h_pxrd'][query_idx]
        results = retriever.query(query_pxrd, top_m=5)
        
        # Check if self is top-1
        if results['indices'][0] == query_idx:
            self_retrieval_success += 1
            avg_self_score += results['scores'][0]
        
        # Average top-2 score
        avg_top2_score += results['scores'][1]
    
    # Compute metrics
    self_retrieval_rate = self_retrieval_success / num_samples
    avg_self_score /= num_samples
    avg_top2_score /= num_samples
    
    print(f"\nRetrieval quality metrics:")
    print(f"  Self-retrieval rate: {self_retrieval_rate:.2%} ({self_retrieval_success}/{num_samples})")
    print(f"  Average self-similarity: {avg_self_score:.6f}")
    print(f"  Average top-2 similarity: {avg_top2_score:.6f}")
    print(f"  Similarity gap: {avg_self_score - avg_top2_score:.6f}")
    
    if self_retrieval_rate == 1.0:
        print("\n✓ Perfect self-retrieval! Index is working correctly.")
    else:
        print(f"\n⚠ Self-retrieval rate < 100%. Check index construction.")


def main():
    parser = argparse.ArgumentParser(description="Test PXRDTemplateRetriever")
    parser.add_argument('--db_path', type=str, required=True,
                       help='Path to database NPZ file')
    parser.add_argument('--index_path', type=str, required=True,
                       help='Path to FAISS index file')
    parser.add_argument('--num_queries', type=int, default=5,
                       help='Number of random queries to test')
    parser.add_argument('--top_m', type=int, default=10,
                       help='Number of templates to retrieve')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PXRD TEMPLATE RETRIEVER TEST")
    print("="*70)
    print(f"Database: {args.db_path}")
    print(f"Index: {args.index_path}")
    print("="*70)
    
    # Initialize retriever
    print("\nInitializing retriever...")
    try:
        retriever = PXRDTemplateRetriever(
            db_npz_path=args.db_path,
            faiss_index_path=args.index_path,
            normalize=True
        )
        print(f"\n{retriever}")
    except Exception as e:
        print(f"\n✗ Failed to initialize retriever: {e}")
        return
    
    # Test 1: Single query
    np.random.seed(42)
    query_idx = np.random.randint(0, retriever.n_samples)
    test_single_query(retriever, query_idx, top_m=args.top_m)
    
    # Test 2: Batch query
    query_indices = np.random.choice(retriever.n_samples, size=min(3, retriever.n_samples), replace=False)
    test_batch_query(retriever, query_indices, top_m=5)
    
    # Test 3: Edge cases
    test_edge_cases(retriever)
    
    # Test 4: Retrieval quality
    test_retrieval_quality(retriever, num_samples=min(args.num_queries, retriever.n_samples))
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
