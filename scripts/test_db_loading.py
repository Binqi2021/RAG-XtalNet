#!/usr/bin/env python
"""
Test script to verify database loading and basic retrieval functionality.

Usage:
    python scripts/test_db_loading.py \
        --db_path outputs/retrieval/hmof100_train_db.npz \
        --index_path outputs/retrieval/hmof100_train_pxrd.index
"""

import argparse
import numpy as np
import faiss


def test_database_loading(db_path, index_path):
    """Test loading and basic operations on the database."""
    
    print("="*60)
    print("DATABASE LOADING TEST")
    print("="*60)
    
    # Load database
    print(f"\n1. Loading database from: {db_path}")
    try:
        db = np.load(db_path)
        print("   ✓ Database loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load database: {e}")
        return False
    
    # Check database fields
    print("\n2. Checking database fields...")
    required_fields = ['h_pxrd', 'h_crystal', 'lattice', 'num_atoms', 'formula']
    for field in required_fields:
        if field in db:
            print(f"   ✓ {field}: {db[field].shape}")
        else:
            print(f"   ✗ Missing field: {field}")
            return False
    
    # Verify data consistency
    print("\n3. Verifying data consistency...")
    N = len(db['h_pxrd'])
    checks = [
        (len(db['h_crystal']) == N, f"h_crystal length matches ({len(db['h_crystal'])} == {N})"),
        (len(db['lattice']) == N, f"lattice length matches ({len(db['lattice'])} == {N})"),
        (len(db['num_atoms']) == N, f"num_atoms length matches ({len(db['num_atoms'])} == {N})"),
        (len(db['formula']) == N, f"formula length matches ({len(db['formula'])} == {N})"),
        (db['h_pxrd'].shape[1] == db['h_crystal'].shape[1], 
         f"embedding dimensions match ({db['h_pxrd'].shape[1]} == {db['h_crystal'].shape[1]})"),
        (db['lattice'].shape[1] == 6, f"lattice has 6 parameters ({db['lattice'].shape[1]} == 6)"),
    ]
    
    all_passed = True
    for passed, msg in checks:
        if passed:
            print(f"   ✓ {msg}")
        else:
            print(f"   ✗ {msg}")
            all_passed = False
    
    if not all_passed:
        return False
    
    # Load FAISS index
    print(f"\n4. Loading FAISS index from: {index_path}")
    try:
        index = faiss.read_index(index_path)
        print("   ✓ FAISS index loaded successfully")
        print(f"   - Index type: {type(index).__name__}")
        print(f"   - Number of vectors: {index.ntotal}")
        print(f"   - Dimension: {index.d}")
    except Exception as e:
        print(f"   ✗ Failed to load FAISS index: {e}")
        return False
    
    # Verify index consistency
    print("\n5. Verifying index consistency...")
    if index.ntotal == N:
        print(f"   ✓ Index size matches database ({index.ntotal} == {N})")
    else:
        print(f"   ✗ Index size mismatch ({index.ntotal} != {N})")
        return False
    
    if index.d == db['h_pxrd'].shape[1]:
        print(f"   ✓ Index dimension matches embeddings ({index.d} == {db['h_pxrd'].shape[1]})")
    else:
        print(f"   ✗ Index dimension mismatch ({index.d} != {db['h_pxrd'].shape[1]})")
        return False
    
    # Test retrieval
    print("\n6. Testing retrieval functionality...")
    try:
        # Use first sample as query
        query = db['h_pxrd'][0:1].astype(np.float32)
        k = min(10, N)
        
        distances, indices = index.search(query, k)
        
        print(f"   ✓ Retrieval successful (k={k})")
        print(f"\n   Top-{k} results:")
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            formula = db['formula'][idx]
            n_atoms = db['num_atoms'][idx]
            print(f"     Rank {i+1}: idx={idx:4d}, dist={dist:.4f}, formula={formula:15s}, n_atoms={n_atoms:3d}")
        
        # Verify self-retrieval
        if indices[0][0] == 0:
            print(f"\n   ✓ Self-retrieval correct (top result is query itself)")
        else:
            print(f"\n   ⚠ Self-retrieval unexpected (top result is {indices[0][0]}, not 0)")
        
        # Check distance values
        if distances[0][0] > 0.99:  # Should be ~1.0 for normalized vectors
            print(f"   ✓ Self-similarity high ({distances[0][0]:.4f} > 0.99)")
        else:
            print(f"   ⚠ Self-similarity low ({distances[0][0]:.4f} <= 0.99)")
        
    except Exception as e:
        print(f"   ✗ Retrieval failed: {e}")
        return False
    
    # Test embedding normalization
    print("\n7. Checking embedding normalization...")
    pxrd_norms = np.linalg.norm(db['h_pxrd'], axis=1)
    crystal_norms = np.linalg.norm(db['h_crystal'], axis=1)
    
    pxrd_normalized = np.allclose(pxrd_norms, 1.0, atol=1e-5)
    crystal_normalized = np.allclose(crystal_norms, 1.0, atol=1e-5)
    
    if pxrd_normalized:
        print(f"   ✓ PXRD embeddings normalized (mean norm: {pxrd_norms.mean():.6f})")
    else:
        print(f"   ⚠ PXRD embeddings not normalized (mean norm: {pxrd_norms.mean():.6f})")
    
    if crystal_normalized:
        print(f"   ✓ Crystal embeddings normalized (mean norm: {crystal_norms.mean():.6f})")
    else:
        print(f"   ⚠ Crystal embeddings not normalized (mean norm: {crystal_norms.mean():.6f})")
    
    # Summary statistics
    print("\n8. Summary statistics:")
    print(f"   - Total samples: {N}")
    print(f"   - Embedding dimension: {db['h_pxrd'].shape[1]}")
    print(f"   - Atoms per structure: {db['num_atoms'].min()}-{db['num_atoms'].max()} (mean: {db['num_atoms'].mean():.1f})")
    print(f"   - Unique formulas: {len(np.unique(db['formula']))}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test database loading and retrieval")
    parser.add_argument('--db_path', type=str, required=True, help='Path to database NPZ file')
    parser.add_argument('--index_path', type=str, required=True, help='Path to FAISS index file')
    
    args = parser.parse_args()
    
    success = test_database_loading(args.db_path, args.index_path)
    
    if not success:
        print("\n" + "="*60)
        print("TESTS FAILED ✗")
        print("="*60)
        exit(1)


if __name__ == "__main__":
    main()
