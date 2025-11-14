#!/usr/bin/env python
"""
Build PXRD-Crystal Retrieval Database

This script generates PXRD and Crystal embeddings using a trained CPCP model,
and constructs a FAISS index for efficient retrieval.

Output Files:
    {save_prefix}_db.npz: Database containing:
        - h_pxrd: PXRD embeddings [N, d]
        - h_crystal: Crystal embeddings [N, d]
        - lattice: Lattice parameters [N, 6] (a, b, c, alpha, beta, gamma)
        - num_atoms: Number of atoms per structure [N]
        - formula: Chemical formulas [N] (if available)
    
    {save_prefix}_pxrd.index: FAISS index for PXRD embeddings (cosine similarity)

Usage:
    # With automatic path generation (recommended)
    python scripts/build_pxrd_crystal_db.py \
        --cpcp_ckpt_path <path_to_cpcp_checkpoint> \
        --data_name hmof_100 \
        --split train \
        --device cuda

    # With custom save prefix
    python scripts/build_pxrd_crystal_db.py \
        --cpcp_ckpt_path <path_to_cpcp_checkpoint> \
        --data_name hmof_100 \
        --split train \
        --save_prefix outputs/custom_path \
        --device cuda
"""

import argparse
import time
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import faiss

from eval_utils import load_model_ckpt, lattices_to_params_shape


def normalize_embeddings(embeddings):
    """L2 normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def extract_formula(atom_types):
    """
    Extract chemical formula from atom types.
    
    Args:
        atom_types: Tensor of atomic numbers
    
    Returns:
        Chemical formula string (e.g., "C6H12O6")
    """
    from collections import Counter
    from xtalnet.common.data_utils import chemical_symbols
    
    atom_counts = Counter(atom_types.cpu().numpy().tolist())
    formula_parts = []
    for atomic_num in sorted(atom_counts.keys()):
        count = atom_counts[atomic_num]
        symbol = chemical_symbols[atomic_num]
        if count == 1:
            formula_parts.append(symbol)
        else:
            formula_parts.append(f"{symbol}{count}")
    
    return "".join(formula_parts)



def build_database(model, dataloader, device):
    """
    Build database by extracting embeddings from all samples.
    
    Args:
        model: CPCP model
        dataloader: PyTorch DataLoader
        device: torch device
    
    Returns:
        Dictionary containing all database fields
    """
    model.eval()
    
    # Lists to accumulate data
    h_pxrd_list = []
    h_crystal_list = []
    lattice_list = []
    num_atoms_list = []
    formula_list = []
    
    print("Extracting embeddings from dataset...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Move batch to device
            batch = batch.to(device)
            
            # Get embeddings using inference method
            outputs = model.inference(batch)
            
            # Extract PXRD embeddings [B, d]
            pxrd_feat = outputs['pxrd_feat'].cpu().numpy()
            h_pxrd_list.append(pxrd_feat)
            
            # Extract Crystal embeddings [B, d]
            crystal_feat = outputs['atom_feat'].cpu().numpy()
            h_crystal_list.append(crystal_feat)
            
            # Extract lattice parameters
            # outputs['lattices'] is [B, 3, 3], convert to [B, 6] (a, b, c, alpha, beta, gamma)
            lattices = outputs['lattices']
            lengths, angles = lattices_to_params_shape(lattices)
            # Concatenate lengths and angles: [B, 3] + [B, 3] -> [B, 6]
            lattice_params = torch.cat([lengths, angles], dim=-1).cpu().numpy()
            lattice_list.append(lattice_params)
            
            # Extract number of atoms [B]
            num_atoms = outputs['num_atoms'].cpu().numpy()
            num_atoms_list.append(num_atoms)
            
            # Extract chemical formulas
            # Split atom_types by num_atoms to get per-structure formulas
            atom_types = outputs['atom_types']
            start_idx = 0
            batch_formulas = []
            for n_atoms in num_atoms:
                structure_atoms = atom_types[start_idx:start_idx + n_atoms]
                formula = extract_formula(structure_atoms)
                batch_formulas.append(formula)
                start_idx += n_atoms
            formula_list.extend(batch_formulas)
    
    # Concatenate all batches
    h_pxrd = np.concatenate(h_pxrd_list, axis=0)  # [N, d]
    h_crystal = np.concatenate(h_crystal_list, axis=0)  # [N, d]
    lattice = np.concatenate(lattice_list, axis=0)  # [N, 6]
    num_atoms = np.concatenate(num_atoms_list, axis=0)  # [N]
    formula = np.array(formula_list)  # [N]
    
    # Create database dictionary
    db = {
        'h_pxrd': h_pxrd,
        'h_crystal': h_crystal,
        'lattice': lattice,
        'num_atoms': num_atoms,
        'formula': formula
    }
    
    return db


def build_faiss_index(embeddings, normalize=True):
    """
    Build FAISS index for cosine similarity search.
    
    Args:
        embeddings: numpy array [N, d]
        normalize: whether to L2 normalize (for cosine similarity)
    
    Returns:
        FAISS index
    """
    N, d = embeddings.shape
    
    # Normalize for cosine similarity
    if normalize:
        embeddings = normalize_embeddings(embeddings)
    
    # Create flat inner product index (equivalent to cosine similarity after normalization)
    index = faiss.IndexFlatIP(d)
    
    # Add vectors to index
    index.add(embeddings.astype(np.float32))
    
    return index



def print_statistics(db, index):
    """Print database and index statistics."""
    print("\n" + "="*60)
    print("DATABASE STATISTICS")
    print("="*60)
    
    N, d = db['h_pxrd'].shape
    print(f"Number of samples: {N}")
    print(f"Embedding dimension: {d}")
    print(f"\nEmbedding shapes:")
    print(f"  h_pxrd: {db['h_pxrd'].shape}")
    print(f"  h_crystal: {db['h_crystal'].shape}")
    print(f"  lattice: {db['lattice'].shape}")
    print(f"  num_atoms: {db['num_atoms'].shape}")
    print(f"  formula: {db['formula'].shape}")
    
    print(f"\nNumber of atoms statistics:")
    print(f"  Min: {db['num_atoms'].min()}")
    print(f"  Max: {db['num_atoms'].max()}")
    print(f"  Mean: {db['num_atoms'].mean():.2f}")
    print(f"  Median: {np.median(db['num_atoms']):.2f}")
    
    print(f"\nLattice parameters statistics (a, b, c, alpha, beta, gamma):")
    for i, param in enumerate(['a', 'b', 'c', 'alpha', 'beta', 'gamma']):
        values = db['lattice'][:, i]
        print(f"  {param}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}")
    
    print(f"\nExample formulas (first 10):")
    for i, formula in enumerate(db['formula'][:10]):
        print(f"  {i}: {formula}")
    
    print(f"\nExample PXRD embedding (first sample, first 10 dims):")
    print(f"  {db['h_pxrd'][0, :10]}")
    
    print(f"\nExample Crystal embedding (first sample, first 10 dims):")
    print(f"  {db['h_crystal'][0, :10]}")
    
    print(f"\nFAISS Index:")
    print(f"  Type: IndexFlatIP (Inner Product)")
    print(f"  Number of vectors: {index.ntotal}")
    print(f"  Dimension: {index.d}")
    print(f"  Is trained: {index.is_trained}")
    
    print("="*60 + "\n")


def get_default_save_prefix(data_name, split):
    """Generate default save prefix if not provided."""
    return f"outputs/retrieval/{data_name}_{split}"


def main(args):
    """Main function to build database and index."""

    # Generate save prefix if not provided
    if args.save_prefix is None:
        args.save_prefix = get_default_save_prefix(args.data_name, args.split)

    print("="*60)
    print("BUILDING PXRD-CRYSTAL RETRIEVAL DATABASE")
    print("="*60)
    print(f"CPCP checkpoint: {args.cpcp_ckpt_path}")
    print(f"Data name: {args.data_name}")
    print(f"Split: {args.split}")
    print(f"Save prefix: {args.save_prefix}")
    print(f"Device: {args.device}")
    print("="*60 + "\n")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load CPCP model and data
    print("Loading CPCP model and dataset...")
    model_path = Path(args.cpcp_ckpt_path).parent
    
    # Load model with appropriate data split
    model, dataloader, cfg = load_model_ckpt(
        model_path, 
        args.cpcp_ckpt_path, 
        load_data=True, 
        testing=(args.split == 'test')
    )
    
    # If not test split, need to get train or val loader
    if args.split != 'test':
        # Reload with testing=False to get train/val loaders
        import hydra
        from hydra.experimental import compose
        from hydra import initialize_config_dir
        
        with initialize_config_dir(str(model_path)):
            cfg = compose(config_name='hparams')
            cfg.data.datamodule.batch_size.train = 1
            cfg.data.datamodule.batch_size.val = 1
            datamodule = hydra.utils.instantiate(
                cfg.data.datamodule, _recursive_=False, scaler_path=model_path
            )
            datamodule.setup()
            
            if args.split == 'train':
                dataloader = datamodule.train_dataloader(shuffle=False)
            elif args.split == 'val':
                dataloader = datamodule.val_dataloader()[0]
            else:
                raise ValueError(f"Invalid split: {args.split}. Must be 'train', 'val', or 'test'")
    
    # Move model to device
    model = model.to(device)
    print(f"Model loaded successfully. Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Build database
    start_time = time.time()
    db = build_database(model, dataloader, device)
    db_time = time.time() - start_time
    print(f"\nDatabase built in {db_time:.2f} seconds")
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    start_time = time.time()
    index = build_faiss_index(db['h_pxrd'], normalize=True)
    index_time = time.time() - start_time
    print(f"FAISS index built in {index_time:.2f} seconds")
    
    # Print statistics
    print_statistics(db, index)
    
    # Save database
    db_path = f"{args.save_prefix}_db.npz"
    print(f"Saving database to: {db_path}")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(db_path, **db)
    print(f"Database saved successfully")
    
    # Save FAISS index
    index_path = f"{args.save_prefix}_pxrd.index"
    print(f"Saving FAISS index to: {index_path}")
    faiss.write_index(index, index_path)
    print(f"FAISS index saved successfully")
    
    print("\n" + "="*60)
    print("DATABASE CONSTRUCTION COMPLETE")
    print("="*60)
    print(f"Total time: {db_time + index_time:.2f} seconds")
    print(f"Output files:")
    print(f"  - {db_path}")
    print(f"  - {index_path}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build PXRD-Crystal retrieval database with FAISS index"
    )
    parser.add_argument(
        '--cpcp_ckpt_path',
        type=str,
        required=True,
        help='Path to CPCP model checkpoint'
    )
    parser.add_argument(
        '--data_name',
        type=str,
        default='hmof_100',
        help='Dataset name (e.g., hmof_100, hmof_400)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Data split to use'
    )
    parser.add_argument(
        '--save_prefix',
        type=str,
        default=None,
        help='Output file prefix (auto-generated if not provided, e.g., outputs/retrieval/hmof_100_train)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for computation'
    )
    
    args = parser.parse_args()
    main(args)
