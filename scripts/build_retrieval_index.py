"""Build retrieval index from CPCP embeddings."""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from xtalnet.retrieval.index_builder import IndexBuilder


def main():
    parser = argparse.ArgumentParser(description='Build crystal structure retrieval index')
    parser.add_argument('--cpcp_ckpt', type=str, 
                       default='ckpt/hmof_100/CPCP/hmof_100_cpcp.ckpt',
                       help='Path to CPCP checkpoint')
    parser.add_argument('--data_path', type=str,
                       default='data/hmof_100/train.lmdb',
                       help='Path to training LMDB')
    parser.add_argument('--output_dir', type=str,
                       default='retrieval_indices/hmof_100',
                       help='Output directory for index')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding extraction')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit number of samples (for testing)')
    
    args = parser.parse_args()
    
    # Build index
    builder = IndexBuilder(
        cpcp_ckpt=args.cpcp_ckpt,
        data_path=args.data_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )
    
    index_path = builder.build_and_save(max_samples=args.max_samples)
    print(f"\n{'='*60}")
    print(f"SUCCESS! Index saved to: {index_path}")
    print(f"{'='*60}\n")
    print("Next steps:")
    print(f"  1. Test retrieval: python scripts/test_retrieval.py --index_path {index_path}")
    print(f"  2. Use in training: Set retrieval_index_path={index_path} in config")


if __name__ == '__main__':
    main()

