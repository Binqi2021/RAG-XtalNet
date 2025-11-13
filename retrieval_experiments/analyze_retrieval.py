"""All-in-one retrieval analysis script."""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from torch_geometric.loader import DataLoader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from xtalnet.retrieval.retriever import CrystalRetriever
from xtalnet.pl_modules.cpcp_module import CPCPModule
from xtalnet.pl_data.dataset import CrystMOFLMDBDataset


class RetrievalAnalyzer:
    def __init__(self, 
                 index_path='retrieval_indices/hmof_100',
                 cpcp_ckpt='ckpt/hmof_100/CPCP/hmof_100_cpcp.ckpt',
                 data_path='data/hmof_100/train.lmdb',
                 output_dir='retrieval_experiments'):
        
        self.output_dir = Path(output_dir)
        self.vis_dir = self.output_dir / 'visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading components...")
        self.retriever = CrystalRetriever(index_path, device='cpu')
        self.model = CPCPModule.load_from_checkpoint(cpcp_ckpt, map_location='cpu')
        self.model.eval()
        self.dataset = CrystMOFLMDBDataset(name='train', path=data_path, 
                                           use_pxrd=True, is_training=False)
        
        self.results = {
            'index': {},
            'quality': {},
            'speed': {},
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def test_quality(self, n_samples=100):
        """Test retrieval quality."""
        print(f"\nTesting retrieval quality on {n_samples} samples...")
        
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        
        recalls = {1: [], 5: [], 10: [], 20: []}
        similarities = []
        
        for i, batch in enumerate(tqdm(loader, total=n_samples, desc="Testing")):
            if i >= n_samples:
                break
            
            with torch.no_grad():
                pxrd_emb = self.model.pxrd_encoder(batch)['cls_token']
            
            results = self.retriever.search(pxrd_emb, k=20)
            query_idx = batch.idx[0].item()
            retrieved_indices = results['indices'][0]
            
            # Calculate recall@k
            for k in recalls.keys():
                recalls[k].append(1.0 if query_idx in retrieved_indices[:k] else 0.0)
            
            # Record top-1 similarity
            similarities.append(results['distances'][0][0])
        
        # Save results
        self.results['quality'] = {
            'recall@1': float(np.mean(recalls[1])),
            'recall@5': float(np.mean(recalls[5])),
            'recall@10': float(np.mean(recalls[10])),
            'recall@20': float(np.mean(recalls[20])),
            'similarity_mean': float(np.mean(similarities)),
            'similarity_std': float(np.std(similarities)),
            'similarity_min': float(np.min(similarities)),
            'similarity_max': float(np.max(similarities)),
            'n_samples': n_samples
        }
        
        self.recalls = recalls
        self.similarities = similarities
        
        print(f"✓ Recall@1: {self.results['quality']['recall@1']:.2%}")
        print(f"✓ Similarity: {self.results['quality']['similarity_mean']:.4f}")
    
    def test_speed(self, n_queries=100):
        """Benchmark query speed."""
        print(f"\nBenchmarking speed on {n_queries} queries...")
        
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        
        # Single query timing
        times = []
        for i, batch in enumerate(tqdm(loader, total=n_queries, desc="Speed test")):
            if i >= n_queries:
                break
            
            with torch.no_grad():
                pxrd_emb = self.model.pxrd_encoder(batch)['cls_token']
            
            import time
            start = time.time()
            _ = self.retriever.search(pxrd_emb, k=5)
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)
        
        self.results['speed'] = {
            'single_query_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'n_queries': n_queries
        }
        
        print(f"✓ Query time: {self.results['speed']['single_query_ms']:.2f} ± "
              f"{self.results['speed']['std_ms']:.2f} ms")
    
    def plot_recall(self):
        """Plot Recall@k curve."""
        print("Plotting recall curve...")
        
        k_values = [1, 5, 10, 20]
        recall_values = [self.results['quality'][f'recall@{k}'] for k in k_values]
        
        plt.figure(figsize=(8, 6))
        plt.plot(k_values, recall_values, marker='o', linewidth=2, markersize=8)
        plt.xlabel('k (number of neighbors)', fontsize=12)
        plt.ylabel('Recall@k', fontsize=12)
        plt.title('Retrieval Recall@k Performance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        
        # Add value labels
        for k, r in zip(k_values, recall_values):
            plt.text(k, r + 0.02, f'{r:.1%}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.vis_dir / 'recall_curve.png'}")
    
    def plot_similarity(self):
        """Plot similarity distribution."""
        print("Plotting similarity distribution...")
        
        plt.figure(figsize=(10, 6))
        plt.hist(self.similarities, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Cosine Similarity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Top-1 Similarity Distribution', fontsize=14, fontweight='bold')
        plt.axvline(np.mean(self.similarities), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(self.similarities):.3f}')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'similarity_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.vis_dir / 'similarity_dist.png'}")
    
    def plot_speed(self):
        """Plot speed benchmark."""
        print("Plotting speed benchmark...")
        
        scenarios = ['Single\nQuery', 'Mean', 'Min', 'Max']
        times = [
            self.results['speed']['single_query_ms'],
            self.results['speed']['single_query_ms'],
            self.results['speed']['min_ms'],
            self.results['speed']['max_ms']
        ]
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(scenarios, times, color=colors, edgecolor='black', linewidth=1.5)
        plt.ylabel('Time (ms)', fontsize=12)
        plt.title('Query Speed Benchmark (CPU)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, time in zip(bars, times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.1f}ms', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'speed_benchmark.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.vis_dir / 'speed_benchmark.png'}")
    
    def plot_tsne(self, n_samples=500):
        """Plot t-SNE visualization of embeddings."""
        print(f"Generating t-SNE visualization ({n_samples} samples)...")
        
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("⚠ sklearn not installed, skipping t-SNE plot")
            return
        
        loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        
        embeddings = []
        atom_counts = []
        
        for i, batch in enumerate(tqdm(loader, total=n_samples, desc="Extracting embeddings")):
            if i >= n_samples:
                break
            
            with torch.no_grad():
                emb = self.model.pxrd_encoder(batch)['cls_token']
                embeddings.append(emb.cpu().numpy())
                atom_counts.append(batch.num_atoms[0].item())
        
        embeddings = np.vstack(embeddings)
        
        # t-SNE
        print("Running t-SNE (this may take a minute)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=atom_counts, cmap='viridis', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Number of Atoms')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.title('t-SNE Visualization of PXRD Embeddings', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'embedding_tsne.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {self.vis_dir / 'embedding_tsne.png'}")
    
    def save_results(self):
        """Save metrics to JSON."""
        print("\nSaving results...")
        
        # Add index info
        import pickle
        with open('retrieval_indices/hmof_100/config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        self.results['index'] = {
            'n_samples': config['n_samples'],
            'embedding_dim': config['embedding_dim'],
            'size_mb': 372,  # From log
            'build_time_sec': 480
        }
        
        # Save JSON
        output_file = self.output_dir / 'metrics.json'
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✓ Saved: {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("RETRIEVAL SYSTEM ANALYSIS SUMMARY")
        print("="*60)
        print(f"Index: {self.results['index']['n_samples']} samples, "
              f"{self.results['index']['embedding_dim']}D")
        print(f"Recall@1: {self.results['quality']['recall@1']:.1%}")
        print(f"Recall@10: {self.results['quality']['recall@10']:.1%}")
        print(f"Similarity: {self.results['quality']['similarity_mean']:.4f}")
        print(f"Query Speed: {self.results['speed']['single_query_ms']:.2f} ms")
        print("="*60)
    
    def run(self):
        """Run complete analysis."""
        print("="*60)
        print("RETRIEVAL SYSTEM ANALYSIS")
        print("="*60)
        
        # Collect metrics
        self.test_quality(n_samples=100)
        self.test_speed(n_queries=100)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.plot_recall()
        self.plot_similarity()
        self.plot_speed()
        self.plot_tsne(n_samples=500)
        
        # Save results
        self.save_results()
        
        print(f"\n✓ Analysis complete! Check {self.output_dir}/")


if __name__ == '__main__':
    analyzer = RetrievalAnalyzer()
    analyzer.run()

