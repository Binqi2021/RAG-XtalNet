"""Quick test to verify retrieval is working correctly."""

import torch
from xtalnet.retrieval.retriever import CrystalRetriever
from xtalnet.pl_modules.cpcp_module import CPCPModule
from xtalnet.pl_data.dataset import CrystMOFLMDBDataset
from torch_geometric.loader import DataLoader

# Load components
print("Loading retriever...")
retriever = CrystalRetriever('retrieval_indices/hmof_100', device='cpu')

print("Loading CPCP model...")
model = CPCPModule.load_from_checkpoint(
    'ckpt/hmof_100/CPCP/hmof_100_cpcp.ckpt',
    map_location='cpu'
)
model.eval()

# Load TRAINING dataset (not validation!)
print("Loading TRAINING dataset...")
train_dataset = CrystMOFLMDBDataset(
    name='train',
    path='data/hmof_100/train.lmdb',
    use_pxrd=True,
    is_training=False
)

# Test on first 10 samples from training set
loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

correct = 0
total = 10

print(f"\nTesting retrieval on {total} training samples...")
print("=" * 60)

for i, batch in enumerate(loader):
    if i >= total:
        break
    
    with torch.no_grad():
        pxrd_emb = model.pxrd_encoder(batch)['cls_token']
    
    results = retriever.search(pxrd_emb, k=5)
    query_idx = batch.idx[0].item()
    retrieved_idx = results['indices'][0][0]
    
    is_correct = (query_idx == retrieved_idx)
    correct += is_correct
    
    status = "✓" if is_correct else "✗"
    print(f"{status} Query idx={query_idx:5d} → Retrieved idx={retrieved_idx:5d} "
          f"(similarity={results['distances'][0][0]:.4f})")

print("=" * 60)
print(f"Recall@1: {correct}/{total} = {100*correct/total:.1f}%")
print("")

if correct == total:
    print("✓✓✓ PERFECT! Retrieval system working correctly! ✓✓✓")
else:
    print("⚠ Some queries didn't match. Check if expected.")

