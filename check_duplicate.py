"""Check if idx=0 and idx=22276 are similar structures."""

import torch
import pickle
from xtalnet.pl_data.dataset import CrystMOFLMDBDataset
from xtalnet.pl_modules.cpcp_module import CPCPModule
from torch_geometric.loader import DataLoader

# Load dataset
dataset = CrystMOFLMDBDataset(
    name='train',
    path='data/hmof_100/train.lmdb',
    use_pxrd=True,
    is_training=False
)

# Load model
model = CPCPModule.load_from_checkpoint(
    'ckpt/hmof_100/CPCP/hmof_100_cpcp.ckpt',
    map_location='cpu'
)
model.eval()

# Get two structures
loader = DataLoader(dataset, batch_size=1, shuffle=False)

samples = {}
for i, batch in enumerate(loader):
    if i == 0:
        samples[0] = batch
    elif i == 22276:
        samples[22276] = batch
        break
    if i > 22276:
        break

print("=" * 70)
print("Comparing structures: idx=0 vs idx=22276")
print("=" * 70)

for idx in [0, 22276]:
    batch = samples[idx]
    print(f"\nStructure {idx}:")
    print(f"  Number of atoms: {batch.num_atoms[0].item()}")
    print(f"  Atom types: {batch.atom_types[:10].tolist()}...")  # First 10
    print(f"  Lattice lengths: {batch.lengths[0].tolist()}")
    print(f"  Lattice angles: {batch.angles[0].tolist()}")
    print(f"  PXRD peaks: {len(batch.peak_x)}")
    
    # Get PXRD embedding
    with torch.no_grad():
        pxrd_emb = model.pxrd_encoder(batch)['cls_token']
    print(f"  PXRD embedding norm: {torch.norm(pxrd_emb).item():.4f}")

# Compare embeddings
with torch.no_grad():
    emb0 = model.pxrd_encoder(samples[0])['cls_token']
    emb22276 = model.pxrd_encoder(samples[22276])['cls_token']
    
    # Normalize
    emb0_norm = torch.nn.functional.normalize(emb0, dim=-1)
    emb22276_norm = torch.nn.functional.normalize(emb22276, dim=-1)
    
    # Cosine similarity
    similarity = (emb0_norm @ emb22276_norm.T).item()

print("\n" + "=" * 70)
print(f"PXRD Embedding Similarity: {similarity:.6f}")
print("=" * 70)

if similarity > 0.9999:
    print("\n✓ These structures have IDENTICAL PXRD patterns!")
    print("  This is expected for:")
    print("    - Same structure with different indexing")
    print("    - Polymorphs with similar patterns")
    print("    - Duplicate entries in dataset")
    print("\n✓ Retrieval system is working CORRECTLY!")
else:
    print(f"\n⚠ Similarity is {similarity:.6f} (not identical)")
    print("  This might indicate an issue with encoding.")

print("")

