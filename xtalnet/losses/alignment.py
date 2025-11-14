"""
Alignment Loss for RAG-XtalNet

This module implements an alignment loss that encourages CCSG's hidden representations
to align with CPCP's crystal embeddings in the latent space. This is an optional
training enhancement that can be disabled via configuration.

The alignment loss uses InfoNCE (contrastive learning) to pull together representations
of the same structure from CCSG and CPCP, while pushing apart different structures.
"""

from typing import List
import torch
import torch.nn.functional as F


def alignment_loss(
    hidden_list: List[torch.Tensor],
    crystal_emb: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Compute alignment loss between CCSG hidden states and CPCP crystal embeddings.
    
    This loss encourages CCSG's intermediate representations to align with CPCP's
    crystal embeddings using InfoNCE (contrastive learning). For each layer's hidden
    state, we compute cosine similarity with the crystal embeddings and apply
    contrastive loss where:
    - Positive pairs: same structure (diagonal elements)
    - Negative pairs: different structures (off-diagonal elements)
    
    Args:
        hidden_list: List of hidden states from CCSG layers, each [B, d]
                    Can be a single element list for simplicity
        crystal_emb: Crystal embeddings from CPCP, shape [B, d]
        temperature: Temperature parameter for InfoNCE loss (default: 0.1)
                    Lower temperature makes the loss more discriminative
    
    Returns:
        Scalar loss value (averaged over all layers and samples)
    
    Example:
        >>> hidden_list = [hidden1, hidden2]  # Each [B, 512]
        >>> crystal_emb = cpcp_model.encode(batch)  # [B, 512]
        >>> loss = alignment_loss(hidden_list, crystal_emb, temperature=0.1)
    
    Note:
        This is an optional training enhancement for RAG-XtalNet.
        Can be disabled by setting use_alignment_loss=False in config.
    """
    if len(hidden_list) == 0:
        return torch.tensor(0.0, device=crystal_emb.device)
    
    total_loss = 0.0
    num_layers = len(hidden_list)
    
    # Normalize crystal embeddings once
    crystal_emb_norm = F.normalize(crystal_emb, dim=-1)  # [B, d]
    
    for hidden in hidden_list:
        # Normalize hidden states
        hidden_norm = F.normalize(hidden, dim=-1)  # [B, d]
        
        # Compute cosine similarity matrix
        # similarity[i, j] = cosine_sim(hidden[i], crystal_emb[j])
        similarity = torch.matmul(hidden_norm, crystal_emb_norm.T)  # [B, B]
        
        # Scale by temperature
        logits = similarity / temperature  # [B, B]
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # InfoNCE loss (cross-entropy with diagonal as targets)
        loss = F.cross_entropy(logits, labels)
        
        total_loss += loss
    
    # Average over all layers
    avg_loss = total_loss / num_layers
    
    return avg_loss


def compute_alignment_metrics(
    hidden_list: List[torch.Tensor],
    crystal_emb: torch.Tensor,
) -> dict:
    """
    Compute alignment metrics for monitoring (without gradients).
    
    Args:
        hidden_list: List of hidden states from CCSG layers, each [B, d]
        crystal_emb: Crystal embeddings from CPCP, shape [B, d]
    
    Returns:
        Dictionary containing:
            - avg_cosine_sim: Average cosine similarity of positive pairs
            - top1_accuracy: Fraction of samples where top-1 match is correct
            - top5_accuracy: Fraction of samples where top-5 contains correct match
    """
    if len(hidden_list) == 0:
        return {
            'avg_cosine_sim': 0.0,
            'top1_accuracy': 0.0,
            'top5_accuracy': 0.0
        }
    
    metrics = {
        'avg_cosine_sim': 0.0,
        'top1_accuracy': 0.0,
        'top5_accuracy': 0.0
    }
    
    crystal_emb_norm = F.normalize(crystal_emb, dim=-1)
    
    for hidden in hidden_list:
        hidden_norm = F.normalize(hidden, dim=-1)
        similarity = torch.matmul(hidden_norm, crystal_emb_norm.T)  # [B, B]
        
        # Average cosine similarity of positive pairs (diagonal)
        diag_sim = torch.diag(similarity).mean().item()
        metrics['avg_cosine_sim'] += diag_sim
        
        # Top-1 accuracy
        top1_indices = similarity.argmax(dim=1)
        labels = torch.arange(similarity.size(0), device=similarity.device)
        top1_acc = (top1_indices == labels).float().mean().item()
        metrics['top1_accuracy'] += top1_acc
        
        # Top-5 accuracy
        top5_indices = similarity.topk(5, dim=1)[1]
        top5_acc = (top5_indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics['top5_accuracy'] += top5_acc
    
    # Average over layers
    num_layers = len(hidden_list)
    for key in metrics:
        metrics[key] /= num_layers
    
    return metrics
