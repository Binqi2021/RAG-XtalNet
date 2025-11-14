import math, copy
from argparse import Namespace
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.data import Batch

from typing import Any, Dict, Tuple, List

import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support
from tqdm import tqdm

from xtalnet.common.utils import PROJECT_ROOT
from xtalnet.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc)

from xtalnet.pl_modules.diff_utils import d_log_p_wrapped_normal
from .bert import BertModel
from .diff_utils import RegressionHead

MAX_ATOMIC_NUM=100


class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        # norms = grad_norm(self, norm_type=2)
        parameters = self.parameters()
        grads = [p.grad for p in parameters if p.grad is not None]
        first_device = grads[0].device
        norms = []
        foreach = None
        grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[torch.Tensor]]] \
        = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])  # type: ignore[assignment]
        for ((device, _), [grads]) in grouped_grads.items():
            if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                norms.extend(torch._foreach_norm(grads, 2.0))
            elif foreach:
                raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                norms.extend([torch.norm(g, 2.0) for g in grads])
        total_norm = torch.norm(torch.stack([norm.to(first_device) for norm in norms]), 2.0)
        self.log_dict({'grad_norm_total': total_norm})

### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pxrd_encoder = hydra.utils.instantiate(self.hparams.pxrd_encoder, args=Namespace(), _recursive_=False)
        if 'pretrained' in self.hparams and self.hparams.pretrained:
            self.load_state_dict(torch.load(self.hparams.pretrained)['state_dict'], strict=False)
            print('succeffully load pretrained model')
        if 'freeze_pxrd_encoder' in self.hparams and self.hparams.freeze_pxrd_encoder:
            for param in self.pxrd_encoder.parameters():
                param.requires_grad = False
            print('freeze pxrd encoder params')

        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler)
        self.time_dim = self.hparams.time_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = self.hparams.cost_lattice < 1e-5
        self.keep_coords = self.hparams.cost_coord < 1e-5
        
        self.decoder = hydra.utils.instantiate(self.hparams.crystal_encoder, latent_dim = self.hparams.latent_dim + self.hparams.time_dim, _recursive_=False)
        
        # ALIGNMENT: Initialize CPCP model for alignment loss (optional)
        self.cpcp_model = None
        self.use_alignment_loss = getattr(self.hparams, 'use_alignment_loss', False)
        self.alignment_lambda = getattr(self.hparams, 'alignment_lambda', 0.1)
        self.alignment_temperature = getattr(self.hparams, 'alignment_temperature', 0.1)
        self.alignment_start_epoch = getattr(self.hparams, 'alignment_start_epoch', 0)
        
        if self.use_alignment_loss:
            cpcp_ckpt_path = getattr(self.hparams, 'cpcp_ckpt_path', None)
            if cpcp_ckpt_path:
                print(f'\n{"="*60}')
                print('ALIGNMENT LOSS ENABLED')
                print(f'{"="*60}')
                print(f'CPCP checkpoint: {cpcp_ckpt_path}')
                print(f'Lambda: {self.alignment_lambda}')
                print(f'Temperature: {self.alignment_temperature}')
                print(f'Start epoch: {self.alignment_start_epoch}')
                print(f'{"="*60}\n')
                
                # Load CPCP model
                from xtalnet.pl_modules.cpcp_module import CPCPModule
                self.cpcp_model = CPCPModule.load_from_checkpoint(cpcp_ckpt_path, strict=False)
                self.cpcp_model.eval()
                self.cpcp_model.requires_grad_(False)
                print('CPCP model loaded and frozen for alignment loss')
            else:
                print('WARNING: use_alignment_loss=True but cpcp_ckpt_path not provided. Alignment loss disabled.')
                self.use_alignment_loss = False
        
        


    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        """
        encode crystal structures to latents.
        """
        result = self.pxrd_encoder(batch)
        return result['cls_token']


    def pxrd_forward(self, batch):
        pxrd_feat = self.pxrd_encoder(batch)['cls_token']
        results = dict()
        results['pxrd_feat'] = pxrd_feat
        return results

    def forward(self, batch, return_hidden=False):
        
        cpcp_results = self.pxrd_forward(batch)
        pxrd_cls = cpcp_results['pxrd_feat']

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]
        beta = self.beta_scheduler.betas[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.


        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        # Training: no RAG, template_emb=None, optionally return hidden states
        decoder_output = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch, pxrd_cls, template_emb=None, return_hidden=return_hidden)
        
        if return_hidden:
            pred_l, pred_x, hidden_list = decoder_output
        else:
            pred_l, pred_x = decoder_output
            hidden_list = None

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)

        loss = (
            self.hparams.cost_lattice * loss_lattice +
            self.hparams.cost_coord * loss_coord
            )

        loss_dict = {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
        }
        
        if return_hidden:
            loss_dict['hidden_list'] = hidden_list
        
        return loss_dict

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5, retriever=None, rag_top_m=4, rag_strength=1.0):
        """
        Sample crystal structures with optional RAG enhancement.
        
        Args:
            batch: Input batch
            step_lr: Step size for Langevin dynamics
            retriever: Optional PXRDTemplateRetriever for RAG
            rag_top_m: Number of templates to retrieve
            rag_strength: Strength of RAG gating (multiplier for gate effect)
        
        Returns:
            Final structure and trajectory
        """

        batch_size = batch.num_graphs

        l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        time_start = self.beta_scheduler.timesteps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}

        pxrd_cls = self.encode(batch)
        
        # RAG: Prepare template embeddings if retriever is provided
        template_emb = None
        if retriever is not None:
            import numpy as np
            # Get PXRD embeddings for retrieval
            pxrd_feat_np = pxrd_cls.cpu().numpy()  # [B, d]
            
            # Retrieve templates for each sample in batch
            template_embs_list = []
            for i in range(batch_size):
                # Query retriever for top-M similar structures
                results = retriever.query(pxrd_feat_np[i], top_m=rag_top_m)
                # Average the retrieved crystal embeddings
                avg_template = np.mean(results['h_crystal'], axis=0)  # [d]
                template_embs_list.append(avg_template)
            
            # Stack and convert to tensor
            template_emb = torch.from_numpy(np.stack(template_embs_list, axis=0)).float().to(self.device)  # [B, d]
            print(f"RAG: Retrieved {rag_top_m} templates per sample, template_emb shape: {template_emb.shape}")
        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            # RAG: Pass template_emb to decoder (scaled by rag_strength)
            scaled_template_emb = template_emb * rag_strength if template_emb is not None else None
            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch, pxrd_cls, scaled_template_emb)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # Predictor

            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            # RAG: Pass template_emb to decoder (scaled by rag_strength)
            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch, pxrd_cls, scaled_template_emb)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack

    def clip_batch(self, batch, n_nodes=350):
        if batch.num_atoms.max() > n_nodes:
            n = len(batch)
            idx = batch.num_atoms.argsort(descending=True)[:n//2]
            batch = Batch.from_data_list(batch[idx])
        return batch

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        batch = self.clip_batch(batch)
        
        # ALIGNMENT: Determine if we should compute alignment loss
        use_alignment = (
            self.use_alignment_loss and 
            self.cpcp_model is not None and 
            self.current_epoch >= self.alignment_start_epoch
        )
        
        # Forward pass with optional hidden states
        output_dict = self(batch, return_hidden=use_alignment)
        
        # Base diffusion loss
        loss = output_dict['loss']
        log_dict, _ = self.compute_stats(output_dict, prefix='train')
        
        # ALIGNMENT: Add alignment loss if enabled
        if use_alignment:
            from xtalnet.losses import alignment_loss
            
            # Get crystal embeddings from CPCP
            with torch.no_grad():
                lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
                crystal_emb = self.cpcp_model.crystal_encoder(
                    batch.atom_types, 
                    batch.frac_coords, 
                    lattices, 
                    batch.num_atoms, 
                    batch.batch
                )
                crystal_emb = F.normalize(crystal_emb, dim=-1)
            
            # Compute alignment loss
            hidden_list = output_dict['hidden_list']
            loss_align = alignment_loss(
                hidden_list, 
                crystal_emb, 
                temperature=self.alignment_temperature
            )
            
            # Combine losses
            total_loss = loss + self.alignment_lambda * loss_align
            
            # Log alignment loss
            log_dict['train_loss_alignment'] = loss_align
            log_dict['train_loss_total'] = total_loss
            log_dict['train_alignment_lambda'] = self.alignment_lambda
            
            # Compute alignment metrics (for monitoring)
            from xtalnet.losses.alignment import compute_alignment_metrics
            with torch.no_grad():
                metrics = compute_alignment_metrics(hidden_list, crystal_emb)
                log_dict['train_align_cosine_sim'] = metrics['avg_cosine_sim']
                log_dict['train_align_top1_acc'] = metrics['top1_accuracy']
            
            loss = total_loss

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            sync_dist=True
        )

        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=1,
            sync_dist=True
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
            batch_size=1,
            sync_dist=True
        )
        return loss

    def compute_stats(self, output_dict, prefix):
        loss = output_dict['loss']
        log_dict = dict()
        for k, v in output_dict.items():
            if 'loss' in k:
                log_dict[f"{prefix}_{k}"] = v

        return log_dict, loss

    