import time
import argparse
import torch
import os

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Batch
from pytorch_lightning import seed_everything

from eval_utils import (
    lattices_to_params_shape,
    load_model_ckpt,
    lattices_to_params_shape,
    load_dataset
)

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group

import copy

import numpy as np


def diffusion(
    loader,
    ccsg_model,
    cpcp_model,
    num_evals,
    begin_idx=0,
    end_idx=-1,
    step_lr=1e-5,
    retriever=None,  # RAG: Optional retriever
    rag_top_m=4,  # RAG: Number of templates
    rag_strength=1.0,  # RAG: Gating strength
):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    pxrd_feat, atom_feat = [], []
    score_list = []
    for idx, batch in enumerate(loader):
        if idx < begin_idx:
            continue
        if idx == end_idx:
            break

        if torch.cuda.is_available():
            batch.cuda()
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        atom_feat_i = []
        input_data_list = input_data_list + batch.to_data_list()
        for eval_idx in range(num_evals):
            print(f"batch {idx} / {len(loader)}, sample {eval_idx} / {num_evals}")
            # RAG: Pass retriever and RAG parameters to sample
            outputs, _ = ccsg_model.sample(
                batch, 
                step_lr=step_lr,
                retriever=retriever,
                rag_top_m=rag_top_m,
                rag_strength=rag_strength
            )
            batch_frac_coords.append(outputs["frac_coords"].detach().cpu())
            batch_num_atoms.append(outputs["num_atoms"].detach().cpu())
            batch_atom_types.append(outputs["atom_types"].detach().cpu())
            batch_lattices.append(outputs["lattices"].detach().cpu())

            new_batch = copy.deepcopy(batch)
            new_batch.frac_coords = outputs["frac_coords"]
            new_batch.lengths, new_batch.angles = lattices_to_params_shape(
                outputs["lattices"]
            )
            atom_feat_i.append(
                cpcp_model.inference(new_batch)["atom_feat"].detach().cpu()
            )
        pxrd_feat_i = cpcp_model.inference(new_batch)["pxrd_feat"].detach().cpu()
        atom_feat_i = torch.stack(atom_feat_i, dim=0)
        score_list.append(torch.cosine_similarity(atom_feat_i, pxrd_feat_i, dim=-1))
        atom_feat.append(atom_feat_i)
        pxrd_feat.append(pxrd_feat_i)
        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)

    return dict(
        frac_coords=frac_coords,
        num_atoms=num_atoms,
        atom_types=atom_types,
        lattices=lattices,
        lengths=lengths,
        angles=angles,
        input_data_batch=input_data_batch,
        atom_feat=atom_feat,
        pxrd_feat=pxrd_feat,
        score_list=score_list,
    )


def main(args):
    seed_everything(42)
    # load_data if do reconstruction.
    ccsg_model_path = Path(os.path.dirname(args.ccsg_ckpt_path))
    save_path = Path(args.save_path)
    ccsg_model, test_loader, cfg = load_model_ckpt(
        ccsg_model_path, args.ccsg_ckpt_path, load_data=True,
        data_root_override='/public/home/huangtianqi/XtalNet/data/hmof_100'
    )
    ccsg_model.eval()
    cpcp_model_path = Path(os.path.dirname(args.cpcp_ckpt_path))
    cpcp_model, _, _ = load_model_ckpt(cpcp_model_path, args.cpcp_ckpt_path)
    cpcp_model.eval()
    if torch.cuda.is_available():
        ccsg_model.to("cuda")
        cpcp_model.to("cuda")

    # RAG: Initialize retriever if database path is provided
    retriever = None
    if args.retrieval_db_npz:
        print(f"\n{'='*60}")
        print("RAG MODE ENABLED")
        print(f"{'='*60}")
        print(f"Database: {args.retrieval_db_npz}")
        print(f"Index: {args.retrieval_index}")
        print(f"Top-M: {args.rag_top_m}")
        print(f"Strength: {args.rag_strength}")
        print(f"{'='*60}\n")
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from xtalnet.retrieval import PXRDTemplateRetriever
        
        retriever = PXRDTemplateRetriever(
            db_npz_path=args.retrieval_db_npz,
            faiss_index_path=args.retrieval_index,
            normalize=True
        )
        print(f"Retriever initialized: {retriever.n_samples} samples\n")

    print("Evaluate the ccsg model.")

    start_time = time.time()
    out_dict = diffusion(
        test_loader,
        ccsg_model,
        cpcp_model,
        num_evals=args.num_evals,
        step_lr=args.step_lr,
        begin_idx=args.begin_idx,
        end_idx=args.end_idx,
        retriever=retriever,  # RAG: Pass retriever
        rag_top_m=args.rag_top_m,  # RAG: Pass top_m
        rag_strength=args.rag_strength,  # RAG: Pass strength
    )

    ccsg_out_name = f"eval_ccsg_{args.label}_number-eval{args.num_evals}_b{args.begin_idx}_e{args.end_idx}.pt"

    torch.save(
        {
            "eval_setting": args,
            "input_data_batch": out_dict["input_data_batch"],
            "frac_coords": out_dict["frac_coords"],
            "num_atoms": out_dict["num_atoms"],
            "atom_types": out_dict["atom_types"],
            "lattices": out_dict["lattices"],
            "lengths": out_dict["lengths"],
            "angles": out_dict["angles"],
            "atom_feat": out_dict["atom_feat"],
            "pxrd_feat": out_dict["pxrd_feat"],
            "score_list": out_dict["score_list"],  # list of tensor
            "time": time.time() - start_time,
        },
        save_path / ccsg_out_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccsg_ckpt_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--cpcp_ckpt_path", required=True)
    parser.add_argument("--step_lr", default=1e-5, type=float)
    parser.add_argument("--num_evals", default=1, type=int)
    parser.add_argument("--label", default="")
    parser.add_argument("--begin_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=-1, type=int)
    
    # RAG: New arguments for retrieval-augmented generation
    parser.add_argument("--retrieval_db_npz", default="", type=str,
                       help="Path to retrieval database NPZ file (enables RAG if provided)")
    parser.add_argument("--retrieval_index", default="", type=str,
                       help="Path to FAISS index file")
    parser.add_argument("--rag_top_m", default=4, type=int,
                       help="Number of templates to retrieve for RAG")
    parser.add_argument("--rag_strength", default=1.0, type=float,
                       help="Strength of RAG gating (multiplier for gate effect)")
    
    args = parser.parse_args()
    main(args)
