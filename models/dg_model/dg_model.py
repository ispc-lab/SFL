# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


from .modules.encoders.single import PerResidueEncoder
from .modules.encoders.pair import ResiduePairEncoder
from .modules.encoders.attn import GAEncoder
from .utils.protein.constants import BBHeavyAtom


class DG_Network(nn.Module):

    def __init__(self, encoder_node_feat_dim=128, max_num_atoms=15, encoder_pair_feat_dim=64, device='cpu'):
        super().__init__()
        self.device = device
        res_dim = encoder_node_feat_dim

        # Encoding
        self.single_encoder = PerResidueEncoder(
            feat_dim=encoder_node_feat_dim,
            max_num_atoms=max_num_atoms,
        )

        self.single_fusion = nn.Sequential(
            nn.Linear(res_dim, res_dim), nn.ReLU(),
            nn.Linear(res_dim, res_dim)
        )
        self.mut_bias = nn.Embedding(
            num_embeddings=2,
            embedding_dim=res_dim,
            padding_idx=0,
        )
        self.pair_encoder = ResiduePairEncoder(
            feat_dim=encoder_pair_feat_dim,
            max_num_atoms=max_num_atoms,
        )
        self.attn_encoder = GAEncoder()

    def encode(self, batch):
        N, L = batch['aa'].shape[:2]
        mask_residue = batch['mask_atoms'][:, :, BBHeavyAtom.CA]
        chi = batch['chi'] * (1 - batch['mut_flag'].float())[:, :, None]

        x = self.single_encoder(
            aa=batch['aa'],
            phi=batch['phi'], phi_mask=batch['phi_mask'],
            psi=batch['psi'], psi_mask=batch['psi_mask'],
            chi=chi, chi_mask=batch['chi_mask'],
            mask_residue=mask_residue,
        )

        b = self.mut_bias(batch['mut_flag'].long())
        x = x + b
        
        z = self.pair_encoder(
            aa=batch['aa'],
            res_nb=batch['res_nb'], chain_nb=batch['chain_nb'],
            pos_atoms=batch['pos_atoms'], mask_atoms=batch['mask_atoms'],
        )
        
        x = self.attn_encoder(
            pos_atoms=batch['pos_atoms'],
            res_feat=x, pair_feat=z,
            mask=mask_residue
        )
        return x

    def forward(self, batch):
        h = self.encode(batch)
        H = h.max(dim=1)[0]
        return H, None
