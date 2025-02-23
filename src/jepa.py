import math
import numpy as np
import torch.nn as nn

import functools
import os, math, gc, importlib
import torch



from ijepa.utils.tensors import (
    trunc_normal_,
    repeat_interleave_batch
)


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x,positional=False):
        # print('positional encoding:', self.pe[:, :x.size(1), :].shape)
        if positional:
            return self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)

class Encoder(nn.Module):
    """Transformer for NLP incorporating SequencePatchEmbed and patch masking."""
    def __init__(self, args):
        super().__init__()
        # Initialize SequencePatchEmbed module
        self.args = args
        self.embed     = Embeddings(args.vocab_size, args.embed_dim)        # Transformer blocks
        self.pos_emb            = PositionalEncoding(args.embed_dim, 0.1)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=args.embed_dim,
                nhead=args.encoder_num_heads,
                dim_feedforward=int(args.embed_dim * args.mlp_ratio),
                dropout=args.drop_rate,
                batch_first=True
            ) for _ in range(args.encoder_depth)
        ])

        self.norm = nn.LayerNorm(args.embed_dim)
        self.init_std = args.init_std
        self.apply(self._init_weights)
        # self.fix_init_weight()


    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, masks=None):
        # Convert input token sequences into patched embeddings
        x = self.embed(x)
        pos_emb = self.pos_emb(x)

        # Apply masks if provided
        if masks is not None:
            x = apply_masks(x, masks)  # Updated to align with your provided masking function
            # print(x.shape)
        # print('check:',x.shape,x.type)
        # Pass through transformer blocks
        # print('before:',x.shape)
        for block in self.blocks:
            x = block(x)
        # print('after:',x.shape)

        x = self.norm(x)

        return x



class Predictor(nn.Module):
    """NLP Predictor based on modified Transformer architecture incorporating sequence patch embedding and masking."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.predictor_embed = nn.Linear(args.embed_dim, args.predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, args.predictor_embed_dim))
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # Stochastic depth decay rule

        self.predictor_pos_embed            = PositionalEncoding(args.predictor_embed_dim, 0.1)

        self.predictor_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=args.predictor_embed_dim,
                nhead=args.predictor_num_heads,
                dim_feedforward=int(args.predictor_embed_dim * args.mlp_ratio),
                dropout=args.drop_rate,
                batch_first=True
            ) for i in range(args.predictor_depth)
        ])


        self.predictor_norm = nn.LayerNorm(args.predictor_embed_dim)
        self.predictor_proj = nn.Linear(args.predictor_embed_dim, args.embed_dim, bias=True)

        self.init_std = args.init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        # self.fix_init_weight()


    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, original, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        B = x.size(0)  # Batch size

        # Map from encoder dimension to predictor dimension
        x = self.predictor_embed(x)
        positions= self.predictor_pos_embed(original,True)
        # Add positional embedding to x tokens
        x_pos_embed = positions.repeat(B, 1, 1)
        # print(x_pos_embed.shape,len(masks_x))
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # Concat mask tokens to x
        positions= self.predictor_pos_embed(original,True)
        # Add positional embedding to x tokens
        pos_embs = positions.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        # pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))

        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs
        # x = x.repeat(len(masks), 1, 1)
        # print(x.shape)

        x = torch.cat([x, pred_tokens], dim=1)

        # Forward prop through predictor blocks
        for blk in self.predictor_blocks:
            x = blk(x)
        x = self.predictor_norm(x)

        # Return predictions for masked tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x



def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: tensor containing indices of patches to keep for each batch, shape [B, M] where M is the number of patches to keep
    """
    B, N, D = x.shape  # Batch size, Number of patches, Feature dimension
    M = masks.size(1)  # Number of patches to keep

    # Initialize a tensor to store the selected patches
    selected_x = torch.empty(B, M, D, dtype=x.dtype, device=x.device)

    # Apply the mask to each item in the batch individually
    for i in range(B):
        mask_keep = masks[i].unsqueeze(-1).expand(-1, D)  # Expand mask for feature dimension
        selected_x[i] = torch.gather(x[i], dim=0, index=mask_keep)

    return selected_x
