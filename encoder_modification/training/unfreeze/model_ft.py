"""MimiSpeaker with partial Mimi encoder_transformer unfreezing.

Probe results:
  encoder_transformer = ProjectedTransformer
    .transformer = StreamingTransformer
      .layers[0..7] = 8 layers, each ~3.15M params (25.19M total)
    .output_projs = ModuleList
  Each layer: self_attn, norm1, norm2, linear1(2048), linear2, layer_scale_1, layer_scale_2
  Output: list of 1 tensor, shape [B, 512, T]

Architecture:
  Pre-conv embeddings (frozen, from disk)  [B, 512, T] at 25Hz
      -> Mimi layers 0..4 (frozen) + layers 5..7 (unfrozen, LR=1e-5)
      -> output_projs (unfrozen)
      -> Stride-2 downsample (25Hz -> 12.5Hz)
      -> Linear projection (512 -> 384)
      -> Sinusoidal PE
      -> 4x Adapter Transformer (384-dim, LR=1e-4)
      -> Linear head -> 4 sigmoids
"""
import math
import copy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import D_MODEL, N_HEADS, FF_DIM, N_LAYERS, DROPOUT, MAX_SPEAKERS, ALPHA


def sinusoidal_pe(length, dim, device="cpu"):
    pos = torch.arange(length, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * -(math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MimiSpeakerFT(nn.Module):
    def __init__(self, mimi_transformer, n_unfreeze=3, input_dim=512):
        super().__init__()

        # Choice: deep copy to avoid modifying the original Mimi.
        self.mimi_tf = copy.deepcopy(mimi_transformer)

        # Access: et.transformer.layers[0..7]
        layers = self.mimi_tf.transformer.layers
        n_total = len(layers)
        self.n_freeze = max(0, n_total - n_unfreeze)
        self.n_unfreeze = min(n_unfreeze, n_total)

        # Freeze everything first
        for p in self.mimi_tf.parameters():
            p.requires_grad = False

        # Unfreeze top n_unfreeze layers
        # Choice: 3 out of 8 = layers 5,6,7. Adds ~9.5M trainable Mimi params.
        # Total ~16.8M (adapter 7.3M + mimi 9.5M). All 8 = 25M risks overfitting.
        # Alternative: 2 layers (conservative, ~6.3M) or 4 (aggressive, ~12.6M).
        for i in range(self.n_freeze, n_total):
            for p in layers[i].parameters():
                p.requires_grad = True

        # Choice: unfreeze output_projs too — sits between transformer and our adapter.
        # Freezing it would bottleneck gradients between two trainable blocks.
        for p in self.mimi_tf.output_projs.parameters():
            p.requires_grad = True

        frozen = sum(p.numel() for p in self.mimi_tf.parameters() if not p.requires_grad)
        unfrozen = sum(p.numel() for p in self.mimi_tf.parameters() if p.requires_grad)
        print(f"Mimi transformer: {n_total} layers, freeze {self.n_freeze}, unfreeze {self.n_unfreeze}")
        print(f"  frozen: {frozen:,}, unfrozen: {unfrozen:,}")

        # Adapter
        self.proj = nn.Linear(input_dim, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, activation="gelu",
        )
        self.adapter = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, MAX_SPEAKERS)

        for p in self.adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def mimi_params(self):
        return [p for p in self.mimi_tf.parameters() if p.requires_grad]

    def adapter_params(self):
        return list(self.proj.parameters()) + list(self.adapter.parameters()) + list(self.head.parameters())

    def forward(self, x):
        # x: (B, D, T) pre-conv embeddings, 25Hz
        tf_out = self.mimi_tf(x)
        if isinstance(tf_out, (list, tuple)):
            tf_out = tf_out[0]

        x = tf_out.transpose(1, 2)  # (B, D, T) -> (B, T, D)
        x = x[:, ::2, :]            # stride-2: 25Hz -> 12.5Hz
        x = self.proj(x)
        pe = sinusoidal_pe(x.shape[1], D_MODEL, x.device)
        x = x + pe.unsqueeze(0)
        x = self.adapter(x)
        return self.head(x)


def sort_loss(pred, labels):
    return F.binary_cross_entropy_with_logits(pred, labels)


def pil_loss(pred, labels):
    B, T, K = pred.shape
    perms = list(itertools.permutations(range(K)))
    best = torch.full((B,), float("inf"), device=pred.device)
    for perm in perms:
        permuted = labels[:, :, list(perm)]
        loss = F.binary_cross_entropy_with_logits(pred, permuted, reduction="none").mean(dim=(1, 2))
        best = torch.minimum(best, loss)
    return best.mean()


def hybrid_loss(pred, labels):
    return ALPHA * sort_loss(pred, labels) + (1 - ALPHA) * pil_loss(pred, labels)