import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import D_MODEL, N_HEADS, FF_DIM, N_LAYERS, DROPOUT, MAX_SPEAKERS, ALPHA, INPUT_DIM


def sinusoidal_pe(length, dim, device="cpu"):
    pos = torch.arange(length, device=device).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * -(math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# PyTorch requires nn.Module for trainable models.
class FusionSpeaker(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.proj = nn.Linear(input_dim, D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, MAX_SPEAKERS)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj(x)
        pe = sinusoidal_pe(x.shape[1], D_MODEL, x.device)
        x = x + pe.unsqueeze(0)
        x = self.encoder(x)
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
