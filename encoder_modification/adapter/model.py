import math
import torch
import torch.nn as nn


class SinusoidalPE(nn.Module):
    # Choice: sinusoidal (fixed) over learned positional embeddings.
    # Fixed sinusoidal generalizes to unseen lengths without retraining.
    # Alternative was RoPE (better for variable-length streaming) but
    # adds complexity we don't need until Phase 2 (AOSC cache). When we
    # add the cache, we'll switch to relative positional encoding.

    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SortformerAdapter(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_speakers=4,
        num_layers=4,
        nhead=8,
        ff_dim=1024,
        dropout=0.1,
        max_len=2048,
    ):
        super().__init__()

        # Choice: 4 layers, 8 heads, ff_dim=1024.
        # Sortformer uses 18 layers on raw mel-spectrograms via NEST.
        # We sit on Mimi's already-rich 512-dim embeddings, so fewer
        # layers suffice. 4 layers × 8 heads ≈ 8M params.
        # Alternative was 3 layers (lighter) or 6 (heavier). 4 is
        # a middle ground — we tune this if needed.

        # Choice: ff_dim=1024 (2x ratio) over standard 2048 (4x).
        # Keeps parameter count lower since this is an adapter, not
        # a standalone model. Full Sortformer uses 192 hidden with
        # standard 4x ff, giving similar total capacity.

        self.pos_enc = SinusoidalPE(embed_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            # Choice: GELU over ReLU. Standard in modern Transformers,
            # smoother gradients. Marginal difference at this scale.
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Choice: single linear layer to K outputs, no hidden layers.
        # The Transformer already builds speaker-discriminative features.
        # Adding an MLP head risks overfitting on small data.
        self.head = nn.Linear(embed_dim, num_speakers)

    def forward(self, x, padding_mask=None):
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        logits = self.head(x)

        # Choice: sigmoid, not softmax. Each speaker is independent.
        # Softmax forces sum-to-1, killing overlap detection.
        return torch.sigmoid(logits)

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
