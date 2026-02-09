import torch
import torchaudio.functional
import numpy as np
from torch.utils.data import DataLoader
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from adapter.model import SortformerAdapter
from adapter.loss import hybrid_loss
from adapter.eval import eval_batch
from adapter.data import MixtureDataset, collate_mixtures
import config


# -- Training hyperparameters --

# Choice: AdamW over Adam. Weight decay helps regularization on small data.
# Sortformer uses AdamW with 1e-3 decay. We follow suit.
LR = 3e-4
WEIGHT_DECAY = 1e-3

# Choice: 50 epochs for initial validation on VoxCeleb mixtures.
# Real training on NeMo-generated data will be longer (24K+ steps
# like Sortformer). This is a proof-of-concept run.
EPOCHS = 50

BATCH_SIZE = 8  # Choice: small batch, limited by Mimi inference per mixture

# Choice: evaluate every 5 epochs. Training is short, frequent eval is cheap.
EVAL_EVERY = 5

# Choice: save best model by DER on a held-out portion.
# Alternative was save every epoch, but disk waste.
SAVE_PATH = "adapter_best.pt"

# Choice: 4 speakers max. PersonaPlex is 2-speaker, but Sortformer
# handles up to 4 and the extra slots cost nothing (4 sigmoids vs 2).
# Gives headroom for multi-party conversations.
NUM_SPEAKERS = 4

# Choice: 4 Transformer layers. Sortformer uses 18 on raw features,
# but Mimi embeddings are already rich — fewer layers suffice.
# 4 layers × 8 heads × 512 dim ≈ 8M params.
ADAPTER_LAYERS = 4
ADAPTER_HEADS = 8
ADAPTER_FF_DIM = 1024  # Choice: 2x ratio (not 4x) to keep adapter lightweight
ADAPTER_DROPOUT = 0.1


def group_by_speaker(ds):
    """Group dataset utterances by speaker ID for mixture generation."""
    speakers = {}
    for row in ds:
        uid = row["id"]
        spk = uid.split("+")[0]
        audio = row["audio"]["array"]
        sr = row["audio"]["sampling_rate"]
        # Choice: resample at load time using torchaudio.functional.resample.
        # Alternative was sox/ffmpeg preprocessing but adds a dependency.
        if sr != config.MIMI_SAMPLE_RATE:
            audio_t = torch.from_numpy(audio).float()
            audio_t = torchaudio.functional.resample(audio_t, sr, config.MIMI_SAMPLE_RATE)
            audio = audio_t.numpy().astype(np.float32)
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append(audio)
    return speakers


def train(mimi, ds):
    """Train adapter on simulated mixtures from single-speaker data.

    mimi: frozen Mimi model (on GPU)
    ds: HuggingFace dataset with 'id' and 'audio' fields
    """
    speakers = group_by_speaker(ds)
    print(f"grouped {sum(len(v) for v in speakers.values())} utterances from {len(speakers)} speakers")

    spk_list = sorted(speakers.keys())
    split = int(len(spk_list) * 0.8)
    train_spk = {s: speakers[s] for s in spk_list[:split]}
    val_spk = {s: speakers[s] for s in spk_list[split:]}
    print(f"train speakers: {len(train_spk)}, val speakers: {len(val_spk)}")

    print("generating training mixtures...")
    train_ds = MixtureDataset(train_spk, mimi, config.MIMI_SAMPLE_RATE, num_mixtures=2000)
    print("generating validation mixtures...")
    val_ds = MixtureDataset(val_spk, mimi, config.MIMI_SAMPLE_RATE, num_mixtures=200)

    # Choice: num_workers=0 because MixtureDataset runs Mimi inference
    # internally (needs GPU access). Multiprocessing + CUDA = pain.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_mixtures, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_mixtures, num_workers=0)

    device = next(mimi.parameters()).device
    model = SortformerAdapter(
        embed_dim=config.EMBED_DIM,
        num_speakers=NUM_SPEAKERS,
        num_layers=ADAPTER_LAYERS,
        nhead=ADAPTER_HEADS,
        ff_dim=ADAPTER_FF_DIM,
        dropout=ADAPTER_DROPOUT,
    ).to(device)
    print(f"adapter parameters: {model.param_count():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Choice: cosine annealing LR schedule. Sortformer uses this.
    # Alternative was inverse sqrt (also used in Sortformer for ASR).
    # Cosine is simpler, works well for fixed-epoch training.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_der = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for emb, labels, mask in train_loader:
            emb = emb.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            pred = model(emb, padding_mask=mask)
            loss = hybrid_loss(pred, labels, alpha=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            model.eval()
            val_der = 0.0
            val_batches = 0

            with torch.no_grad():
                for emb, labels, mask in val_loader:
                    emb = emb.to(device)
                    labels = labels.to(device)
                    mask = mask.to(device)

                    pred = model(emb, padding_mask=mask)
                    val_der += eval_batch(pred, labels, mask)
                    val_batches += 1

            avg_der = val_der / max(val_batches, 1)
            print(f"epoch {epoch + 1:3d}: loss={avg_loss:.4f}  val_DER={avg_der:.4f}  lr={scheduler.get_last_lr()[0]:.6f}")

            if avg_der < best_der:
                best_der = avg_der
                torch.save(model.state_dict(), SAVE_PATH)
                print(f"  saved best model (DER={best_der:.4f})")
        else:
            print(f"epoch {epoch + 1:3d}: loss={avg_loss:.4f}")

    print(f"training complete. best val DER: {best_der:.4f}")
    return model
