import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import FRAME_HZ, SEG_S, MAX_SPEAKERS, BATCH

SEG_FRAMES = int(SEG_S * FRAME_HZ)  # 1125 frames for 90s


# PyTorch DataLoader requires Dataset class (__len__ + __getitem__ protocol).
class DiarDataset(Dataset):
    def __init__(self, manifest_path):
        entries = [json.loads(l) for l in open(manifest_path) if l.strip()]
        self.entries = [
            e for e in entries
            if e.get("emb_filepath") and e.get("labels_filepath")
        ]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        emb = np.load(e["emb_filepath"])       # (D, T_emb)
        # Mimi encoder runs at 25Hz but labels are at 12.5Hz (80ms frames).
        # Choice: stride-2 downsample — simple, preserves sharp speaker boundaries.
        # Alternative: average-pool pairs — smoother but blurs transitions.
        emb = emb[:, ::2]                      # (D, T_emb//2) → 12.5Hz
        labels = np.load(e["labels_filepath"])  # (T_lbl, K)

        D, T_emb = emb.shape
        T_lbl = labels.shape[0]
        T_min = min(T_emb, T_lbl)

        # Random crop to SEG_FRAMES, pad if shorter
        max_start = max(0, T_min - SEG_FRAMES)
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        end = min(start + SEG_FRAMES, T_emb, T_lbl)
        length = end - start

        emb_out = np.zeros((D, SEG_FRAMES), dtype=np.float32)
        lbl_out = np.zeros((SEG_FRAMES, MAX_SPEAKERS), dtype=np.float32)
        emb_out[:, :length] = emb[:, start:end]
        lbl_out[:length, :] = labels[start:end, :]

        return torch.from_numpy(emb_out), torch.from_numpy(lbl_out), length


def make_loader(manifest_path, shuffle=True):
    ds = DiarDataset(manifest_path)
    # Choice: 4 workers keeps GPU fed on 16-core instance.
    # Alternative: 8 workers, diminishing returns for .npy loading.
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                      num_workers=4, pin_memory=True, drop_last=True)