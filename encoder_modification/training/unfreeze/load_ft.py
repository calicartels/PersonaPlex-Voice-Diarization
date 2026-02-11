"""DataLoader for pre-conv embeddings used in fine-tuning pipeline.
Loads pre-transformer Mimi embeddings and labels, crops to fixed length."""
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import FRAME_HZ, SEG_S, MAX_SPEAKERS, BATCH

# Choice: 90s segment at 25Hz (pre-downsample) = 2250 frames.
# The stride-2 downsample to 12.5Hz happens inside the model now,
# so we load at the native 25Hz rate.
SEG_FRAMES_25HZ = int(SEG_S * 25)  # 2250 frames for 90s at 25Hz
SEG_FRAMES_12HZ = int(SEG_S * FRAME_HZ)  # 1125 frames for labels at 12.5Hz


class DiarDatasetFT(Dataset):
    """Dataset that loads pre-conv embeddings (25Hz) and labels (12.5Hz)."""

    def __init__(self, manifest_path):
        entries = [json.loads(l) for l in open(manifest_path) if l.strip()]
        self.entries = [
            e for e in entries
            if e.get("preconv_emb_filepath") and e.get("labels_filepath")
        ]
        if not self.entries:
            # Choice: fallback to original emb_filepath if preconv not available.
            # This lets us test the pipeline with existing embeddings.
            self.entries = [
                e for e in entries
                if e.get("emb_filepath") and e.get("labels_filepath")
            ]
            if self.entries:
                print(f"WARNING: using post-transformer embeddings (no preconv_emb_filepath found)")
                self.use_preconv = False
            else:
                print(f"WARNING: no valid entries in {manifest_path}")
                self.use_preconv = False
        else:
            self.use_preconv = True

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]

        if self.use_preconv:
            emb = np.load(e["preconv_emb_filepath"])  # (D, T) at 25Hz
            # NO stride-2 here — model handles downsampling
        else:
            emb = np.load(e["emb_filepath"])  # (D, T) at 25Hz post-transformer
            # Still no stride-2 — model handles it

        labels = np.load(e["labels_filepath"])  # (T_lbl, K) at 12.5Hz

        D, T_emb = emb.shape
        T_lbl = labels.shape[0]

        # Align: emb is at 25Hz, labels at 12.5Hz. After stride-2, emb becomes 12.5Hz.
        T_emb_downsampled = T_emb // 2
        T_min = min(T_emb_downsampled, T_lbl)

        # Random crop (in 12.5Hz frame space)
        seg_frames_lbl = SEG_FRAMES_12HZ
        max_start = max(0, T_min - seg_frames_lbl)
        start_lbl = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        end_lbl = min(start_lbl + seg_frames_lbl, T_lbl)
        length = end_lbl - start_lbl

        # Convert label-space crop to embedding-space (2x because 25Hz vs 12.5Hz)
        start_emb = start_lbl * 2
        end_emb = min(end_lbl * 2, T_emb)
        seg_frames_emb = SEG_FRAMES_25HZ

        emb_out = np.zeros((D, seg_frames_emb), dtype=np.float32)
        lbl_out = np.zeros((seg_frames_lbl, MAX_SPEAKERS), dtype=np.float32)

        emb_len = end_emb - start_emb
        emb_out[:, :emb_len] = emb[:, start_emb:end_emb]
        lbl_out[:length, :] = labels[start_lbl:end_lbl, :]

        return torch.from_numpy(emb_out), torch.from_numpy(lbl_out), length


def make_loader_ft(manifest_path, shuffle=True):
    ds = DiarDatasetFT(manifest_path)
    # Choice: 4 workers same as before.
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle,
                      num_workers=4, pin_memory=True, drop_last=True)