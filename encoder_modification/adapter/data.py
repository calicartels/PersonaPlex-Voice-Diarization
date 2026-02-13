import random
import numpy as np
import torch
from torch.utils.data import Dataset

# Choice: simulated mixtures from single-speaker clips for initial validation.
# Real training will use NeMo speech data simulator (Fisher, LibriSpeech sources)
# with controlled overlap ratios, as in Sortformer paper. This simple mixer
# validates the architecture before investing in the full data pipeline.


def mix_utterances(wav1, wav2, sr, overlap_ratio=0.3):
    # Choice: default 0.3 matches Sortformer's training config
    # (overlap_ratio=0.12 for simulated data, but we want more
    # overlap exposure for the adapter to learn from).
    n1, n2 = len(wav1), len(wav2)

    # Place wav2 after wav1, pulled back by overlap amount
    shorter = min(n1, n2)
    overlap_samples = int(shorter * overlap_ratio)
    offset = n1 - overlap_samples

    total_len = max(n1, offset + n2)
    mixed = np.zeros(total_len, dtype=np.float32)
    mixed[:n1] += wav1
    mixed[offset:offset + n2] += wav2

    # Frame-level labels at 12.5Hz (80ms per frame)
    frame_step = sr / 12.5  # samples per frame
    num_frames = int(total_len / frame_step)

    labels = np.zeros((num_frames, 2), dtype=np.float32)
    for t in range(num_frames):
        center = int(t * frame_step)
        # Speaker 0 active if within wav1 range
        if center < n1:
            labels[t, 0] = 1.0
        # Speaker 1 active if within wav2 range (offset-relative)
        if offset <= center < offset + n2:
            labels[t, 1] = 1.0

    return mixed, labels


class MixtureDataset(Dataset):
    # Choice: generate 2-speaker mixtures only for initial validation.
    # 3-4 speaker mixtures need more sophisticated mixing (multiple
    # offsets, energy normalization). We validate the architecture
    # on 2 speakers first, scale up with NeMo simulator later.

    def __init__(self, speakers, mimi, sr, num_mixtures=5000, max_speakers=4,
                 overlap_range=(0.1, 0.5)):
        self.speakers = speakers
        self.speaker_ids = list(speakers.keys())
        self.mimi = mimi
        self.sr = sr
        self.num_mixtures = num_mixtures
        self.max_speakers = max_speakers
        self.overlap_range = overlap_range

    def __len__(self):
        return self.num_mixtures

    def __getitem__(self, idx):
        # Pick 2 different speakers
        spk_pair = random.sample(self.speaker_ids, 2)
        wav1 = random.choice(self.speakers[spk_pair[0]])
        wav2 = random.choice(self.speakers[spk_pair[1]])

        overlap = random.uniform(*self.overlap_range)
        mixed, labels = mix_utterances(wav1, wav2, self.sr, overlap)

        # Run mixture through frozen Mimi
        wav_t = torch.from_numpy(mixed).float().unsqueeze(0).unsqueeze(0)
        wav_t = wav_t.to(next(self.mimi.parameters()).device)

        with torch.no_grad():
            emb = self.mimi._encode_to_unquantized_latent(wav_t)
            # [1, 512, T_frames] -> [T_frames, 512]
            emb = emb.squeeze(0).permute(1, 0).cpu()

        # Align label length to embedding length
        T = emb.size(0)
        if labels.shape[0] > T:
            labels = labels[:T]
        elif labels.shape[0] < T:
            pad = np.zeros((T - labels.shape[0], labels.shape[1]), dtype=np.float32)
            labels = np.concatenate([labels, pad], axis=0)

        # Pad to max_speakers channels (fill unused with zeros)
        if labels.shape[1] < self.max_speakers:
            pad = np.zeros((T, self.max_speakers - labels.shape[1]), dtype=np.float32)
            labels = np.concatenate([labels, pad], axis=1)

        return emb, torch.from_numpy(labels)


def collate_mixtures(batch):
    # Choice: zero-pad shorter sequences and return padding mask.
    # Alternative was bucketing by length (fewer wasted frames) but
    # adds complexity. With 2-speaker mixtures from VoxCeleb (4-70s),
    # length variance is moderate.
    embs, labels = zip(*batch)
    max_t = max(e.size(0) for e in embs)

    B = len(batch)
    D = embs[0].size(1)
    K = labels[0].size(1)

    padded_emb = torch.zeros(B, max_t, D)
    padded_lab = torch.zeros(B, max_t, K)
    mask = torch.ones(B, max_t, dtype=torch.bool)

    for i, (e, l) in enumerate(zip(embs, labels)):
        t = e.size(0)
        padded_emb[i, :t] = e
        padded_lab[i, :t] = l
        mask[i, :t] = False

    return padded_emb, padded_lab, mask
