import argparse
import sys
import torch
import torchaudio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import median_filter
from config import CKPT, SAMPLE_RATE, FRAME_S, MAX_SPEAKERS, PERSONAPLEX_REPO, MIMI_CHECKPOINT


def ensure_checkpoint():
    checkpoint_path = Path(MIMI_CHECKPOINT)
    if checkpoint_path.exists():
        return
    
    print(f"Checkpoint not found at {MIMI_CHECKPOINT}")
    print("Downloading from HuggingFace...")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    from huggingface_hub import hf_hub_download
    downloaded = hf_hub_download(
        repo_id="nvidia/personaplex-7b-v1",
        filename="tokenizer-e351c8d8-checkpoint125.safetensors",
        local_dir=str(checkpoint_path.parent),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded to: {downloaded}")


def load_mimi(device):
    ensure_checkpoint()
    sys.path.insert(0, PERSONAPLEX_REPO)
    from moshi.models.loaders import get_mimi
    mimi = get_mimi(MIMI_CHECKPOINT, device=device)
    mimi.eval()
    return mimi


def diarize(audio_path, mimi, model, device, threshold=0.5):
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0).to(device)

    with torch.no_grad():
        emb = mimi.encoder(wav)
        if hasattr(mimi, "encoder_transformer") and mimi.encoder_transformer is not None:
            emb = mimi.encoder_transformer(emb)
        pred = model(emb).squeeze(0).cpu().numpy()

    labels = (pred > threshold).astype(float)
    for k in range(MAX_SPEAKERS):
        labels[:, k] = median_filter(labels[:, k], size=5)
    return labels


def to_rttm(labels, sid):
    lines = []
    for k in range(labels.shape[1]):
        active = labels[:, k]
        in_seg, start = False, 0
        for t in range(len(active)):
            if active[t] and not in_seg:
                start, in_seg = t, True
            elif not active[t] and in_seg:
                on, dur = start * FRAME_S, (t - start) * FRAME_S
                lines.append(f"SPEAKER {sid} 1 {on:.3f} {dur:.3f} <NA> <NA> spk{k} <NA> <NA>")
                in_seg = False
        if in_seg:
            on, dur = start * FRAME_S, (len(active) - start) * FRAME_S
            lines.append(f"SPEAKER {sid} 1 {on:.3f} {dur:.3f} <NA> <NA> spk{k} <NA> <NA>")
    return "\n".join(lines)


def plot_timeline(labels, out_path):
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(14, 3))
    times = np.arange(labels.shape[0]) * FRAME_S
    for k in range(labels.shape[1]):
        if labels[:, k].sum() == 0:
            continue
        ax.fill_between(times, k, k + 0.8, where=labels[:, k].astype(bool),
                        color=colors[k], alpha=0.7, label=f"Speaker {k}")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([i + 0.4 for i in range(MAX_SPEAKERS)])
    ax.set_yticklabels([f"Spk {i}" for i in range(MAX_SPEAKERS)])
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


# argparse genuinely needed: user provides audio path + output dir
parser = argparse.ArgumentParser()
parser.add_argument("--audio", required=True)
parser.add_argument("--out", default="demo_output")
parser.add_argument("--threshold", type=float, default=0.5)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(args.out)
out.mkdir(exist_ok=True)

print(f"Loading models on {device}...")
mimi = load_mimi(device)

# Detect dim
test = torch.randn(1, 1, SAMPLE_RATE).to(device)
with torch.no_grad():
    d = mimi.encoder(test)
    if hasattr(mimi, "encoder_transformer") and mimi.encoder_transformer is not None:
        d = mimi.encoder_transformer(d)

from model import MimiSpeaker
model = MimiSpeaker(input_dim=d.shape[1]).to(device)
model.load_state_dict(torch.load(CKPT / "best.pt", map_location=device))
model.eval()

print(f"Diarizing {args.audio}...")
labels = diarize(args.audio, mimi, model, device, args.threshold)

sid = Path(args.audio).stem
n_spk = sum(labels[:, k].sum() > 0 for k in range(MAX_SPEAKERS))

rttm_path = out / f"{sid}.rttm"
rttm_path.write_text(to_rttm(labels, sid))

plot_timeline(labels, out / f"{sid}_timeline.png")

print(f"{n_spk} speakers detected -> {rttm_path}")