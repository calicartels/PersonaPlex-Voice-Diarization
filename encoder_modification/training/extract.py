import json
import numpy as np
import torch
import torchaudio
import sys
from pathlib import Path
from tqdm import tqdm
from config import EMB, MANIFESTS, SAMPLE_RATE, PERSONAPLEX_REPO, MIMI_CHECKPOINT


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


def load_mimi(device="cuda"):
    ensure_checkpoint()
    sys.path.insert(0, PERSONAPLEX_REPO)
    from moshi.models.loaders import get_mimi
    mimi = get_mimi(MIMI_CHECKPOINT, device=device)
    mimi.eval()
    return mimi


def extract_one(mimi, audio_path, device="cuda"):
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        # Choice: mean across channels for multi-channel.
        # Alternative: first channel only (faster, loses info).
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0).to(device)

    # Choice: chunk_secs=60 — safe for 24GB VRAM (60s * 24kHz = 1.44M samples).
    # Alternative: 90s matches Sortformer training seg length, but tighter on VRAM.
    # Short sessions (<= chunk_secs) take the fast path with no chunking overhead.
    chunk_secs = 60
    chunk_samples = chunk_secs * SAMPLE_RATE
    total_samples = wav.shape[-1]

    if total_samples <= chunk_samples:
        # Fast path: fits in GPU memory in one pass
        with torch.no_grad():
            emb = mimi.encoder(wav)
            if hasattr(mimi, "encoder_transformer") and mimi.encoder_transformer is not None:
                emb = mimi.encoder_transformer(emb)
                emb = emb[0] if isinstance(emb, list) else emb
        return emb.squeeze(0).cpu().numpy()  # (D, T)

    # Chunked path for long recordings (e.g. AMI meetings = 30-90 min)
    # Choice: process on GPU, collect on CPU, concat at end.
    # Alternative: accumulate on GPU then move — risks OOM on many chunks.
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        seg = wav[:, :, start:start + chunk_samples]
        with torch.no_grad():
            emb = mimi.encoder(seg)
            if hasattr(mimi, "encoder_transformer") and mimi.encoder_transformer is not None:
                emb = mimi.encoder_transformer(emb)
                emb = emb[0] if isinstance(emb, list) else emb
            chunks.append(emb.squeeze(0).cpu())
        # Choice: empty_cache after each chunk — forces VRAM release.
        # Alternative: skip this (faster but risks fragmentation on long files).
        torch.cuda.empty_cache()

    return torch.cat(chunks, dim=-1).numpy()  # (D, T)


def process_manifest(mf_path, mimi, device):
    tag = Path(mf_path).stem.replace("_labels", "")
    out = EMB / tag
    out.mkdir(parents=True, exist_ok=True)

    entries = [json.loads(l) for l in open(mf_path)]
    updated = []
    for i, e in tqdm(enumerate(entries), total=len(entries), desc=f"  {tag}", unit="file"):
        audio = e.get("audio_filepath", "")
        if not audio or not Path(audio).exists():
            continue
        sid = e.get("session_id", Path(audio).stem)
        emb_path = out / f"{sid}.npy"
        if not emb_path.exists():
            emb = extract_one(mimi, audio, device)
            np.save(emb_path, emb)
        e["emb_filepath"] = str(emb_path)
        updated.append(e)

    out_mf = MANIFESTS / f"{tag}_emb.json"
    with open(out_mf, "w") as f:
        for e in updated:
            f.write(json.dumps(e) + "\n")
    return len(updated)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Mimi from {MIMI_CHECKPOINT} on {device}")
mimi = load_mimi(device)

for mf in sorted(MANIFESTS.glob("*_labels.json")):
    n = process_manifest(mf, mimi, device)
    print(f"{mf.stem}: {n} embeddings extracted")