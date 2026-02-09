import json
import numpy as np
import torch
import torchaudio
import sys
from pathlib import Path
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

    with torch.no_grad():
        # Choice: encoder + encoder_transformer output (continuous, pre-RVQ).
        # Richest representation before quantization.
        # Alternative: discrete RVQ codes, but loses fine-grained speaker info.
        emb = mimi.encoder(wav)
        if hasattr(mimi, "encoder_transformer") and mimi.encoder_transformer is not None:
            emb = mimi.encoder_transformer(emb)
    return emb.squeeze(0).cpu().numpy()  # (D, T)


def process_manifest(mf_path, mimi, device):
    tag = Path(mf_path).stem.replace("_labels", "")
    out = EMB / tag
    out.mkdir(parents=True, exist_ok=True)

    entries = [json.loads(l) for l in open(mf_path)]
    updated = []
    for i, e in enumerate(entries):
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
        if (i + 1) % 200 == 0:
            print(f"  {tag}: {i+1}/{len(entries)}")

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