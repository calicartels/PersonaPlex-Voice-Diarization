"""Extract embeddings from Mimi's conv encoder ONLY (before encoder_transformer).
These get fed through unfrozen transformer layers during training."""
import json
import numpy as np
import torch
import torchaudio
import sys, os
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMB, MANIFESTS, SAMPLE_RATE, PERSONAPLEX_REPO, MIMI_CHECKPOINT


def load_mimi(device="cuda"):
    checkpoint_path = Path(MIMI_CHECKPOINT)
    if not checkpoint_path.exists():
        from huggingface_hub import hf_hub_download
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="nvidia/personaplex-7b-v1",
            filename="tokenizer-e351c8d8-checkpoint125.safetensors",
            local_dir=str(checkpoint_path.parent),
            local_dir_use_symlinks=False,
        )
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
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0).to(device)

    # Choice: chunk at 60s — safe for 24GB VRAM.
    chunk_secs = 60
    chunk_samples = chunk_secs * SAMPLE_RATE
    total_samples = wav.shape[-1]

    if total_samples <= chunk_samples:
        with torch.no_grad():
            # KEY: conv encoder only, NOT encoder_transformer
            emb = mimi.encoder(wav)
        return emb.squeeze(0).cpu().numpy()

    chunks = []
    for start in range(0, total_samples, chunk_samples):
        seg = wav[:, :, start:start + chunk_samples]
        with torch.no_grad():
            emb = mimi.encoder(seg)
            chunks.append(emb.squeeze(0).cpu())
        torch.cuda.empty_cache()

    return torch.cat(chunks, dim=-1).numpy()


def process_manifest(mf_path, mimi, device):
    tag = Path(mf_path).stem.replace("_labels", "")
    # Choice: separate dir "preconv" to not overwrite post-transformer embeddings.
    out = EMB / f"{tag}_preconv"
    out.mkdir(parents=True, exist_ok=True)

    entries = [json.loads(l) for l in open(mf_path)]
    updated = []
    for e in tqdm(entries, desc=f"  {tag}", unit="file"):
        audio = e.get("audio_filepath", "")
        if not audio or not Path(audio).exists():
            continue
        sid = e.get("session_id", Path(audio).stem)
        emb_path = out / f"{sid}.npy"
        if not emb_path.exists():
            emb = extract_one(mimi, audio, device)
            np.save(emb_path, emb)
        e["preconv_emb_filepath"] = str(emb_path)
        updated.append(e)

    out_mf = MANIFESTS / f"{tag}_preconv.json"
    with open(out_mf, "w") as f:
        for e in updated:
            f.write(json.dumps(e) + "\n")
    return len(updated)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Mimi on {device} (conv encoder only)")
mimi = load_mimi(device)

manifests = sorted(MANIFESTS.glob("*_labels.json"))
for mf in tqdm(manifests, desc="Manifests", unit="manifest"):
    n = process_manifest(mf, mimi, device)
    print(f"{mf.stem}: {n} pre-conv embeddings")