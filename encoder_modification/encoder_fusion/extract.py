import json
import tempfile
import numpy as np
import torch
import torchaudio
import sys
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import EMB, MANIFESTS, SAMPLE_RATE, TITANET_SR, TITANET_WIN_S, TITANET_HOP_S, PERSONAPLEX_REPO, MIMI_CHECKPOINT


def ensure_checkpoint():
    checkpoint_path = Path(MIMI_CHECKPOINT)
    if checkpoint_path.exists():
        return
    print(f"Checkpoint not found at {MIMI_CHECKPOINT}")
    print("Downloading from HuggingFace...")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="nvidia/personaplex-7b-v1",
        filename="tokenizer-e351c8d8-checkpoint125.safetensors",
        local_dir=str(checkpoint_path.parent),
        local_dir_use_symlinks=False,
    )


def load_mimi(device="cuda"):
    ensure_checkpoint()
    sys.path.insert(0, PERSONAPLEX_REPO)
    from moshi.models.loaders import get_mimi
    mimi = get_mimi(MIMI_CHECKPOINT, device=device)
    mimi.eval()
    return mimi


def extract_mimi_one(mimi, audio_path, device="cuda"):
    wav, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0).to(device)
    chunk_secs = 60
    chunk_samples = chunk_secs * SAMPLE_RATE
    total_samples = wav.shape[-1]
    if total_samples <= chunk_samples:
        with torch.no_grad():
            emb = mimi.encoder(wav)
            if hasattr(mimi, "encoder_transformer") and mimi.encoder_transformer is not None:
                emb = mimi.encoder_transformer(emb)
                emb = emb[0] if isinstance(emb, list) else emb
        return emb.squeeze(0).cpu().numpy()
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        seg = wav[:, :, start:start + chunk_samples]
        with torch.no_grad():
            emb = mimi.encoder(seg)
            if hasattr(mimi, "encoder_transformer") and mimi.encoder_transformer is not None:
                emb = mimi.encoder_transformer(emb)
                emb = emb[0] if isinstance(emb, list) else emb
            chunks.append(emb.squeeze(0).cpu())
        torch.cuda.empty_cache()
    return torch.cat(chunks, dim=-1).numpy()


def load_titanet(device="cuda"):
    from nemo.collections.asr.models import EncDecSpeakerLabelModel
    model = EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
    model = model.to(device).eval()
    return model


def extract_titanet_sliding(titanet, audio_path, device):
    wav, sr = torchaudio.load(audio_path)
    if sr != TITANET_SR:
        wav = torchaudio.functional.resample(wav, sr, TITANET_SR)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    sig = wav.squeeze(0).numpy()
    duration_s = len(sig) / TITANET_SR
    win_samples = int(TITANET_WIN_S * TITANET_SR)
    hop_samples = int(TITANET_HOP_S * TITANET_SR)
    embs = []
    starts = list(range(0, len(sig) - win_samples + 1, hop_samples))
    it = tqdm(starts, desc="  TitaNet", unit="win", leave=False) if len(starts) > 20 else starts
    for start in it:
        chunk = sig[start:start + win_samples]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            torchaudio.save(f.name, torch.from_numpy(chunk).unsqueeze(0), TITANET_SR)
        with torch.no_grad():
            emb = titanet.get_embedding(f.name)
        Path(f.name).unlink()
        embs.append(emb.squeeze().cpu().numpy())
    if not embs:
        win_samples = min(win_samples, len(sig))
        chunk = np.pad(sig, (0, max(0, win_samples - len(sig))), mode="edge")[:win_samples]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            torchaudio.save(f.name, torch.from_numpy(chunk).unsqueeze(0), TITANET_SR)
        with torch.no_grad():
            emb = titanet.get_embedding(f.name)
        Path(f.name).unlink()
        embs = [emb.squeeze().cpu().numpy()]
    embs = np.stack(embs).T
    return embs.astype(np.float32)


def extract_fused(mimi, titanet, audio_path, device):
    mimi_emb = extract_mimi_one(mimi, audio_path, device)
    titanet_emb = extract_titanet_sliding(titanet, audio_path, device)
    T_mimi = mimi_emb.shape[1]
    T_titanet = titanet_emb.shape[1]
    if T_titanet < 2:
        titanet_resized = np.tile(titanet_emb, (1, T_mimi))[:, :T_mimi]
    else:
        zoom_factor = T_mimi / T_titanet
        titanet_resized = zoom(titanet_emb, (1, zoom_factor), order=1)
        titanet_resized = titanet_resized[:, :T_mimi].astype(np.float32)
    fused = np.concatenate([mimi_emb, titanet_resized], axis=0)
    return fused


def process_manifest(mf_path, mimi, titanet, device):
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
        if emb_path.exists():
            e["emb_filepath"] = str(emb_path)
            updated.append(e)
            continue
        fused = extract_fused(mimi, titanet, audio, device)
        np.save(emb_path, fused)
        e["emb_filepath"] = str(emb_path)
        updated.append(e)
    out_mf = MANIFESTS / f"{tag}_fusion_emb.json"
    with open(out_mf, "w") as f:
        for e in updated:
            f.write(json.dumps(e) + "\n")
    return len(updated)


device = "cuda" if torch.cuda.is_available() else "cpu"
ensure_checkpoint()
print("Loading Mimi...")
mimi = load_mimi(device)
print("Loading TitaNet...")
titanet = load_titanet(device)
for mf in sorted(MANIFESTS.glob("*_labels.json")):
    n = process_manifest(mf, mimi, titanet, device)
    print(f"{mf.stem}: {n} fused embeddings")
