import sys
import os
import numpy as np
import torch
import torchaudio.functional as AF
from tqdm import tqdm
import config

EMB_CACHE = os.path.join(config.CACHE_DIR, "baseline_embeddings.npz")


def load_mimi():
    # Add personaplex repo to path so we can import moshi
    sys.path.insert(0, config.PERSONAPLEX_REPO)
    from moshi.models.loaders import get_mimi

    # Choice: get_mimi handles weight loading and model construction.
    # Alternative was manually instantiating MimiModel and loading state_dict,
    # but get_mimi already does all of that with the right kwargs.
    print(f"loading Mimi from {config.MIMI_CHECKPOINT}")
    mimi = get_mimi(config.MIMI_CHECKPOINT, device=config.DEVICE)
    mimi.eval()
    return mimi


def extract_one(mimi, audio_array, sr):
    wav = torch.from_numpy(audio_array).float()

    if sr != config.MIMI_SAMPLE_RATE:
        wav = AF.resample(wav, sr, config.MIMI_SAMPLE_RATE)

    # Mimi expects [B, 1, T_samples]
    wav = wav.unsqueeze(0).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        # Returns [B, 512, T_frames] before quantization
        emb = mimi._encode_to_unquantized_latent(wav)
        # Choice: mean pool over time -> [512].
        # See config.py for rationale on pooling method.
        emb = emb.mean(dim=2).squeeze(0)

    return emb.cpu().numpy()


def extract_all(ds, force=False):
    if os.path.exists(EMB_CACHE) and not force:
        print(f"loading cached embeddings from {EMB_CACHE}")
        data = np.load(EMB_CACHE)
        return dict(zip(data["ids"], data["embeddings"]))

    mimi = load_mimi()
    ids = []
    embeddings = []

    for row in tqdm(ds, desc="Extracting embeddings", unit="utterance"):
        uid = row["id"]
        audio = row["audio"]
        emb = extract_one(mimi, audio["array"], audio["sampling_rate"])
        ids.append(uid)
        embeddings.append(emb)

    print(f"extracted {len(ids)} embeddings")

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.savez(EMB_CACHE, ids=np.array(ids), embeddings=np.stack(embeddings))
    print(f"cached to {EMB_CACHE}")

    return dict(zip(ids, embeddings))