import os
os.environ["NO_TORCH_COMPILE"] = "1"

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
import soundfile as sf
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "moshi"))
from moshi.models.loaders import get_mimi

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import setup_style, COLOR_PALETTE

def load_audio(audio_path, sample_rate=24000):
    data, sr = sf.read(audio_path)
    waveform = torch.from_numpy(data if data.ndim == 1 else data.T).float()
    waveform = waveform.unsqueeze(0) if data.ndim == 1 else waveform
    resampler = torchaudio.transforms.Resample(sr, sample_rate)
    waveform = resampler(waveform)
    waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
    return waveform.unsqueeze(0)

def extract_embedding(mimi, audio):
    with torch.no_grad():
        features = mimi._encode_to_unquantized_latent(audio)
        embedding = features.mean(dim=2).squeeze(0).cpu().numpy()
    return embedding

def temporal_consistency(mimi, audio_files, output_dir):
    embeddings = []
    for audio_path in audio_files:
        audio = load_audio(Path(audio_path))
        embedding = extract_embedding(mimi, audio)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            similarities.append(sim)
    
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    setup_style()
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=30, edgecolor='black', color=COLOR_PALETTE[0], alpha=0.7)
    plt.xlabel('Cosine Similarity', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title('Temporal Consistency: Same Voice Across Different Samples', fontsize=13, fontweight='bold', pad=10)
    plt.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.4f}')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.savefig(output_dir / 'temporal_consistency.png')
    plt.close()
    
    return {'mean_similarity': float(mean_sim), 'std_similarity': float(std_sim), 'min_similarity': float(np.min(similarities)), 'max_similarity': float(np.max(similarities))}

def sentence_variation(mimi, audio_files, sentence_ids, output_dir):
    embeddings = {}
    for audio_path, sent_id in zip(audio_files, sentence_ids):
        audio = load_audio(Path(audio_path))
        embedding = extract_embedding(mimi, audio)
        embeddings.setdefault(sent_id, []).append(embedding)
    
    sentence_means = {sid: np.mean(embs, axis=0) for sid, embs in embeddings.items()}
    
    similarities = []
    sent_pairs = []
    for i, (sid1, emb1) in enumerate(sentence_means.items()):
        for j, (sid2, emb2) in enumerate(sentence_means.items()):
            if i < j:
                sim = 1 - cosine(emb1, emb2)
                similarities.append(sim)
                sent_pairs.append(f"{sid1}-{sid2}")
    
    mean_sim = np.mean(similarities)
    
    setup_style()
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(similarities)), similarities, color=COLOR_PALETTE[1], edgecolor='black', linewidth=0.5)
    plt.xlabel('Sentence Pair', fontsize=12, fontweight='bold')
    plt.ylabel('Cosine Similarity', fontsize=12, fontweight='bold')
    plt.title('Sentence-to-Sentence Variation', fontsize=13, fontweight='bold', pad=10)
    plt.xticks(range(len(sent_pairs)), sent_pairs, rotation=45, ha='right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'sentence_variation.png')
    plt.close()
    
    return {'mean_sentence_similarity': float(mean_sim), 'sentence_similarities': {pair: float(sim) for pair, sim in zip(sent_pairs, similarities)}}

def voice_space(mimi, audio_files, output_dir, labels=None):
    embeddings = []
    for audio_path in audio_files:
        audio = load_audio(Path(audio_path))
        embedding = extract_embedding(mimi, audio)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    setup_style()
    plt.figure(figsize=(10, 8))
    if labels:
        unique_labels = list(set(labels))
        colors = sns.color_palette("husl", len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = [j for j, l in enumerate(labels) if l == label]
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=label, alpha=0.7, c=[colors[i]], s=60, edgecolors='black', linewidths=0.5)
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=60, edgecolors='black', linewidths=0.5)
    
    plt.xlabel(f'Voice Similarity Dimension 1 ({pca.explained_variance_ratio_[0]:.2%} Variance)', fontsize=12, fontweight='bold')
    plt.ylabel(f'Voice Similarity Dimension 2 ({pca.explained_variance_ratio_[1]:.2%} Variance)', fontsize=12, fontweight='bold')
    plt.title('Single Voice in 2D Space (PCA Projection)', fontsize=13, fontweight='bold', pad=10)
    if labels:
        plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.savefig(output_dir / 'voice_space_2d.png')
    plt.close()

def main():
    weights_path = Path(__file__).parent.parent.parent / "weights" / "tokenizer-e351c8d8-checkpoint125.safetensors"
    mimi = get_mimi(str(weights_path), device='cpu')
    mimi.eval()
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    main()
