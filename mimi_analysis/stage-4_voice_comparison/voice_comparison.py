import os
os.environ["NO_TORCH_COMPILE"] = "1"

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cosine, euclidean
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "moshi"))
from moshi.models.loaders import get_mimi

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import setup_style, COLOR_PALETTE, SIMILARITY_CMAP

def load_audio(audio_path, sample_rate=24000):
    data, sr = sf.read(audio_path)
    waveform = torch.from_numpy(data if data.ndim == 1 else data.T).float()
    waveform = waveform.unsqueeze(0) if data.ndim == 1 else waveform
    resampler = torchaudio.transforms.Resample(sr, sample_rate)
    waveform = resampler(waveform)
    waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
    return waveform.unsqueeze(0)

def extract_features(mimi, audio):
    with torch.no_grad():
        features = mimi._encode_to_unquantized_latent(audio)
    return features

def pool_features(features):
    return features.mean(dim=2)

def process_voices(mimi, audio_files):
    all_features = {}
    all_embeddings = {}
    for voice_id, files in audio_files.items():
        voice_features = []
        voice_embeddings = []
        for audio_path in files:
            audio = load_audio(audio_path)
            features = extract_features(mimi, audio)
            embedding = pool_features(features)
            voice_features.append(features.cpu())
            voice_embeddings.append(embedding.cpu())
        all_features[voice_id] = voice_features
        all_embeddings[voice_id] = voice_embeddings
    return all_features, all_embeddings

def similarity_matrix(all_embeddings):
    labels = []
    embeddings = []
    for voice_id, emb_list in all_embeddings.items():
        for i, emb in enumerate(emb_list):
            labels.append(f"{voice_id}_{i+1}")
            embeddings.append(emb.squeeze(0).numpy())
    n = len(embeddings)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = 1 - cosine(embeddings[i], embeddings[j])
    return sim_matrix, labels

def similarity_heatmap(sim_matrix, labels, output_dir):
    setup_style()
    plt.figure(figsize=(12, 10))
    sns.heatmap(sim_matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt='.3f', cmap=SIMILARITY_CMAP, center=0.5, vmin=0, vmax=1, linewidths=0.5, linecolor='gray')
    plt.title('Cosine Similarity Between Voice Embeddings', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Sample', fontsize=12, fontweight='bold')
    plt.ylabel('Sample', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.savefig(output_dir / 'voice_similarity_matrix.png')
    plt.close()

def separability(all_embeddings):
    intra_distances = []
    for voice_id, emb_list in all_embeddings.items():
        embeddings = [e.squeeze(0).numpy() for e in emb_list]
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                intra_distances.append(euclidean(embeddings[i], embeddings[j]))
    inter_distances = []
    voice_ids = list(all_embeddings.keys())
    for i in range(len(voice_ids)):
        for j in range(i + 1, len(voice_ids)):
            emb_i = [e.squeeze(0).numpy() for e in all_embeddings[voice_ids[i]]]
            emb_j = [e.squeeze(0).numpy() for e in all_embeddings[voice_ids[j]]]
            for ei in emb_i:
                for ej in emb_j:
                    inter_distances.append(euclidean(ei, ej))
    return {'intra_mean': float(np.mean(intra_distances)), 'intra_std': float(np.std(intra_distances)), 'inter_mean': float(np.mean(inter_distances)), 'inter_std': float(np.std(inter_distances)), 'separability_ratio': float(np.mean(inter_distances) / np.mean(intra_distances))}

def projection_2d(all_embeddings, output_dir):
    labels = []
    embeddings = []
    colors_map = {}
    colors = sns.color_palette("husl", len(all_embeddings))
    for idx, (voice_id, emb_list) in enumerate(all_embeddings.items()):
        colors_map[voice_id] = colors[idx]
        for emb in emb_list:
            labels.append(voice_id)
            embeddings.append(emb.squeeze(0).numpy())
    embeddings = np.array(embeddings)
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)
    setup_style()
    plt.figure(figsize=(10, 8))
    for voice_id in set(labels):
        mask = np.array(labels) == voice_id
        plt.scatter(embeddings_pca[mask, 0], embeddings_pca[mask, 1], c=[colors_map[voice_id]], label=voice_id, s=150, alpha=0.7, edgecolors='black', linewidths=1.5)
    plt.xlabel(f'Voice Similarity Dimension 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12, fontweight='bold')
    plt.ylabel(f'Voice Similarity Dimension 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12, fontweight='bold')
    plt.title('Voice Embeddings - 2D Projection', fontsize=13, fontweight='bold', pad=10)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.savefig(output_dir / 'voice_embeddings_2d_pca.png')
    plt.close()
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_tsne = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    for voice_id in set(labels):
        mask = np.array(labels) == voice_id
        plt.scatter(embeddings_tsne[mask, 0], embeddings_tsne[mask, 1], c=[colors_map[voice_id]], label=voice_id, s=150, alpha=0.7, edgecolors='black', linewidths=1.5)
    plt.xlabel('Voice Similarity Dimension 1 (t-SNE)', fontsize=12, fontweight='bold')
    plt.ylabel('Voice Similarity Dimension 2 (t-SNE)', fontsize=12, fontweight='bold')
    plt.title('Voice Embeddings - t-SNE 2D Projection', fontsize=13, fontweight='bold', pad=10)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.savefig(output_dir / 'voice_embeddings_2d_tsne.png')
    plt.close()

def projection_3d(all_embeddings, output_dir, method='pca'):
    labels = []
    embeddings = []
    colors_map = {}
    colors = sns.color_palette("husl", len(all_embeddings))
    for idx, (voice_id, emb_list) in enumerate(all_embeddings.items()):
        colors_map[voice_id] = colors[idx]
        for emb in emb_list:
            labels.append(voice_id)
            embeddings.append(emb.squeeze(0).numpy())
    embeddings = np.array(embeddings)
    if method.lower() == 'pca':
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
        explained_var = pca.explained_variance_ratio_
    else:
        tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_3d = tsne.fit_transform(embeddings)
        explained_var = None
    setup_style()
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    for voice_id in set(labels):
        mask = np.array(labels) == voice_id
        ax.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2], c=[colors_map[voice_id]], label=voice_id, s=100, alpha=0.7, edgecolors='black', linewidths=1)
    if explained_var is not None:
        ax.set_xlabel(f'Voice Similarity Dimension 1 ({explained_var[0]:.1%} Variance)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Voice Similarity Dimension 2 ({explained_var[1]:.1%} Variance)', fontsize=11, fontweight='bold')
        ax.set_zlabel(f'Voice Similarity Dimension 3 ({explained_var[2]:.1%} Variance)', fontsize=11, fontweight='bold')
    else:
        ax.set_xlabel('Voice Similarity Dimension 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('Voice Similarity Dimension 2', fontsize=11, fontweight='bold')
        ax.set_zlabel('Voice Similarity Dimension 3', fontsize=11, fontweight='bold')
    ax.set_title(f'Voice Embeddings - {method.upper()} 3D Projection', fontsize=13, fontweight='bold', pad=15)
    ax.legend(fontsize=10, framealpha=0.9)
    plt.savefig(output_dir / f'voice_embeddings_3d_{method}.png')
    plt.close()

def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent
    weights_path = base_dir / "weights" / "tokenizer-e351c8d8-checkpoint125.safetensors"
    voices_dir = base_dir / "assets" / "voices"
    output_dir = script_dir / "outputs"
    audio_files = {}
    for audio_file in sorted(voices_dir.glob("*.wav")):
        voice_id = audio_file.stem
        audio_files.setdefault(voice_id, []).append(str(audio_file))
    mimi = get_mimi(str(weights_path), device='cpu')
    mimi.eval()
    all_features, all_embeddings = process_voices(mimi, audio_files)
    output_dir.mkdir(exist_ok=True, parents=True)
    sim_matrix, labels = similarity_matrix(all_embeddings)
    similarity_heatmap(sim_matrix, labels, output_dir)
    metrics = separability(all_embeddings)
    projection_2d(all_embeddings, output_dir)
    projection_3d(all_embeddings, output_dir, method='pca')

if __name__ == "__main__":
    main()
