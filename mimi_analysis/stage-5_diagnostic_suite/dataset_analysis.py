import os
os.environ["NO_TORCH_COMPILE"] = "1"

import sys
from pathlib import Path
import sqlite3
import numpy as np
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, euclidean
from scipy.cluster.hierarchy import dendrogram, linkage
from tqdm import tqdm
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "moshi"))
from moshi.models.loaders import get_mimi

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Synthetic_data"))
from config import OUTPUT_DIR, DB_PATH, SAMPLE_RATE

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import setup_style, COLOR_PALETTE, SIMILARITY_CMAP

def load_audio(filepath, sample_rate=SAMPLE_RATE):
    audio, sr = sf.read(filepath)
    audio_tensor = torch.from_numpy(audio).float()
    audio_tensor = audio_tensor.unsqueeze(0) if audio_tensor.ndim == 1 else audio_tensor
    resampler = torchaudio.transforms.Resample(sr, sample_rate)
    audio_resampled = resampler(audio_tensor).numpy()
    audio_resampled = audio_resampled.mean(axis=0, keepdims=True) if audio_resampled.ndim > 1 else audio_resampled
    audio_final = audio_resampled[np.newaxis, np.newaxis, :] if audio_resampled.ndim == 1 else audio_resampled[np.newaxis, :, :]
    return torch.from_numpy(audio_final).float()

def extract_embedding(mimi, audio):
    with torch.no_grad():
        features = mimi._encode_to_unquantized_latent(audio)
        embedding = features.mean(dim=2).squeeze(0).cpu().numpy()
    return embedding

def extract_all(mimi, db, output_dir, batch_size=8):
    cursor = db.cursor()
    cursor.execute("SELECT sample_id, voice_id, filepath, augmentation_type, sentence_id FROM samples WHERE filepath IS NOT NULL")
    samples = cursor.fetchall()
    embeddings_cache = {}
    
    for i in tqdm(range(0, len(samples), batch_size), desc="Extracting embeddings"):
        batch = samples[i:i+batch_size]
        batch_audio = []
        batch_info = []
        for sample_id, voice_id, filepath, aug_type, sent_id in batch:
            full_path = OUTPUT_DIR / filepath
            audio = load_audio(full_path)
            batch_audio.append(audio)
            batch_info.append((sample_id, voice_id, aug_type, sent_id))
        
        max_length = max(a.shape[2] for a in batch_audio)
        padded_audio = []
        for audio in batch_audio:
            if audio.shape[2] < max_length:
                pad_length = max_length - audio.shape[2]
                padding = torch.zeros(1, 1, pad_length, dtype=audio.dtype)
                audio = torch.cat([audio, padding], dim=2)
            padded_audio.append(audio)
        
        batch_tensor = torch.cat(padded_audio, dim=0)
        with torch.no_grad():
            batch_features = mimi._encode_to_unquantized_latent(batch_tensor)
            batch_embeddings = batch_features.mean(dim=2).cpu().numpy()
        
        for (sample_id, voice_id, aug_type, sent_id), embedding in zip(batch_info, batch_embeddings):
            embeddings_cache[sample_id] = {'embedding': embedding, 'voice_id': voice_id, 'augmentation': aug_type, 'sentence_id': sent_id}
    
    cache_path = output_dir / "embeddings_cache.pkl"
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings_cache, f)
    return embeddings_cache

def load_cache(output_dir):
    cache_path = output_dir / "embeddings_cache.pkl"
    return pickle.load(open(cache_path, 'rb'))

def acoustics_semantics(embeddings_cache, output_dir):
    clean_samples = {k: v for k, v in embeddings_cache.items() if v['augmentation'] == 'clean'}
    by_voice = {}
    by_sentence = {}
    for sample_id, data in clean_samples.items():
        by_voice.setdefault(data['voice_id'], []).append(data['embedding'])
        by_sentence.setdefault(data['sentence_id'], []).append(data['embedding'])
    
    intra_voice_sims = []
    for voice_id, embeddings in by_voice.items():
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                intra_voice_sims.append(1 - cosine(embeddings[i], embeddings[j]))
    
    inter_voice_sims = []
    for sent_id, embeddings in by_sentence.items():
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                inter_voice_sims.append(1 - cosine(embeddings[i], embeddings[j]))
    
    results = {'intra_voice_mean': float(np.mean(intra_voice_sims)), 'intra_voice_std': float(np.std(intra_voice_sims)), 'inter_voice_mean': float(np.mean(inter_voice_sims)), 'inter_voice_std': float(np.std(inter_voice_sims))}
    
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(intra_voice_sims, bins=50, alpha=0.7, label='Same Voice, Different Text', color=COLOR_PALETTE[0], edgecolor='black', linewidth=0.5)
    ax.hist(inter_voice_sims, bins=50, alpha=0.7, label='Same Text, Different Voices', color=COLOR_PALETTE[1], edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Acoustics vs Semantics: Voice Identity vs Text Content', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.savefig(output_dir / 'acoustics_vs_semantics.png')
    plt.close()
    return results

def intra_consistency(embeddings_cache):
    by_voice = {}
    for sample_id, data in embeddings_cache.items():
        if data['augmentation'] == 'clean':
            by_voice.setdefault(data['voice_id'], []).append(data['embedding'])
    
    voice_sims = {}
    for voice_id, embeddings in by_voice.items():
        sims = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sims.append(1 - cosine(embeddings[i], embeddings[j]))
        voice_sims[voice_id] = {'mean': float(np.mean(sims)), 'std': float(np.std(sims)), 'min': float(np.min(sims)), 'max': float(np.max(sims))}
    
    overall_mean = float(np.mean([v['mean'] for v in voice_sims.values()]))
    overall_std = float(np.std([v['mean'] for v in voice_sims.values()]))
    return {'mean': overall_mean, 'std': overall_std}

def inter_separability(embeddings_cache):
    voice_embeddings = {}
    for sample_id, data in embeddings_cache.items():
        if data['augmentation'] == 'clean':
            voice_embeddings.setdefault(data['voice_id'], []).append(data['embedding'])
    
    voice_means = {voice_id: np.mean(embs, axis=0) for voice_id, embs in voice_embeddings.items()}
    voice_ids = list(voice_means.keys())
    
    inter_sims = []
    for i in range(len(voice_ids)):
        for j in range(i+1, len(voice_ids)):
            inter_sims.append(1 - cosine(voice_means[voice_ids[i]], voice_means[voice_ids[j]]))
    
    intra_dists = [euclidean(voice_embeddings[vid][i], voice_embeddings[vid][j]) for vid in voice_embeddings for i in range(len(voice_embeddings[vid])) for j in range(i+1, len(voice_embeddings[vid])) if len(voice_embeddings[vid]) > 1]
    inter_dists = [euclidean(voice_means[voice_ids[i]], voice_means[voice_ids[j]]) for i in range(len(voice_ids)) for j in range(i+1, len(voice_ids))]
    
    intra_dist_mean = float(np.mean(intra_dists))
    inter_dist_mean = float(np.mean(inter_dists))
    separability = float(inter_dist_mean / intra_dist_mean if intra_dist_mean > 0 else float('inf'))
    
    return {'inter_mean': float(np.mean(inter_sims)), 'inter_std': float(np.std(inter_sims)), 'separability_ratio': separability}

def robustness(embeddings_cache, output_dir):
    by_voice_sent = {}
    for sample_id, data in embeddings_cache.items():
        key = (data['voice_id'], data['sentence_id'])
        by_voice_sent.setdefault(key, {})[data['augmentation']] = data['embedding']
    
    light_sims = []
    medium_sims = []
    for key, augs in by_voice_sent.items():
        if 'clean' not in augs:
            continue
        clean_emb = augs['clean']
        if 'light_noise' in augs:
            light_sims.append(1 - cosine(clean_emb, augs['light_noise']))
        if 'medium_noise' in augs:
            medium_sims.append(1 - cosine(clean_emb, augs['medium_noise']))
    
    results = {'light_mean': float(np.mean(light_sims)), 'light_std': float(np.std(light_sims)), 'medium_mean': float(np.mean(medium_sims)), 'medium_std': float(np.std(medium_sims))}
    
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(light_sims, bins=50, alpha=0.7, label='Light Noise', color=COLOR_PALETTE[2], edgecolor='black', linewidth=0.5)
    ax.hist(medium_sims, bins=50, alpha=0.7, label='Medium Noise', color=COLOR_PALETTE[3], edgecolor='black', linewidth=0.5)
    ax.axvline(0.85, color='green', linestyle='--', linewidth=2, label='Light Threshold (0.85)', alpha=0.8)
    ax.axvline(0.75, color='orange', linestyle='--', linewidth=2, label='Medium Threshold (0.75)', alpha=0.8)
    ax.set_xlabel('Cosine Similarity to Clean Audio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Robustness to Audio Augmentations', fontsize=13, fontweight='bold', pad=10)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.savefig(output_dir / 'robustness.png')
    plt.close()
    return results

def voice_space(embeddings_cache, output_dir):
    clean_data = [(data['voice_id'], data['embedding']) for data in embeddings_cache.values() if data['augmentation'] == 'clean']
    voice_ids = [v for v, _ in clean_data]
    embeddings = np.array([e for _, e in clean_data])
    pca2d = PCA(n_components=2)
    emb_pca2d = pca2d.fit_transform(embeddings)
    
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    unique_voices = list(set(voice_ids))
    colors = sns.color_palette("husl", len(unique_voices))
    for i, voice_id in enumerate(unique_voices):
        mask = np.array(voice_ids) == voice_id
        ax.scatter(emb_pca2d[mask, 0], emb_pca2d[mask, 1], c=[colors[i]], label=voice_id, s=60, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel(f'Voice Similarity Dimension 1 ({pca2d.explained_variance_ratio_[0]*100:.1f}% Variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Voice Similarity Dimension 2 ({pca2d.explained_variance_ratio_[1]*100:.1f}% Variance)', fontsize=12, fontweight='bold')
    ax.set_title('Voice Embeddings in 2D Space (PCA Projection)', fontsize=13, fontweight='bold', pad=10)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_dir / 'voice_space_pca2d.png')
    plt.close()
    
    voice_means = {}
    for voice_id, emb in clean_data:
        voice_means.setdefault(voice_id, []).append(emb)
    mean_embeddings = {v: np.mean(embs, axis=0) for v, embs in voice_means.items()}
    voice_list = list(mean_embeddings.keys())
    mean_emb_array = np.array([mean_embeddings[v] for v in voice_list])
    linkage_matrix = linkage(mean_emb_array, method='ward')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=voice_list, leaf_rotation=90, ax=ax)
    ax.set_xlabel('Voice ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Euclidean Distance', fontsize=12, fontweight='bold')
    ax.set_title('Voice Clustering Dendrogram (Hierarchical Clustering)', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'voice_clustering.png')
    plt.close()

def generate_report(all_results, output_dir):
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("SYNTHETIC DATASET ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write("TEST 1: ACOUSTICS VS SEMANTICS\n")
        f.write("-"*70 + "\n")
        t1 = all_results['test1']
        f.write(f"Same voice, different text: {t1['intra_voice_mean']:.4f} ± {t1['intra_voice_std']:.4f}\n")
        f.write(f"Same text, different voices: {t1['inter_voice_mean']:.4f} ± {t1['inter_voice_std']:.4f}\n")
        f.write(f"Result: {'PASS - Captures acoustics' if t1['intra_voice_mean'] > t1['inter_voice_mean'] else 'FAIL'}\n\n")
        f.write("TEST 2: INTRA-SPEAKER CONSISTENCY\n")
        f.write("-"*70 + "\n")
        t2 = all_results['test2']
        f.write(f"Mean consistency: {t2['mean']:.4f} ± {t2['std']:.4f}\n")
        f.write(f"Result: {'PASS' if t2['mean'] > 0.80 else 'FAIL'}\n\n")
        f.write("TEST 3: INTER-SPEAKER SEPARABILITY\n")
        f.write("-"*70 + "\n")
        t3 = all_results['test3']
        f.write(f"Inter-speaker similarity: {t3['inter_mean']:.4f} ± {t3['inter_std']:.4f}\n")
        f.write(f"Separability ratio: {t3['separability_ratio']:.4f}\n")
        f.write(f"Result: {'PASS' if t3['separability_ratio'] > 2.0 else 'FAIL'}\n\n")
        f.write("TEST 4: ROBUSTNESS\n")
        f.write("-"*70 + "\n")
        t4 = all_results['test4']
        f.write(f"Light noise: {t4['light_mean']:.4f} ± {t4['light_std']:.4f}\n")
        f.write(f"Medium noise: {t4['medium_mean']:.4f} ± {t4['medium_std']:.4f}\n")
        f.write(f"Result: {'PASS' if t4['light_mean'] > 0.85 and t4['medium_mean'] > 0.75 else 'PARTIAL'}\n\n")

def run_analysis():
    output_dir = Path(__file__).parent / "outputs" / "analysis_results"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    weights_path = Path(__file__).parent.parent.parent / "weights" / "tokenizer-e351c8d8-checkpoint125.safetensors"
    mimi = get_mimi(str(weights_path), device='cpu')
    mimi.eval()
    
    db = sqlite3.connect(DB_PATH)
    cache_path = output_dir / "embeddings_cache.pkl"
    embeddings_cache = load_cache(output_dir) if cache_path.exists() else extract_all(mimi, db, output_dir)
    
    results = {
        'test1': acoustics_semantics(embeddings_cache, output_dir),
        'test2': intra_consistency(embeddings_cache),
        'test3': inter_separability(embeddings_cache),
        'test4': robustness(embeddings_cache, output_dir)
    }
    
    voice_space(embeddings_cache, output_dir)
    generate_report(results, output_dir)
    db.close()
    return results

def main():
    run_analysis()

if __name__ == "__main__":
    main()
