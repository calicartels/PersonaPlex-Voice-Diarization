import os
os.environ["NO_TORCH_COMPILE"] = "1"

import sys
from pathlib import Path
import sqlite3
import numpy as np
import torch
import torchaudio
import soundfile as sf
from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "moshi"))
from moshi.models.loaders import get_mimi

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Synthetic_data"))
from config import OUTPUT_DIR, DB_PATH, SAMPLE_RATE

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import setup_style, COLOR_PALETTE, SIMILARITY_CMAP

def mean_pooling(features):
        return features.mean(dim=2)
    
def max_pooling(features):
        return features.max(dim=2)[0]
    
def stats_pooling(features):
        mean = features.mean(dim=2)
        std = features.std(dim=2)
        return torch.cat([mean, std], dim=1)
    
def first_frame(features):
        return features[:, :, 0]
    
def last_frame(features):
        return features[:, :, -1]

def l2_normalize(embedding):
        norm = np.linalg.norm(embedding)
    return embedding if norm == 0 else embedding / norm
    
def whitening(embeddings):
        mean = embeddings.mean(axis=0)
        centered = embeddings - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        epsilon = 1e-5
        whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + epsilon)) @ eigenvectors.T
    whitened = centered @ whitening_matrix.T
        params = {'mean': mean, 'whitening_matrix': whitening_matrix}
        return whitened, params
    
def pca_reduce(embeddings, n_components=256):
        from sklearn.decomposition import PCA
        max_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=max_components)
        reduced = pca.fit_transform(embeddings)
        params = {'pca': pca, 'explained_variance': pca.explained_variance_ratio_.sum()}
        return reduced, params

def load_samples(db):
    cursor = db.cursor()
    cursor.execute("SELECT sample_id, voice_id, filepath, augmentation_type, sentence_id, sentence_text FROM samples WHERE augmentation_type = 'clean' AND filepath IS NOT NULL")
    samples = []
        for row in cursor.fetchall():
        samples.append({'sample_id': row[0], 'voice_id': row[1], 'filepath': row[2], 'augmentation': row[3], 'sentence_id': row[4], 'sentence_text': row[5]})
    return samples
    
def load_audio(filepath, target_duration=None, sample_rate=SAMPLE_RATE):
        audio, sr = sf.read(filepath)
            audio_tensor = torch.from_numpy(audio).float()
    audio_tensor = audio_tensor.unsqueeze(0) if audio_tensor.ndim == 1 else audio_tensor
    resampler = torchaudio.transforms.Resample(sr, sample_rate)
    audio_resampled = resampler(audio_tensor).numpy()
    audio_resampled = audio_resampled[:int(target_duration * sample_rate)] if target_duration and len(audio_resampled) > int(target_duration * sample_rate) else audio_resampled
    audio_resampled = audio_resampled.mean(axis=0, keepdims=True) if audio_resampled.ndim > 1 else audio_resampled
    audio_final = audio_resampled[np.newaxis, np.newaxis, :] if audio_resampled.ndim == 1 else audio_resampled[np.newaxis, :, :]
    return torch.from_numpy(audio_final).float()
    
def extract_layer(mimi, audio, layer_idx=None):
        with torch.no_grad():
        features = mimi.encoder(audio)
        if layer_idx is None or mimi.encoder_transformer is None:
                return features
        transformer = mimi.encoder_transformer
                features = features.transpose(1, 2)
        features = transformer.input_proj(features) if transformer.input_proj is not None else features
            num_layers = len(transformer.transformer.layers)
            target_layer = layer_idx if layer_idx >= 0 else num_layers
        for layer in transformer.transformer.layers[:target_layer]:
                features = layer(features)
        return features.transpose(1, 2)
            
def test_config(mimi, samples, dataset_dir, extraction_point, pooling_method, audio_duration=None, post_processing=None):
        config_name = f"{extraction_point}_{pooling_method}"
    config_name = f"{config_name}_{audio_duration}s" if audio_duration else config_name
    config_name = f"{config_name}_{post_processing}" if post_processing else config_name
        
        embeddings_list = []
        sample_info = []
        
    for sample in tqdm(samples, desc=config_name[:30], leave=False):
        filepath = dataset_dir / sample['filepath']
        audio = load_audio(filepath, audio_duration)
        layer_map = {'layer0_encoder': None, 'layer2_transformer': 2, 'layer4_transformer': 4, 'layer6_transformer': 6, 'layer8_transformer': 8, 'final': -1}
                layer_idx = layer_map.get(extraction_point)
        features = extract_layer(mimi, audio, layer_idx)
        pooling_funcs = {'mean_pooling': mean_pooling, 'max_pooling': max_pooling, 'stats_pooling': stats_pooling, 'first_frame': first_frame, 'last_frame': last_frame}
        embedding = pooling_funcs[pooling_method](features)
                embedding = embedding.squeeze(0).cpu().numpy()
                embeddings_list.append(embedding)
                sample_info.append(sample)
                
    if not embeddings_list:
            return None
        
        embeddings_array = np.array(embeddings_list)
    post_proc_map = {'l2_norm': lambda x: np.array([l2_normalize(emb) for emb in x]), 'whitening': lambda x: whitening(x)[0], 'pca': lambda x: pca_reduce(x, min(256, x.shape[1]))[0]}
    embeddings_array = post_proc_map.get(post_processing, lambda x: x)(embeddings_array)
        
    metrics = compute_metrics(embeddings_array, sample_info)
    metrics.update({'config_name': config_name, 'extraction_point': extraction_point, 'pooling_method': pooling_method, 'audio_duration': audio_duration, 'post_processing': post_processing})
        return metrics
    
def compute_metrics(embeddings, sample_info):
        by_voice = {}
        by_sentence = {}
        for i, sample in enumerate(sample_info):
        by_voice.setdefault(sample['voice_id'], []).append(embeddings[i])
        by_sentence.setdefault(sample['sentence_id'], []).append(embeddings[i])
        
        intra_voice_sims = []
        for voice_id, embs in by_voice.items():
            for i in range(len(embs)):
                for j in range(i+1, len(embs)):
                intra_voice_sims.append(1 - cosine(embs[i], embs[j]))
        
        inter_voice_sims = []
        for sent_id, embs in by_sentence.items():
            for i in range(len(embs)):
                for j in range(i+1, len(embs)):
                inter_voice_sims.append(1 - cosine(embs[i], embs[j]))
        
        intra_dists = []
        for voice_id, embs in by_voice.items():
            for i in range(len(embs)):
                for j in range(i+1, len(embs)):
                intra_dists.append(euclidean(embs[i], embs[j]))
        
        voice_means = {v: np.mean(embs, axis=0) for v, embs in by_voice.items()}
        voice_ids = list(voice_means.keys())
        inter_dists = []
        for i in range(len(voice_ids)):
            for j in range(i+1, len(voice_ids)):
            inter_dists.append(euclidean(voice_means[voice_ids[i]], voice_means[voice_ids[j]]))
        
        intra_voice_mean = float(np.mean(intra_voice_sims)) if intra_voice_sims else 0.0
        inter_voice_mean = float(np.mean(inter_voice_sims)) if inter_voice_sims else 0.0
        intra_dist_mean = float(np.mean(intra_dists)) if intra_dists else 0.0
        inter_dist_mean = float(np.mean(inter_dists)) if inter_dists else 0.0
        separability = float(inter_dist_mean / intra_dist_mean) if intra_dist_mean > 0 else 0.0
        acoustics_score = float(intra_voice_mean - inter_voice_mean)
        
    return {'intra_voice_similarity': intra_voice_mean, 'inter_voice_similarity': inter_voice_mean, 'acoustics_score': acoustics_score, 'separability_ratio': separability, 'intra_dist_mean': intra_dist_mean, 'inter_dist_mean': inter_dist_mean, 'num_voices': int(len(by_voice)), 'num_samples': int(len(embeddings)), 'embedding_dim': int(embeddings.shape[1])}
    
def convert_types(obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
        return [convert_types(item) for item in obj]
        elif pd.isna(obj):
            return None
    elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    
def save_results(results, output_dir):
    df = pd.DataFrame(results)
    csv_path = output_dir / 'diagnostic_results.csv'
        df.to_csv(csv_path, index=False)
    json_path = output_dir / 'diagnostic_results.json'
    results_native = convert_types(results)
        with open(json_path, 'w') as f:
            json.dump(results_native, f, indent=2)
    
def generate_plots(results, output_dir):
    df = pd.DataFrame(results)
        core_results = df[df['audio_duration'].isna() & df['post_processing'].isna()]
        
        if not core_results.empty:
        pivot_acoustics = core_results.pivot(index='extraction_point', columns='pooling_method', values='acoustics_score')
        pivot_sep = core_results.pivot(index='extraction_point', columns='pooling_method', values='separability_ratio')
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        setup_style()
        sns.heatmap(pivot_acoustics, annot=True, fmt='.3f', cmap=SIMILARITY_CMAP, center=0, ax=axes[0], cbar_kws={'label': 'Acoustics Score'}, linewidths=0.5, linecolor='gray')
        axes[0].set_title('Acoustics Score: Voice vs Text Discrimination', fontsize=13, fontweight='bold', pad=10)
        axes[0].set_xlabel('Pooling Method', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Extraction Point', fontsize=12, fontweight='bold')
        sns.heatmap(pivot_sep, annot=True, fmt='.3f', cmap=SIMILARITY_CMAP, center=2.0, ax=axes[1], cbar_kws={'label': 'Separability Ratio'}, linewidths=0.5, linecolor='gray')
        axes[1].set_title('Separability Ratio: Inter-Speaker vs Intra-Speaker Distance', fontsize=13, fontweight='bold', pad=10)
        axes[1].set_xlabel('Pooling Method', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Extraction Point', fontsize=12, fontweight='bold')
            plt.tight_layout()
        plt.savefig(output_dir / 'extraction_pooling_heatmap.png')
            plt.close()
        
        duration_results = df[df['audio_duration'].notna()]
        if not duration_results.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            durations = duration_results.groupby('audio_duration')
        setup_style()
        bars1 = axes[0].bar(range(len(durations)), durations['acoustics_score'].mean(), color=COLOR_PALETTE[0], edgecolor='black', linewidth=1.2, alpha=0.8)
        axes[0].bar_label(bars1, fmt='%.3f', padding=3)
            axes[0].set_xticks(range(len(durations)))
        axes[0].set_xticklabels([f"{d}s" for d in durations.groups.keys()], fontsize=11)
        axes[0].set_ylabel('Acoustics Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Effect of Audio Duration on Voice Discrimination', fontsize=13, fontweight='bold', pad=10)
        axes[0].axhline(0, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        bars2 = axes[1].bar(range(len(durations)), durations['separability_ratio'].mean(), color=COLOR_PALETTE[1], edgecolor='black', linewidth=1.2, alpha=0.8)
        axes[1].bar_label(bars2, fmt='%.2f', padding=3)
            axes[1].set_xticks(range(len(durations)))
        axes[1].set_xticklabels([f"{d}s" for d in durations.groups.keys()], fontsize=11)
        axes[1].set_ylabel('Separability Ratio', fontsize=12, fontweight='bold')
        axes[1].set_title('Effect of Audio Duration on Speaker Separation', fontsize=13, fontweight='bold', pad=10)
        axes[1].axhline(2.0, color='green', linestyle='--', alpha=0.6, linewidth=1.5, label='Target (2.0)')
        axes[1].legend(fontsize=10, framealpha=0.9)
        axes[1].grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
        plt.savefig(output_dir / 'duration_effect.png')
            plt.close()
        
        postproc_results = df[df['post_processing'].notna()]
        if not postproc_results.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            methods = postproc_results.groupby('post_processing')
        setup_style()
        bars1 = axes[0].bar(range(len(methods)), methods['acoustics_score'].mean(), color=COLOR_PALETTE[4], edgecolor='black', linewidth=1.2, alpha=0.8)
        axes[0].bar_label(bars1, fmt='%.3f', padding=3)
            axes[0].set_xticks(range(len(methods)))
        axes[0].set_xticklabels(methods.groups.keys(), rotation=45, ha='right', fontsize=11)
        axes[0].set_ylabel('Acoustics Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Effect of Post-Processing on Voice Discrimination', fontsize=13, fontweight='bold', pad=10)
        axes[0].axhline(0, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        bars2 = axes[1].bar(range(len(methods)), methods['separability_ratio'].mean(), color=COLOR_PALETTE[5], edgecolor='black', linewidth=1.2, alpha=0.8)
        axes[1].bar_label(bars2, fmt='%.2f', padding=3)
            axes[1].set_xticks(range(len(methods)))
        axes[1].set_xticklabels(methods.groups.keys(), rotation=45, ha='right', fontsize=11)
        axes[1].set_ylabel('Separability Ratio', fontsize=12, fontweight='bold')
        axes[1].set_title('Effect of Post-Processing on Speaker Separation', fontsize=13, fontweight='bold', pad=10)
        axes[1].axhline(2.0, color='green', linestyle='--', alpha=0.6, linewidth=1.5, label='Target (2.0)')
        axes[1].legend(fontsize=10, framealpha=0.9)
        axes[1].grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
        plt.savefig(output_dir / 'postprocessing_effect.png')
            plt.close()
            
def run_diagnostic(dataset_dir, weights_path):
    output_dir = Path(__file__).parent / "outputs" / "diagnostic_results"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    mimi = get_mimi(str(weights_path), device='cpu')
    mimi.eval()
    
    db = sqlite3.connect(DB_PATH)
    samples = load_samples(db)
    results = []
    
    extraction_points = ['layer0_encoder', 'layer2_transformer', 'layer4_transformer', 'layer6_transformer', 'layer8_transformer', 'final']
    pooling_methods = ['mean_pooling', 'max_pooling', 'stats_pooling', 'first_frame', 'last_frame']
    audio_durations = [None, 1.0, 3.0, 5.0]
    post_processing_methods = [None, 'l2_norm', 'whitening', 'pca']
    
    for extraction in extraction_points:
        for pooling in pooling_methods:
            result = test_config(mimi, samples, dataset_dir, extraction, pooling, None, None)
            if result:
                results.append(result)
    
    if not results:
        db.close()
        return
    
    best_config = max(results, key=lambda x: x['acoustics_score'])
    
    for duration in audio_durations[1:]:
        result = test_config(mimi, samples, dataset_dir, best_config['extraction_point'], best_config['pooling_method'], duration, None)
        if result:
            results.append(result)
    
    for post_proc in post_processing_methods[1:]:
        result = test_config(mimi, samples, dataset_dir, best_config['extraction_point'], best_config['pooling_method'], None, post_proc)
        if result:
            results.append(result)
    
    save_results(results, output_dir)
    generate_plots(results, output_dir)
    db.close()

def main():
    dataset_dir = Path(__file__).parent.parent.parent / "Synthetic_data"
    weights_path = Path(__file__).parent.parent.parent / "weights" / "tokenizer-e351c8d8-checkpoint125.safetensors"
    run_diagnostic(dataset_dir, weights_path)

if __name__ == "__main__":
    main()
