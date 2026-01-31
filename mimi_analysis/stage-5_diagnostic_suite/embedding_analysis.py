import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis, skew
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Synthetic_data"))
from config import OUTPUT_DIR

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import setup_style, COLOR_PALETTE, HEATMAP_CMAP

def activation_patterns(embeddings_matrix, output_dir):
    thresholds = [0.01, 0.05, 0.1]
    for threshold in thresholds:
        active_counts = (np.abs(embeddings_matrix) > threshold).sum(axis=0)
        sparsity = 1 - (active_counts / len(embeddings_matrix))
    
    plt.figure(figsize=(14, 8))
    sample_indices = np.random.choice(len(embeddings_matrix), min(20, len(embeddings_matrix)), replace=False)
    sample_embeddings = embeddings_matrix[sample_indices]
    setup_style()
    sns.heatmap(sample_embeddings.T, cmap=HEATMAP_CMAP, cbar=True, cbar_kws={'label': 'Activation Value'}, linewidths=0.5, linecolor='gray')
    plt.xlabel('Sample Index', fontsize=12, fontweight='bold')
    plt.ylabel('Channel Index (0-511)', fontsize=12, fontweight='bold')
    plt.title('Embedding Activation Patterns Across Channels', fontsize=13, fontweight='bold', pad=10)
    plt.savefig(output_dir / 'activation_heatmap.png')
    plt.close()

def channel_stats(embeddings_matrix, output_dir):
    channel_means = embeddings_matrix.mean(axis=0)
    channel_stds = embeddings_matrix.std(axis=0)
    channel_mins = embeddings_matrix.min(axis=0)
    channel_maxs = embeddings_matrix.max(axis=0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(channel_means, bins=50, color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Mean Value', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Distribution of Channel Means', fontsize=13, fontweight='bold', pad=10)
    axes[0, 1].hist(channel_stds, bins=50, color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Std Dev', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Distribution of Channel Std Devs', fontsize=13, fontweight='bold', pad=10)
    axes[1, 0].hist(channel_mins, bins=50, color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Min Value', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Distribution of Channel Minimums', fontsize=13, fontweight='bold', pad=10)
    axes[1, 1].hist(channel_maxs, bins=50, color='orange', alpha=0.7)
    axes[1, 1].set_xlabel('Max Value', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Distribution of Channel Maximums', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'channel_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()

def discriminative_channels(clean_data, embeddings_matrix, output_dir, top_k=50):
    by_voice = {}
    for sid, data in clean_data.items():
        by_voice.setdefault(data['voice_id'], []).append(data['embedding'])
    
    discriminative_scores = []
    for channel_idx in tqdm(range(512), desc="Analyzing channels"):
        intra_vars = []
        for voice_id, embeddings in by_voice.items():
            channel_values = [emb[channel_idx] for emb in embeddings]
            intra_vars.append(np.var(channel_values))
        intra_var = np.mean(intra_vars)
        voice_means = [np.mean([emb[channel_idx] for emb in embeddings]) for embeddings in by_voice.values()]
        inter_var = np.var(voice_means)
        score = inter_var / (intra_var + 1e-10)
        discriminative_scores.append((channel_idx, score, inter_var, intra_var))
    
    discriminative_scores.sort(key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(discriminative_scores, columns=['channel', 'score', 'inter_var', 'intra_var'])
    df.to_csv(output_dir / 'discriminative_channels.csv', index=False)
    top_channels = [c for c, _, _, _ in discriminative_scores[:top_k]]
    plt.figure(figsize=(14, 8))
    sample_indices = np.random.choice(len(embeddings_matrix), min(50, len(embeddings_matrix)), replace=False)
    sample_embeddings = embeddings_matrix[sample_indices][:, top_channels]
    sns.heatmap(sample_embeddings.T, cmap='coolwarm', center=0, cbar=True)
    plt.xlabel('Sample Index', fontsize=12, fontweight='bold')
    plt.ylabel(f'Top {top_k} Discriminative Channels', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_k} Most Discriminative Channels', fontsize=13, fontweight='bold', pad=10)
    plt.savefig(output_dir / 'top_discriminative_channels.png', dpi=150, bbox_inches='tight')
    plt.close()
    return top_channels

def value_distributions(embeddings_matrix, output_dir):
    all_values = embeddings_matrix.flatten()
    mean = np.mean(all_values)
    std = np.std(all_values)
    skewness = skew(all_values)
    kurt = kurtosis(all_values)
    percentiles = np.percentile(all_values, [5, 25, 50, 75, 95])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(all_values, bins=100, color='purple', alpha=0.7)
    axes[0].axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')
    axes[0].set_xlabel('Value', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribution of All Embedding Values', fontsize=13, fontweight='bold', pad=10)
    axes[0].legend(fontsize=11)
    from scipy import stats
    stats.probplot(all_values[:10000], dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_xlabel('Theoretical Quantiles', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Sample Quantiles', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'value_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

def correlation_structure(embeddings_matrix, output_dir):
    sample_size = min(100, embeddings_matrix.shape[1])
    sample_channels = np.random.choice(512, sample_size, replace=False)
    sampled_embeddings = embeddings_matrix[:, sample_channels]
    corr_matrix = np.corrcoef(sampled_embeddings.T)
    upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
    high_corr_threshold = 0.7
    high_corr_count = (np.abs(upper_tri) > high_corr_threshold).sum()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, vmin=-1, vmax=1, square=True, cbar=True)
    plt.title(f'Channel Correlation Matrix (sample of {sample_size} channels)', fontsize=13, fontweight='bold', pad=10)
    plt.xlabel('Channel Index', fontsize=12, fontweight='bold')
    plt.ylabel('Channel Index', fontsize=12, fontweight='bold')
    plt.savefig(output_dir / 'channel_correlations.png', dpi=150, bbox_inches='tight')
    plt.close()

def dimensionality(embeddings_matrix, output_dir):
    pca = PCA()
    pca.fit(embeddings_matrix)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    thresholds = [0.80, 0.90, 0.95, 0.99]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(range(1, 51), pca.explained_variance_ratio_[:50], 'o-')
    axes[0].set_xlabel('Component Index', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Explained Variance Ratio', fontsize=12, fontweight='bold')
    axes[0].set_title('Scree Plot (First 50 Components)', fontsize=13, fontweight='bold', pad=10)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(range(1, len(cumsum)+1), cumsum)
    for threshold in thresholds:
        n_dims = np.argmax(cumsum >= threshold) + 1
        axes[1].axhline(threshold, color='red', linestyle='--', alpha=0.5)
        axes[1].axvline(n_dims, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12, fontweight='bold')
    axes[1].set_title('Cumulative Variance Explained', fontsize=13, fontweight='bold', pad=10)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'dimensionality.png', dpi=150, bbox_inches='tight')
    plt.close()

def voice_pairs(clean_data, output_dir, num_pairs=5):
    by_voice = {}
    for sid, data in clean_data.items():
        by_voice.setdefault(data['voice_id'], []).append(data['embedding'])
    voice_means = {v: np.mean(embs, axis=0) for v, embs in by_voice.items()}
    voice_ids = list(voice_means.keys())
    np.random.seed(42)
    pairs = []
    for _ in range(min(num_pairs, len(voice_ids)//2)):
        v1, v2 = np.random.choice(voice_ids, 2, replace=False)
        pairs.append((v1, v2))
    for v1, v2 in pairs:
        emb1 = voice_means[v1]
        emb2 = voice_means[v2]
        diff = emb2 - emb1
        top_diffs = np.argsort(np.abs(diff))[-20:][::-1]
        plt.figure(figsize=(14, 5))
        plt.subplot(1, 2, 1)
        plt.plot(emb1, label=v1, alpha=0.7)
        plt.plot(emb2, label=v2, alpha=0.7)
        plt.xlabel('Channel Index', fontsize=12, fontweight='bold')
        plt.ylabel('Embedding Value', fontsize=12, fontweight='bold')
        plt.title(f'{v1} vs {v2} - Raw Embeddings', fontsize=13, fontweight='bold', pad=10)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.bar(range(len(diff)), diff, alpha=0.7)
        plt.xlabel('Channel Index', fontsize=12, fontweight='bold')
        plt.ylabel('Difference Value', fontsize=12, fontweight='bold')
        plt.title(f'{v1} - {v2} Difference', fontsize=13, fontweight='bold', pad=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'pair_comparison_{v1}_{v2}.png', dpi=150, bbox_inches='tight')
        plt.close()

def run_analysis(embeddings_cache, output_dir):
    output_dir.mkdir(exist_ok=True, parents=True)
    clean_data = {sid: data for sid, data in embeddings_cache.items() if data['augmentation'] == 'clean'}
    embeddings_matrix = np.array([data['embedding'] for data in clean_data.values()])
    activation_patterns(embeddings_matrix, output_dir)
    channel_stats(embeddings_matrix, output_dir)
    top_channels = discriminative_channels(clean_data, embeddings_matrix, output_dir, top_k=50)
    value_distributions(embeddings_matrix, output_dir)
    correlation_structure(embeddings_matrix, output_dir)
    dimensionality(embeddings_matrix, output_dir)
    voice_pairs(clean_data, output_dir, num_pairs=5)

def main():
    import pickle
    from dataset_analysis import run_analysis as dataset_run
    cache_path = Path(__file__).parent / "outputs" / "analysis_results" / "embeddings_cache.pkl"
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            embeddings_cache = pickle.load(f)
    else:
        dataset_run()
        with open(cache_path, 'rb') as f:
            embeddings_cache = pickle.load(f)
    run_analysis(embeddings_cache, Path(__file__).parent / "outputs" / "analysis_results" / "embedding_deep_dive")

if __name__ == "__main__":
    main()
