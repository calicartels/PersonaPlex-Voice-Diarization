import os
os.environ["NO_TORCH_COMPILE"] = "1"

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys
from sklearn.decomposition import PCA
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "moshi"))
from moshi.models.loaders import get_mimi

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_style import setup_style, COLOR_PALETTE, HEATMAP_CMAP

def load_audio(audio_path, sample_rate=24000):
    data, sr = sf.read(audio_path)
    waveform = torch.from_numpy(data if data.ndim == 1 else data.T).float()
    waveform = waveform.unsqueeze(0) if data.ndim == 1 else waveform
    resampler = torchaudio.transforms.Resample(sr, sample_rate)
    waveform = resampler(waveform)
    waveform = waveform.mean(dim=0, keepdim=True) if waveform.shape[0] > 1 else waveform
    return waveform.unsqueeze(0)

def encoder_step(mimi, audio):
    with torch.no_grad():
        features = mimi.encoder(audio)
    return features

def transformer_step(mimi, features):
    with torch.no_grad():
        refined_features, = mimi.encoder_transformer(features)
    return refined_features

def quantizer_step(mimi, features):
    with torch.no_grad():
        codes = mimi.quantizer.encode(features)
    return codes

def pipeline_viz(mimi, audio, output_dir, sample_rate=24000):
    setup_style()
    output_dir.mkdir(exist_ok=True, parents=True)
    
    features = encoder_step(mimi, audio)
    refined_features = transformer_step(mimi, features)
    codes = quantizer_step(mimi, refined_features)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    audio_np = audio[0, 0].cpu().numpy()
    time_audio = np.arange(len(audio_np)) / sample_rate
    axes[0].plot(time_audio, audio_np, linewidth=0.8, color=COLOR_PALETTE[0], alpha=0.8)
    axes[0].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=12, fontweight='bold')
    axes[0].set_title('Raw Audio Waveform', fontsize=13, fontweight='bold', pad=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    features_np = features[0, :32, :].cpu().numpy()
    im1 = axes[1].imshow(features_np, aspect='auto', cmap=HEATMAP_CMAP, origin='lower')
    axes[1].set_xlabel('Time Frames', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Feature Channel Index', fontsize=12, fontweight='bold')
    axes[1].set_title('Encoder Features (First 32 of 512 Channels)', fontsize=13, fontweight='bold', pad=10)
    plt.colorbar(im1, ax=axes[1], label='Feature Value')
    
    refined_np = refined_features[0, :32, :].cpu().numpy()
    im2 = axes[2].imshow(refined_np, aspect='auto', cmap=HEATMAP_CMAP, origin='lower')
    axes[2].set_xlabel('Time Frames', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Feature Channel Index', fontsize=12, fontweight='bold')
    axes[2].set_title('Transformer-Refined Features', fontsize=13, fontweight='bold', pad=10)
    plt.colorbar(im2, ax=axes[2], label='Feature Value')
    
    codes_np = codes[0].cpu().numpy()
    im3 = axes[3].imshow(codes_np, aspect='auto', cmap='tab20', origin='lower', vmin=0, vmax=mimi.cardinality-1)
    axes[3].set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Codebook Index', fontsize=12, fontweight='bold')
    axes[3].set_title('Quantized Codes', fontsize=13, fontweight='bold', pad=10)
    plt.colorbar(im3, ax=axes[3], label='Code Value')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'encoding_pipeline.png')
    plt.close()
    
    return features, refined_features, codes

def projection_3d(features, output_dir, method='pca', sample_frames=100):
    setup_style()
    output_dir.mkdir(exist_ok=True, parents=True)
    
    features_np = features[0].cpu().numpy().T
    features_np = features_np[np.linspace(0, features_np.shape[0] - 1, sample_frames, dtype=int)] if features_np.shape[0] > sample_frames else features_np
    
    if method.lower() == 'pca':
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(features_np)
        explained_var = pca.explained_variance_ratio_
    else:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, features_np.shape[0] - 1))
        features_3d = tsne.fit_transform(features_np)
        explained_var = None
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = np.linspace(0, 1, len(features_3d))
    scatter = ax.scatter(features_3d[:, 0], features_3d[:, 1], features_3d[:, 2], c=colors, cmap='viridis', s=30, alpha=0.7, edgecolors='black', linewidths=0.3)
    
    if explained_var is not None:
        ax.set_xlabel(f'Feature Embedding Dimension 1 ({explained_var[0]:.1%} Variance)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'Feature Embedding Dimension 2 ({explained_var[1]:.1%} Variance)', fontsize=11, fontweight='bold')
        ax.set_zlabel(f'Feature Embedding Dimension 3 ({explained_var[2]:.1%} Variance)', fontsize=11, fontweight='bold')
    else:
        ax.set_xlabel('Feature Embedding Dimension 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature Embedding Dimension 2', fontsize=11, fontweight='bold')
        ax.set_zlabel('Feature Embedding Dimension 3', fontsize=11, fontweight='bold')
    
    ax.set_title(f'Encoder Features - {method.upper()} 3D Projection', fontsize=13, fontweight='bold', pad=15)
    plt.colorbar(scatter, ax=ax, label='Normalized Time Position')
    
    plt.savefig(output_dir / f'encoder_features_3d_{method}.png')
    plt.close()
    
    return features_3d

def main():
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent.parent
    weights_path = base_dir / "weights" / "tokenizer-e351c8d8-checkpoint125.safetensors"
    output_dir = script_dir / "outputs"
    audio_path = base_dir / "assets" / "test" / "input_service.wav"
    
    mimi = get_mimi(str(weights_path), device='cpu')
    mimi.eval()
    
    audio = load_audio(str(audio_path))
    
    features, refined_features, codes = pipeline_viz(mimi, audio, output_dir)
    projection_3d(features, output_dir, method='pca', sample_frames=200)
    projection_3d(refined_features, output_dir, method='pca', sample_frames=200)

if __name__ == "__main__":
    main()
