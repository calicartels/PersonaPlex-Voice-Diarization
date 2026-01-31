import os
os.environ["NO_TORCH_COMPILE"] = "1"

import sys
from pathlib import Path
import torch
import numpy as np
import json
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "moshi"))
from moshi.models.loaders import get_mimi

def extract_embedding(mimi, audio):
    with torch.no_grad():
        features = mimi._encode_to_unquantized_latent(audio)
        embedding = features.mean(dim=2).squeeze(0).cpu().numpy()
    return embedding

def short_audio(mimi, sample_rate, durations=[0.1, 0.5, 1.0]):
    results = {}
    reference_audio = torch.randn(1, 1, int(3.0 * sample_rate))
    ref_embedding = extract_embedding(mimi, reference_audio)
    
    for duration in durations:
        samples = int(duration * sample_rate)
        short_audio = torch.randn(1, 1, samples)
        embedding = extract_embedding(mimi, short_audio)
        sim = 1 - cosine(embedding, ref_embedding)
        results[f'{duration}s'] = {'similarity': float(sim), 'shape': list(embedding.shape)}
    
    return results

def long_audio(mimi, sample_rate, durations=[5.0, 10.0, 30.0]):
    results = {}
    reference_audio = torch.randn(1, 1, int(3.0 * sample_rate))
    ref_embedding = extract_embedding(mimi, reference_audio)
    
    for duration in durations:
        samples = int(duration * sample_rate)
        long_audio = torch.randn(1, 1, samples)
        embedding = extract_embedding(mimi, long_audio)
        sim = 1 - cosine(embedding, ref_embedding)
        results[f'{duration}s'] = {'similarity': float(sim), 'shape': list(embedding.shape)}
    
    return results

def silence(mimi, sample_rate):
    silence = torch.zeros(1, 1, int(3.0 * sample_rate))
    noise = torch.randn(1, 1, int(3.0 * sample_rate))
    
    silence_emb = extract_embedding(mimi, silence)
    noise_emb = extract_embedding(mimi, noise)
    
    sim = 1 - cosine(silence_emb, noise_emb)
    
    return {
        'silence_embedding_norm': float(np.linalg.norm(silence_emb)),
        'noise_embedding_norm': float(np.linalg.norm(noise_emb)),
        'similarity': float(sim)
    }

def pure_noise(mimi, sample_rate, noise_types=['gaussian', 'uniform']):
    results = {}
    reference_audio = torch.randn(1, 1, int(3.0 * sample_rate))
    ref_emb = extract_embedding(mimi, reference_audio)
    
    for noise_type in noise_types:
        noise = torch.randn(1, 1, int(3.0 * sample_rate)) if noise_type == 'gaussian' else torch.rand(1, 1, int(3.0 * sample_rate)) * 2 - 1
        noise_emb = extract_embedding(mimi, noise)
        sim = 1 - cosine(noise_emb, ref_emb)
        results[noise_type] = {'similarity': float(sim), 'embedding_norm': float(np.linalg.norm(noise_emb))}
    
    return results

def variable_length(mimi, sample_rate):
    lengths = [0.5, 1.0, 2.0, 3.0, 5.0]
    embeddings = []
    
    for length in lengths:
        samples = int(length * sample_rate)
        audio = torch.randn(1, 1, samples)
        embedding = extract_embedding(mimi, audio)
        embeddings.append(embedding)
    
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            similarities.append(sim)
    
    mean_sim = np.mean(similarities)
    std_sim = np.std(similarities)
    
    return {'mean_similarity': float(mean_sim), 'std_similarity': float(std_sim), 'lengths_tested': lengths}

def save_results(results, output_dir):
    json_path = output_dir / "edge_test_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

def run_all(weights_path):
    sample_rate = 24000
    mimi = get_mimi(str(weights_path), device='cpu')
    mimi.eval()
    
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    results = {
        'short_audio': short_audio(mimi, sample_rate),
        'long_audio': long_audio(mimi, sample_rate),
        'silence': silence(mimi, sample_rate),
        'pure_noise': pure_noise(mimi, sample_rate),
        'variable_length': variable_length(mimi, sample_rate)
    }
    
    save_results(results, output_dir)

def main():
    weights_path = Path(__file__).parent.parent.parent / "weights" / "tokenizer-e351c8d8-checkpoint125.safetensors"
    run_all(weights_path)

if __name__ == "__main__":
    main()
