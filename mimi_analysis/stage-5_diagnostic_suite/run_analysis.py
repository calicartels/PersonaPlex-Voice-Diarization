import sys
from pathlib import Path
import torch
import pickle

from dataset_analysis import run_analysis as dataset_run
from embedding_analysis import run_analysis as embedding_run

def main():
    validate_dataset()
    results = dataset_run()
    cache_path = Path(__file__).parent / "outputs" / "analysis_results" / "embeddings_cache.pkl"
    with open(cache_path, 'rb') as f:
        embeddings_cache = pickle.load(f)
    embedding_run(embeddings_cache, Path(__file__).parent / "outputs" / "analysis_results" / "embedding_deep_dive")

if __name__ == "__main__":
    main()
