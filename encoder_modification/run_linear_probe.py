import os
import numpy as np
from data.pairs import load_pairs
from probe.train import train_probe
from probe.eval import eval_probe
import config

EMB_CACHE = os.path.join(config.CACHE_DIR, "baseline_embeddings.npz")

if not os.path.exists(EMB_CACHE):
    print(f"ERROR: no cached embeddings at {EMB_CACHE}")
    print("run run_baseline.py first to extract embeddings")
    exit(1)

print("loading cached embeddings")
data = np.load(EMB_CACHE)
embeddings = dict(zip(data["ids"], data["embeddings"]))
print(f"  loaded {len(embeddings)} embeddings")

pairs = load_pairs()

W, b = train_probe(embeddings)
eer = eval_probe(embeddings, pairs, W)

