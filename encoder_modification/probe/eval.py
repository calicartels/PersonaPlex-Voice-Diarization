import numpy as np
from baseline.eval import compute_eer


def project_embeddings(embeddings, W):
    # Choice: use W without bias. The bias shifts logits for classification
    # but doesn't help cosine similarity (which is scale/shift invariant
    # after normalization). Including bias would just add noise.
    projected = {}
    for uid, emb in embeddings.items():
        projected[uid] = emb @ W.T
    return projected


def eval_probe(embeddings, pairs, W):
    print("projecting embeddings through learned weights")
    proj = project_embeddings(embeddings, W)
    print(f"  original dim: {next(iter(embeddings.values())).shape[0]}")
    print(f"  projected dim: {next(iter(proj.values())).shape[0]}")
    print("computing EER on projected embeddings")
    return compute_eer(proj, pairs)

