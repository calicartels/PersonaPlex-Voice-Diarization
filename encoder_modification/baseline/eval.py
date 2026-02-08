import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from tqdm import tqdm


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def compute_scores(embeddings, pairs):
    # Choice: compute cosine similarity for each pair.
    # Alternative was PLDA or LDA-reduced cosine, but those require
    # a separate training set. Raw cosine is the right baseline since
    # we're measuring what's in the embeddings without any learning.
    scores = []
    labels = []
    missing = 0

    for label, id1, id2 in tqdm(pairs, desc="Computing similarities", unit="pair"):
        if id1 not in embeddings or id2 not in embeddings:
            missing += 1
            continue
        s = cosine_similarity(embeddings[id1], embeddings[id2])
        scores.append(s)
        labels.append(label)

    if missing > 0:
        print(f"warning: {missing}/{len(pairs)} pairs skipped (missing utterances)")

    return np.array(scores), np.array(labels)


def compute_eer(embeddings, pairs):
    scores, labels = compute_scores(embeddings, pairs)

    # Choice: scipy brentq on interpolated ROC for EER.
    # This is the standard method used in VoxSRC challenges.
    # Alternative was manual threshold sweep but brentq is exact.
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    print(f"EER: {eer * 100:.2f}%")
    print(f"  pairs evaluated: {len(scores)}")
    print(f"  positive pairs: {labels.sum()}")
    print(f"  score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  mean pos score: {scores[labels == 1].mean():.4f}")
    print(f"  mean neg score: {scores[labels == 0].mean():.4f}")

    return eer