import torch
import torch.nn.functional as F
from itertools import permutations


def bce(pred, target):
    return F.binary_cross_entropy(pred, target, reduction="mean")


def arrival_time(labels):
    K = labels.size(1)
    first_frame = torch.full((K,), labels.size(0) + 1, device=labels.device)

    for k in range(K):
        active = (labels[:, k] > 0.5).nonzero(as_tuple=True)[0]
        if len(active) > 0:
            first_frame[k] = active[0]

    return torch.argsort(first_frame)


def sort_labels(labels):
    B, T, K = labels.shape
    sorted_labels = torch.zeros_like(labels)

    for b in range(B):
        order = arrival_time(labels[b])
        sorted_labels[b] = labels[b, :, order]

    return sorted_labels


def sort_loss(pred, labels):
    # Forces the model to assign speaker 0 to whoever speaks first,
    # speaker 1 to whoever speaks second, etc. This eliminates
    # permutation ambiguity at inference — no Hungarian algorithm needed.
    sorted_gt = sort_labels(labels)
    return bce(pred, sorted_gt)


def pil_loss(pred, labels):
    # Choice: brute-force over all permutations. For K=4 this is 24
    # permutations — trivial. Alternative was Hungarian algorithm (O(K^3))
    # but with K=4 the overhead of setting it up exceeds brute force.
    B, T, K = pred.shape
    perms = list(permutations(range(K)))
    best_loss = None

    for perm in perms:
        perm_labels = labels[:, :, list(perm)]
        loss = bce(pred, perm_labels)
        if best_loss is None or loss < best_loss:
            best_loss = loss

    return best_loss


def hybrid_loss(pred, labels, alpha=0.5):
    # Choice: alpha=0.5, equal weight. Sortformer paper found this
    # outperforms either loss alone. Sort Loss teaches arrival-time
    # ordering, PIL catches cases where sorting is ambiguous.
    # Alternative was alpha=0.7 (more sort) or 0.3 (more PIL).
    # 0.5 is the paper's recommendation and our starting point.
    ls = sort_loss(pred, labels)
    lp = pil_loss(pred, labels)
    return alpha * ls + (1 - alpha) * lp
