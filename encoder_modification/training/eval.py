import numpy as np
import torch
from scipy.ndimage import median_filter
from tqdm import tqdm
from config import MANIFESTS, CKPT, MAX_SPEAKERS
from model import MimiSpeaker
from load import make_loader


def compute_der(pred, true):
    pred_any = pred.any(axis=1)
    true_any = true.any(axis=1)
    speech = true_any.sum()
    if speech == 0:
        return 0.0

    fa = (pred_any & ~true_any).sum()
    miss = (~pred_any & true_any).sum()

    # Choice: frame-level confusion (no collar). Simple and fast.
    # Alternative: pyannote.metrics for collar-aware DER, but adds dependency.
    both = pred_any & true_any
    confusion = 0
    for t in range(len(pred)):
        if both[t]:
            confusion += np.abs(pred[t] - true[t]).sum() / 2
    return (fa + miss + confusion) / speech


def evaluate(model, loader, device, threshold):
    model.eval()
    ders = []
    with torch.no_grad():
        for emb, labels, lengths in tqdm(loader, desc="Eval", unit="batch", leave=False):
            pred = torch.sigmoid(model(emb.to(device))).cpu().numpy()
            labels = labels.numpy()
            for i in range(pred.shape[0]):
                T = lengths[i].item()
                p = (pred[i, :T] > threshold).astype(float)
                l = labels[i, :T]
                # Choice: median filter k=5 removes short flickering.
                # Sortformer uses similar onset/offset post-processing.
                for k in range(MAX_SPEAKERS):
                    p[:, k] = median_filter(p[:, k], size=5)
                ders.append(compute_der(p, l))
    return np.mean(ders) * 100


device = "cuda" if torch.cuda.is_available() else "cpu"
val_loader = make_loader(MANIFESTS / "val.json", shuffle=False)
input_dim = next(iter(val_loader))[0].shape[1]

model = MimiSpeaker(input_dim=input_dim).to(device)
ckpt = torch.load(CKPT / "best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
# Choice: sweep thresholds 0.3-0.6 to find optimal.
# Sortformer tunes post-processing per dataset on dev splits.
best_t, best_der = 0.5, 100.0
for t in tqdm([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6], desc="Threshold sweep", unit="thresh"):
    der = evaluate(model, val_loader, device, t)
    print(f"  threshold={t:.2f}  DER={der:.2f}%")
    if der < best_der:
        best_t, best_der = t, der

print(f"\nBest: threshold={best_t:.2f}, DER={best_der:.2f}%")