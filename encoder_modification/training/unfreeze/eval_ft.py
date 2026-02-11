"""Evaluate fine-tuned MimiSpeaker. Per-speaker-count DER + confusion breakdown."""
import sys
import numpy as np
import torch
from scipy.ndimage import median_filter
from config import MANIFESTS, CKPT, MAX_SPEAKERS, PERSONAPLEX_REPO, MIMI_CHECKPOINT
from model_ft import MimiSpeakerFT
from load_ft import make_loader_ft


def compute_der_detailed(pred, true):
    pred_any = pred.any(axis=1)
    true_any = true.any(axis=1)
    speech = true_any.sum()
    if speech == 0:
        return 0, 0, 0, 0
    fa = (pred_any & ~true_any).sum()
    miss = (~pred_any & true_any).sum()
    both = pred_any & true_any
    confusion = 0
    for t in range(len(pred)):
        if both[t]:
            confusion += np.abs(pred[t] - true[t]).sum() / 2
    return fa, miss, confusion, speech


def evaluate(model, loader, device, threshold):
    model.eval()
    results = []
    with torch.no_grad():
        for emb, labels, lengths in loader:
            pred = torch.sigmoid(model(emb.to(device))).cpu().numpy()
            labels_np = labels.numpy()
            for i in range(pred.shape[0]):
                T = lengths[i].item()
                p = (pred[i, :T] > threshold).astype(float)
                l = labels_np[i, :T]
                for k in range(MAX_SPEAKERS):
                    p[:, k] = median_filter(p[:, k], size=5)
                n_spk = (l.mean(axis=0) > 0.01).sum()
                fa, miss, conf, speech = compute_der_detailed(p, l)
                if speech > 0:
                    results.append((n_spk, fa, miss, conf, speech))
    return results


def print_results(results, threshold):
    total_fa = sum(r[1] for r in results)
    total_miss = sum(r[2] for r in results)
    total_conf = sum(r[3] for r in results)
    total_speech = sum(r[4] for r in results)
    if total_speech == 0:
        return 100.0

    fa_pct = 100 * total_fa / total_speech
    miss_pct = 100 * total_miss / total_speech
    conf_pct = 100 * total_conf / total_speech
    der = fa_pct + miss_pct + conf_pct

    print(f"  t={threshold:.2f}  DER={der:.2f}%  FA={fa_pct:.1f}% Miss={miss_pct:.1f}% Conf={conf_pct:.1f}%")

    for n_spk in sorted(set(r[0] for r in results)):
        subset = [r for r in results if r[0] == n_spk]
        s_fa = sum(r[1] for r in subset)
        s_miss = sum(r[2] for r in subset)
        s_conf = sum(r[3] for r in subset)
        s_speech = sum(r[4] for r in subset)
        if s_speech > 0:
            s_der = 100 * (s_fa + s_miss + s_conf) / s_speech
            s_conf_pct = 100 * s_conf / s_speech
            print(f"    {n_spk}-spk ({len(subset)} samples): DER={s_der:.1f}% Conf={s_conf_pct:.1f}%")
    return der


# ---- Main ----
device = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, PERSONAPLEX_REPO)
from moshi.models.loaders import get_mimi
mimi = get_mimi(MIMI_CHECKPOINT, device=device)
mimi.eval()

val_mf = MANIFESTS / "val_ft.json"
if not val_mf.exists():
    val_mf = MANIFESTS / "val.json"
val_loader = make_loader_ft(val_mf, shuffle=False)

ckpt_path = CKPT / "best_ft.pt"
if not ckpt_path.exists():
    print(f"No checkpoint at {ckpt_path}")
    sys.exit(1)

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
n_unfreeze = ckpt.get("n_unfreeze", ckpt.get("config", {}).get("n_unfreeze", 3))

model = MimiSpeakerFT(
    mimi_transformer=mimi.encoder_transformer,
    n_unfreeze=n_unfreeze,
    input_dim=512,
).to(device)
model.load_state_dict(ckpt["model_state_dict"])

best_t, best_der = 0.5, 100.0
for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    results = evaluate(model, val_loader, device, t)
    der = print_results(results, t)
    if der < best_der:
        best_t, best_der = t, der

print(f"\nBest: threshold={best_t:.2f}, DER={best_der:.2f}%")