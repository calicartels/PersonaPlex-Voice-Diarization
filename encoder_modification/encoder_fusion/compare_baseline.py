import sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import median_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import MANIFESTS, CKPT, MAX_SPEAKERS
from model import FusionSpeaker
from load import make_loader

mod_root = Path(__file__).resolve().parent.parent
train_dir = mod_root / "training"

import importlib.util
train_config = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("train_config", train_dir / "config.py")
)
train_config.__loader__.exec_module(train_config)
TRAIN_MF = train_config.MANIFESTS
TRAIN_CKPT = train_config.CKPT

_saved_config = sys.modules.get("config")
sys.modules["config"] = train_config
train_model = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("train_model", train_dir / "model.py")
)
train_model.__loader__.exec_module(train_model)
TrainMimiSpeaker = train_model.MimiSpeaker

train_load = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("train_load", train_dir / "load.py")
)
train_load.__loader__.exec_module(train_load)
if _saved_config is not None:
    sys.modules["config"] = _saved_config
train_make_loader = train_load.make_loader


def compute_der(pred, true):
    pred_any = pred.any(axis=1)
    true_any = true.any(axis=1)
    speech = true_any.sum()
    if speech == 0:
        return 0.0
    fa = (pred_any & ~true_any).sum()
    miss = (~pred_any & true_any).sum()
    both = pred_any & true_any
    confusion = 0
    for t in range(len(pred)):
        if both[t]:
            confusion += np.abs(pred[t] - true[t]).sum() / 2
    return (fa + miss + confusion) / speech


def eval_model(model, loader, device, threshold):
    model.eval()
    ders = []
    with torch.no_grad():
        for emb, labels, lengths in loader:
            pred = torch.sigmoid(model(emb.to(device))).cpu().numpy()
            labels = labels.numpy()
            for i in range(pred.shape[0]):
                T = lengths[i].item()
                p = (pred[i, :T] > threshold).astype(float)
                l = labels[i, :T]
                for k in range(MAX_SPEAKERS):
                    p[:, k] = median_filter(p[:, k], size=5)
                ders.append(compute_der(p, l))
    return np.mean(ders) * 100


device = "cuda" if torch.cuda.is_available() else "cpu"

print("=== Mimi-only (baseline) ===")
mimi_der = None
if (TRAIN_CKPT / "best.pt").exists():
    val_loader = train_make_loader(TRAIN_MF / "val.json", shuffle=False)
    input_dim = next(iter(val_loader))[0].shape[1]
    model = TrainMimiSpeaker(input_dim=input_dim).to(device)
    ckpt = torch.load(TRAIN_CKPT / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_t, best_der = 0.5, 100.0
    for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        der = eval_model(model, val_loader, device, t)
        if der < best_der:
            best_t, best_der = t, der
    mimi_der = best_der
    print(f"  Best: threshold={best_t:.2f}, DER={best_der:.2f}%")
else:
    print("  Skip: training/checkpoints/best.pt not found")

print("\n=== Fusion (Mimi + TitaNet) ===")
fusion_der = None
if (CKPT / "best.pt").exists():
    val_loader = make_loader(MANIFESTS / "val_fusion.json", shuffle=False)
    input_dim = next(iter(val_loader))[0].shape[1]
    model = FusionSpeaker(input_dim=input_dim).to(device)
    ckpt = torch.load(CKPT / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_t, best_der = 0.5, 100.0
    for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        der = eval_model(model, val_loader, device, t)
        if der < best_der:
            best_t, best_der = t, der
    fusion_der = best_der
    print(f"  Best: threshold={best_t:.2f}, DER={best_der:.2f}%")
else:
    print("  Skip: checkpoints_fusion/best.pt not found")

print("\n=== Comparison ===")
if mimi_der is not None and fusion_der is not None:
    delta = fusion_der - mimi_der
    print(f"  Mimi-only:  DER={mimi_der:.2f}%")
    print(f"  Fusion:     DER={fusion_der:.2f}%")
    print(f"  Delta:      {delta:+.2f}% ( Fusion - Mimi )")
elif mimi_der is not None:
    print(f"  Mimi-only:  DER={mimi_der:.2f}%")
elif fusion_der is not None:
    print(f"  Fusion:     DER={fusion_der:.2f}%")
