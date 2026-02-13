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

compare_mode = "--compare" in sys.argv

mod_root = Path(__file__).resolve().parent.parent
train_dir = mod_root / "training"


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
                ders.append((p, l))
    return ders


def compute_der_from_pairs(pairs):
    total_fa, total_miss, total_conf, total_speech = 0, 0, 0, 0
    for p, l in pairs:
        fa, miss, conf, speech = compute_der_detailed(p, l)
        total_fa += fa
        total_miss += miss
        total_conf += conf
        total_speech += speech
    if total_speech == 0:
        return 100.0
    return 100 * (total_fa + total_miss + total_conf) / total_speech


def evaluate(model, loader, device, threshold, detailed=False):
    model.eval()
    results = []
    with torch.no_grad():
        for emb, labels, lengths in tqdm(loader, desc="Eval", unit="batch", leave=False):
            pred = torch.sigmoid(model(emb.to(device))).cpu().numpy()
            labels_np = labels.numpy()
            for i in range(pred.shape[0]):
                T = lengths[i].item()
                p = (pred[i, :T] > threshold).astype(float)
                l = labels_np[i, :T]
                for k in range(MAX_SPEAKERS):
                    p[:, k] = median_filter(p[:, k], size=5)
                if detailed:
                    n_spk = (l.mean(axis=0) > 0.01).sum()
                    fa, miss, conf, speech = compute_der_detailed(p, l)
                    if speech > 0:
                        results.append((n_spk, fa, miss, conf, speech))
                else:
                    fa, miss, conf, speech = compute_der_detailed(p, l)
                    if speech > 0:
                        results.append((fa + miss + conf) / speech)
    if detailed:
        return results
    return np.mean(results) * 100


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


device = "cuda" if torch.cuda.is_available() else "cpu"

if compare_mode:
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
            pairs = eval_model(model, val_loader, device, t)
            der = compute_der_from_pairs(pairs)
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
            pairs = eval_model(model, val_loader, device, t)
            der = compute_der_from_pairs(pairs)
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
else:
    val_loader = make_loader(MANIFESTS / "val_fusion.json", shuffle=False)
    input_dim = next(iter(val_loader))[0].shape[1]

    model = FusionSpeaker(input_dim=input_dim).to(device)
    ckpt = torch.load(CKPT / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    best_t, best_der = 0.5, 100.0
    for t in tqdm([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6], desc="Threshold sweep", unit="thresh"):
        results = evaluate(model, val_loader, device, t, detailed=True)
        der = print_results(results, t)
        if der < best_der:
            best_t, best_der = t, der

    print(f"\nBest: threshold={best_t:.2f}, DER={best_der:.2f}%")
