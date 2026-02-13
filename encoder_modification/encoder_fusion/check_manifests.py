import json
from pathlib import Path
import numpy as np
from config import MANIFESTS


def check_manifest(path):
    if not path.exists():
        print(f"  Missing: {path}")
        return
    entries = [json.loads(l) for l in open(path) if l.strip()]
    n = len(entries)
    emb_ok = 0
    lbl_ok = 0
    shape_ok = 0
    for e in entries:
        emb = e.get("emb_filepath", "")
        lbl = e.get("labels_filepath", "")
        if emb and Path(emb).exists():
            emb_ok += 1
        if lbl and Path(lbl).exists():
            lbl_ok += 1
        if emb and lbl and Path(emb).exists() and Path(lbl).exists():
            emb_arr = np.load(emb)
            lbl_arr = np.load(lbl)
            D, T_emb = emb_arr.shape
            T_lbl, K = lbl_arr.shape
            T_min = min(T_emb // 2, T_lbl)
            if T_min > 0:
                shape_ok += 1
    print(f"  {path.name}: {n} entries, emb_ok={emb_ok}, lbl_ok={lbl_ok}, shape_ok={shape_ok}")


print("Fusion manifests:")
for mf in sorted(MANIFESTS.glob("*_fusion*.json")):
    check_manifest(mf)
print("\nTrain/Val:")
check_manifest(MANIFESTS / "train_fusion.json")
check_manifest(MANIFESTS / "val_fusion.json")
