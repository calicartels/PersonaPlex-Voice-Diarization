"""Merge pre-conv embedding manifests into train/val splits for fine-tuning."""
import json
import random
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MANIFESTS

# Choice: same 3x oversample ratio for real data as original pipeline.
OVERSAMPLE = {"ami": 3, "voxconverse": 3}

random.seed(42)
all_entries = []

for mf in sorted(MANIFESTS.glob("*_preconv.json")):
    entries = [json.loads(l) for l in open(mf) if l.strip()]
    entries = [e for e in entries if e.get("preconv_emb_filepath") and e.get("labels_filepath")]

    tag = mf.stem.replace("_preconv", "")
    rate = 1
    for key, r in OVERSAMPLE.items():
        if key in tag:
            rate = r
    batch = entries * rate
    all_entries.extend(batch)
    print(f"{tag}: {len(entries)} x {rate} = {len(batch)}")

if not all_entries:
    print("No preconv manifests found. Falling back to existing _emb.json manifests...")
    for mf in sorted(MANIFESTS.glob("*_emb.json")):
        entries = [json.loads(l) for l in open(mf) if l.strip()]
        entries = [e for e in entries if e.get("emb_filepath") and e.get("labels_filepath")]
        tag = mf.stem.replace("_emb", "")
        rate = 1
        for key, r in OVERSAMPLE.items():
            if key in tag:
                rate = r
        batch = entries * rate
        all_entries.extend(batch)
        print(f"{tag}: {len(entries)} x {rate} = {len(batch)}")

random.shuffle(all_entries)

# Choice: 95/5 split same as original.
split = int(len(all_entries) * 0.95)
train, val = all_entries[:split], all_entries[split:]

for name, data in [("train_ft", train), ("val_ft", val)]:
    out = MANIFESTS / f"{name}.json"
    with open(out, "w") as f:
        for e in data:
            f.write(json.dumps(e) + "\n")

print(f"Train: {len(train)}, Val: {len(val)}")