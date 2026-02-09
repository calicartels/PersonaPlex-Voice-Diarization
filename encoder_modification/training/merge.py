import json
import random
from config import MANIFESTS

# Choice: 3x oversample real data to reach ~30% effective weight.
# Matches Sortformer's 70/30 sim/real ratio.
# Alternative: sampling weights in DataLoader, but explicit copies are simpler.
OVERSAMPLE = {"ami": 3, "voxconverse": 3}

random.seed(42)
all_entries = []

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

# Choice: 95/5 train/val split. Maximizes training data,
# 5% still gives enough samples for stable DER estimates.
split = int(len(all_entries) * 0.95)
train, val = all_entries[:split], all_entries[split:]

for name, data in [("train", train), ("val", val)]:
    out = MANIFESTS / f"{name}.json"
    with open(out, "w") as f:
        for e in data:
            f.write(json.dumps(e) + "\n")

print(f"Train: {len(train)}, Val: {len(val)}")