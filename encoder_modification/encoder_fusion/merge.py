import json
import random
from config import MANIFESTS, OVERSAMPLE

random.seed(42)
all_entries = []
for mf in sorted(MANIFESTS.glob("*_fusion_emb.json")):
    entries = [json.loads(l) for l in open(mf) if l.strip()]
    entries = [e for e in entries if e.get("emb_filepath") and e.get("labels_filepath")]
    tag = mf.stem.replace("_fusion_emb", "")
    rate = 1
    for key, r in OVERSAMPLE.items():
        if key in tag:
            rate = r
            break
    batch = entries * rate
    all_entries.extend(batch)
    print(f"{tag}: {len(entries)} x {rate} = {len(batch)}")
random.shuffle(all_entries)
split = int(len(all_entries) * 0.95)
train, val = all_entries[:split], all_entries[split:]
for name, data in [("train_fusion", train), ("val_fusion", val)]:
    out = MANIFESTS / f"{name}.json"
    with open(out, "w") as f:
        for e in data:
            f.write(json.dumps(e) + "\n")
print(f"Train: {len(train)}, Val: {len(val)}")
