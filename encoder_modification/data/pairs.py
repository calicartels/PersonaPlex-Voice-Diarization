import os
import urllib.request
import config
from data.load import normalize_pairs_path


def load_pairs():
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    local_path = os.path.join(config.CACHE_DIR, "veri_test2.txt")

    if not os.path.exists(local_path):
        print(f"downloading pairs from {config.PAIRS_URL}")
        urllib.request.urlretrieve(config.PAIRS_URL, local_path)

    pairs = []
    with open(local_path) as f:
        for line in f:
            parts = line.strip().split()
            # format: "1 id10270/5r0dWxy17C8/00001.wav id10270/5r0dWxy17C8/00002.wav"
            label = int(parts[0])
            id1 = normalize_pairs_path(parts[1])
            id2 = normalize_pairs_path(parts[2])
            pairs.append((label, id1, id2))

    print(f"loaded {len(pairs)} pairs ({sum(p[0] for p in pairs)} positive)")
    return pairs