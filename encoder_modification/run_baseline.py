import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data import load_voxceleb, load_pairs
from baseline import extract_all, compute_eer


ds = load_voxceleb()
pairs = load_pairs()
embeddings = extract_all(ds)
eer = compute_eer(embeddings, pairs)

# Verdict
if eer < 0.15:
    print(f"\nresult: strong speaker signal (EER={eer*100:.1f}%).")
elif eer < 0.20:
    print(f"\nresult: moderate speaker signal (EER={eer*100:.1f}%).")
elif eer < 0.30:
    print(f"\nresult: weak speaker signal (EER={eer*100:.1f}%).")
else:
    print(f"\nresult: poor speaker signal (EER={eer*100:.1f}%).")