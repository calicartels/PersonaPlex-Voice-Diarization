import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from data import load_voxceleb
from baseline.extract import load_mimi
from adapter.train import train
import config

if not torch.cuda.is_available():
    print("ERROR: CUDA not available. This script requires a GPU.")
    print("Mimi inference during data generation needs GPU acceleration.")
    exit(1)

print(f"using device: {config.DEVICE}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

ds = load_voxceleb()
mimi = load_mimi()
model = train(mimi, ds)

