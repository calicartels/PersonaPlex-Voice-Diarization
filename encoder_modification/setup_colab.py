# Colab setup helper script
# Run this in a Colab cell after cloning the repo

import os
from pathlib import Path

# Default Colab paths
COLAB_ROOT = "/content/PersonaPlex-Voice-Diarization"
COLAB_CHECKPOINT = "/content/weights/tokenizer-e351c8d8-checkpoint125.safetensors"

# Set environment variables
os.environ["ENCODER_MODIFICATION_ROOT"] = COLAB_ROOT
os.environ["MIMI_CHECKPOINT"] = COLAB_CHECKPOINT

# Verify
import config
print(f"✓ PROJECT_ROOT: {config.PROJECT_ROOT}")
print(f"✓ PERSONAPLEX_REPO: {config.PERSONAPLEX_REPO}")
print(f"✓ MIMI_CHECKPOINT: {config.MIMI_CHECKPOINT}")
print(f"✓ Checkpoint exists: {os.path.exists(config.MIMI_CHECKPOINT)}")
print(f"✓ Moshi folder exists: {os.path.exists(config.PERSONAPLEX_REPO)}")
print(f"✓ Device: {config.DEVICE}")

