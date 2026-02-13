import os
from pathlib import Path
import torch

# -- Paths (edit these) --

# Get the root of this project (PersonaPlex-Voice-Diarization)
# Can be overridden via ENCODER_MODIFICATION_ROOT env var (useful for Colab)
PROJECT_ROOT = Path(os.getenv("ENCODER_MODIFICATION_ROOT", Path(__file__).parent.parent))

# Local moshi folder is already in this repo.
# We add this to sys.path so we can import moshi.models.loaders.get_mimi
# Can be overridden via PERSONAPLEX_REPO env var (useful for Colab)
PERSONAPLEX_REPO = os.getenv("PERSONAPLEX_REPO", str(PROJECT_ROOT / "moshi"))

# Mimi checkpoint file. Already downloaded in weights/ folder.
# File: tokenizer-e351c8d8-checkpoint125.safetensors
# Source: https://huggingface.co/nvidia/personaplex-7b-v1
# Choice: using the PersonaPlex checkpoint rather than Kyutai's original
# because we want to test the exact encoder that PersonaPlex fine-tuned on.
# Can be overridden via MIMI_CHECKPOINT env var (useful for Colab)
MIMI_CHECKPOINT = os.getenv("MIMI_CHECKPOINT", str(PROJECT_ROOT / "weights" / "tokenizer-e351c8d8-checkpoint125.safetensors"))

# Where to cache extracted embeddings and downloaded data.
CACHE_DIR = os.path.expanduser("~/.cache/encoder_modification")

# -- Dataset --

# Choice: Codec-SUPERB mirror has VoxCeleb1 test split (4874 utterances, 1.3GB)
# pre-packaged as parquet with audio. No access form needed.
# Alternative was ProgramComputer/voxceleb (full dataset, 30GB+) but overkill for Step 0.
HF_DATASET = "Codec-SUPERB/Voxceleb1_test_original"
HF_SPLIT = "test"

# Official VoxCeleb1-O verification pairs (37,720 pairs).
# Choice: veri_test2.txt (cleaned version) over veri_test.txt (original).
# The cleaned version removes pairs where one utterance is corrupted.
# Metadata files are freely downloadable, no password.
PAIRS_URL = "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt"

# -- Audio --

# Mimi expects 24kHz mono. VoxCeleb1 is 16kHz.
# Choice: resample at load time using torchaudio.functional.resample.
# Alternative was sox/ffmpeg preprocessing but adds a dependency and disk usage.
MIMI_SAMPLE_RATE = 24000
VOXCELEB_SAMPLE_RATE = 16000

# -- Extraction --

# Choice: process one utterance at a time (batch_size=1).
# VoxCeleb clips vary from 4s to 69s, so batching requires padding.
# With 4874 clips this runs in ~10-15 min on a single GPU, fast enough.
# Auto-detect device: use CUDA if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- Embedding --

# Choice: mean pooling over time frames.
# Alternative was mean+std concatenation (doubles dim to 1024, slightly better
# for trained systems). But for a raw cosine-similarity baseline, mean pool
# is standard and sufficient. We're measuring a go/no-go signal, not optimizing.
POOL_MODE = "mean"

# Mimi embedding dim, confirmed from code inspection.
EMBED_DIM = 512