from pathlib import Path
import os
import sys

# Choice: allow override via env var for Colab/vast.ai
ROOT = Path(os.getenv("DIARIZATION_ROOT", "/workspace/diarization"))

# Get project root (PersonaPlex-Voice-Diarization)
# Can be overridden via ENCODER_MODIFICATION_ROOT env var
PROJECT_ROOT = Path(os.getenv("ENCODER_MODIFICATION_ROOT", Path(__file__).parent.parent.parent))

# Local moshi folder for Mimi model loading
PERSONAPLEX_REPO = os.getenv("PERSONAPLEX_REPO", str(PROJECT_ROOT / "moshi"))

# Mimi checkpoint path
MIMI_CHECKPOINT = os.getenv("MIMI_CHECKPOINT", str(PROJECT_ROOT / "weights" / "tokenizer-e351c8d8-checkpoint125.safetensors"))
RAW = ROOT / "raw"
SIM = ROOT / "simulated"
EMB = ROOT / "embeddings"
LABELS = ROOT / "labels"
MANIFESTS = ROOT / "manifests"
CKPT = ROOT / "checkpoints"
NEMO = ROOT / "NeMo"

# Choice: 24kHz matches Mimi's native sample rate.
# Alternative: 16kHz (speech standard), but requires resampling at inference.
SAMPLE_RATE = 24000
# NeMo simulator outputs 16kHz, we resample at extraction time.
SIM_SR = 16000

# Choice: 12.5 Hz frame rate = Mimi stride 1920 at 24kHz = 80ms.
# Matches Sortformer's NEST encoder frame rate exactly.
FRAME_HZ = 12.5
FRAME_S = 0.08
MAX_SPEAKERS = 4

# Choice: 3 configs, ~500h total. Covers speaker count + overlap variation.
# Alternative: 7 configs (Sortformer's 5000h plan), overkill for 8M param adapter.
SIM_CONFIGS = {
    "2spk":      dict(n_sessions=3000, n_speakers=2, overlap=0.12, length=300),
    "4spk":      dict(n_sessions=2000, n_speakers=4, overlap=0.25, length=300),
    "3spk_long": dict(n_sessions=1000, n_speakers=3, overlap=0.18, length=600),
}

# Choice: 384 hidden, 4 layers = ~7.3M params. Fits 500h data budget.
# Alternative: 512 hidden = ~12M, risks overfitting on this data size.
D_MODEL = 384
N_HEADS = 6
FF_DIM = 1536
N_LAYERS = 4
# Choice: 0.5 dropout matches Sortformer's Transformer layers.
# Alternative: 0.3 converges faster but higher overfit risk.
DROPOUT = 0.5

# Choice: AdamW with Sortformer's exact schedule (inverse sqrt + warmup).
# Alternative: cosine annealing (PersonaPlex uses it), marginal difference.
LR = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-3
WARMUP = 2500
BATCH = 4
SEG_S = 90
# Choice: alpha=0.5 for hybrid loss (equal Sort + PIL), Sortformer's default.
# Alternative: alpha=0.7 biases toward Sort Loss for stronger arrival-time ordering.
ALPHA = 0.5
MAX_EPOCHS = 20
VAL_EVERY = 500