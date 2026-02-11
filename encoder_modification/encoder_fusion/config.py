from pathlib import Path
import os

ROOT = Path(os.getenv("DIARIZATION_ROOT", "/workspace/diarization"))
PROJECT_ROOT = Path(os.getenv("ENCODER_MODIFICATION_ROOT", Path(__file__).parent.parent.parent))
PERSONAPLEX_REPO = os.getenv("PERSONAPLEX_REPO", str(PROJECT_ROOT / "moshi"))
MIMI_CHECKPOINT = os.getenv("MIMI_CHECKPOINT", str(PROJECT_ROOT / "weights" / "tokenizer-e351c8d8-checkpoint125.safetensors"))

RAW = ROOT / "raw"
SIM = ROOT / "simulated"
EMB = ROOT / "embeddings_fusion"
LABELS = ROOT / "labels"
MANIFESTS = ROOT / "manifests"
CKPT = ROOT / "checkpoints_fusion"

SAMPLE_RATE = 24000
TITANET_SR = 16000
FRAME_HZ = 12.5
FRAME_S = 0.08
MAX_SPEAKERS = 4
D_MIMI = 512
D_TITANET = 192
INPUT_DIM = D_MIMI + D_TITANET

TITANET_WIN_S = 1.5
TITANET_HOP_S = 0.2

D_MODEL = 384
N_HEADS = 6
FF_DIM = 1536
N_LAYERS = 4
DROPOUT = 0.1
LR = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-3
WARMUP = 2500
BATCH = 4
SEG_S = 90
ALPHA = 0.5
MAX_EPOCHS = 40
VAL_EVERY = 500

OVERSAMPLE = {"ami": 3, "voxconverse": 3}
