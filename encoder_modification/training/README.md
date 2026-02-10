# MimiSpeaker: Speaker Diarization Adapter on Frozen Mimi Codec Embeddings

Train a lightweight 7.3M-parameter Transformer adapter on frozen Mimi encoder embeddings for speaker diarization, using Sort Loss + PIL (Hybrid Loss) inspired by [Sortformer](https://arxiv.org/abs/2409.06656) and [PersonaPlex].

## Requirements

- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or better
- **RAM**: 32GB+ (377GB available on tested instance)
- **Disk**: ~150GB free
- **OS**: Ubuntu 22.04+ with CUDA 12.4+ drivers
- **Tested on**: vast.ai RTX 4090 instance, CUDA 13.0 driver (backward-compatible with cu124 wheels)
- **Budget**: ~$5–10 on vast.ai ($0.31/hr × ~6–8 hours total)

## Quick Start (Full Pipeline)

If you just want to run everything end-to-end, copy-paste the blocks below in order.

---

### Step 0: Environment Setup

```bash
# Create workspace
mkdir -p /workspace/diarization
cd /workspace/diarization

# Create isolated venv — do NOT use --system-site-packages
python -m venv .venv
source .venv/bin/activate

# Verify CUDA version (driver must be >= 12.4)
python -c "import subprocess; r=subprocess.run(['nvcc','--version'],capture_output=True,text=True); print(r.stdout)" 2>/dev/null || echo "no nvcc"
nvidia-smi | head -4
```

### Step 1: Install PyTorch (CUDA 12.4 wheels)

> **Critical**: Install torch FIRST, then NeMo, then force-reinstall torch.
> NeMo's pip install silently overwrites torch with a different CUDA build.

```bash
# Install torch ecosystem — all from same cu124 index
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

# Verify all match
python -c "import torch; print('torch:', torch.__version__, torch.version.cuda)"
python -c "import torchaudio; print('torchaudio:', torchaudio.__version__)"
python -c "import torchvision; print('torchvision:', torchvision.__version__)"
# Expected: torch 2.5.1, cuda 12.4, torchaudio 2.5.1, torchvision 0.20.1
```

### Step 2: Clone Repos

```bash
cd /workspace/diarization

# Main repo
git clone https://github.com/calicartels/PersonaPlex-Voice-Diarization.git
cd PersonaPlex-Voice-Diarization
git checkout colab
cd /workspace/diarization

# NeMo (for multi-speaker simulator)
git clone --depth 1 https://github.com/NVIDIA/NeMo.git
```

### Step 3: Install NeMo + Dependencies

```bash
# Install NeMo toolkit (will overwrite torch — we fix this next)
pip install nemo-toolkit[asr]==2.6.2

# CRITICAL: Force-reinstall our torch versions (NeMo overwrites them)
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --force-reinstall --no-deps

# Verify AGAIN
python -c "import torch; print('torch:', torch.__version__, torch.version.cuda)"
python -c "import torchaudio; print('torchaudio:', torchaudio.__version__)"

# Install remaining deps
pip install soundfile pyyaml matplotlib scipy huggingface_hub moshi wandb
```

### Step 4: HuggingFace Login (Required for Mimi Weights)

The PersonaPlex Mimi checkpoint is gated. You must:

1. **Accept the license** at [https://huggingface.co/nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) — click "Agree and access"
2. **Create a token** at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (needs "Read" permission)
3. **Login on the instance**:

```bash
huggingface-cli login
# Paste your token when prompted
```

### Step 5: Run the Pipeline

```bash
cd /workspace/diarization/PersonaPlex-Voice-Diarization/encoder_modification/training

# Step 5a: Download data (LibriSpeech train-clean-100 + AMI)
# ~28GB LibriSpeech, ~30GB AMI
python download.py

# Step 5b: Simulate multi-speaker sessions (~53 min)
# Generates 6000 sessions: 3000×2spk + 2000×4spk + 1000×3spk_long = ~583 hours
python simulate.py

# Step 5c: Convert RTTM annotations to frame-level labels (12.5Hz, 80ms frames)
python process_rttm.py

# Step 5d: Fix AMI manifest (download.py leaves audio_filepath empty)
python fix_ami_manifest.py

# Step 5e: Extract Mimi embeddings (~30 min)
# Processes audio through frozen Mimi encoder + encoder_transformer
# Long files (AMI meetings) are chunked at 60s to fit 24GB VRAM
python extract.py

# Step 5f: Merge all manifests into train/val splits
python merge.py

# Step 5g: Train the adapter (~20-30 min)
python train.py
# Select option 3 (offline) for wandb if not using cloud logging

# Step 5h: Evaluate
python eval.py
```

---

## Architecture

```
Audio (24kHz) → Mimi Encoder (frozen, 512-dim, 25Hz)
                    ↓
              Stride-2 Downsample (→ 12.5Hz, matches label frame rate)
                    ↓
              Linear Projection (512 → 384)
                    ↓
              Sinusoidal Positional Encoding
                    ↓
              4× Transformer Encoder Layers (384-dim, 6 heads, 1536 FFN)
                    ↓
              Linear Head (384 → 4 speakers)
                    ↓
              Sigmoid → Per-frame, per-speaker activity probabilities
```

**Total trainable parameters**: 7,296,388 (~7.3M)

## Training Details

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| D_MODEL | 384 | Balances capacity vs speed for 4-layer model |
| N_HEADS | 6 | 384/6 = 64 dim per head (standard) |
| FF_DIM | 1536 | 4× D_MODEL (standard ratio) |
| N_LAYERS | 4 | Minimal depth; Sortformer uses 18 but on raw audio |
| DROPOUT | 0.1 | 0.5 was too aggressive — predictions stuck at ~0.5 |
| LR | 1e-4 | With cosine annealing to 1e-6 |
| WARMUP | 2500 steps | Linear warmup |
| BATCH | 4 | Fits 90s segments on 24GB VRAM |
| SEG_DURATION | 90s | Matches Sortformer training segment length |
| ALPHA | 0.5 | Equal weight Sort Loss + PIL (hybrid loss) |
| MAX_EPOCHS | 40 | Early stopping patience=10 |
| MAX_SPEAKERS | 4 | Matches Sortformer's design |

### Loss Function (Hybrid Loss)

```
L = α × L_sort + (1 - α) × L_PIL
```

- **Sort Loss**: BCE with labels sorted by speaker arrival time (speaker who speaks first = channel 0)
- **PIL (Permutation Invariant Loss)**: BCE minimized over all 4!=24 speaker permutations
- **Numerically stable**: Uses `binary_cross_entropy_with_logits` (log-sum-exp trick) on raw logits

## Data Pipeline

### Simulated Data (NeMo Speech Data Simulator)

| Config | Sessions | Speakers | Duration/Session | Total |
|--------|----------|----------|------------------|-------|
| 2spk | 3000 | 2 | ~5 min | ~250h |
| 4spk | 2000 | 4 | ~5 min | ~167h |
| 3spk_long | 1000 | 3 | ~10 min | ~167h |

Source audio: LibriSpeech train-clean-100 (251 speakers, 28,539 utterances)

### Real Data

| Dataset | Sessions | Notes |
|---------|----------|-------|
| AMI IHM | 170 | Real meetings, multi-channel mixed to mono |
| VoxConverse | 0 | Download URL changed; skipped (non-critical) |

### Frame Rate Mismatch Fix

Mimi encoder outputs at **25 Hz** (stride 960, 24000/960=25). Labels are at **12.5 Hz** (80ms frames matching Sortformer). The dataloader applies stride-2 downsampling (`emb[:, ::2]`) to align them.

## Known Issues & Fixes Applied

1. **CUDA version mismatch**: NeMo overwrites torch with different CUDA build → force-reinstall after NeMo
2. **NeMo simulator missing config fields**: YAML needs all 6 sections (rir_generation, background_noise, etc.)
3. **LibriSpeech manifest**: Needs speaker_id, words, and alignments fields for NeMo
4. **AMI empty audio paths**: download.py leaves `audio_filepath: ""` → fix_ami_manifest.py patches it
5. **Mimi encoder_transformer returns list**: Unwrap with `emb[0] if isinstance(emb, list) else emb`
6. **AMI OOM on full meetings**: 85-minute meetings don't fit in 24GB VRAM → chunk at 60s
7. **25Hz vs 12.5Hz frame mismatch**: Stride-2 downsample in dataloader
8. **Dropout 0.5 too aggressive**: Predictions stuck at ~0.5 for 7.3M model → reduced to 0.1
9. **Sigmoid + BCE numerical instability**: Changed to raw logits + `binary_cross_entropy_with_logits`

## Results

### Training

- Best validation loss: ~0.415 (BCE with logits)
- Training completed 20 epochs in ~20 minutes on RTX 4090

### Evaluation (Validation Set)

| Metric | Value |
|--------|-------|
| **Mean DER (FA+Miss only)** | **4.6%** |
| Median DER (FA+Miss) | 2.2% |
| Samples with DER < 30% | 99% |
| Samples with DER > 80% | 0% |
| Best threshold | 0.40 |

> **Note**: The model excels at detecting *when* speech occurs (4.6% FA+Miss) but speaker channel assignment (confusion) adds significant error to the full DER metric. This is expected for a first-pass adapter and validates that Mimi codec embeddings carry diarization-relevant information.

## File Structure

```
encoder_modification/training/
├── config.py           # All hyperparameters and paths
├── download.py         # Download LibriSpeech + AMI + VoxConverse
├── simulate.py         # NeMo multi-speaker session simulator
├── process_rttm.py     # RTTM → frame-level labels (12.5Hz)
├── fix_ami_manifest.py # Patch AMI audio paths in manifest
├── extract.py          # Mimi embedding extraction (chunked for long files)
├── merge.py            # Combine manifests into train/val splits
├── model.py            # MimiSpeaker model + hybrid loss
├── load.py             # Dataset/DataLoader with frame alignment
├── train.py            # Training loop with early stopping
├── eval.py             # DER evaluation with threshold sweep
└── run.sh              # Full pipeline script
```

## References

- **Sortformer**: Park et al., "Seamless Integration of Speaker Diarization and ASR by Bridging Timestamps and Tokens", 2024. [arXiv:2409.06656](https://arxiv.org/abs/2409.06656)
- **Streaming Sortformer**: Medennikov et al., "Speaker Cache-Based Online Speaker Diarization with Arrival-Time Ordering", 2025. [arXiv:2507.18446](https://arxiv.org/abs/2507.18446)
- **PersonaPlex**: Roy et al., "Voice and Role Control for Full Duplex Conversational Speech Models", ICASSP 2026.
- **Moshi/Mimi**: Défossez et al., "A Speech-Text Foundation Model for Real-Time Dialogue", 2024.

