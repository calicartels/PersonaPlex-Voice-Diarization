# encoder_modification

Speaker diarization adapter experiments on Mimi neural audio codec.

The goal is to determine whether Mimi's internal representations carry
enough speaker identity information to support a lightweight diarization
adapter, eliminating the need for a separate speaker encoder in the
PersonaPlex pipeline.

## Background

Mimi is a neural audio codec — it was trained to compress and reconstruct
audio, not to identify speakers. But to reconstruct a voice accurately,
the encoder must implicitly learn something about who is speaking. The
question is whether that implicit knowledge is accessible enough for
downstream speaker tasks, or whether it's too entangled with phoneme
content, prosody, and noise to be useful.

We probe this by tapping Mimi's pre-quantization embeddings — the 512-dim
vectors produced after the SEANet convolutional encoder and 8-layer
Transformer, but before the residual vector quantizer splits them into
codebooks. This is the richest representation Mimi produces.

## What is EER

Equal Error Rate (EER) is the standard metric for speaker verification.
Given a set of audio pairs labeled "same speaker" or "different speaker,"
you compute a similarity score for each pair and sweep a threshold.
At each threshold, some same-speaker pairs fall below it (misses) and
some different-speaker pairs fall above it (false alarms). EER is the
point where the miss rate equals the false alarm rate.

Lower is better. State-of-the-art speaker verification models achieve
EER around 1-2% on VoxCeleb1. A random system would score 50%. For our
purposes, sub-20% means the embeddings carry enough speaker signal to
train an adapter on, and sub-15% means the adapter's job is straightforward.

## Step 0: Baseline Speaker Signal (Original Plan)

Measure how much speaker identity exists in Mimi's pre-quantization
embeddings. No training. Extract embeddings from frozen Mimi, compute
cosine similarity on VoxCeleb1 test pairs, report EER.

**Plan:**
- Extract embeddings from frozen Mimi for all 4874 utterances in the
  VoxCeleb1 test set
- Mean-pool each utterance's frame-level embeddings (512-dim, 12.5 Hz)
  into a single vector
- Compute cosine similarity on the 37,611 official VoxCeleb1-O verification pairs

**Expected outcomes:**
- If EER < 20%, the embeddings carry speaker signal and the adapter is viable
- If EER ~ 50%, embeddings are speaker-agnostic and we need a different approach

## Step 0: Results

**Result: EER = 35.01%**

    pairs evaluated: 37611
    positive pairs:  18802
    score range:     [-0.41, 0.998]
    mean pos score:  0.572
    mean neg score:  0.447

**Analysis:**

The gap between positive and negative mean scores (0.125) confirms that
speaker information exists in these embeddings — same-speaker pairs
consistently score higher. But 35% EER means the distributions overlap
heavily. Raw cosine similarity on mean-pooled embeddings cannot cleanly
separate speakers.

This is expected. Mimi's training objective was audio reconstruction, not
speaker discrimination. Speaker identity is encoded but entangled with
everything else — phonemes, prosody, channel noise. A linear operation
(cosine similarity) on a crude summary (mean pool) is the wrong tool
to extract it.

## Step 0.5: Linear Probe (Original Plan)

The next test asks: is speaker identity linearly separable in these
embeddings? We train a single linear layer (512 -> N speakers) with
cross-entropy loss on VoxCeleb1 test speakers, then project test
embeddings through the learned weight matrix and measure EER with
cosine similarity in the projected space.

**Expected outcomes:**
- If EER drops to 15-20%, the information is present but not axis-aligned,
  and the planned 3-layer Transformer adapter will extract it
- If EER stays above 30%, the signal is genuinely too weak and we move to
  Experiment B (dedicated parallel speaker encoder)

## Step 0.5: Results

**Result: EER = 20.48%**

    pairs evaluated: 37611
    positive pairs:  18802
    score range:     [-0.7709, 0.9996]
    mean pos score:  0.6971
    mean neg score:  0.2778
    training accuracy: 79.6%

**Analysis:**

EER dropped from 35.01% to 20.48% — a 14.5 percentage point improvement.
The mean score gap widened from 0.125 to 0.419, showing much cleaner separation
in the projected space. Training accuracy reached 79.6%, confirming that
speaker identity is linearly separable in these embeddings.

This is the key finding: speaker information exists and is accessible via
a learned linear transform. The 512-dim space contains directions that
separate speakers, but they're not axis-aligned (raw cosine similarity
fails). A 3-layer Transformer adapter should extract this signal even
more effectively, potentially pushing EER below 15%.

## Structure

    encoder_modification/
    ├── config.py             — paths, hyperparams, all knobs in one place
    ├── requirements.txt
    ├── run_baseline.py       — entry point for Step 0 (raw cosine EER)
    ├── run_linear_probe.py   — entry point for Step 0.5 (linear probe EER)
    ├── data/                 — dataset loading and pair parsing
    ├── baseline/             — embedding extraction and EER computation
    └── probe/                — linear probe training and evaluation

## Usage

### Local Setup

    pip install -r requirements.txt

    # Paths are auto-detected from repo structure
    # Override via environment variables if needed:
    # export ENCODER_MODIFICATION_ROOT=/path/to/PersonaPlex-Voice-Diarization
    # export MIMI_CHECKPOINT=/path/to/tokenizer-e351c8d8-checkpoint125.safetensors

    # Step 0: raw cosine baseline
    python run_baseline.py

    # Step 0.5: linear probe
    python run_linear_probe.py

    # Step 1: adapter training (requires GPU)
    python run_adapter.py

### Colab Setup

See [COLAB_SETUP.md](COLAB_SETUP.md) for complete Colab instructions.

Quick start:
1. Clone repo: `!git clone https://github.com/YOUR_USERNAME/PersonaPlex-Voice-Diarization.git`
2. Download checkpoint: `!wget -O /content/weights/tokenizer-*.safetensors https://huggingface.co/nvidia/personaplex-7b-v1/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors`
3. Set env vars: `os.environ["ENCODER_MODIFICATION_ROOT"] = "/content/PersonaPlex-Voice-Diarization"`
4. Run scripts as above

## Dependencies

Mimi model loaded from local PersonaPlex fork (not pip-installed).
VoxCeleb1 test set streamed from HuggingFace (Codec-SUPERB mirror).
VoxCeleb1 dev set for linear probe training via ProgramComputer/voxceleb
or local copy.
