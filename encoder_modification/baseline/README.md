# Exp 1: Baseline (Raw Cosine EER)

Measure intrinsic speaker signal in Mimi embeddings—no training.

## Method

- Extract embeddings from frozen Mimi (512-dim, pre-quantization) for VoxCeleb1 test set
- Mean-pool each utterance → single vector
- Compute cosine similarity on 37,611 verification pairs
- Report EER (Equal Error Rate)

## Result

**EER = 35.01%**

| Metric | Value |
|--------|-------|
| Mean positive-pair score | 0.572 |
| Mean negative-pair score | 0.447 |
| Score gap | 0.125 |

## Why it happened

Speaker info exists (positive pairs score higher) but distributions overlap heavily. Mimi encodes identity but it's entangled with phonemes, prosody, and channel noise. Raw cosine on mean-pooled vectors cannot untangle it.

## Architecture

```
Audio (24kHz)
    │
    ▼
┌─────────────────────┐
│  Mimi Encoder       │  frozen, 512-d @ 12.5Hz
│  (SEANet + 8×Trans) │
└─────────────────────┘
    │
    ▼  mean pool over time
┌─────────────────────┐
│  512-d vector       │  one per utterance
└─────────────────────┘
    │
    ▼  cosine similarity on pairs
┌─────────────────────┐
│  EER 35.01%         │
└─────────────────────┘
```

## Run

```bash
cd encoder_modification && python run_baseline.py
```
