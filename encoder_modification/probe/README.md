# Exp 1.5: Linear Probe

Is speaker identity linearly separable with a learned transform?

## Method

- Train a single linear layer (512 → N speakers) on VoxCeleb1
- Evaluate EER in the projected space

## Result

**EER = 20.48%** (down from 35.01% baseline)

| Metric | Value |
|--------|-------|
| Mean positive-pair score | 0.697 |
| Mean negative-pair score | 0.278 |
| Score gap | 0.419 |
| Training accuracy | 79.6% |

## Why it happened

The linear layer learned directions that separate speakers. Speaker identity is not axis-aligned in the raw space, but it's linearly accessible. A deeper adapter should do even better.

## Architecture

```
Audio → Mimi (frozen) → 512-d embeddings
                            │
                            ▼
                    ┌───────────────┐
                    │ Linear(512→N) │  N = num speakers
                    └───────────────┘
                            │
                            ▼  cosine sim in projected space
                    ┌───────────────┐
                    │ EER 20.48%   │
                    └───────────────┘
```

## Run

```bash
cd encoder_modification && python run_linear_probe.py
```

## Files

- `train.py` — linear classifier on cached embeddings
- `eval.py` — EER in projected space
