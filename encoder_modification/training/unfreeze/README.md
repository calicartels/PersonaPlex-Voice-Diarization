# Exp 2.5: Unfreeze (Fine-Tune Mimi Top Layers)

Unfreeze top 3 layers of Mimi's `encoder_transformer` + adapter. Maybe Mimi has speaker info deeper in the network that fine-tuning can unlock.

## Method

- Extract pre-conv (pre-transformer) embeddings
- Unfreeze top 3 of 8 `encoder_transformer` layers
- Dual learning rates: lower for Mimi, higher for adapter
- Same data, same loss as Exp 2

## Result

**Confusion increased 33% → 46%** — Fine-tuning made speaker discrimination worse.

## Why it happened

Mimi's architecture was optimized for reconstruction. The codec objective doesn't preserve speaker-discriminative information—it likely suppresses it. Modifying Mimi's internals doesn't create capability that wasn't trained in.

## Architecture (modification to Exp 2)

```
Audio
    │
    ▼
┌─────────────────────┐
│  Mimi Encoder       │
│  - Conv: frozen     │
│  - Trans layers     │
│    0-4: frozen      │
│    5-7: TRAINABLE   │  ← only change vs Exp 2
└─────────────────────┘
    │
    ▼  same adapter as Exp 2
┌─────────────────────┐
│  Confusion 46%      │  (worse than 33%)
└─────────────────────┘
```

## Run

```bash
cd encoder_modification/training/unfreeze && bash run_ft.sh
```

Requires Exp 2 pipeline (download, simulate, process_rttm) to have run first.
