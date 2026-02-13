# Exp 3: Encoder Fusion (Mimi + TitaNet)

Concatenate Mimi and TitaNet embeddings, train adapter with Sort Loss + PIL. Mimi gives VAD; TitaNet gives speaker identity.

## Method

- Mimi (frozen) → 512-d embeddings
- TitaNet (frozen) → 192-d embeddings
- Concat(512+192) → same adapter architecture as Exp 2
- Same data pipeline (download, simulate, process_rttm)

## Result

**DER 60.93% vs Mimi-only 65.03%** — −4.1% improvement. Confusion still high (~57%) from segment-level interpolation and overlap.

## Why it happened

TitaNet injects speaker identity. DER dropped. Remaining confusion from frame-level predictions and overlapping speech. But the direction is right: hybrid encoder works.

## Architecture

```
                    ┌─────────────────────┐
                    │  Mimi (frozen)      │  512-d
                    └─────────────────────┘
Audio ──┬───────────────────────┐
        │                       │
        │   ┌─────────────────────┐
        └───│  TitaNet (frozen)   │  192-d
            └─────────────────────┘
                        │
                        ▼
                ┌───────────────┐
                │ Concat 704-d  │  512 + 192
                └───────────────┘
                        │
                        ▼  same adapter as Exp 2
                ┌───────────────┐
                │ DER 60.93%    │  −4.1% vs Mimi-only
                └───────────────┘
```

## Run

```bash
cd encoder_modification && bash encoder_fusion/run.sh
```

Or step by step: download → simulate → process_rttm (from training), then extract → merge → train → eval.

## Eval modes

```bash
python eval.py           # Fusion only, detailed FA/Miss/Conf breakdown
python eval.py --compare # Mimi vs Fusion comparison
```

## Scripts

- `eval.py` — DER, FA/Miss/Conf breakdown; `--compare` for Mimi vs Fusion
- `check_manifests.py` — sanity check manifests
- `upload.py` — HF upload. Set HF_REPO, HF_TOKEN.
