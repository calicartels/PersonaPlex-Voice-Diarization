# Adapter (VoxCeleb Proof-of-Concept)

Lightweight 4-layer Transformer adapter on frozen Mimi, trained on VoxCeleb simulated mixtures. Proof-of-concept before scaling to AMI + NeMo data.

## Method

- Mimi frozen; adapter: Linear(512→384) → 4× Transformer → Head(4 speakers)
- Hybrid loss: Sort Loss + PIL
- Data: VoxCeleb single-speaker utterances mixed into 2-speaker sessions
- Generates mixtures on the fly (no NeMo)

## Architecture

```
Frozen Mimi (512-dim) → Linear → Relative pos enc → 4× Transformer → Sigmoid(4)
```

Sigmoid (not softmax) allows overlap. Hybrid loss (Sort + PIL) handles permutation.

## Run

```bash
cd encoder_modification && python run_adapter.py
```

Requires GPU. Uses HuggingFace VoxCeleb.

## Relation to training/

`training/` uses AMI + NeMo-simulated LibriSpeech for the full pipeline. This adapter is a smaller VoxCeleb-only proof-of-concept.
