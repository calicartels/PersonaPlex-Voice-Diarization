# adapter

Lightweight speaker diarization adapter on frozen Mimi embeddings.

## Architecture

```
Frozen Mimi (512-dim, 12.5Hz) .... what the audio sounds like
Relative positional encoding ...... where each frame sits in time
Transformer encoder (×N layers) .. which frames share a speaker
Linear (512 → 4) ................. confidence per speaker slot
Sigmoid (×4 independent) ......... overlap: multiple speakers can be active
```

## Why each layer

**Positional encoding.** Self-attention without it is permutation invariant —
it cannot learn "speaker 0 arrived before speaker 1." Relative encoding
(not absolute) generalizes across window positions for streaming later.

**Transformer, not MLP/LSTM.** MLP sees one frame, can't compare voices
across time. LSTM forgets early speakers. Self-attention sees everything,
finds which frames share a voice.

**Sigmoid, not softmax.** Softmax forces outputs to sum to 1 — kills
overlap detection. Sigmoid lets speaker 0 = 0.95 and speaker 2 = 0.88
at the same frame.

**Hybrid loss (Sort + PIL).** Sort Loss assigns speakers by arrival time,
eliminating permutation ambiguity at inference. PIL acts as safety net
when arrival order is ambiguous. α=0.5 per Sortformer paper.

## Files

- `model.py` — SortformerAdapter module
- `loss.py` — sort loss, permutation invariant loss, hybrid loss
- `data.py` — simulated 2-speaker mixtures from single-speaker audio
- `train.py` — training loop
- `eval.py` — diarization error rate

## AOSC cache

Not implemented yet. The cache is an inference-time mechanism for
long-form audio — it doesn't change the model architecture. Phase 1
validates the adapter on 90-second fixed segments. Phase 2 adds the
cache for streaming.