# Mimi Diarization Experiments: Full Exploration Summary

Speaker diarization experiments on Mimi neural audio codec embeddings. We systematically probed whether Mimi's internal representations carry enough speaker identity information for diarization, and whether architectural modifications could unlock it.

## Motivation

PersonaPlex uses Mimi as its audio codec. Mimi was trained for reconstruction, not speaker discrimination. The question: can we add a lightweight diarization adapter on top of frozen Mimi, or do we need a separate speaker encoder?

---

## Experiment Timeline

### Step 0: Baseline Speaker Signal (Raw Cosine Similarity)

**Goal:** Measure intrinsic speaker signal in Mimi embeddings—no training.

**Method:**
- Extract embeddings from frozen Mimi (512-dim, pre-quantization) for VoxCeleb1 test set
- Mean-pool each utterance → single vector
- Compute cosine similarity on 37,611 verification pairs
- Report EER (Equal Error Rate)

**Result: EER = 35.01%**

| Metric | Value |
|--------|-------|
| Mean positive-pair score | 0.572 |
| Mean negative-pair score | 0.447 |
| Score gap | 0.125 |

**Conclusion:** Some speaker information exists (positive pairs score higher), but distributions overlap heavily. Raw cosine similarity cannot separate speakers. Mimi encodes identity but it's entangled with phonemes, prosody, and channel noise.

---

### Step 0.5: Linear Probe

**Goal:** Is speaker identity linearly separable with a learned transform?

**Method:**
- Train a single linear layer (512 → N speakers) on VoxCeleb1
- Evaluate EER in the projected space

**Result: EER = 20.48%**

| Metric | Value |
|--------|-------|
| Mean positive-pair score | 0.697 |
| Mean negative-pair score | 0.278 |
| Score gap | 0.419 |
| Training accuracy | 79.6% |

**Conclusion:** Speaker info is linearly accessible. EER dropped 14.5 points. A deeper adapter (Transformer) should extract it even better. Worth trying Step 1 (adapter).

---

### Step 1: Adapter on Frozen Mimi

**Goal:** Train a 4-layer Transformer adapter on Mimi embeddings for diarization.

**Method:**
- Mimi frozen; adapter: Linear(512→384) → 4× Transformer → Head(4 speakers)
- Hybrid loss: Sort Loss + PIL (permutation-invariant)
- Data: LibriSpeech (simulated multi-speaker) + AMI meetings
- ~7.3M trainable parameters

**Results:**
- **VAD (FA + Miss):** 4.6% — excellent speech vs silence detection
- **Speaker confusion:** ~33% for 2-speaker — poor channel assignment

**Conclusion:** Mimi gives strong *when* (VAD) but weak *who* (speaker ID). The model correctly detects speech boundaries but frequently assigns the wrong speaker channel. Speaker signal in Mimi is too weak for reliable diarization.

---

### Step 1.5: Fine-Tuning Mimi Top Layers (Unfreeze Experiment)

**Goal:** Unfreeze top 3 layers of Mimi's `encoder_transformer` + adapter. Maybe Mimi has speaker info deeper in the network that fine-tuning can unlock.

**Method:**
- Extract pre-conv (pre-transformer) embeddings
- Unfreeze top 3 of 8 `encoder_transformer` layers
- Dual learning rates: lower for Mimi, higher for adapter
- Same data, same loss

**Result: Confusion *increased* — 33% → 46% for 2-speaker**

**Conclusion:** Fine-tuning made speaker discrimination worse, not better. Mimi's architecture was not trained for speaker discrimination; modifying its internals does not create that capability. The codec objective (reconstruction) likely discards or suppresses speaker-discriminative information that a dedicated encoder would preserve.

---

## Final Conclusion

| Approach | EER / DER | Verdict |
|----------|-----------|---------|
| Raw Mimi (cosine) | 35% EER | Too weak |
| Linear probe on Mimi | 20% EER | Some signal, not sufficient |
| Adapter (frozen Mimi) | 4.6% FA+Miss, ~33% confusion | Good VAD, poor speaker ID |
| Fine-tuned Mimi + adapter | ~46% confusion | Worse |

**Mimi, as a codec, lacks sufficient speaker-discriminative information.** No amount of adapter depth, fine-tuning, or architectural tweaks on the Mimi side will fix this.

**Recommended path: Hybrid encoder**
- **Mimi (frozen):** acoustic features, VAD
- **TitaNet or ECAPA-TDNN (frozen):** speaker embeddings
- **Adapter:** fuse both via concatenation → predict diarization

No retraining of encoders. Same training data. Adapter learns to combine Mimi's VAD strength with TitaNet's speaker identity.

---

## File References

| Experiment | Entry Point | Location |
|------------|-------------|----------|
| Step 0 | `run_baseline.py` | `encoder_modification/` |
| Step 0.5 | `run_linear_probe.py` | `encoder_modification/` |
| Step 1 | `train.py` | `encoder_modification/training/` |
| Step 1.5 | `run_ft.sh` | `encoder_modification/training/unfreeze/` |
