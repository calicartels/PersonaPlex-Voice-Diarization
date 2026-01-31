# Stage 4: Voice Comparison

## What We're Testing

This stage compares multiple different voices to see how well the encoder separates them in feature space.

## What It Proves

**Voice Comparison (`voice_comparison.py`):**
- **Separability:** Different voices should produce distinct embeddings
- **Similarity Patterns:** Which voices are similar? (accent, gender, age)
- **Clustering:** Do voices naturally cluster by characteristics?
- **Distance Metrics:** Inter-voice distances should be larger than intra-voice distances
- **Proves:** Whether the encoder can distinguish between different speakers effectively

## Why This Matters

For real-time voice diarization, we need:
1. **High inter-speaker distance:** Different voices should be far apart in feature space
2. **Low intra-speaker distance:** Same voice should be close together
3. **Clear boundaries:** There should be a gap between voice clusters

If all voices cluster together, the model can't separate them. If the separability ratio (inter/intra distance) is > 2.0, the model is working well.

## Files

- `voice_comparison.py` - Multi-voice comparison and analysis

## Usage

```bash
python voice_comparison.py
```

## Outputs

All outputs saved to `outputs/` directory.
