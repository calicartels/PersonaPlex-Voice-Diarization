# Stage 3: Single Voice Exploration

## What We're Testing

This stage focuses on understanding how a single voice behaves across different sentences and over time. We analyze consistency and variation within one speaker.

## What It Proves

**Single Voice Analysis (`single_voice.py`):**
- **Temporal Consistency:** Same voice saying different things should produce similar embeddings
- **Sentence Variation:** How much do embeddings vary when the same voice says different sentences?
- **Voice Stability:** Does the encoder capture consistent voice characteristics regardless of content?
- **Proves:** Whether the encoder maintains speaker identity across different utterances

## Why This Matters

For voice diarization to work, the encoder must:
1. Keep the same voice consistent across different sentences (high intra-speaker similarity)
2. Allow some variation for natural speech differences
3. Capture voice characteristics that persist regardless of what's being said

If a single voice's embeddings vary wildly, the model won't be useful for speaker separation.

## Files

- `single_voice.py` - Single voice analysis tools

## Usage

```python
from single_voice import SingleVoiceAnalyzer

analyzer = SingleVoiceAnalyzer(weights_path, device)
analyzer.analyze_temporal_consistency(audio_files)
analyzer.analyze_sentence_variation(audio_files, sentence_ids)
```

## Outputs

All outputs saved to `outputs/` directory.
