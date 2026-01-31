# Stage 6: Edge Case Testing

## What We're Testing

This stage tests how the encoder handles edge cases: very short audio, very long audio, silence, pure noise, and variable-length inputs.

## What It Proves

**Edge Case Tests (`edge_tests.py`):**
- **Short Audio:** Can the model work with < 1 second of audio?
- **Long Audio:** Does the model handle > 10 seconds efficiently?
- **Silence:** What happens with silent audio?
- **Pure Noise:** How does the model respond to non-speech audio?
- **Variable Length:** Are embeddings consistent across different audio lengths?
- **Proves:** The model's robustness and limitations in real-world scenarios

## Why This Matters

Real-world voice diarization faces:
- Short utterances (quick interjections)
- Long conversations (need efficient processing)
- Background noise and silence
- Non-speech audio

If the model fails on these cases, it won't work in production. These tests reveal the model's practical boundaries.

## Files

- `edge_tests.py` - Edge case testing suite

## Usage

```bash
python edge_tests.py
```

## Outputs

- `edge_test_results.json` - Test results
