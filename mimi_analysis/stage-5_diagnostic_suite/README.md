# Stage 5: Diagnostic Suite

## What We're Testing

This is the comprehensive diagnostic suite that systematically tests different configurations to find the best settings for speaker discrimination.

## What It Proves

**Configuration Testing (`config_testing.py`):**
- Tests 6 extraction points (encoder, transformer layers 2/4/6/8, final)
- Tests 5 pooling methods (mean, max, stats, first/last frame)
- Tests 3 audio durations (1s, 3s, 5s)
- Tests 3 post-processing methods (L2 norm, whitening, PCA)
- **Proves:** Which configuration gives the best speaker discrimination

**Dataset Analysis (`dataset_analysis.py`):**
- **Acoustics vs Semantics:** Does the encoder capture voice (acoustics) or text (semantics)?
- **Intra-Speaker Consistency:** How consistent is the same voice across different sentences?
- **Inter-Speaker Separability:** How well are different voices separated?
- **Robustness:** How do augmentations (noise, reverb) affect embeddings?
- **Proves:** Whether the encoder prioritizes speaker identity over content

**Embedding Analysis (`embedding_analysis.py`):**
- Analyzes the 512-dimensional embedding space
- Finds discriminative channels
- Studies activation patterns and correlations
- **Proves:** Which parts of the embedding are most useful for speaker discrimination

## Why This Matters

This comprehensive testing tells us:
1. **Best extraction point:** Where in the model should we extract features?
2. **Best pooling method:** How should we aggregate time-varying features?
3. **Optimal duration:** How much audio do we need?
4. **Post-processing:** Does normalization or dimensionality reduction help?

These results guide us in building an effective voice diarization system.

## Files

- `config_testing.py` - Systematic configuration testing
- `dataset_analysis.py` - Dataset-wide analysis
- `embedding_analysis.py` - Deep embedding space analysis
- `run_analysis.py` - Run complete pipeline

## Usage

```bash
python run_analysis.py
```

## Outputs

- `outputs/diagnostic_results/` - Configuration testing results
- `outputs/analysis_results/` - Dataset and embedding analysis
