# Mimi Encoder Analysis

Comprehensive analysis suite for the Mimi encoder model, organized by analysis stage.

## Problem Statement

**Problem:** Mimi encoder captures semantics (text content) more than acoustics (speaker identity).

**Evidence:**
- Same text, different speakers: 0.88 similarity (HIGH - bad)
- Same speaker, different text: 0.78 similarity (LOWER - bad)
- **Inverted:** Should be opposite

**Root Cause:** Trained for TTS (text → speech), not speaker verification

## Success Metrics

### Primary Metrics

1. **Acoustics Score** = (Intra-voice similarity) - (Inter-voice similarity)
   - **Target:** > 0.10 (positive = captures voice)
   - **Current best:** 0.13 (with PCA)
   - **Baseline:** -0.09 (negative = captures text)

2. **Separability Ratio** = (Inter-speaker distance) / (Intra-speaker distance)
   - **Target:** > 2.0 (different speakers 2x farther apart than same speaker samples)
   - **Current best:** 0.55
   - **Gold standard (WavLM):** ~3.5-4.0

3. **Intra-speaker Consistency**
   - **Target:** > 0.80 similarity
   - **Current:** 0.17
   - **Meaning:** Same speaker samples should cluster together

### Threshold Derivation

**Acoustics Score > 0.10:**
- Derived from speaker verification literature
- Empirically: models with score < 0 unusable
- Score 0.05-0.15: weak but trainable
- Score > 0.3: good baseline

**Separability Ratio > 2.0:**
- Standard in speaker diarization research
- Ratio < 1.0 = broken (inter < intra)
- Ratio 1.0-2.0 = poor separation
- Ratio 2.0-3.5 = acceptable
- Ratio > 3.5 = good (WavLM-level)

**Intra-speaker > 0.80:**
- Speaker verification benchmark
- EER (Equal Error Rate) depends on this
- < 0.70 = unreliable
- 0.70-0.85 = usable
- 0.85+ = good

## Structure

- **stage-1_exploration/** - Initial model exploration and inspection
- **stage-2_architecture/** - Model architecture analysis
- **stage-3_voice_exploration/** - Single voice deep dive
- **stage-4_voice_comparison/** - Multi-voice comparison
- **stage-5_diagnostic_suite/** - Comprehensive diagnostic testing
  - `dataset_analysis.py` - Test 1: Acoustics vs Semantics (CRITICAL), Test 3: Separability ratio (CRITICAL)
  - `config_testing.py` - Systematic exploration (extraction points, pooling methods, post-processing including PCA discovery)
  - `embedding_analysis.py` - Understanding the 512-dim space (discriminative channel analysis, dimensionality analysis validating PCA)
- **stage-6_edge_cases/** - Edge case testing

## Quick Start

Each stage has its own README with usage instructions. Start with stage-1 for basic exploration.

## Dependencies

All stages require:
- PyTorch
- NumPy
- Matplotlib/Seaborn
- scikit-learn
- soundfile

See individual stage READMEs for specific requirements.
