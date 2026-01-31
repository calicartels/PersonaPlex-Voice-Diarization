# PersonaPlex Voice Diarization Analysis

## Goal

Analyze whether the Mimi encoder (from PersonaPlex/Moshi) can be used for speaker diarization. The core question: **Does Mimi capture acoustic (speaker) information or semantic (text) information?**

## Problem

Mimi was trained for text-to-speech (TTS), not speaker recognition. Initial analysis shows it captures text content more than speaker identity:
- Same text, different speakers: HIGH similarity (bad)
- Same speaker, different text: LOWER similarity (bad)
- This is backwards from what we need for diarization

## Solution

Through systematic testing, we discovered that PCA post-processing transforms Mimi embeddings from text-focused to voice-focused, making speaker diarization possible (though not ideal).

## Documentation

### Analysis Suite
- **[mimi_analysis/README.md](mimi_analysis/README.md)** - Complete analysis suite overview, problem statement, and success metrics

### Output Guide
- **[mimi_analysis/OUTPUTS_MASTER_GUIDE.md](mimi_analysis/OUTPUTS_MASTER_GUIDE.md)** - Master guide explaining all generated plots and outputs

### Stage Documentation
- **[mimi_analysis/stage-1_exploration/README.md](mimi_analysis/stage-1_exploration/README.md)** - Initial encoder exploration
- **[mimi_analysis/stage-2_architecture/README.md](mimi_analysis/stage-2_architecture/README.md)** - Model architecture analysis
- **[mimi_analysis/stage-3_voice_exploration/README.md](mimi_analysis/stage-3_voice_exploration/README.md)** - Single voice deep dive
- **[mimi_analysis/stage-4_voice_comparison/README.md](mimi_analysis/stage-4_voice_comparison/README.md)** - Multi-voice comparison
- **[mimi_analysis/stage-5_diagnostic_suite/README.md](mimi_analysis/stage-5_diagnostic_suite/README.md)** - Comprehensive diagnostic testing
- **[mimi_analysis/stage-6_edge_cases/README.md](mimi_analysis/stage-6_edge_cases/README.md)** - Edge case and boundary testing

## Quick Start

1. Read [mimi_analysis/README.md](mimi_analysis/README.md) for the full problem statement and methodology
2. Check [mimi_analysis/OUTPUTS_MASTER_GUIDE.md](mimi_analysis/OUTPUTS_MASTER_GUIDE.md) to understand the results
3. Review stage-specific READMEs for detailed analysis procedures

## Key Findings

- **Before PCA:** Acoustics score = -0.09 (captures text, not voice)
- **After PCA:** Acoustics score = +0.13 (captures voice)
- **Separability:** Improved from 0.55 to ~0.8-1.0 (still below target of 2.0)
- **Verdict:** Usable but not ideal - requires heavy post-processing
