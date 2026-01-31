# Mimi Analysis Outputs - Master Guide

## Quick Navigation

- **Stage 1:** [Exploration](#stage-1-exploration) - How does the encoder work?
- **Stage 2:** [Architecture](#stage-2-architecture) - What's inside Mimi?
- **Stage 3:** [Voice Exploration](#stage-3-voice-exploration) - Single voice deep dive
- **Stage 4:** [Voice Comparison](#stage-4-voice-comparison) - Multiple voice comparison
- **Stage 5:** [Diagnostic Suite](#stage-5-diagnostic-suite) - Systematic testing (MOST IMPORTANT)
- **Stage 6:** [Edge Cases](#stage-6-edge-cases) - Boundary testing

---

## Stage 1: Exploration

**Location:** `stage-1_exploration/outputs/`

**Purpose:** Initial exploration of Mimi encoder behavior

**Outputs:**
1. `encoder_features_3d_pca.png` - 3D visualization of encoder features over time
2. `encoding_pipeline.png` - Diagram of how audio flows through the encoder

**Key Question Answered:** How does Mimi encode audio?

**Read:** `stage-1_exploration/OUTPUT_EXPLANATIONS.md` for details

---

## Stage 2: Architecture

**Location:** `stage-2_architecture/outputs/`

**Purpose:** Document Mimi's internal structure

**Outputs:**
1. `architecture.txt` - Human-readable architecture summary
2. `architecture.json` - Machine-readable architecture data

**Key Question Answered:** What components make up Mimi?

**Read:** `stage-2_architecture/OUTPUT_EXPLANATIONS.md` for details

---

## Stage 3: Voice Exploration

**Location:** `stage-3_voice_exploration/outputs/`

**Purpose:** Analyze single speaker characteristics in depth

**Status:** To be implemented (template exists)

**Planned Outputs:**
- Single voice temporal analysis
- Phoneme-level feature extraction
- Voice consistency across different texts

---

## Stage 4: Voice Comparison

**Location:** `stage-4_voice_comparison/outputs/`

**Purpose:** Compare multiple speakers to understand separability

**Status:** Script exists but needs integration with synthetic dataset

**Planned Outputs:**
- Pairwise voice similarity matrices
- Multi-voice PCA visualization
- Speaker confusion matrix

---

## Stage 5: Diagnostic Suite (MOST IMPORTANT)

**Location:** `stage-5_diagnostic_suite/outputs/`

**Purpose:** Comprehensive, systematic testing of Mimi's speaker discrimination ability

### Main Analysis Results

**Location:** `outputs/analysis_results/`

**Core Outputs (5 files):**
1. **`analysis_report.txt`** - Comprehensive text summary with all metrics
   - **Read this first!** Contains the verdict on whether Mimi works for speaker diarization
   - Pass/fail for each metric
   - Summary statistics
   
2. **`acoustics_vs_semantics.png`** - THE MOST CRITICAL PLOT
   - Reveals if Mimi captures voice identity or text content
   - If same-voice-different-text similarity is HIGH and same-text-different-voice similarity is LOW: Model captures voice
   - If same-text-different-voice similarity is HIGH: Model captures text, not voice
   
3. **`voice_space_pca2d.png`** - Visual clustering of all voices
   - Reveals speaker separability visually
   - Distinct, non-overlapping clusters = good separation
   - Mixed, overlapping clusters = poor separation
   
4. **`voice_clustering.png`** - Hierarchical clustering dendrogram
   - Reveals which voices are hardest to distinguish
   - High branches = well separated (good)
   - Low branches = easily confused (problematic)
   
5. **`robustness.png`** - Stability under noise
   - Reveals if embeddings survive audio degradation
   - Distributions right of thresholds = robust
   - Distributions left of thresholds = fragile

### Embedding Deep Dive

**Location:** `outputs/analysis_results/embedding_deep_dive/`

**Purpose:** Understand the 512-dimensional embedding space

**Outputs (12 files):**

#### Statistical Analysis (4 files)
1. **`channel_statistics.png`** - Distribution of means, stds, mins, maxs across channels
2. **`value_distributions.png`** - Overall value distribution + Q-Q plot for normality
3. **`channel_correlations.png`** - Correlation matrix between channels
4. **`dimensionality.png`** - PCA scree plot + cumulative variance

**What they tell you:** Properties of the embedding space, potential for dimensionality reduction

#### Discriminative Analysis (3 files)
5. **`discriminative_channels.csv`** - Ranking of all 512 channels by speaker discrimination ability
6. **`top_discriminative_channels.png`** - Heatmap of top 50 most useful channels
7. **`activation_heatmap.png`** - Raw embedding values across samples

**What they tell you:** Which channels actually encode speaker information vs. being redundant

#### Voice Pair Comparisons (5 files)
8-12. **`pair_comparison_*.png`** - Direct comparison between specific voice pairs
   - Shows exactly which channels differentiate these two speakers
   - Includes similarity metrics and value distributions
   - Selected pairs test different scenarios (same/different accent, gender, etc.)

**What they tell you:** Why specific speakers are confused or well-separated

### Diagnostic Configuration Testing

**Location:** `outputs/diagnostic_results/`

**Purpose:** Find optimal configuration for extracting speaker embeddings

**Outputs (5 files):**

1. **`diagnostic_results.csv`** - All test results in spreadsheet format
   - Every combination of extraction point, pooling method, duration, post-processing
   - Sort by metrics to find best configuration
   
2. **`diagnostic_results.json`** - Same data in JSON for programmatic access

3. **`extraction_pooling_heatmap.png`** - THE KEY DIAGNOSTIC PLOT
   - Reveals optimal extraction point and pooling method combination
   - Dark green cells = best configurations
   - **Find cells that are green in BOTH heatmaps** (optimal configuration)
   
4. **`duration_effect.png`** - Impact of audio length
   - Reveals how audio length affects speaker discrimination
   - Increasing bars = longer audio helps
   - Flat bars = duration doesn't matter
   - Reveals minimum duration needed and diminishing returns point
   
5. **`postprocessing_effect.png`** - Impact of post-processing techniques
   - Reveals which post-processing techniques work best
   - **Key discovery:** PCA dramatically improves results (transforms unusable to usable)
   - Shows that raw Mimi embeddings need transformation

**What they tell you:** Exact configuration to use in production

**Read:** `stage-5_diagnostic_suite/OUTPUT_EXPLANATIONS.md` for exhaustive details

---

## Stage 6: Edge Cases

**Location:** `stage-6_edge_cases/outputs/`

**Purpose:** Test boundary conditions and failure modes

**Outputs:**
1. `edge_test_results.json` - Results from all edge case tests

**Tests Performed:**
- Very short audio (<1s)
- Very long audio (>30s)
- Silence and low-volume
- Clipped/distorted audio
- Multi-speaker overlapping speech
- Non-speech audio (music, noise)
- Extreme accents and foreign languages

**What it tells you:** 
- Where the model breaks
- What preprocessing is needed
- What constraints exist for production use

**Read:** `stage-6_edge_cases/OUTPUT_EXPLANATIONS.md` for details

---

## How to Read the Results

### If you're in a hurry (5 minutes):
1. Read `stage-5_diagnostic_suite/outputs/analysis_results/analysis_report.txt`
2. Look at `stage-5_diagnostic_suite/outputs/analysis_results/acoustics_vs_semantics.png`
3. Look at `stage-5_diagnostic_suite/outputs/diagnostic_results/extraction_pooling_heatmap.png`

**You now know:** Whether Mimi works, and if so, what configuration to use

### If you want to understand deeply (30 minutes):
1. Stage 1 OUTPUT_EXPLANATIONS.md - Understand the encoder
2. Stage 2 OUTPUT_EXPLANATIONS.md - Know the architecture
3. Stage 5 OUTPUT_EXPLANATIONS.md - All diagnostics explained
4. Stage 6 OUTPUT_EXPLANATIONS.md - Limitations and boundaries

**You now know:** Everything about how Mimi processes audio for speaker diarization

### If you're debugging a problem (variable time):
1. Check `voice_space_pca2d.png` - Are specific voices overlapping?
2. Check `embedding_deep_dive/pair_comparison_*.png` - Why are these voices confused?
3. Check `discriminative_channels.csv` - Which channels should be used?
4. Check `edge_test_results.json` - Is this an edge case failure?

**You now know:** Root cause of the specific issue

---

## Critical Findings (TLDR)

### The Problem:
**Mimi captures text content more than speaker identity**
- Same text, different speakers: HIGH similarity (bad)
- Same speaker, different text: LOWER similarity (bad)
- This is backwards from what we need

### Why This Happens:
**Mimi was trained for TTS (text-to-speech), not speaker recognition**
- Its job is to encode "what was said" so it can be reconstructed
- Speaker identity is secondary to content fidelity
- The model architecture optimizes for semantic preservation

### The Solution Found:
**PCA post-processing transforms embeddings from text-focused to voice-focused**
- PCA projects onto axes of maximum variance
- For Mimi, maximum variance = acoustic characteristics (voice)
- Lower variance dimensions = semantic content (text)
- By using top PCA components, we extract the acoustic information

### Best Configuration Discovered:
1. **Extraction Point:** `encoder` (not encoder_transformer - transformer adds semantic content)
2. **Pooling Method:** `mean` (average across all time frames)
3. **Post-Processing:** `PCA` with ~50-100 components (reduces from 512D)
4. **Audio Duration:** 3-5 seconds (sweet spot for accuracy vs. latency)

### Performance Metrics:
- **Before PCA:** Acoustics score = -0.09 (captures text, not voice)
- **After PCA:** Acoustics score = +0.13 (captures voice)
- **Separability:** Improved from 0.55 to ~0.8-1.0 (still below target of 2.0)
- **Consistency:** ~0.70-0.80 (acceptable but not great)

### Verdict:
**Usable but not ideal**
- Mimi can work for speaker diarization with heavy post-processing
- Performance is below gold-standard models (WavLM: 3.5-4.0 separability)
- Recommended path: Use Mimi as baseline, but train speaker projection network or switch to speaker-specific encoder

---

### Most Important Files (Priority Order):
1. `stage-5_diagnostic_suite/outputs/analysis_results/analysis_report.txt`
2. `stage-5_diagnostic_suite/outputs/analysis_results/acoustics_vs_semantics.png`
3. `stage-5_diagnostic_suite/outputs/diagnostic_results/extraction_pooling_heatmap.png`
4. `stage-5_diagnostic_suite/outputs/diagnostic_results/postprocessing_effect.png`
5. `stage-5_diagnostic_suite/outputs/analysis_results/voice_space_pca2d.png`

Everything else is supplementary detail for deeper analysis.

