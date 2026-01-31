# Stage 5 Diagnostic Suite - Output Explanations

## Overview
This stage provides comprehensive analysis of the Mimi encoder's ability to capture speaker information using our synthetic dataset. It tests the core hypothesis: **Does Mimi capture acoustic (speaker) information or semantic (text) information?**

---

## Analysis Results Outputs

### 1. `analysis_report.txt`
**What it reveals:** Overall verdict on whether Mimi works for speaker diarization

**Key revelations:**
- Pass/fail status for each critical metric
- Whether Mimi captures voice identity or text content
- How consistent same-speaker embeddings are
- How well different speakers are separated
- Stability under noise and augmentation

**Why it matters:** This is the definitive answer to "Can we use Mimi for speaker diarization?" Read the summary section for the final verdict.

---

### 2. `acoustics_vs_semantics.png`
**What it reveals:** THE MOST CRITICAL TEST - Does Mimi capture voice or text?

**Key revelations:**
- If same-voice-different-text similarity is HIGH and same-text-different-voice similarity is LOW: Model captures voice ✅
- If same-text-different-voice similarity is HIGH: Model captures text, not voice ❌
- The relative positions of the distributions reveal the fundamental behavior

**Why it matters:** This single plot tells us if Mimi is fundamentally suitable for speaker diarization. If it captures text more than voice, no amount of post-processing will fix it. This is the make-or-break test.

---

### 3. `voice_space_pca2d.png`
**What it reveals:** Visual confirmation of speaker separability

**Key revelations:**
- Distinct, non-overlapping clusters = good speaker separation
- Mixed, overlapping clusters = poor speaker separation
- Tight clusters = consistent same-speaker embeddings
- Spread clusters = inconsistent same-speaker embeddings
- The variance percentages show how much information is captured in 2D

**Why it matters:** Visual confirmation of speaker separability. In a good system, you should be able to "see" the different voices as separate clusters. If voices are mixed together, the model cannot distinguish them.

---

### 4. `voice_clustering.png`
**What it reveals:** Which speakers are hardest to distinguish

**Key revelations:**
- High branch points = very different speakers (good for diarization)
- Low branch points = similar speakers (potential confusion)
- Cluster groups reveal speakers that share characteristics (gender, accent, etc.)
- The tree structure shows hierarchical relationships between voices

**Why it matters:** Reveals which speakers are hardest to distinguish. If two speakers have a very low branch point, the model might confuse them. This helps identify problematic voice pairs.

---

### 5. `robustness.png`
**What it reveals:** Stability of embeddings when audio quality degrades

**Key revelations:**
- Distributions peaked right of thresholds = robust model (maintains speaker identity despite noise)
- Distributions shifted left = fragile model (loses speaker information under noise)
- Light noise should maintain high similarity (~0.85+)
- Medium noise should maintain moderate similarity (~0.75+)

**Why it matters:** Real-world audio is never perfect. A model that works only on clean audio is useless for production. This test validates real-world applicability.

---

## Embedding Deep Dive Outputs

### 6. `embedding_deep_dive/channel_statistics.png`
**What it reveals:** Which channels are actually "doing work" vs. being redundant

**Key revelations:**
- Wide spread in means = channels have different baseline activation levels
- Wide spread in std devs = some channels are very discriminative (high variation), others are constant (low variation)
- Similar mins/maxs = normalization or activation functions at play
- High std dev channels are likely the most important for speaker discrimination

**Why it matters:** Reveals which channels are actually "doing work" vs. being redundant or dead. This helps identify which dimensions of the 512D space are meaningful.

---

### 7. `embedding_deep_dive/discriminative_channels.csv` & `top_discriminative_channels.png`
**What it reveals:** Which of the 512 channels are most useful for distinguishing speakers

**Key revelations:**
- High score channels = change a lot between speakers but stay consistent for the same speaker (ideal!)
- Low score channels = either don't change between speakers or change randomly (useless)
- The score is inter-voice variance divided by intra-voice variance
- Vertical stripes in heatmap = channels that distinguish specific speakers

**Why it matters:** We could potentially reduce the 512D embeddings to just the top channels without losing speaker discrimination ability. This enables faster processing and simpler models.

---

### 8. `embedding_deep_dive/value_distributions.png`
**What it reveals:** Statistical properties of embedding values

**Key revelations:**
- Bell curve = approximately normal distribution
- Multiple peaks = multi-modal (different "types" of activations)
- Skewed = bias toward low or high values
- Q-Q plot reveals deviations from normality
- Understanding distribution helps choose appropriate distance metrics

**Why it matters:** Understanding the value distribution helps with:
- Choosing appropriate distance metrics (cosine vs. Euclidean)
- Applying normalization or standardization
- Detecting outliers or unusual activations

---

### 9. `embedding_deep_dive/channel_correlations.png`
**What it reveals:** Redundancy and relationships between channels

**Key revelations:**
- Red blocks = groups of channels that move together (redundant)
- Blue blocks = channels that are inversely related
- White areas = independent channels (good!)
- High correlation means we can reduce dimensionality

**Why it matters:** High correlation means redundancy. We could potentially:
- Reduce dimensionality by removing correlated channels
- Apply PCA or other techniques to decorrelate features
- Understand the inherent dimensionality of the representation

---

### 10. `embedding_deep_dive/dimensionality.png`
**What it reveals:** Effective dimensionality of the 512D embedding space

**Key revelations:**
- Steep drop in scree plot = first few components capture most information (good for dimensionality reduction)
- Gradual drop = information is distributed across many dimensions (hard to reduce)
- Cumulative plot shows how many dimensions needed to capture X% of information
- If 90% variance achieved with 50 dimensions instead of 512, we can greatly speed up processing

**Why it matters:** 
- Validates whether PCA is an effective post-processing technique (as discovered in diagnostic testing)
- Reveals the "true" dimensionality of speaker information in Mimi
- Enables optimization by reducing dimensions without losing information

---

### 11. `embedding_deep_dive/activation_heatmap.png`
**What it reveals:** Raw, unprocessed view of encoder outputs

**Key revelations:**
- Horizontal stripes = channels that are consistently active/inactive across all samples
- Vertical stripes = samples that have unusual activation patterns
- Checkerboard patterns = complex interactions between channels and samples
- Blocks of color = groups of related channels or similar samples

**Why it matters:** Provides a raw, unprocessed view of what the encoder actually outputs. Useful for:
- Debugging unexpected behavior
- Identifying dead or saturated channels
- Spotting artifacts or anomalies in the embeddings

---

### 12. `embedding_deep_dive/pair_comparison_*.png` (5 files)
**What it reveals:** Why specific voice pairs are confused or well-separated

**Key revelations:**
- High difference bars = channels that discriminate between these two specific voices
- Similar distributions = voices have similar acoustic characteristics
- Different distributions = voices are acoustically distinct
- Low cosine similarity (<0.5) = easy to distinguish
- High cosine similarity (>0.8) = might be confused by the model
- The top different channels show which dimensions matter for this pair

**Why it matters:** 
- Diagnose specific confusion cases (why does the model mix up these two voices?)
- Validate that the model uses different channels for different voice pairs
- Understand which acoustic characteristics the model captures

**Voice pairs analyzed:**
- Same accent pairs: Test if model distinguishes speakers with similar accents
- Different accent pairs: Test if accent dominates over speaker identity
- Male/female pairs: Test if gender is the primary discriminator
- Similar sounding pairs: Worst-case scenarios for the model

---

## Diagnostic Results Outputs (from config_testing.py)

### 13. `diagnostic_results/diagnostic_results.csv` & `.json`
**What it reveals:** Optimal configuration for extracting speaker embeddings

**Key revelations:**
- Every combination of extraction point, pooling method, duration, and post-processing tested
- Metrics show which configurations work best
- Sort by acoustics_score and separability_ratio to find winners
- Reveals interaction effects between different choices

**Why it matters:** This is the data that answers "What's the best way to use Mimi for speaker diarization?" Find configurations that score high on both acoustics_score (>0.10) and separability_ratio (>2.0).

---

### 14. `diagnostic_results/extraction_pooling_heatmap.png`
**What it reveals:** Optimal extraction point and pooling method combination

**Key revelations:**
- Dark green cells = best configurations for that metric
- Dark red cells = worst configurations
- Horizontal patterns = one extraction point consistently better than others
- Vertical patterns = one pooling method consistently better than others
- Diagonal patterns = interaction effects between extraction and pooling
- Cells green in BOTH heatmaps = optimal configuration

**Why it matters:** This single plot answers: "What's the optimal way to extract speaker embeddings from Mimi?" The best configuration maximizes both voice discrimination and speaker separation.

---

### 15. `diagnostic_results/duration_effect.png`
**What it reveals:** How audio length affects speaker discrimination

**Key revelations:**
- Increasing bars = longer audio provides more speaker information (good!)
- Flat bars = duration doesn't matter (information is in short-term features)
- Decreasing bars = longer audio adds noise or semantic content (bad!)
- Bars crossing thresholds = minimum duration needed for acceptable performance

**Why it matters:** Determines minimum audio length required for speaker diarization:
- Very short clips (<1s): Can we do real-time diarization?
- Medium clips (3-5s): Practical balance of latency and accuracy
- Long clips (full): Maximum possible performance

---

### 16. `diagnostic_results/postprocessing_effect.png`
**What it reveals:** Impact of post-processing techniques on embeddings

**Key revelations:**
- PCA dramatically improves results (turns negative acoustics scores into positive ones)
- This suggests raw embeddings mix acoustic and semantic info
- PCA separates them by projecting onto axes of maximum variance
- L2 norm mostly normalizes, doesn't change information content
- Whitening decorrelates features, might help or hurt

**Why it matters:** Post-processing can dramatically improve results:
- Our discovery: PCA transforms unusable embeddings into usable ones
- This is why PCA is now a core part of our best pipeline
- Shows that raw Mimi embeddings need transformation to work for speaker diarization

---

## Key Findings Summary

### Critical Issues Discovered:
1. **Semantics > Acoustics:** Mimi captures text content more than speaker identity (negative acoustics score)
2. **Low Separability:** Different speakers aren't well separated (ratio < 1.0 without post-processing)
3. **Inconsistent Same-Speaker:** Even the same speaker's embeddings vary significantly (consistency < 0.80)

### Solutions Found:
1. **PCA Post-Processing:** Transforms embeddings to focus on acoustic variance, dramatically improving all metrics
2. **Mean Pooling:** Best way to aggregate temporal features into a single embedding
3. **Encoder Features:** Better than encoder_transformer (transformer adds semantic content)

### Next Steps:
1. Fine-tune a small "speaker projection" network on top of Mimi embeddings
2. Test contrastive learning to explicitly push same-speaker embeddings together
3. Explore other audio encoders specifically trained for speaker recognition (WavLM, ECAPA-TDNN)
