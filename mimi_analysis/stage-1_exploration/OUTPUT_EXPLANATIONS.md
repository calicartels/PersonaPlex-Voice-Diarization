# Stage 1 Exploration - Output Explanations

## Overview
This stage explores the Mimi encoder's internal workings by processing sample audio and visualizing the encoding pipeline and feature representations.

---

## Outputs

### 1. `encoding_pipeline.png`
**What it reveals:** The audio processing flow through Mimi

**Key revelations:**
- Audio is transformed into 512-dimensional features at each time frame, not a single embedding
- The encoder produces temporal sequences that need aggregation (pooling) for speaker representation
- The unquantized latent features are the primary output we analyze for voice characteristics

**Why it matters:** Understanding this pipeline is crucial for knowing where to extract speaker information. The unquantized latent features are what we analyze for voice characteristics.

---

### 2. `encoder_features_3d_pca.png`
**What it reveals:** How encoder features evolve over time and their temporal structure

**Key revelations:**
- Features change throughout the audio, showing temporal dynamics
- Tight clustering indicates consistent features across the audio segment
- Wide spread indicates significant variation in acoustic characteristics
- The explained variance shows how much information is captured in the top dimensions

**Why it matters:** This reveals the temporal structure of the encoded audio. For speaker diarization, we want features that are consistent for the same speaker but distinct across different speakers. The trajectory shows whether features are stable or highly variable.

---

## Key Insights from Stage 1

1. **Encoder Architecture:** Mimi uses a convolutional encoder (SEANet) that produces 512-dimensional features at each time frame

2. **Temporal Representation:** Audio is encoded as a sequence of feature vectors, not a single embedding. We need to aggregate these (pooling) to get a single speaker representation

3. **Feature Dimensionality:** The 512D feature space is high-dimensional. PCA helps us visualize it, but the full dimensionality contains more information

4. **Next Questions:** 
   - Do these features capture speaker identity or just speech content?
   - Which part of the encoder is most important for speaker information?
   - How should we pool the temporal features into a single embedding?
