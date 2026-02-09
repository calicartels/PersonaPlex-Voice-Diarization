# probe

Linear probe to test whether speaker identity is linearly separable
in Mimi's pre-quantization embeddings.

## Files

- `train.py` — trains a linear classifier (512 -> num_speakers) on cached embeddings
- `eval.py` — projects embeddings through learned weights, computes EER

## Method

Loads the .npz cache from Step 0. Extracts speaker labels from utterance IDs.
Trains a single linear layer with cross-entropy loss to classify speakers.
After training, uses the weight matrix as a projection: each 512-dim embedding
is mapped to a speaker-discriminative subspace. EER is computed with cosine
similarity on the projected embeddings using the same VoxCeleb1-O pairs.

## Why train and test on the same set

We only have VoxCeleb1 test (40 speakers, 4874 utterances). This is not
a generalization test — it's a capacity test. We ask: does the 512-dim
space contain linear directions that separate speakers? If the probe can't
separate them even with full access to labels, the signal is too weak.
If it can, the 3-layer Transformer adapter has a real chance on unseen speakers.

