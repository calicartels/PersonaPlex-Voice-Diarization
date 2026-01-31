# Stage 1: Model Exploration

## What We're Testing

This stage explores the basic structure and behavior of the Mimi encoder. We want to understand how audio flows through the model and what transformations happen at each step.

## What It Proves

**Model Inspection (`model_inspect.py`):**
- Verifies the model loads correctly
- Shows parameter counts and model size
- Confirms feature extraction works
- **Proves:** The model is functional and can process audio

**Encoder Explorer (`encoder_explorer.py`):**
- Visualizes the step-by-step transformation: audio → encoder → transformer → quantizer
- Shows how time compression works (24kHz audio → lower frame rate)
- Reveals feature dimensions and shapes at each stage
- **Proves:** The encoder pipeline works as expected and we understand the data flow

## Why This Matters

Before testing voice separation, we need to confirm the model works correctly. This stage establishes a baseline and helps us understand what features are available for speaker discrimination.

## Files

- `model_inspect.py` - Basic model inspection
- `encoder_explorer.py` - Step-by-step pipeline visualization

## Usage

```bash
python model_inspect.py
python encoder_explorer.py
```

## Outputs

All outputs saved to `outputs/` directory.
