# Stage 2: Architecture Analysis

## What We're Testing

This stage analyzes the Mimi model's internal architecture: layer structure, parameter distribution, and activation shapes at each stage.

## What It Proves

**Architecture Analysis (`architecture_analysis.py`):**
- Maps out all model components (encoder, transformer, quantizer, decoder)
- Counts parameters per component
- Shows activation shapes at each processing stage
- Analyzes weight distribution across components
- **Proves:** We understand the model's structure and can identify where speaker information might be encoded

## Why This Matters

Understanding the architecture helps us:
- Know which layers to extract features from
- Understand model capacity and complexity
- Identify bottlenecks or key components
- Make informed decisions about feature extraction points

## Files

- `architecture_analysis.py` - Complete architecture breakdown

## Usage

```bash
python architecture_analysis.py
```

## Outputs

- `architecture.json` - Machine-readable architecture data
- `architecture.txt` - Human-readable summary
