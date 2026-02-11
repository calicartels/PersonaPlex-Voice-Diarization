#!/bin/bash
set -e

echo "MimiSpeaker Fine-Tune Pipeline"
echo "Unfreezes top 3 Mimi encoder_transformer layers + adapter"

cd "$(dirname "$0")"

echo "Step 1: Extract pre-conv embeddings"
python extract_preconv.py

echo "Step 2: Merge manifests"
python merge_ft.py

echo "Step 3: Train (Mimi layers 5-7 + adapter)"
python train_ft.py

echo "Step 4: Evaluate"
python eval_ft.py

echo "Step 5: Upload checkpoint"
if [ -n "$HF_REPO" ]; then
    cd ..
    python upload_checkpoint.py
    cd unfreeze
else
    echo "  Skipping upload (set HF_REPO env var to enable)"
fi

echo "Done."

