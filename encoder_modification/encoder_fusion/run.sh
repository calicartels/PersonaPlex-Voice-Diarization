#!/bin/bash
set -e
cd "$(dirname "$0")/.."
ROOT="${DIARIZATION_ROOT:-/workspace/diarization}"
TRAIN="$(pwd)/training"
FUSION="$(pwd)/encoder_fusion"

if [ "$1" = "--check" ]; then
    cd "$FUSION" && python check_manifests.py
    exit 0
fi

REQ="$(cd "$(dirname "$0")/../.." && pwd)/requirements.txt"
python -c "import torch, nemo" 2>/dev/null || pip install -q -r "$REQ" --timeout 120 --retries 5

echo "Step 1: Download data"
python "$TRAIN/download.py"

echo "Step 2: Simulate multi-speaker sessions"
SIM_DONE=true
for cfg in sim_2spk sim_4spk sim_3spk_long; do
    mf="$ROOT/manifests/${cfg}.json"
    if [ ! -s "$mf" ]; then
        SIM_DONE=false
        break
    fi
done
if $SIM_DONE; then
    echo "  Simulated data exists, skipping"
else
    python "$TRAIN/simulate.py"
fi

echo "Step 3: Convert RTTM to frame labels"
LBL_DONE=true
for cfg in sim_2spk sim_4spk sim_3spk_long; do
    mf="$ROOT/manifests/${cfg}_labels.json"
    if [ ! -s "$mf" ]; then
        LBL_DONE=false
        break
    fi
done
if $LBL_DONE; then
    echo "  Labels exist, skipping"
else
    python "$TRAIN/process_rttm.py"
fi

echo "Step 4: Extract Mimi + TitaNet fused embeddings"
cd "$FUSION"
python extract.py

echo "Step 5: Merge manifests"
python merge.py

echo "Step 6: Train"
python train.py

echo "Step 7: Evaluate (with --compare for Mimi vs Fusion)"
python eval.py --compare

echo "Step 8: Upload to HuggingFace"
if [ -n "$HF_REPO" ] || [ -n "$HF_TOKEN" ]; then
    HF_REPO="${HF_REPO:-TMVishnu/personaplex-voice-diarization}" python upload.py
else
    echo "  Skipping upload (set HF_REPO and HF_TOKEN to enable)"
fi

echo "Done. Checkpoint: $ROOT/checkpoints_fusion/best.pt"
