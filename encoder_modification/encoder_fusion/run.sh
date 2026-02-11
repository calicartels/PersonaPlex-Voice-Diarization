#!/bin/bash
set -e
cd "$(dirname "$0")/.."
ROOT="${DIARIZATION_ROOT:-/workspace/diarization}"
TRAIN="$(pwd)/training"
FUSION="$(pwd)/encoder_fusion"

python -c "import torch, nemo" 2>/dev/null || pip install -q -r "$TRAIN/requirements.txt" --timeout 120 --retries 5

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

echo "Step 4: Fix AMI manifest"
python "$TRAIN/fix_ami_manifest.py"

echo "Step 5: Extract Mimi + TitaNet fused embeddings"
cd "$FUSION"
python extract.py

echo "Step 6: Merge manifests"
python merge.py

echo "Step 7: Train"
python train.py

echo "Step 8: Evaluate"
python eval.py

echo "Done. Checkpoint: $ROOT/checkpoints_fusion/best.pt"
