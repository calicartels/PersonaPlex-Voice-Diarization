#!/bin/bash
set -e

echo "MimiSpeaker Diarization Pipeline"
echo "Target: 1x RTX 4090, 16 CPU, 717GB disk"

# Choice: skip pip if torch already importable. Saves 1-2 min on reruns.
# Alternative: always run pip (safer but slower).
python -c "import torch, nemo" 2>/dev/null && echo "Dependencies already installed" \
    || pip install -q -r requirements.txt --timeout 120 --retries 5

echo "Step 1: Download data"
python download.py

echo "Step 2: Simulate multi-speaker sessions"
# Choice: check for manifest files to skip simulation.
# Each sim config produces a manifest in /workspace/diarization/manifests/
# If all 3 exist with content, skip the 53-min simulation step.
SIM_DONE=true
for cfg in sim_2spk sim_4spk sim_3spk_long; do
    mf="/workspace/diarization/manifests/${cfg}.json"
    if [ ! -s "$mf" ]; then
        SIM_DONE=false
        break
    fi
done

if $SIM_DONE; then
    echo "  Simulated data already exists, skipping"
else
    python simulate.py
fi

echo "Step 3: Convert RTTM to frame labels"
# Choice: check if label manifests exist. Fast step (~30s) so rerunning is fine,
# but skip avoids unnecessary disk writes.
LBL_DONE=true
for cfg in sim_2spk sim_4spk sim_3spk_long; do
    mf="/workspace/diarization/manifests/${cfg}_labels.json"
    if [ ! -s "$mf" ]; then
        LBL_DONE=false
        break
    fi
done

if $LBL_DONE; then
    echo "  Labels already exist, skipping"
else
    python process_rttm.py
fi

echo "Step 3.5: Fix AMI manifest (audio paths)"
python fix_ami_manifest.py

echo "Step 4: Extract Mimi embeddings"
# extract.py already skips existing .npy files internally
python extract.py

echo "Step 5: Merge manifests"
# Fast (<1s), always rerun to pick up any new data
python merge.py

echo "Step 6: Train"
# train.py resumes from CKPT/latest.pt if it exists
python train.py

echo "Step 7: Evaluate"
python eval.py

echo "Step 8: Upload checkpoint to HuggingFace"
if [ -n "$HF_REPO" ]; then
    python upload_checkpoint.py
else
    echo "  Skipping upload (set HF_REPO env var to enable)"
fi

echo "Done. Checkpoint: /workspace/diarization/checkpoints/best.pt"
echo "Demo: python demo.py --audio meeting.wav"