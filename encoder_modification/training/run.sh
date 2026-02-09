#!/bin/bash
set -e

echo "MimiSpeaker Diarization Pipeline"
echo "Target: 1x RTX 4090, 16 CPU, 717GB disk"

pip install -q -r requirements.txt

echo "Step 1: Download data"
python download.py

echo "Step 2: Simulate multi-speaker sessions"
python simulate.py

echo "Step 3: Convert RTTM to frame labels"
python process_rttm.py

echo "Step 4: Extract Mimi embeddings"
python extract.py

echo "Step 5: Merge manifests"
python merge.py

echo "Step 6: Train"
python train.py

echo "Step 7: Evaluate"
python eval.py

echo "Done. Checkpoint: /workspace/diarization/checkpoints/best.pt"
echo "Demo: python demo.py --audio meeting.wav"