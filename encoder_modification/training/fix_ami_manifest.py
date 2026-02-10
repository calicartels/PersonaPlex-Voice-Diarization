import json
from pathlib import Path

manifest = Path("/workspace/diarization/manifests/ami_labels.json")
audio_dir = Path("/workspace/diarization/raw/ami/audio")

lines = manifest.read_text().strip().split("\n")
out = []
patched = 0
for line in lines:
    d = json.loads(line)
    wav_path = audio_dir / f"{d['session_id']}.Mix-Headset.wav"
    if wav_path.exists():
        d["audio_filepath"] = str(wav_path)
        patched += 1
    else:
        print(f"WARNING: missing audio for {d['session_id']}")
    out.append(json.dumps(d))

manifest.write_text("\n".join(out) + "\n")
print(f"Patched {patched}/{len(out)} entries")