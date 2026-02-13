import json
import numpy as np
import soundfile as sf
from pathlib import Path
from config import FRAME_S, MAX_SPEAKERS, LABELS, MANIFESTS, SIM, RAW, SIM_CONFIGS


def parse_rttm(path):
    segs = []
    for line in open(path):
        p = line.strip().split()
        if p[0] != "SPEAKER":
            continue
        segs.append((p[7], float(p[3]), float(p[3]) + float(p[4])))
    return segs  # (speaker_id, onset, offset)


def to_frame_labels(segs, duration_s):
    n_frames = int(np.ceil(duration_s / FRAME_S))

    # Choice: sort speakers by first appearance (arrival time order).
    # Required for Sort Loss. PIL doesn't care, but hybrid loss needs it.
    first_onset = {}
    for spk, on, _ in segs:
        first_onset[spk] = min(first_onset.get(spk, float("inf")), on)
    sorted_spks = sorted(first_onset, key=lambda s: first_onset[s])
    spk2idx = {s: i for i, s in enumerate(sorted_spks)}

    labels = np.zeros((n_frames, MAX_SPEAKERS), dtype=np.float32)
    for spk, on, off in segs:
        idx = spk2idx[spk]
        if idx >= MAX_SPEAKERS:
            continue
        s = int(on / FRAME_S)
        e = min(int(off / FRAME_S) + 1, n_frames)
        labels[s:e, idx] = 1.0
    return labels


def process_manifest(manifest_path, tag):
    lbl_dir = LABELS / tag
    lbl_dir.mkdir(parents=True, exist_ok=True)

    entries = [json.loads(l) for l in open(manifest_path)]
    updated = []
    for e in entries:
        rttm = e.get("rttm_filepath", "")
        if not rttm or not Path(rttm).exists():
            continue
        segs = parse_rttm(rttm)
        if not segs:
            continue

        sid = e.get("session_id", Path(rttm).stem)
        labels = to_frame_labels(segs, e["duration"])
        lbl_path = lbl_dir / f"{sid}.npy"
        np.save(lbl_path, labels)

        e["labels_filepath"] = str(lbl_path)
        updated.append(e)

    out = MANIFESTS / f"{tag}_labels.json"
    with open(out, "w") as f:
        for e in updated:
            f.write(json.dumps(e) + "\n")
    return len(updated)


def fix_manifest_paths(manifest_path, audio_dir):
    """Patch audio_filepath in manifest when paths are wrong (e.g. Colab)."""
    if not manifest_path.exists():
        return 0
    lines = manifest_path.read_text().strip().split("\n")
    out = []
    patched = 0
    for line in lines:
        if not line.strip():
            continue
        d = json.loads(line)
        wav_path = Path(audio_dir) / f"{d['session_id']}.Mix-Headset.wav"
        if wav_path.exists():
            d["audio_filepath"] = str(wav_path)
            patched += 1
        elif not d.get("audio_filepath") or not Path(d["audio_filepath"]).exists():
            print(f"WARNING: missing audio for {d['session_id']}")
        out.append(json.dumps(d))
    manifest_path.write_text("\n".join(out) + "\n")
    if patched:
        print(f"Patched {patched}/{len(out)} AMI audio paths")
    return patched


def process_rttm_dir(rttm_dir, audio_dir, tag):
    lbl_dir = LABELS / tag
    lbl_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for rttm in sorted(Path(rttm_dir).glob("*.rttm")):
        segs = parse_rttm(rttm)
        if not segs:
            continue
        sid = rttm.stem
        audio = None
        # Choice: AMI uses .Mix-Headset.wav; generic uses .wav/.flac
        for name in [f"{sid}.Mix-Headset.wav", f"{sid}.wav", f"{sid}.flac"]:
            c = Path(audio_dir) / name
            if c.exists():
                audio = c
                break
        dur = sf.info(str(audio)).duration if audio else max(off for _, _, off in segs) + 1.0
        labels = to_frame_labels(segs, dur)
        lbl_path = lbl_dir / f"{sid}.npy"
        np.save(lbl_path, labels)
        entries.append({
            "audio_filepath": str(audio) if audio else "",
            "duration": dur,
            "labels_filepath": str(lbl_path),
            "rttm_filepath": str(rttm),
            "session_id": sid,
            "tag": tag,
        })

    mf = MANIFESTS / f"{tag}_labels.json"
    with open(mf, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return len(entries)


# Simulated
for name in SIM_CONFIGS:
    tag = f"sim_{name}"
    mf = MANIFESTS / f"{tag}.json"
    if mf.exists():
        print(f"{tag}: {process_manifest(mf, tag)} labeled")

# AMI
ami_rttm = RAW / "ami" / "rttm"
ami_audio = RAW / "ami" / "audio"
if ami_rttm.exists() and list(ami_rttm.glob("*.rttm")):
    print(f"ami: {process_rttm_dir(ami_rttm, ami_audio, 'ami')} labeled")
    fix_manifest_paths(MANIFESTS / "ami_labels.json", ami_audio)

# VoxConverse
vox_rttm = RAW / "voxconverse" / "voxconverse"
if vox_rttm.exists():
    print(f"voxconverse: {process_rttm_dir(vox_rttm, RAW / 'voxconverse' / 'audio', 'voxconverse')} labeled")