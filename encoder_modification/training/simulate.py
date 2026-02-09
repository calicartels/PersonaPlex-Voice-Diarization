import json
import yaml
import subprocess
import soundfile as sf
from pathlib import Path
from config import RAW, SIM, MANIFESTS, NEMO, SIM_SR, SIM_CONFIGS


def run(cmd, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def build_source_manifest():
    manifest = MANIFESTS / "librispeech_source.json"
    if manifest.exists():
        return manifest
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    ls_root = RAW / "librispeech" / "LibriSpeech" / "train-clean-100"
    entries = []
    for flac in sorted(ls_root.rglob("*.flac")):
        txt_dir = flac.parent
        trans = txt_dir / f"{txt_dir.name}.trans.txt"
        text = ""
        if trans.exists():
            for line in open(trans):
                if line.startswith(flac.stem):
                    text = line.strip().split(" ", 1)[1]
                    break
        info = sf.info(str(flac))
        entries.append({
            "audio_filepath": str(flac),
            "duration": info.duration,
            "text": text,
        })

    with open(manifest, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return manifest


def write_nemo_config(name, params, source_manifest):
    out_dir = SIM / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = SIM / f"{name}.yaml"

    # Choice: NeMo multispeaker_simulator with Hydra YAML config.
    # This is Sortformer's exact data generation tool.
    # Alternative: custom mixer, but NeMo handles turn-taking dynamics better.
    cfg = {
        "data_simulator": {
            "manifest_filepath": str(source_manifest),
            "sr": SIM_SR,
            "random_seed": 42,
            "session_config": {
                "num_speakers": params["n_speakers"],
                "num_sessions": params["n_sessions"],
                "session_length": params["length"],
            },
            "session_params": {
                # Choice: overlap_prob varies per config (0.12-0.25).
                # Sortformer default is 0.12, AMI has ~16% measured overlap.
                "overlap_prob": params["overlap"],
                # Choice: 0.5s mean silence matches natural turn-taking.
                # Alternative: 0.8s (slower pace), but less data-efficient.
                "mean_silence": 0.5,
                "mean_silence_var": 0.1,
                "mean_overlap": 0.4,
                "mean_overlap_var": 0.1,
                # Choice: 0.1 silence ratio. Sortformer used 0.1 as default.
                "mean_silence_ratio": 0.1,
            },
            "outputs": {
                "output_dir": str(out_dir),
                "output_filename": "session",
            },
        }
    }
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return cfg_path


def run_one(name, source_manifest):
    params = SIM_CONFIGS[name]
    est_h = params["n_sessions"] * params["length"] / 3600

    write_nemo_config(name, params, source_manifest)

    # NeMo simulator entry point
    run(
        f"python tools/speech_data_simulator/multispeaker_simulator.py "
        f"--config-path {SIM} --config-name {name}.yaml",
        cwd=str(NEMO),
    )

    # Build manifest from generated files
    out_dir = SIM / name
    entries = []
    for rttm in sorted(out_dir.glob("*.rttm")):
        sid = rttm.stem
        wav = out_dir / f"{sid}.wav"
        if not wav.exists():
            continue
        info = sf.info(str(wav))
        entries.append({
            "audio_filepath": str(wav),
            "duration": info.duration,
            "num_speakers": params["n_speakers"],
            "rttm_filepath": str(rttm),
            "session_id": sid,
            "tag": f"sim_{name}",
        })

    mf = MANIFESTS / f"sim_{name}.json"
    with open(mf, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return len(entries), est_h


source = build_source_manifest()
print(f"Source manifest: {source}")

total_h = 0
for name in SIM_CONFIGS:
    print(f"Simulating {name}...")
    n, h = run_one(name, source)
    total_h += h
    print(f"  {name}: {n} sessions, ~{h:.0f}h")

print(f"Total: ~{total_h:.0f}h simulated")