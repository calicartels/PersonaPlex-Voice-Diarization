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
        info = sf.info(str(flac))
        spk = flac.parts[-3]
        chapter = flac.parts[-2]
        # Choice: trans file is {spk}-{chapter}.trans.txt (LibriSpeech convention).
        # Original code used {chapter}.trans.txt which doesn't exist.
        trans = flac.parent / f"{spk}-{chapter}.trans.txt"
        text = ""
        if trans.exists():
            for line in open(trans):
                if line.startswith(flac.stem):
                    text = line.strip().split(" ", 1)[1]
                    break
        words = text.split() if text else []
        # NeMo simulator requires min 2 alignments per utterance
        if len(words) < 2:
            continue
        # Choice: fake alignments by spacing words evenly across duration.
        # Same approach as Sortformer paper Eq.24 (syllable-based approximation).
        # Alternative: run Montreal Forced Aligner, but adds heavy dependency.
        n = len(words)
        step = info.duration / (n + 1)
        alignments = [round(step * (i + 1), 3) for i in range(n)]
        entries.append({
            "audio_filepath": str(flac),
            "duration": info.duration,
            "speaker_id": spk,
            "words": words,
            "alignments": alignments,
        })

    with open(manifest, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"  Manifest: {len(entries)} utterances from {ls_root}")
    return manifest


def write_nemo_config(name, params, source_manifest):
    out_dir = SIM / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = SIM / f"{name}.yaml"

    # Choice: full NeMo schema with all required sections.
    # RIR/augmentation/noise disabled — clean simulated data matches
    # Sortformer's default simulation setup.
    cfg = {
        "data_simulator": {
            "manifest_filepath": str(source_manifest),
            "sr": SIM_SR,
            "random_seed": 42,
            "multiprocessing_chunksize": 10000,
            "session_config": {
                "num_speakers": params["n_speakers"],
                "num_sessions": params["n_sessions"],
                "session_length": params["length"],
            },
            "session_params": {
                "max_audio_read_sec": 20.0,
                "sentence_length_params": [0.4, 0.05],
                "dominance_var": 0.11,
                "min_dominance": 0.05,
                "turn_prob": 0.875,
                "min_turn_prob": 0.5,
                # Choice: NeMo defaults for turn-taking dynamics.
                # Alternative: custom overlap/silence from our SIM_CONFIGS,
                # but NeMo's mean_overlap already captures this via params["overlap"].
                "mean_silence": 0.15,
                "mean_silence_var": 0.01,
                "per_silence_var": 900,
                "per_silence_min": 0.0,
                "per_silence_max": -1,
                "mean_overlap": params["overlap"],
                "mean_overlap_var": 0.01,
                "per_overlap_var": 900,
                "per_overlap_min": 0.0,
                "per_overlap_max": -1,
                "start_window": True,
                "window_type": "hamming",
                "window_size": 0.05,
                "start_buffer": 0.1,
                "split_buffer": 0.1,
                "release_buffer": 0.1,
                "normalize": True,
                "normalization_type": "equal",
                "normalization_var": 0.1,
                "min_volume": 0.75,
                "max_volume": 1.25,
                "end_buffer": 0.5,
            },
            "outputs": {
                "output_dir": str(out_dir),
                "output_filename": "session",
                "overwrite_output": True,
                "output_precision": 3,
            },
            "background_noise": {
                "add_bg": False,
                "background_manifest": None,
                "num_noise_files": 10,
                "snr": 60,
                "snr_min": None,
                "snr_max": None,
            },
            "segment_augmentor": {
                "add_seg_aug": False,
                "augmentor": {
                    "gain": {"prob": 0.5, "min_gain_dbfs": -10.0, "max_gain_dbfs": 10.0}
                },
            },
            "session_augmentor": {
                "add_sess_aug": False,
                "augmentor": {
                    "white_noise": {"prob": 1.0, "min_level": -90, "max_level": -46}
                },
            },
            "speaker_enforcement": {
                "enforce_num_speakers": True,
                "enforce_time": [0.25, 0.75],
            },
            "segment_manifest": {
                "window": 0.5,
                "shift": 0.25,
                "step_count": 50,
                "deci": 3,
            },
            "rir_generation": {
                "use_rir": False,
                "toolkit": "nemo",
                "room_config": {
                    "room_sz": [[3, 3, 2.5], [10, 6, 4]],
                    "pos_src": [[[1, 1, 1.5], [1.5, 2, 1.5]]],
                    "noise_src_pos": [[1.5, 2.5, 1.5], [2, 2.5, 1.5]],
                    "mic_config": {
                        "num_channels": 2,
                        "pos_rcv": [[[[0.5, 1.5], [0.5, 1.5], [0.5, 1.5]],
                                      [[0.5, 1.5], [0.5, 1.5], [0.5, 1.5]]]],
                        "orV_rcv": None,
                        "mic_pattern": "omni",
                    },
                    "absorbtion_params": {
                        "abs_weights": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                        "T60": 0.1,
                        "att_diff": 15.0,
                        "att_max": 60.0,
                    },
                },
            },
        }
    }
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return cfg_path


def run_nemo_simulator(name):
    if not NEMO.exists():
        run(f"git clone --depth 1 https://github.com/NVIDIA/NeMo.git {NEMO}")
    run(f"python tools/speech_data_simulator/multispeaker_simulator.py "
        f"--config-path {SIM} --config-name {name}.yaml", cwd=str(NEMO))


def run_one(name, source_manifest):
    params = SIM_CONFIGS[name]
    est_h = params["n_sessions"] * params["length"] / 3600

    write_nemo_config(name, params, source_manifest)
    run_nemo_simulator(name)

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