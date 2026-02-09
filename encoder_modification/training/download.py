import subprocess
import sys
from pathlib import Path
from tqdm import tqdm
from config import RAW, NEMO


def run(cmd):
    subprocess.run(cmd, shell=True, check=True)


def download_librispeech():
    out = RAW / "librispeech"
    out.mkdir(parents=True, exist_ok=True)
    target = out / "LibriSpeech" / "train-clean-100"
    if target.exists():
        return target
    # Choice: train-clean-100 only (28GB, 251 speakers).
    # Alternative: add train-clean-360 for more speaker diversity,
    # but 251 speakers is enough to simulate 500h with varied voices.
    url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    tar_path = out / "tc100.tar.gz"
    print("  Downloading LibriSpeech (28GB)")
    run(f"wget --progress=bar:force -c {url} -O {tar_path}")
    print("  Extracting...")
    run(f"tar -xzf {tar_path} -C {out}")
    tar_path.unlink()
    return target


def download_ami():
    out = RAW / "ami"
    audio_dir = out / "audio"
    rttm_dir = out / "rttm"
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    rttm_dir.mkdir(parents=True, exist_ok=True)
    
    wavs = list(audio_dir.glob("*.wav"))
    rttms = list(rttm_dir.glob("*.rttm"))
    
    if wavs and rttms:
        print(f"  AMI already downloaded: {len(wavs)} WAVs, {len(rttms)} RTTMs")
        return out
    
    print("  Running AMI download script")
    subprocess.run([sys.executable, str(Path(__file__).parent / "download_ami.py")], check=False)
    return out


def download_voxconverse():
    out = RAW / "voxconverse"
    out.mkdir(parents=True, exist_ok=True)
    repo = out / "voxconverse"
    if not repo.exists():
        run(f"git clone --depth 1 https://github.com/joonson/voxconverse.git {repo}")
    run("pip install yt-dlp -q")
    audio = out / "audio"
    audio.mkdir(exist_ok=True)
    run(f"cd {repo} && python download_videos.py --save_path {audio} || true")
    return out


def clone_nemo():
    if NEMO.exists():
        return NEMO
    print("  Cloning NeMo repository...")
    run(f"git clone --progress --depth 1 https://github.com/NVIDIA/NeMo.git {NEMO}")
    return NEMO


print("Downloading LibriSpeech...")
print(f"  -> {download_librispeech()}")

print("Downloading AMI...")
print(f"  -> {download_ami()}")

print("Downloading VoxConverse...")
print(f"  -> {download_voxconverse()}")

print("Cloning NeMo...")
print(f"  -> {clone_nemo()}")