import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from config import RAW

AMI_BASE = "https://groups.inf.ed.ac.uk/ami"
AUDIO_BASE = f"{AMI_BASE}/AMICorpusMirror/amicorpus"
ANNOTATIONS_URL = f"{AMI_BASE}/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"

OUT_DIR = RAW / "ami"
AUDIO_DIR = OUT_DIR / "audio"
RTTM_DIR = OUT_DIR / "rttm"
ANNOT_DIR = OUT_DIR / "annotations"


def all_meeting_ids():
    ids = []
    for prefix, start, end in [("ES", 2002, 2016), ("IS", 1000, 1009), ("TS", 3003, 3012)]:
        for num in range(start, end + 1):
            for part in "abcd":
                ids.append(f"{prefix}{num}{part}")
    ids += [
        "EN2001a", "EN2001b", "EN2001d", "EN2001e",
        "EN2002a", "EN2002b", "EN2002c", "EN2002d",
        "EN2003a", "EN2004a", "EN2005a",
        "EN2006a", "EN2006b",
        "EN2009b", "EN2009c", "EN2009d",
        "IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011",
        "IN1001", "IN1002", "IN1005", "IN1007", "IN1008", "IN1009",
        "IN1012", "IN1013", "IN1014", "IN1016",
    ]
    return ids


def download_audio(meeting_ids):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, mid in enumerate(meeting_ids):
        wav_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        if wav_path.exists():
            continue
        url = f"{AUDIO_BASE}/{mid}/audio/{mid}.Mix-Headset.wav"
        ret = subprocess.run(
            ["wget", "-q", "--tries=3", "--timeout=30", "-O", str(wav_path), url],
            capture_output=True,
        )
        if ret.returncode != 0:
            wav_path.unlink(missing_ok=True)
            failed.append(mid)
        if (i + 1) % 20 == 0:
            print(f"  audio: {i+1}/{len(meeting_ids)}")
    print(f"audio done: {len(meeting_ids) - len(failed)} ok, {len(failed)} failed")
    if failed:
        print(f"  failed: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    return failed


def download_annotations():
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUT_DIR / "annotations.zip"
    if not zip_path.exists():
        subprocess.run(
            ["wget", "-q", "--tries=3", "-O", str(zip_path), ANNOTATIONS_URL],
            check=True,
        )
    subprocess.run(
        ["unzip", "-o", "-q", str(zip_path), "segments/*", "-d", str(ANNOT_DIR)],
        check=True,
    )
    print(f"annotations extracted to {ANNOT_DIR}")


def parse_segments_xml(xml_path):
    ns = {"nite": "http://nite.sourceforge.net/"}
    tree = ET.parse(xml_path)
    segs = []
    for seg in tree.iter():
        start = seg.get("transcriber_start")
        end = seg.get("transcriber_end")
        if start is not None and end is not None:
            s, e = float(start), float(end)
            if e > s:
                segs.append((s, e))
    return segs


def convert_to_rttm(meeting_ids):
    RTTM_DIR.mkdir(parents=True, exist_ok=True)
    seg_dir = ANNOT_DIR / "segments"
    converted = 0
    for mid in meeting_ids:
        rttm_path = RTTM_DIR / f"{mid}.rttm"
        lines = []
        for xml_file in sorted(seg_dir.glob(f"{mid}.*.segments.xml")):
            spk = xml_file.stem.split(".")[1]
            for start, end in parse_segments_xml(xml_file):
                dur = round(end - start, 3)
                lines.append(f"SPEAKER {mid} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>")
        if lines:
            lines.sort(key=lambda l: float(l.split()[3]))
            rttm_path.write_text("\n".join(lines) + "\n")
            converted += 1
    print(f"rttm: {converted} meetings converted to {RTTM_DIR}")


EVAL_MEETINGS = {
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "IS1009a", "IS1009b", "IS1009c", "IS1009d",
    "TS3003a", "TS3003b", "TS3003c", "TS3003d",
    "EN2002a", "EN2002b", "EN2002c", "EN2002d",
}
DEV_MEETINGS = {
    "ES2011a", "ES2011b", "ES2011c", "ES2011d",
    "IS1008a", "IS1008b", "IS1008c", "IS1008d",
    "TS3004a", "TS3004b", "TS3004c", "TS3004d",
    "IB4005",
    "EN2001a", "EN2001b", "EN2001d", "EN2001e",
}


def write_split_lists(meeting_ids):
    train = [m for m in meeting_ids if m not in EVAL_MEETINGS and m not in DEV_MEETINGS]
    dev = [m for m in meeting_ids if m in DEV_MEETINGS]
    evl = [m for m in meeting_ids if m in EVAL_MEETINGS]
    for name, ids in [("train", train), ("dev", dev), ("eval", evl)]:
        (OUT_DIR / f"{name}.list").write_text("\n".join(sorted(ids)) + "\n")
    print(f"splits: train={len(train)}, dev={len(dev)}, eval={len(evl)}")


if __name__ == "__main__":
    meeting_ids = all_meeting_ids()
    print(f"AMI corpus: {len(meeting_ids)} meetings")

    print("downloading annotations...")
    download_annotations()

    print("downloading audio (headset mix, ~5GB total)...")
    failed = download_audio(meeting_ids)

    available = [m for m in meeting_ids if (AUDIO_DIR / f"{m}.Mix-Headset.wav").exists()]
    print(f"converting {len(available)} meetings to RTTM...")
    convert_to_rttm(available)
    write_split_lists(available)

    print(f"\ndone. output structure:")
    print(f"  {AUDIO_DIR}/  — {len(list(AUDIO_DIR.glob('*.wav')))} WAV files")
    print(f"  {RTTM_DIR}/   — RTTM files")
    print(f"  {OUT_DIR}/    — train.list, dev.list, eval.list")

