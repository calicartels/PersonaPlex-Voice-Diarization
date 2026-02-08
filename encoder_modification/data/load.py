from datasets import load_dataset
import config


def load_voxceleb():
    # Choice: streaming=False because we need random access for pair evaluation.
    # 4874 utterances, ~1.3GB download, cached by HF after first run.
    # Alternative: streaming=True saves disk but requires sequential access
    # and re-download each run. Not worth it for 1.3GB.
    print(f"loading {config.HF_DATASET} split={config.HF_SPLIT}")
    ds = load_dataset(config.HF_DATASET, split=config.HF_SPLIT, cache_dir=config.CACHE_DIR)
    print(f"loaded {len(ds)} utterances")
    return ds


def get_speaker_id(utterance_id):
    # id format: "id10270+5r0dWxy17C8+00001.wav"
    return utterance_id.split("+")[0]


def normalize_pairs_path(path):
    # pairs file uses "/": "id10270/5r0dWxy17C8/00001.wav"
    # dataset uses "+": "id10270+5r0dWxy17C8+00001.wav"
    return path.replace("/", "+")