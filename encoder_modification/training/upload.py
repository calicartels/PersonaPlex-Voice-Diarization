import os
from config import CKPT


def upload_to_hf(repo_id, checkpoint_path, token=None):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=f"checkpoints/{checkpoint_path.name}",
        repo_id=repo_id,
        repo_type="model",
    )


def upload_best(repo_id=None, token=None):
    best = CKPT / "best.pt"
    if not best.exists():
        return False

    if not repo_id:
        return False

    upload_to_hf(repo_id, best, token)
    return True


repo_id = os.getenv("HF_REPO")
token = os.getenv("HF_TOKEN")
upload_best(repo_id, token)
