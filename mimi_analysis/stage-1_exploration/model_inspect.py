import sys
from pathlib import Path
import torch
from typing import Tuple, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "moshi"))
from moshi.models.loaders import get_mimi

def inspect_mimi(weights_path: str | Path, output_path: str | Path | None = None) -> Tuple[Any, torch.Tensor, Dict[str, Any]]:
    """Inspect Mimi model weights and extract baseline features."""
    weights_path = Path(weights_path)
    mimi = get_mimi(str(weights_path), device='cpu')
    mimi.eval()
    
    dummy_audio = torch.randn(1, 1, 24000)
    with torch.no_grad():
        features = mimi._encode_to_unquantized_latent(dummy_audio)
    
    stats = {
        'shape': list(features.shape),
        'mean': features.mean().item(),
        'std': features.std().item(),
        'min': features.min().item(),
        'max': features.max().item(),
    }
    
    if output_path:
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
    
    return mimi, features, stats
