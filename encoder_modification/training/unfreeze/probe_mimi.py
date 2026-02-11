import sys, os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PERSONAPLEX_REPO, MIMI_CHECKPOINT

sys.path.insert(0, PERSONAPLEX_REPO)
from moshi.models.loaders import get_mimi

device = "cuda" if torch.cuda.is_available() else "cpu"
mimi = get_mimi(MIMI_CHECKPOINT, device=device)
mimi.eval()

print("=== Mimi top-level modules ===")
for name, _ in mimi.named_children():
    print(f"  {name}")

print("\n=== encoder_transformer structure ===")
et = mimi.encoder_transformer
print(f"Type: {type(et)}")
for name, mod in et.named_children():
    print(f"  {name}: {type(mod).__name__}")

print("\n=== encoder_transformer layer list ===")
for attr in ["layers", "transformer", "blocks", "encoder"]:
    if hasattr(et, attr):
        layers = getattr(et, attr)
        print(f"  Found: et.{attr}, type={type(layers).__name__}, len={len(layers) if hasattr(layers, '__len__') else 'N/A'}")
        if hasattr(layers, '__len__'):
            for i, layer in enumerate(layers):
                n_params = sum(p.numel() for p in layer.parameters())
                print(f"    layer[{i}]: {type(layer).__name__}, params={n_params:,}")

print("\n=== Parameter counts ===")
enc_params = sum(p.numel() for p in mimi.encoder.parameters())
et_params = sum(p.numel() for p in et.parameters())
print(f"  encoder (conv): {enc_params:,}")
print(f"  encoder_transformer: {et_params:,}")

print("\n=== encoder output shape ===")
test = torch.randn(1, 1, 24000).to(device)
with torch.no_grad():
    conv_out = mimi.encoder(test)
    print(f"  encoder output: {conv_out.shape}")
    tf_out = et(conv_out)
    if isinstance(tf_out, (list, tuple)):
        print(f"  encoder_transformer output: list of {len(tf_out)}, first={tf_out[0].shape}")
    else:
        print(f"  encoder_transformer output: {tf_out.shape}")

print("\n=== Full named_parameters (first 30) ===")
params = list(et.named_parameters())
for i, (name, p) in enumerate(params):
    if i >= 30:
        print(f"  ... ({len(params)} total)")
        break
    print(f"  {name}: {p.shape}")
