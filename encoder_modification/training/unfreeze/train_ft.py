"""Training loop with partial Mimi unfreezing and differential learning rates."""
import sys, os
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MANIFESTS, CKPT, LR, WEIGHT_DECAY, WARMUP, MAX_EPOCHS, VAL_EVERY, PERSONAPLEX_REPO, MIMI_CHECKPOINT

    sys.path.insert(0, PERSONAPLEX_REPO)
    from moshi.models.loaders import get_mimi
from model_ft import MimiSpeakerFT, hybrid_loss
from load_ft import make_loader_ft

try:
    import wandb
    wandb.init(project="mimi-speaker-diarization-ft")
    USE_WANDB = True
except ImportError:
    USE_WANDB = False

# Choice: 1e-5 for Mimi layers, 1e-4 for adapter. Lower LR prevents
# catastrophic forgetting of pretrained representations.
MIMI_LR = 1e-5
N_UNFREEZE = 3


def lr_fn(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    return (WARMUP ** 0.5) * (step ** -0.5)


def validate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for emb, labels, _ in loader:
            emb, labels = emb.to(device), labels.to(device)
            total += hybrid_loss(model(emb), labels).item()
            n += 1
    model.train()
    return total / max(n, 1)


device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT.mkdir(parents=True, exist_ok=True)

mimi = get_mimi(MIMI_CHECKPOINT, device=device)
mimi.eval()

train_mf = MANIFESTS / "train_ft.json"
val_mf = MANIFESTS / "val_ft.json"
if not train_mf.exists():
    train_mf = MANIFESTS / "train.json"
    val_mf = MANIFESTS / "val.json"

train_loader = make_loader_ft(train_mf, shuffle=True)
val_loader = make_loader_ft(val_mf, shuffle=False)

model = MimiSpeakerFT(mimi.encoder_transformer, n_unfreeze=N_UNFREEZE, input_dim=512).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"MimiSpeakerFT: {n_params:,} trainable params, {N_UNFREEZE} Mimi layers unfrozen")

# Choice: separate param groups with different LRs.
opt = torch.optim.AdamW([
    {"params": model.mimi_params(), "lr": MIMI_LR},
    {"params": model.adapter_params(), "lr": LR},
], weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

best_val = float("inf")
patience, wait = 10, 0
step = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    ep_loss, ep_n = 0.0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", unit="batch")
    for emb, labels, _ in pbar:
        emb, labels = emb.to(device), labels.to(device)
        loss = hybrid_loss(model(emb), labels)

        opt.zero_grad()
        loss.backward()
        # Choice: tighter clip for Mimi params (0.5) vs adapter (1.0)
        # to prevent destroying pretrained representations.
        torch.nn.utils.clip_grad_norm_(model.mimi_params(), 0.5)
        torch.nn.utils.clip_grad_norm_(model.adapter_params(), 1.0)
        opt.step()
        sched.step()

        ep_loss += loss.item()
        ep_n += 1
        step += 1
        pbar.set_postfix(loss=f"{ep_loss/ep_n:.4f}", lr=f"{opt.param_groups[1]['lr']:.2e}")

        if USE_WANDB:
            wandb.log({"train/loss": loss.item(), "train/lr": opt.param_groups[1]["lr"]}, step=step)

        if step % VAL_EVERY == 0:
            vl = validate(model, val_loader, device)
            tqdm.write(f"step {step}: train={ep_loss/ep_n:.4f} val={vl:.4f}")
            if USE_WANDB:
                wandb.log({"val/loss": vl}, step=step)

    vl = validate(model, val_loader, device)
    train_loss = ep_loss / ep_n
    print(f"Epoch {epoch+1}: train={train_loss:.4f} val={vl:.4f}")

    checkpoint = {
        "epoch": epoch + 1, "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "best_val_loss": best_val, "train_loss": train_loss, "val_loss": vl,
        "n_unfreeze": N_UNFREEZE,
    }

    if vl < best_val:
        best_val = vl
        wait = 0
        torch.save(checkpoint, CKPT / "best_ft.pt")
    else:
        wait += 1

    torch.save(checkpoint, CKPT / "latest_ft.pt")

    if wait >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

if USE_WANDB:
    wandb.finish()

print(f"Best val loss: {best_val:.4f} -> {CKPT / 'best_ft.pt'}")
