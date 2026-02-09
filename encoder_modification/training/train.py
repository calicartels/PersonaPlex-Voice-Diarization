import torch
import json
from pathlib import Path
from config import MANIFESTS, CKPT, LR, WEIGHT_DECAY, WARMUP, MAX_EPOCHS, VAL_EVERY, D_MODEL, N_HEADS, FF_DIM, N_LAYERS, DROPOUT, ALPHA, BATCH
from model import MimiSpeaker, hybrid_loss
from load import make_loader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed")


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


def save_checkpoint(model, opt, sched, epoch, step, best_val, train_loss, val_loss, input_dim, is_best=False):
    checkpoint = {
        "epoch": epoch + 1,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "best_val_loss": best_val,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": {
            "input_dim": input_dim,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "ff_dim": FF_DIM,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup": WARMUP,
            "alpha": ALPHA,
            "batch_size": BATCH,
        }
    }
    
    if is_best:
        torch.save(checkpoint, CKPT / "best.pt")
        torch.save(checkpoint, CKPT / f"best_epoch_{epoch+1}_val_{val_loss:.4f}.pt")
    
    if (epoch + 1) % 5 == 0:
        torch.save(checkpoint, CKPT / f"epoch_{epoch+1}.pt")
    
    torch.save(checkpoint, CKPT / "latest.pt")


def load_checkpoint(checkpoint_path, model, opt=None, sched=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if opt is not None:
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
    if sched is not None:
        sched.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


device = "cuda" if torch.cuda.is_available() else "cpu"
CKPT.mkdir(parents=True, exist_ok=True)

if WANDB_AVAILABLE:
    wandb.init(
        project="mimi-speaker-diarization",
        config={
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "dropout": DROPOUT,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup": WARMUP,
            "alpha": ALPHA,
            "batch_size": BATCH,
            "max_epochs": MAX_EPOCHS,
        }
    )

train_loader = make_loader(MANIFESTS / "train.json", shuffle=True)
val_loader = make_loader(MANIFESTS / "val.json", shuffle=False)

sample_emb = next(iter(train_loader))[0]
input_dim = sample_emb.shape[1]

model = MimiSpeaker(input_dim=input_dim).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"MimiSpeaker: {n_params:,} params, input_dim={input_dim}")

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

start_epoch = 0
step = 0
best_val = float("inf")

resume_path = CKPT / "latest.pt"
if resume_path.exists():
    print(f"Resuming from {resume_path}")
    ckpt = load_checkpoint(resume_path, model, opt, sched)
    start_epoch = ckpt["epoch"]
    step = ckpt["step"]
    best_val = ckpt["best_val_loss"]
    print(f"Resumed: epoch={start_epoch}, step={step}, best_val={best_val:.4f}")

patience, wait = 5, 0

for epoch in range(start_epoch, MAX_EPOCHS):
    model.train()
    ep_loss, ep_n = 0.0, 0

    for emb, labels, _ in train_loader:
        emb, labels = emb.to(device), labels.to(device)
        loss = hybrid_loss(model(emb), labels)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        ep_loss += loss.item()
        ep_n += 1
        step += 1

        if WANDB_AVAILABLE:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": opt.param_groups[0]["lr"],
                "step": step,
            }, step=step)

        if step % VAL_EVERY == 0:
            vl = validate(model, val_loader, device)
            lr = opt.param_groups[0]["lr"]
            print(f"  step {step}: train={ep_loss/ep_n:.4f} val={vl:.4f} lr={lr:.2e}")
            if WANDB_AVAILABLE:
                wandb.log({"val/loss": vl, "val/step": step}, step=step)

    vl = validate(model, val_loader, device)
    train_loss = ep_loss / ep_n
    print(f"Epoch {epoch+1}: train={train_loss:.4f} val={vl:.4f}")

    is_best = vl < best_val
    if is_best:
        best_val = vl
        wait = 0
    else:
        wait += 1

    save_checkpoint(model, opt, sched, epoch, step, best_val, train_loss, vl, input_dim, is_best=is_best)

    if WANDB_AVAILABLE:
        wandb.log({
            "epoch": epoch + 1,
            "epoch/train_loss": train_loss,
            "epoch/val_loss": vl,
            "epoch/best_val_loss": best_val,
        }, step=step)

    if wait >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

if WANDB_AVAILABLE:
    wandb.finish()

print(f"Best val loss: {best_val:.4f} -> {CKPT / 'best.pt'}")
