import torch
from config import MANIFESTS, CKPT, LR, WEIGHT_DECAY, WARMUP, MAX_EPOCHS, VAL_EVERY
from model import MimiSpeaker, hybrid_loss
from load import make_loader


def lr_fn(step):
    # Choice: inverse square root with linear warmup. Sortformer's exact schedule.
    # Alternative: cosine annealing (PersonaPlex uses it), marginal difference.
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

train_loader = make_loader(MANIFESTS / "train.json", shuffle=True)
val_loader = make_loader(MANIFESTS / "val.json", shuffle=False)

# Auto-detect Mimi embedding dim from first batch
sample_emb = next(iter(train_loader))[0]
input_dim = sample_emb.shape[1]

model = MimiSpeaker(input_dim=input_dim).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"MimiSpeaker: {n_params:,} params, input_dim={input_dim}")

# Choice: AdamW matching Sortformer. Alternative: Adam, but weight decay helps.
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

best_val = float("inf")
# Choice: patience=5 for early stopping. Prevents overfitting 8M params on 500h.
# Alternative: no early stopping, but wastes compute if loss plateaus.
patience, wait = 5, 0
step = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    ep_loss, ep_n = 0.0, 0

    for emb, labels, _ in train_loader:
        emb, labels = emb.to(device), labels.to(device)
        loss = hybrid_loss(model(emb), labels)

        opt.zero_grad()
        loss.backward()
        # Choice: grad clip 1.0, standard for Transformers.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        ep_loss += loss.item()
        ep_n += 1
        step += 1

        if step % VAL_EVERY == 0:
            vl = validate(model, val_loader, device)
            lr = opt.param_groups[0]["lr"]
            print(f"  step {step}: train={ep_loss/ep_n:.4f} val={vl:.4f} lr={lr:.2e}")

    vl = validate(model, val_loader, device)
    print(f"Epoch {epoch+1}: train={ep_loss/ep_n:.4f} val={vl:.4f}")

    if vl < best_val:
        best_val = vl
        wait = 0
        torch.save(model.state_dict(), CKPT / "best.pt")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), CKPT / f"epoch_{epoch+1}.pt")

print(f"Best val loss: {best_val:.4f} -> {CKPT / 'best.pt'}")