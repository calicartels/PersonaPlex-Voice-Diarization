import numpy as np
import torch
import torch.nn as nn
import os
import config

PROBE_CACHE = os.path.join(config.CACHE_DIR, "probe_weights.npz")

# Choice: Adam over SGD. Standard for quick convergence on small data.
# SGD would need LR tuning and momentum selection for no benefit here.
LR = 1e-3

# Choice: 100 epochs. 4874 samples with a linear layer converges in ~30,
# but 100 is cheap insurance. Training takes <1 min on CPU.
EPOCHS = 100

# Choice: 256. Fits the 4874 samples in ~19 batches.
# Full-batch (4874) also works but mini-batch gives smoother gradients.
BATCH_SIZE = 256


def parse_speaker_id(uid):
    return uid.split("+")[0]


def train_probe(embeddings, force=False):
    if os.path.exists(PROBE_CACHE) and not force:
        print(f"loading cached probe weights from {PROBE_CACHE}")
        data = np.load(PROBE_CACHE)
        return data["W"], data["b"]

    uids = list(embeddings.keys())
    spk_ids = [parse_speaker_id(u) for u in uids]
    unique_spk = sorted(set(spk_ids))
    spk2idx = {s: i for i, s in enumerate(unique_spk)}
    num_speakers = len(unique_spk)
    print(f"training linear probe: {len(uids)} utterances, {num_speakers} speakers")

    X = np.stack([embeddings[u] for u in uids])
    y = np.array([spk2idx[s] for s in spk_ids])

    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()

    # Choice: projection dim = num_speakers (40 for VoxCeleb1 test).
    # This is the standard linear probe — one weight per speaker.
    # Alternative was an arbitrary wider projection (e.g. 128, 256) but
    # that changes the question from "linearly separable?" to
    # "separable with a learned feature transform?" which is a different test.
    model = nn.Linear(config.EMBED_DIM, num_speakers)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(X_t))
        X_shuf = X_t[perm]
        y_shuf = y_t[perm]

        total_loss = 0.0
        correct = 0

        for i in range(0, len(X_t), BATCH_SIZE):
            xb = X_shuf[i:i + BATCH_SIZE]
            yb = y_shuf[i:i + BATCH_SIZE]

            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            acc = correct / len(X_t) * 100
            avg_loss = total_loss / len(X_t)
            print(f"  epoch {epoch + 1:3d}: loss={avg_loss:.4f}  acc={acc:.1f}%")

    W = model.weight.detach().numpy()
    b = model.bias.detach().numpy()

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    np.savez(PROBE_CACHE, W=W, b=b)
    print(f"cached probe weights to {PROBE_CACHE}")

    return W, b

