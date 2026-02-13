import numpy as np


def compute_der(pred, labels, threshold=0.5, collar=0.0, frame_dur=0.08):
    # Choice: collar=0.0 for strict evaluation during training.
    # DIHARD uses 0.0, CALLHOME uses 0.25. We use 0.0 to
    # not mask errors — we want to see real performance.
    pred_bin = (pred > threshold).astype(np.float32)

    ref_count = labels.sum(axis=1)
    hyp_count = pred_bin.sum(axis=1)
    correct = (labels * pred_bin).sum(axis=1)

    total_speech = ref_count.sum()
    if total_speech == 0:
        return 0.0

    missed = (ref_count - correct).sum()       # speakers we missed
    false_alarm = (hyp_count - correct).sum()  # speakers we hallucinated
    confusion = min(missed, false_alarm)       # wrong speaker assignments
    missed -= confusion
    false_alarm -= confusion

    der = (missed + false_alarm + confusion) / total_speech
    return float(der)


def eval_batch(pred, labels, mask=None, threshold=0.5):
    pred_np = pred.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    ders = []
    for b in range(pred_np.shape[0]):
        if mask is not None:
            valid = ~mask[b].cpu().numpy()
            p = pred_np[b][valid]
            l = labels_np[b][valid]
        else:
            p = pred_np[b]
            l = labels_np[b]

        ders.append(compute_der(p, l, threshold=threshold))

    return np.mean(ders)
