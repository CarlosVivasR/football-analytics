"""Scoring rules for 1X2 forecasts and a paired bootstrap for comparing them.

P is an (n, 3) array of home/draw/away probabilities, y the observed outcome
coded 0/1/2. All per-match scores are lower-is-better.
"""

import numpy as np


def _one_hot(y, k=3):
    oh = np.zeros((len(y), k))
    oh[np.arange(len(y)), y] = 1.0
    return oh


def rps(P, y):
    """Ranked probability score per match."""
    diff = np.cumsum(P, axis=1) - np.cumsum(_one_hot(y, P.shape[1]), axis=1)
    return np.sum(diff ** 2, axis=1) / (P.shape[1] - 1)


def brier(P, y):
    return np.sum((P - _one_hot(y, P.shape[1])) ** 2, axis=1)


def log_loss(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def ece_home_win(P, y, n_bins=10):
    """Expected calibration error of the home-win probability, quantile bins."""
    order = np.argsort(P[:, 0])
    p = P[order, 0]
    hit = (np.asarray(y)[order] == 0).astype(float)
    ece = 0.0
    for b in np.array_split(np.arange(len(p)), n_bins):
        if len(b):
            ece += len(b) / len(p) * abs(p[b].mean() - hit[b].mean())
    return float(ece)


def reliability(P, y, edges=(1 / 3, 0.40, 0.50, 0.60, 0.70, 0.80, 1.0001)):
    """Mean confidence vs accuracy of the top pick, binned by confidence."""
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == np.asarray(y)).astype(float)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            rows.append({"lo": round(float(lo), 3), "hi": round(float(min(hi, 1.0)), 3),
                         "pred": round(float(conf[m].mean()), 4),
                         "obs": round(float(correct[m].mean()), 4), "n": int(m.sum())})
    return rows


def summarise(P, y):
    return {
        "rps": round(float(np.mean(rps(P, y))), 5),
        "brier": round(float(np.mean(brier(P, y))), 5),
        "logloss": round(float(np.mean(log_loss(P, y))), 5),
        "acc": round(float(np.mean(np.argmax(P, axis=1) == y)), 5),
        "ece": round(ece_home_win(P, y), 5),
        "n": int(len(y)),
    }


def paired_bootstrap(score_a, score_b, n_boot=5000, seed=20260613):
    """Bootstrap the mean per-match difference a - b over the same matches.

    Returns (mean, ci_low, ci_high, two-sided p). Negative means a is better.
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(score_a) - np.asarray(score_b)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means >= 0).mean(), (means <= 0).mean())
    return float(diff.mean()), float(lo), float(hi), float(min(p, 1.0))
