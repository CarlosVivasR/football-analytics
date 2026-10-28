"""Bookmaker odds: de-vigging, joining to results, and blending with the model."""

import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

from .data import team_key
from .dixon_coles import fit_dixon_coles, predict_1x2
from .scoring import rps

ODDS_COLUMNS = ["date", "home", "away", "home_goals", "away_goals",
                "odds_H_avg", "odds_D_avg", "odds_A_avg"]


def shin_devig(odds):
    """Implied probabilities from decimal 1X2 odds using Shin's (1993) method.

    Falls back to proportional normalisation when the book has no overround or
    the insider share z cannot be bracketed.
    """
    r = 1.0 / np.asarray(odds, dtype=float)
    total = r.sum()
    if total <= 1.0:
        return r / total

    def probs(z):
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * r * r / total) - z) / (2.0 * (1.0 - z))

    def excess(z):
        return probs(z).sum() - 1.0

    f_lo, f_hi = excess(0.0), excess(0.999)
    if f_lo == 0:
        z = 0.0
    elif np.sign(f_lo) == np.sign(f_hi):
        return r / total
    else:
        z = brentq(excess, 0.0, 0.999, xtol=1e-10)
    p = probs(z)
    return p / p.sum()


def log_pool(p_model, p_market, w):
    """Logarithmic opinion pool; w is the weight on the model (1 = model only)."""
    b = np.clip(p_model, 1e-12, 1.0) ** w * np.clip(p_market, 1e-12, 1.0) ** (1.0 - w)
    return b / b.sum(axis=-1, keepdims=True)


def load_odds(path):
    """Read a user-supplied odds file (see README for the columns)."""
    d = pd.read_csv(path, parse_dates=["date"])
    missing = set(ODDS_COLUMNS) - set(d.columns)
    if missing:
        raise ValueError(f"odds file is missing columns: {sorted(missing)}")
    d = d.dropna(subset=["odds_H_avg", "odds_D_avg", "odds_A_avg"]).copy()
    # Some result strings are awarded or after extra time, so use the goals.
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    d = d.dropna(subset=["home_goals", "away_goals"]).copy()
    hg, ag = d["home_goals"].astype(int), d["away_goals"].astype(int)
    d["y"] = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    if "competition" not in d.columns:
        d["competition"] = ""
    return d


def join_odds(odds, results):
    """Match odds rows to results on date and team keys, in either home/away order.

    `swapped` marks rows where the odds file lists the teams in the opposite
    order to the results file; the results file carries the venue information.
    """
    lookup = {}
    for r in results.itertuples(index=False):
        key = (r.date.strftime("%Y-%m-%d"), team_key(r.home_team), team_key(r.away_team))
        lookup.setdefault(key, (r.home_team, r.away_team, bool(r.neutral)))

    rows = []
    for r in odds.itertuples(index=False):
        day, kh, ka = r.date.strftime("%Y-%m-%d"), team_key(r.home), team_key(r.away)
        swapped = False
        hit = lookup.get((day, kh, ka))
        if hit is None:
            hit = lookup.get((day, ka, kh))
            swapped = hit is not None
        if hit is None:
            continue
        rows.append({"date": r.date, "home": r.home, "away": r.away,
                     "res_home": hit[0], "res_away": hit[1], "neutral": hit[2],
                     "swapped": swapped, "oH": r.odds_H_avg, "oD": r.odds_D_avg,
                     "oA": r.odds_A_avg, "y": int(r.y), "competition": r.competition})
    return pd.DataFrame(rows)


def predict_walk_forward(joined, results, step_days=90, verbose=True, **fit_kw):
    """Dixon-Coles 1X2 for each joined match, in the odds file's home/away order.

    `joined` must already be sorted; the returned rows follow its order. The model
    for each block is fitted on results strictly before the block's origin.
    """
    dates = joined["date"].reset_index(drop=True)
    start = joined["date"].min().normalize()
    end = joined["date"].max().normalize() + pd.Timedelta(days=step_days)
    P = np.full((len(joined), 3), np.nan)
    for origin in pd.date_range(start, end, freq=f"{step_days}D"):
        in_block = (dates >= origin) & (dates < origin + pd.Timedelta(days=step_days))
        pos = np.flatnonzero(in_block.to_numpy())
        if len(pos) == 0:
            continue
        model = fit_dixon_coles(results, origin, maxiter=300, **fit_kw)
        for k in pos:
            m = joined.iloc[k]
            p = predict_1x2(model, m["res_home"], m["res_away"], m["neutral"])
            P[k] = p[::-1] if m["swapped"] else p
        if verbose:
            print(f"  origin {origin.date()}: {len(pos)} matches", file=sys.stderr)
    return P


def tune_weight(P_model, P_market, y, train_frac=0.6):
    """Fit the pool weight on the first part of a chronological sample, score the rest."""
    cut = int(round(len(y) * train_frac))
    train, test = slice(0, cut), slice(cut, len(y))

    def mean_rps(sl, w):
        return float(np.mean(rps(log_pool(P_model[sl], P_market[sl], w), y[sl])))

    grid = np.linspace(0.0, 1.0, 101)
    w0 = grid[int(np.argmin([mean_rps(train, w) for w in grid]))]
    res = minimize(lambda x: mean_rps(train, float(np.clip(x[0], 0, 1))), x0=[w0],
                   method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 500})
    w_star = float(np.clip(res.x[0], 0, 1))
    return {"n_train": cut, "n_test": len(y) - cut, "w_star": round(w_star, 4),
            "test_rps_at_w_star": round(mean_rps(test, w_star), 6),
            "test_rps_model": round(mean_rps(test, 1.0), 6),
            "test_rps_market": round(mean_rps(test, 0.0), 6)}
