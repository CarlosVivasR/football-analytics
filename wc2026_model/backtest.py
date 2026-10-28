"""Walk-forward backtest of the Dixon-Coles model against Elo and climatology.

The model is refitted every `step_days` on all matches strictly before the
origin and used to predict the next block, so no forecast sees its own result.
Elo rating differences come from a single chronological replay, which is
pre-match by construction; the ordered-logit map is fitted on the calibration
era before the test period.
"""

import sys

import numpy as np
import pandas as pd

from . import elo, scoring
from .dixon_coles import fit_dixon_coles, predict_1x2

TEST_START = pd.Timestamp("2022-01-01")
CAL_START = pd.Timestamp("2018-01-01")
REFIT_STEP_DAYS = 90


def walk_forward(df, test_start=TEST_START, step_days=REFIT_STEP_DAYS, verbose=True, **fit_kw):
    """Return (P, rows): 1X2 forecasts and the df index of each forecast match."""
    origins = pd.date_range(test_start, df["date"].max(), freq=f"{step_days}D")
    probs, rows = [], []
    for origin in origins:
        block = df[(df["date"] >= origin) & (df["date"] < origin + pd.Timedelta(days=step_days))]
        if block.empty:
            continue
        model = fit_dixon_coles(df, origin, maxiter=300, **fit_kw)
        for i, m in block.iterrows():
            probs.append(predict_1x2(model, m["home_team"], m["away_team"], m["neutral"]))
            rows.append(i)
        if verbose:
            print(f"  origin {origin.date()}: {len(block)} matches (total {len(rows)})",
                  file=sys.stderr)
    return np.array(probs), np.array(rows)


def climatology(df, test_start=TEST_START, years=20):
    pre = df[(df["date"] < test_start) & (df["date"] >= test_start - pd.Timedelta(days=365 * years))]
    return np.array([(pre["outcome"] == k).mean() for k in range(3)])


def run_backtest(df, test_start=TEST_START, cal_start=CAL_START, **fit_kw):
    """Score Dixon-Coles, Elo and climatology on the same test matches."""
    P_dc, rows = walk_forward(df, test_start, **fit_kw)
    y = df.loc[rows, "outcome"].to_numpy()
    major = df.loc[rows, "is_major"].to_numpy()

    diffs, _ = elo.replay(df)
    cal = (df["date"] >= cal_start) & (df["date"] < test_start)
    s, tau = elo.fit_ordered_logit(diffs[cal.to_numpy()], df.loc[cal, "outcome"].to_numpy())
    P_elo = elo.elo_probs(diffs[rows], s, tau)
    P_clim = np.tile(climatology(df, test_start), (len(y), 1))

    out = {"overall": {}, "major": {}}
    for name, P in (("dixon_coles", P_dc), ("elo", P_elo), ("climatology", P_clim)):
        out["overall"][name] = scoring.summarise(P, y)
        out["major"][name] = scoring.summarise(P[major], y[major])

    rps_dc = scoring.rps(P_dc, y)
    comparisons = {}
    for name, P in (("elo", P_elo), ("climatology", P_clim)):
        for label, mask in (("overall", slice(None)), ("major", major)):
            mean, lo, hi, p = scoring.paired_bootstrap(rps_dc[mask], scoring.rps(P, y)[mask])
            comparisons[f"dc_minus_{name}_{label}"] = {
                "mean": round(mean, 5), "ci95": [round(lo, 5), round(hi, 5)], "p": round(p, 4)}
    out["paired_bootstrap_rps"] = comparisons
    out["reliability_dc"] = scoring.reliability(P_dc, y)
    out["elo_ordered_logit"] = {"s": round(s, 4), "tau": round(tau, 4)}
    out["protocol"] = (f"walk-forward, refit every {REFIT_STEP_DAYS} days on data before the origin; "
                       f"test from {test_start.date()}; Elo map fitted on {cal_start.date()} to "
                       f"{test_start.date()}")
    return out
