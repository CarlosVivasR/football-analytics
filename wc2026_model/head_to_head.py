"""Paired comparison of the walk-forward model against de-vigged bookmaker odds.

    python -m wc2026_model.head_to_head --odds path/to/odds.csv --out outputs/h2h.json

The odds file is not distributed with this repository (see README for the
expected columns). Market probabilities use Shin's de-vig; the blend is a log
opinion pool with weight w on the model.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import dixon_coles as dc
from .data import download_results, load_results
from .market import join_odds, load_odds, log_pool, predict_walk_forward, shin_devig, tune_weight
from .scoring import paired_bootstrap, rps


def by_favourite_strength(P_model, P_market, y, edges=(0, 0.45, 0.60, 0.75, 1.01)):
    fav = P_market.max(axis=1)
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fav > lo) & (fav <= hi)
        if m.any():
            out[f"{lo:.2f}-{min(hi, 1):.2f}"] = {
                "n": int(m.sum()),
                "model": round(float(rps(P_model[m], y[m]).mean()), 4),
                "market": round(float(rps(P_market[m], y[m]).mean()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--odds", required=True, help="CSV of 1X2 odds with results")
    ap.add_argument("--results", default="data/results.csv", help="martj42 results.csv")
    ap.add_argument("--out", default="outputs/h2h.json")
    ap.add_argument("--since", help="score only matches on or after this date")
    ap.add_argument("--w-ref", type=float, default=0.30, help="reference blend weight")
    ap.add_argument("--spread", type=float, default=dc.SPREAD)
    args = ap.parse_args()

    if not Path(args.results).exists():
        download_results(args.results)
    results = load_results(args.results)
    odds = load_odds(args.odds)
    joined = join_odds(odds, results)
    print(f"joined {len(joined)} of {len(odds)} odds rows", file=sys.stderr)

    # Require a year of history before the first forecast.
    joined = joined[joined["date"] > results["date"].min() + pd.Timedelta(days=365)]
    joined = joined.sort_values("date", kind="mergesort").reset_index(drop=True)

    P_market = np.array([shin_devig((r.oH, r.oD, r.oA)) for r in joined.itertuples()])
    P_model = predict_walk_forward(joined, results, spread=args.spread)

    keep = ~np.isnan(P_model).any(axis=1)
    if args.since:
        keep &= (joined["date"] >= pd.Timestamp(args.since)).to_numpy()
    joined, P_model, P_market = joined[keep].reset_index(drop=True), P_model[keep], P_market[keep]
    y = joined["y"].to_numpy()

    base = np.array([(y == k).mean() for k in range(3)])
    P_ref = log_pool(P_model, P_market, args.w_ref)
    scores = {"market": rps(P_market, y), "model": rps(P_model, y), "blend_ref": rps(P_ref, y),
              "climatology": rps(np.tile(base, (len(y), 1)), y)}

    def ci(a, b):
        mean, lo, hi, p = paired_bootstrap(scores[a], scores[b])
        return {"mean": round(mean, 5), "ci95": [round(lo, 5), round(hi, 5)], "p": round(p, 4)}

    out = {
        "n": int(len(y)),
        "date_range": [str(joined["date"].min().date()), str(joined["date"].max().date())],
        "swapped_orientation": int(joined["swapped"].sum()),
        "median_overround": round(float(np.median(1 / joined.oH + 1 / joined.oD + 1 / joined.oA)), 4),
        "outcome_rates_hda": [round(float(b), 4) for b in base],
        "spread": args.spread,
        "w_ref": args.w_ref,
        "rps": {k: round(float(v.mean()), 5) for k, v in scores.items()},
        "paired_bootstrap": {"model_minus_market": ci("model", "market"),
                             "blend_ref_minus_market": ci("blend_ref", "market")},
        "by_market_favourite_prob": by_favourite_strength(P_model, P_market, y),
        "favourite_agreement": round(float((P_model.argmax(1) == P_market.argmax(1)).mean()), 4),
        "weight_tuning": tune_weight(P_model, P_market, y),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("n", "rps", "paired_bootstrap", "weight_tuning")}, indent=2))


if __name__ == "__main__":
    main()
