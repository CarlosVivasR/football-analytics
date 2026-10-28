"""Fit the Dixon-Coles and Elo ratings on all available history and run the backtest.

    python -m wc2026_model.fit_ratings --data data/results.csv --out outputs/ratings.json

The results file is downloaded from the martj42 repository if it is missing.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import dixon_coles as dc
from . import elo
from .backtest import CAL_START, TEST_START, run_backtest
from .data import download_results, load_results


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data/results.csv", help="martj42 results.csv")
    ap.add_argument("--out", default="outputs/ratings.json")
    ap.add_argument("--cutoff", help="only use matches before this date (YYYY-MM-DD)")
    ap.add_argument("--skip-backtest", action="store_true")
    args = ap.parse_args()

    if not Path(args.data).exists():
        print(f"downloading results to {args.data}", file=sys.stderr)
        download_results(args.data)
    df = load_results(args.data, cutoff=args.cutoff)
    snapshot = df["date"].max()
    print(f"{len(df)} matches, {df['date'].min().date()} to {snapshot.date()}", file=sys.stderr)

    model = dc.fit_dixon_coles(df, snapshot + pd.Timedelta(days=1), maxiter=600)
    print(f"Dixon-Coles: mu={model['mu']:.4f} gamma={model['gamma']:.4f} rho={model['rho']:.4f}",
          file=sys.stderr)

    diffs, ratings = elo.replay(df)
    cal = (df["date"] >= CAL_START) & (df["date"] < TEST_START)
    s, tau = elo.fit_ordered_logit(diffs[cal.to_numpy()], df.loc[cal, "outcome"].to_numpy())

    out = {
        "snapshot_date": str(snapshot.date()),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dixon_coles": {
            "mu": round(model["mu"], 4), "gamma": round(model["gamma"], 4),
            "rho": round(model["rho"], 4),
            "half_life_days": dc.HALF_LIFE_DAYS, "friendly_weight": dc.FRIENDLY_WEIGHT,
            "ridge": dc.RIDGE, "spread": dc.SPREAD, "max_goals": dc.MAX_GOALS,
            "teams": {t: {"atk": round(float(model["atk"][t]), 4),
                          "def": round(float(model["def"][t]), 4)} for t in model["atk"]},
        },
        "elo": {
            "hfa_points": elo.ELO_HFA, "divisor": elo.ELO_DIVISOR,
            "s": round(s, 4), "tau": round(tau, 4),
            "ratings": {t: round(float(r), 1) for t, r in sorted(ratings.items())},
        },
    }

    if not args.skip_backtest:
        print("walk-forward backtest", file=sys.stderr)
        out["backtest"] = run_backtest(df)
        for label in ("overall", "major"):
            for name, m in out["backtest"][label].items():
                print(f"{label:8s} {name:12s} RPS={m['rps']:.5f} acc={m['acc']:.3f} n={m['n']}")
        for k, v in out["backtest"]["paired_bootstrap_rps"].items():
            print(f"{k:32s} {v['mean']:+.5f} CI {v['ci95']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
