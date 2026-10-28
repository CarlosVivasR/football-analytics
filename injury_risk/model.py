"""Primary Cox model (M1) for the first severe overuse injury in the window.

    python -m injury_risk.model --processed data/processed --out outputs

Unit: one player at one tournament. Time runs in days from the start of the
observation window; players without an event are censored at its end.
"""

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .tournaments import TOP5_COMPS, window

PENALIZER = 0.05
DURATION, EVENT = "time_to_event_days", "had_event"
COVARIATES = [
    "age_at_tournament",
    "log_market_value",
    "is_top5_player",
    "club_minutes_12m",
    "log_prior_inj_24m",
    "prior_severe_count_24m",
    "has_hamstring_history",
    "has_acl_history",
    "had_surgery_24m",
    "recurrence_history_count",
    "days_since_last_injury_capped",
]


def survival_table(features, history, outcomes):
    df = features.merge(history, on=["tournament_id", "player_id"], how="left")
    for c in history.columns.drop(["tournament_id", "player_id"]):
        df[c] = df[c].fillna(99999 if c.startswith("days_since") else 0)

    primary = outcomes[(outcomes["mechanism"] == "overuse") & (outcomes["is_severe"] == True)].copy()  # noqa: E712
    primary["from_dt"] = pd.to_datetime(primary["from_dt"], errors="coerce")
    first = primary.sort_values("from_dt").groupby(["player_id", "tournament_id"]).first().reset_index()

    df["window_start"] = df["tournament_id"].map(lambda t: window(t)[0])
    df["window_days"] = (df["tournament_id"].map(lambda t: window(t)[1]) - df["window_start"]).dt.days
    df = df.merge(first[["player_id", "tournament_id", "from_dt"]].rename(columns={"from_dt": "event_date"}),
                  on=["player_id", "tournament_id"], how="left")
    df[EVENT] = df["event_date"].notna()
    df[DURATION] = ((df["event_date"] - df["window_start"]).dt.days
                    .where(df[EVENT], df["window_days"]).clip(lower=1))
    return df


def m1_design(df):
    """Add the M1 covariates and drop incomplete rows. Returns (table, covariate names)."""
    df = df.copy()
    df["log_market_value"] = np.log1p(df["market_value_at_tournament_eur"].fillna(0))
    df["is_top5_player"] = df["current_club_competition_id"].isin(TOP5_COMPS).astype(int)
    df["log_prior_inj_24m"] = np.log1p(df["prior_inj_count_24m"])
    df["days_since_last_injury_capped"] = df["days_since_last_injury"].clip(upper=730)
    df = pd.get_dummies(df, columns=["position_broad"], prefix="pos", drop_first=True)
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    covariates = COVARIATES + pos_cols
    df = df.dropna(subset=covariates + [DURATION, EVENT]).copy()
    df[pos_cols] = df[pos_cols].astype(int)
    df[EVENT] = df[EVENT].astype(int)
    return df, covariates


def load_m1(processed):
    p = Path(processed)
    df = survival_table(pd.read_csv(p / "features.csv", low_memory=False),
                        pd.read_csv(p / "history.csv"),
                        pd.read_csv(p / "outcomes.csv", low_memory=False))
    return m1_design(df)


def fit_cox(df, covariates):
    cph = CoxPHFitter(penalizer=PENALIZER)
    cph.fit(df[covariates + [DURATION, EVENT]], duration_col=DURATION, event_col=EVENT)
    return cph


def forest_plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = summary.sort_values("exp(coef)")
    hr, lo, hi = s["exp(coef)"], s["exp(coef) lower 95%"], s["exp(coef) upper 95%"]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, max(5, 0.4 * len(s))))
    ax.errorbar(hr, y, xerr=[hr - lo, hi - hr], fmt="o", color="black", ecolor="grey", capsize=3, lw=1)
    ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(s.index)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio (95% CI, log scale)")
    ax.set_title("Primary Cox model (M1)", loc="left")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--processed", default="data/processed", help="output folder of injury_risk.features")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, covariates = load_m1(args.processed)
    n_events = int(df[EVENT].sum())
    print(f"n={len(df)}, events={n_events}, events per covariate={n_events / len(covariates):.1f}")

    cph = fit_cox(df, covariates)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cph.print_summary()
    (out / "cox_m1_summary.txt").write_text(f"n={len(df)}, events={n_events}\n\n{buf.getvalue()}")
    cph.summary.to_csv(out / "cox_m1.csv")
    forest_plot(cph.summary, out / "forest_plot_m1.png")
    print(buf.getvalue())


if __name__ == "__main__":
    main()
