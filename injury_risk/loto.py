"""Leave-one-tournament-out validation of the primary Cox model.

    python -m injury_risk.loto --processed data/processed --out outputs

For each tournament, M1 is fitted on the other 14 and scored on the held-out one
with Harrell's C. The headline figure pools concordant pairs within tournaments
across all folds, so players are only compared with others exposed in the same
window. Confidence intervals resample players within tournaments.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from .model import DURATION, EVENT, PENALIZER, fit_cox, load_m1

N_BOOT = 2000
SEED = 20261028
MIN_EVENTS_CI = 3  # with one or two events a bootstrap CI is degenerate


def pair_counts(t, e, risk):
    """Concordant pairs (ties count half) and comparable pairs for Harrell's C.

    A pair is comparable if i had the event and t_i < t_j, or t_i == t_j with j
    censored, as in lifelines.utils.concordance_index.
    """
    t, e, risk = np.asarray(t, float), np.asarray(e, int), np.asarray(risk, float)
    ev = np.flatnonzero(e == 1)
    if len(ev) == 0:
        return 0.0, 0
    ti, ri = t[ev][:, None], risk[ev][:, None]
    comparable = (ti < t[None, :]) | ((ti == t[None, :]) & (e[None, :] == 0))
    conc = ((ri > risk[None, :]) & comparable).sum() + 0.5 * ((ri == risk[None, :]) & comparable).sum()
    return float(conc), int(comparable.sum())


def boot_ci(stat, groups, rng):
    """Percentile 95% CI; each index array in `groups` is resampled separately."""
    vals = []
    for _ in range(N_BOOT):
        v = stat([rng.choice(g, size=len(g), replace=True) for g in groups])
        if v is not None and np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi, len(vals)


def run(df, covariates, rng):
    df = df.copy()
    full = fit_cox(df, covariates)
    df["risk_in"] = full.predict_partial_hazard(df[covariates]).values
    df["risk_oof"] = np.nan
    tournaments = sorted(df["tournament_id"].unique())
    for tid in tournaments:
        test = df["tournament_id"] == tid
        cph = fit_cox(df[~test], covariates)
        df.loc[test, "risk_oof"] = cph.predict_partial_hazard(df.loc[test, covariates]).values

    t, e = df[DURATION].values, df[EVENT].values
    r_oof, r_in = df["risk_oof"].values, df["risk_in"].values
    groups = {tid: np.flatnonzero(df["tournament_id"].values == tid) for tid in tournaments}

    def c_index(ix, risk):
        conc, comp = pair_counts(t[ix], e[ix], risk[ix])
        return conc / comp if comp else None

    rows = []
    for tid, ix in groups.items():
        n_ev = int(e[ix].sum())
        lo = hi = np.nan
        n_boot = 0
        if n_ev >= MIN_EVENTS_CI:
            lo, hi, n_boot = boot_ci(lambda idx: c_index(idx[0], r_oof), [ix], rng)
        rows.append({"tournament": tid, "n": len(ix), "events": n_ev, "c_loto": c_index(ix, r_oof),
                     "ci_lo": lo, "ci_hi": hi, "n_boot_valid": n_boot, "c_insample": c_index(ix, r_in)})
    per_t = pd.DataFrame(rows)

    def within(idx, risk):
        conc = comp = 0
        for ix in idx:
            a, b = pair_counts(t[ix], e[ix], risk[ix])
            conc, comp = conc + a, comp + b
        return conc / comp if comp else None

    def pooled(idx):
        j = np.concatenate(idx)
        return concordance_index(t[j], -r_oof[j], e[j])

    g = list(groups.values())
    c_in_pooled = concordance_index(t, -r_in, e)
    conc, comp = pair_counts(t, e, r_in)
    assert abs(conc / comp - c_in_pooled) < 1e-9, "pair counter disagrees with lifelines"

    # Order matters: the bootstrap draws share one generator.
    summary = {
        "loto_within": (within(g, r_oof), *boot_ci(lambda idx: within(idx, r_oof), g, rng)[:2]),
        "insample_within": (within(g, r_in), *boot_ci(lambda idx: within(idx, r_in), g, rng)[:2]),
        "loto_pooled": (concordance_index(t, -r_oof, e), *boot_ci(pooled, g, rng)[:2]),
        "insample_pooled": c_in_pooled,
    }
    return per_t, summary


def report(per_t, s, covariates, n, n_events):
    klub = int(per_t.loc[per_t["tournament"] == "KLUB2025", "events"].sum())
    lines = [
        "Leave-one-tournament-out validation, primary model M1",
        f"CoxPHFitter(penalizer={PENALIZER}), {len(covariates)} covariates: " + ", ".join(covariates),
        f"n={n} player-tournaments, events={n_events} ({n_events / len(covariates):.1f} per covariate); "
        f"KLUB2025 events = {klub} ({100 * klub / n_events:.0f}%)",
        f"Bootstrap: {N_BOOT} resamples within tournament, percentile 95% CI, seed={SEED}",
        "",
        f"{'tournament':10s} {'n':>5s} {'events':>6s} {'C_LOTO':>7s} {'95% CI':>17s} {'C_in-sample':>12s}",
    ]
    for r in per_t.itertuples():
        ci = f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]" if r.events >= MIN_EVENTS_CI else "not estimable"
        lines.append(f"{r.tournament:10s} {r.n:5d} {r.events:6d} {r.c_loto:7.3f} {ci:>17s} {r.c_insample:12.3f}")
    lines += [
        "",
        "LOTO C-index, within-tournament pairs:      {:.3f}  [{:.3f}, {:.3f}]".format(*s["loto_within"]),
        "In-sample C-index, within-tournament pairs: {:.3f}  [{:.3f}, {:.3f}]".format(*s["insample_within"]),
        "LOTO C-index, all pairs:                    {:.3f}  [{:.3f}, {:.3f}]".format(*s["loto_pooled"]),
        "In-sample C-index, all pairs:               {:.3f}".format(s["insample_pooled"]),
        "Optimism (in-sample minus LOTO, all pairs): {:+.3f}".format(s["insample_pooled"] - s["loto_pooled"][0]),
        "",
        f"{(per_t['events'] <= 4).sum()} of {len(per_t)} tournaments have 4 or fewer events; "
        f"no CI is given below {MIN_EVENTS_CI} events.",
    ]
    return "\n".join(lines) + "\n"


def plot(per_t, s, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_agg, lo_agg, hi_agg = s["loto_within"]
    order = per_t.sort_values(["events", "tournament"], ascending=[True, False]).reset_index(drop=True)
    y = np.arange(len(order)) + 2
    fig, ax = plt.subplots(figsize=(8, 7))
    for yi, r in zip(y, order.itertuples()):
        if r.events < MIN_EVENTS_CI:
            ax.plot(r.c_loto, yi, "o", ms=7, mfc="white", mec="C0", mew=1.5)
        else:
            ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color="C0", lw=2, alpha=0.5)
            ax.plot(r.c_loto, yi, "o", ms=7, color="C0")
    ax.plot([lo_agg, hi_agg], [0, 0], color="black", lw=2.5)
    ax.plot(c_agg, 0, "D", ms=8, color="black")
    ax.axvline(0.5, color="grey", lw=1, ls="--")
    ax.axvline(s["insample_pooled"], color="grey", lw=1, ls=":")
    ax.set_yticks(list(y) + [0])
    ax.set_yticklabels([f"{r.tournament} ({r.events} ev)" for r in order.itertuples()]
                       + [f"Aggregate ({int(order['events'].sum())} ev)"])
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Held-out C-index with 95% bootstrap CI (hollow: fewer than {MIN_EVENTS_CI} events)")
    ax.set_title("Leave-one-tournament-out validation of M1", loc="left")
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
    per_t, summary = run(df, covariates, np.random.default_rng(SEED))
    text = report(per_t, summary, covariates, len(df), int(df[EVENT].sum()))
    (out / "loto.txt").write_text(text)
    per_t.to_csv(out / "loto_per_tournament.csv", index=False)
    plot(per_t, summary, out / "loto_cindex.png")
    print(text)


if __name__ == "__main__":
    main()
