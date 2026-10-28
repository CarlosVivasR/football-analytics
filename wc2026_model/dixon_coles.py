"""Time-weighted Dixon-Coles model for international matches.

log lambda_home = mu + atk_i - def_j + gamma * home_adv
log lambda_away = mu + atk_j - def_i

home_adv is 0 at neutral venues. Matches are weighted by exponential time decay
and friendlies are down-weighted. Attack and defence get a small ridge penalty and
are centred to sum to zero afterwards. rho (the Dixon-Coles low-score correction)
is fitted in a second step with the rates held fixed.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

HALF_LIFE_DAYS = 1460
FRIENDLY_WEIGHT = 0.7
RIDGE = 0.03
# Global multiplier on the centred ratings. Chosen on a pre-2022 validation slice to
# correct under-dispersion in heavy mismatches; 1.0 switches it off.
SPREAD = 1.10
MAX_GOALS = 10


def dc_tau(x, y, lam_h, lam_a, rho):
    """Dixon-Coles adjustment factor for the 0-0, 0-1, 1-0 and 1-1 cells."""
    t = np.ones_like(lam_h, dtype=float)
    t = np.where((x == 0) & (y == 0), 1.0 - lam_h * lam_a * rho, t)
    t = np.where((x == 0) & (y == 1), 1.0 + lam_h * rho, t)
    t = np.where((x == 1) & (y == 0), 1.0 + lam_a * rho, t)
    t = np.where((x == 1) & (y == 1), 1.0 - rho, t)
    return np.clip(t, 1e-9, None)


def match_weights(d, ref_date, half_life=HALF_LIFE_DAYS, friendly_weight=FRIENDLY_WEIGHT):
    age = (ref_date - d["date"]).dt.days.to_numpy().astype(float)
    w = np.exp(-np.log(2) / half_life * age)
    w = w * np.where(d["is_friendly"].to_numpy(), friendly_weight, 1.0)
    return w / w.mean()


def fit_dixon_coles(df, ref_date, half_life=HALF_LIFE_DAYS, friendly_weight=FRIENDLY_WEIGHT,
                    ridge=RIDGE, spread=SPREAD, maxiter=400):
    """Fit on all matches strictly before ref_date.

    Returns a dict with mu, gamma, rho and per-team attack/defence ratings.
    """
    d = df[df["date"] < ref_date]
    teams = sorted(set(d["home_team"]) | set(d["away_team"]))
    idx = {t: k for k, t in enumerate(teams)}
    n = len(teams)
    home_i = d["home_team"].map(idx).to_numpy()
    away_i = d["away_team"].map(idx).to_numpy()
    hg = d["home_score"].to_numpy().astype(float)
    ag = d["away_score"].to_numpy().astype(float)
    adv = d["home_adv"].to_numpy()
    w = match_weights(d, ref_date, half_life, friendly_weight)

    def nll_and_grad(p):
        mu, gamma = p[0], p[1]
        atk, dfn = p[2:2 + n], p[2 + n:]
        eta_h = mu + atk[home_i] - dfn[away_i] + gamma * adv
        eta_a = mu + atk[away_i] - dfn[home_i]
        lam_h = np.exp(np.clip(eta_h, -10, 4))
        lam_a = np.exp(np.clip(eta_a, -10, 4))
        nll = np.sum(w * (lam_h - hg * eta_h)) + np.sum(w * (lam_a - ag * eta_a))
        nll += ridge * (np.sum(atk ** 2) + np.sum(dfn ** 2))

        rh = w * (lam_h - hg)
        ra = w * (lam_a - ag)
        g_atk = np.zeros(n)
        g_def = np.zeros(n)
        np.add.at(g_atk, home_i, rh)
        np.add.at(g_atk, away_i, ra)
        np.add.at(g_def, away_i, -rh)
        np.add.at(g_def, home_i, -ra)
        grad = np.concatenate([[rh.sum() + ra.sum(), (rh * adv).sum()],
                               g_atk + 2 * ridge * atk,
                               g_def + 2 * ridge * dfn])
        return nll, grad

    p0 = np.concatenate([[0.0, 0.25], np.zeros(2 * n)])
    res = minimize(nll_and_grad, p0, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxfun": maxiter * 100, "ftol": 1e-9})
    mu, gamma = res.x[0], res.x[1]
    atk, dfn = res.x[2:2 + n], res.x[2 + n:]

    # Centre the ratings (the means move into mu, rates unchanged), then spread them.
    mu = mu + atk.mean() - dfn.mean()
    atk = (atk - atk.mean()) * spread
    dfn = (dfn - dfn.mean()) * spread

    lam_h = np.exp(np.clip(mu + atk[home_i] - dfn[away_i] + gamma * adv, -10, 4))
    lam_a = np.exp(np.clip(mu + atk[away_i] - dfn[home_i], -10, 4))

    def rho_nll(r):
        return -np.sum(w * np.log(dc_tau(hg, ag, lam_h, lam_a, r[0])))

    rr = minimize(rho_nll, [0.0], method="L-BFGS-B", bounds=[(-0.15, 0.25)],
                  options={"maxiter": 80})

    return {
        "mu": float(mu), "gamma": float(gamma), "rho": float(rr.x[0]),
        "atk": dict(zip(teams, atk)), "def": dict(zip(teams, dfn)),
    }


def expected_goals(model, home, away, neutral):
    """Expected goals for both sides. Unknown teams get the average rating (0)."""
    atk, dfn = model["atk"], model["def"]
    adv = 0.0 if neutral else 1.0
    lam_h = np.exp(model["mu"] + atk.get(home, 0.0) - dfn.get(away, 0.0) + model["gamma"] * adv)
    lam_a = np.exp(model["mu"] + atk.get(away, 0.0) - dfn.get(home, 0.0))
    return float(np.clip(lam_h, 1e-4, 30)), float(np.clip(lam_a, 1e-4, 30))


def score_grid(model, home, away, neutral, max_goals=MAX_GOALS):
    """Scoreline probabilities; rows are home goals, columns away goals."""
    lam_h, lam_a = expected_goals(model, home, away, neutral)
    goals = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
    x, y = np.meshgrid(goals, goals, indexing="ij")
    tau = dc_tau(x.ravel(), y.ravel(), np.full(x.size, lam_h), np.full(x.size, lam_a),
                 model["rho"])
    grid = grid * tau.reshape(grid.shape)
    return grid / grid.sum()


def predict_1x2(model, home, away, neutral):
    grid = score_grid(model, home, away, neutral)
    p = np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])
    return p / p.sum()
