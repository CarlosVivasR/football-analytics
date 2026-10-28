"""World Football Elo style ratings with an ordered-logit map to 1X2 probabilities."""

import numpy as np
from scipy.optimize import minimize

ELO_INIT = 1500.0
ELO_HFA = 65.0
ELO_DIVISOR = 400.0


def importance_k(tournament):
    t = tournament.lower()
    if "world cup" in t and "qualif" not in t:
        return 60.0
    if t in ("uefa euro", "copa américa", "copa america", "african cup of nations",
             "afc asian cup", "gold cup") or "nations league" in t:
        return 50.0
    if "qualif" in t:
        return 40.0
    if "friendly" in t:
        return 20.0
    return 35.0


def margin_multiplier(goal_diff, winner_rating_diff):
    gd = abs(goal_diff)
    if gd <= 1:
        g = 1.0
    elif gd == 2:
        g = 1.5
    else:
        g = (11.0 + gd) / 8.0
    return g * (2.2 / (winner_rating_diff * 0.001 + 2.2))


def replay(df):
    """Run Elo through all matches in date order.

    Returns the pre-match rating difference (home minus away, home advantage
    included) for every row, and the final ratings.
    """
    ratings = {}
    diffs = np.empty(len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        r_home = ratings.get(row.home_team, ELO_INIT)
        r_away = ratings.get(row.away_team, ELO_INIT)
        r_home_adj = r_home + (0.0 if row.neutral else ELO_HFA)
        diffs[i] = r_home_adj - r_away

        gd = row.home_score - row.away_score
        score = 1.0 if gd > 0 else 0.0 if gd < 0 else 0.5
        expected = 1.0 / (1.0 + 10 ** ((r_away - r_home_adj) / ELO_DIVISOR))
        if gd != 0:
            winner_diff = (r_home_adj - r_away) if gd > 0 else (r_away - r_home_adj)
            mult = margin_multiplier(gd, winner_diff)
        else:
            mult = 1.0
        delta = importance_k(row.tournament) * mult * (score - expected)
        ratings[row.home_team] = r_home + delta
        ratings[row.away_team] = r_away - delta
    return diffs, ratings


def elo_probs(diff, s, tau):
    """Ordered-logit style 1X2 probabilities from a rating difference."""
    z = diff / s
    p_home = 1.0 / (1.0 + np.exp(-(z - tau)))
    p_away = 1.0 / (1.0 + np.exp(-(-z - tau)))
    p_draw = np.clip(1.0 - p_home - p_away, 1e-6, None)
    total = p_home + p_draw + p_away
    return np.vstack([p_home / total, p_draw / total, p_away / total]).T


def fit_ordered_logit(diff, y):
    """Maximum-likelihood scale s and draw threshold tau."""
    def nll(params):
        s, tau = params
        if s <= 1.0 or tau <= 0:
            return 1e9
        p = elo_probs(diff, s, tau)[np.arange(len(y)), y]
        return -np.mean(np.log(np.clip(p, 1e-12, 1.0)))

    res = minimize(nll, x0=[200.0, 0.5], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 5000})
    return float(res.x[0]), float(res.x[1])
