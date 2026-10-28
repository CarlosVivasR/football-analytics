import numpy as np
import pandas as pd

from wc2026_model import elo, scoring
from wc2026_model.data import team_key
from wc2026_model.dixon_coles import fit_dixon_coles, predict_1x2, score_grid
from wc2026_model.market import log_pool, shin_devig


def test_rps_perfect_and_uniform():
    y = np.array([0, 1, 2])
    assert np.allclose(scoring.rps(np.eye(3), y), 0)
    uniform = np.full((1, 3), 1 / 3)
    # Home win under a uniform forecast: ((2/3)^2 + (1/3)^2) / 2
    assert np.isclose(scoring.rps(uniform, np.array([0]))[0], 5 / 18)


def test_paired_bootstrap_sign():
    a, b = np.full(200, 0.1), np.full(200, 0.2)
    mean, lo, hi, _ = scoring.paired_bootstrap(a, b, n_boot=200)
    assert np.isclose(mean, -0.1) and hi < 0


def test_shin_devig():
    p = shin_devig((1.9, 3.6, 4.2))
    assert np.isclose(p.sum(), 1)
    assert p[0] > 1 / 1.9 / (1 / 1.9 + 1 / 3.6 + 1 / 4.2)  # favourite gains relative to proportional
    fair = shin_devig((2.0, 4.0, 4.0))
    assert np.allclose(fair, [0.5, 0.25, 0.25])


def test_log_pool_limits():
    pm, pk = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.3, 0.5])
    assert np.allclose(log_pool(pm, pk, 1.0), pm)
    assert np.allclose(log_pool(pm, pk, 0.0), pk)


def test_team_key_aliases():
    assert team_key("USA") == team_key("United States")
    assert team_key("Curaçao") == "curacao"


def _synthetic_results(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    strength = {"A": 0.6, "B": 0.2, "C": -0.2, "D": -0.6}
    teams = list(strength)
    rows = []
    for k in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        rows.append({"date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=k),
                     "home_team": h, "away_team": a,
                     "home_score": rng.poisson(np.exp(0.1 + strength[h] - strength[a] / 2)),
                     "away_score": rng.poisson(np.exp(0.1 + strength[a] - strength[h] / 2)),
                     "neutral": True, "is_friendly": False})
    df = pd.DataFrame(rows)
    df["home_adv"] = 0.0
    return df


def test_dixon_coles_recovers_ordering():
    df = _synthetic_results()
    model = fit_dixon_coles(df, df["date"].max() + pd.Timedelta(days=1), spread=1.0)
    atk = model["atk"]
    assert atk["A"] > atk["B"] > atk["C"] > atk["D"]
    assert np.isclose(score_grid(model, "A", "D", True).sum(), 1)
    p = predict_1x2(model, "A", "D", True)
    assert np.isclose(p.sum(), 1) and p[0] > p[2]


def test_elo_is_zero_sum():
    df = _synthetic_results(200)
    df["tournament"] = "Friendly"
    _, ratings = elo.replay(df)
    assert np.isclose(sum(ratings.values()), 4 * elo.ELO_INIT)
