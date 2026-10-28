import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

from injury_risk.features import build_history, classify_mechanism, parse_date, parse_days
from injury_risk.loto import pair_counts
from injury_risk.tournaments import window


def test_classify_mechanism():
    assert classify_mechanism("Hamstring injury") == "overuse"
    assert classify_mechanism("Broken collarbone") == "contact"
    assert classify_mechanism("Cruciate ligament tear") == "overuse"  # "tear" wins over knee
    assert classify_mechanism("Knee injury") == "ambiguous_knee"
    assert classify_mechanism("Covid-19") == "not_injury"
    assert classify_mechanism(None) == "unknown"


def test_parsers():
    assert parse_days("35 days") == 35
    assert parse_days(None) is None
    assert parse_date("05/06/2018") == pd.Timestamp("2018-06-05")
    assert parse_date("2018-06-05") == pd.Timestamp("2018-06-05")
    assert pd.isna(parse_date(""))


def test_window():
    w0, w1 = window("WC2018")
    assert w0 == pd.Timestamp("2018-05-15") and w1 == pd.Timestamp("2018-07-29")


def test_history_ignores_injuries_inside_window():
    squads = pd.DataFrame({"tournament_id": ["WC2018"], "player_id": [1]})
    injuries = pd.DataFrame({
        "player_id": [1, 1, 1],
        "from_dt": pd.to_datetime(["2017-10-01", "2018-01-10", "2018-06-01"]),
        "days_missed": [40, 10, 50],
        "text": ["hamstring strain", "hamstring injury", "calf injury"],
    })
    h = build_history(squads, injuries).iloc[0]
    assert h["prior_inj_count_24m"] == 2
    assert h["prior_severe_count_24m"] == 1
    assert h["recurrence_history_count"] == 1
    assert h["days_since_last_injury"] == (pd.Timestamp("2018-05-15") - pd.Timestamp("2018-01-10")).days


def test_pair_counts_matches_lifelines():
    rng = np.random.default_rng(1)
    t = rng.integers(1, 60, 300).astype(float)
    e = rng.integers(0, 2, 300)
    risk = rng.normal(size=300)
    conc, comp = pair_counts(t, e, risk)
    assert np.isclose(conc / comp, concordance_index(t, -risk, e))
