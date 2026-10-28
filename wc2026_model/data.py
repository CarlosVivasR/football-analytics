"""Download and load the martj42 international results dataset (CC0)."""

import re
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

MAJOR_PATTERN = (r"FIFA World Cup|UEFA Euro|Copa Am|African Cup of Nations|AFC Asian Cup|"
                 r"UEFA Nations League|CONCACAF Nations|Gold Cup|Confederations")

# Alternative spellings mapped to the normalised martj42 name.
ALIASES = {
    "usa": "unitedstates",
    "bosniaherzegovina": "bosniaandherzegovina",
    "czechia": "czechrepublic",
    "turkiye": "turkey",
    "congodr": "drcongo",
    "caboverde": "capeverde",
    "korearepublic": "southkorea",
    "republicofkorea": "southkorea",
    "unitedstatesofamerica": "unitedstates",
}


def download_results(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(RESULTS_URL, path)
    return path


def load_results(path, cutoff=None):
    """Load played matches, sorted by date, with outcome coded 0=home, 1=draw, 2=away.

    If `cutoff` is given, only matches strictly before that date are kept.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    if cutoff is not None:
        df = df[df["date"] < pd.Timestamp(cutoff)].copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    df = df.sort_values("date").reset_index(drop=True)
    df["is_friendly"] = df["tournament"].str.contains("Friendly", case=False, na=False)
    df["is_major"] = df["tournament"].str.contains(MAJOR_PATTERN, case=False, regex=True, na=False)
    df["home_adv"] = (~df["neutral"]).astype(float)
    df["outcome"] = np.select([df["home_score"] > df["away_score"],
                               df["home_score"] == df["away_score"]], [0, 1], 2)
    return df


def team_key(name):
    """Lowercase, accent-free, letters-only team key with aliases resolved."""
    s = unicodedata.normalize("NFD", str(name))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z]", "", s.lower())
    return ALIASES.get(s, s)
