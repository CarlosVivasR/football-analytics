"""Tournaments in scope and the observation window around each one."""

import pandas as pd

# (first match, final) for each tournament.
TOURNAMENTS = {
    "WC2010": ("2010-06-11", "2010-07-11"),
    "WC2014": ("2014-06-12", "2014-07-13"),
    "WC2018": ("2018-06-14", "2018-07-15"),
    "EURO2008": ("2008-06-07", "2008-06-29"),
    "EURO2012": ("2012-06-08", "2012-07-01"),
    "EURO2016": ("2016-06-10", "2016-07-10"),
    "EURO2020": ("2021-06-11", "2021-07-11"),
    "EURO2024": ("2024-06-14", "2024-07-14"),
    "COPA2011": ("2011-07-01", "2011-07-24"),
    "COPA2015": ("2015-06-11", "2015-07-04"),
    "COPA2016": ("2016-06-03", "2016-06-26"),
    "COPA2019": ("2019-06-14", "2019-07-07"),
    "COPA2021": ("2021-06-13", "2021-07-10"),
    "COPA2024": ("2024-06-20", "2024-07-14"),
    "KLUB2025": ("2025-06-15", "2025-07-13"),
}

DAYS_BEFORE = 30
DAYS_AFTER = 14

# Top-five league competition ids in the transfermarkt-datasets tables.
TOP5_COMPS = {"GB1", "ES1", "IT1", "L1", "FR1"}


def start(tid):
    return pd.Timestamp(TOURNAMENTS[tid][0])


def window(tid):
    """Observation window: 30 days before the first match to 14 days after the final."""
    first, final = TOURNAMENTS[tid]
    return (pd.Timestamp(first) - pd.Timedelta(days=DAYS_BEFORE),
            pd.Timestamp(final) + pd.Timedelta(days=DAYS_AFTER))
