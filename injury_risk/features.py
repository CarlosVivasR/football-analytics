"""Build the pre-tournament covariates and the outcome table.

    python -m injury_risk.features --squads squads.csv --injuries injuries.csv
        --transfermarkt-datasets path/to/csvs --out data/processed

Inputs are described in the README. Three files are written to --out:
features.csv (workload, value, age, position), history.csv (injury history
before the window) and outcomes.csv (injuries that start inside the window).
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from .tournaments import TOURNAMENTS, start, window

# Keyword lists are matched as substrings of the lower-cased injury description,
# in the order not-injury, contact, overuse, knee, ankle.
NOT_INJURY = [
    "rest", "quarantine", "covid", "ban ", "suspension",
    "personal reasons", "personal", "illness", "stomach flu",
    "flu", "influenza", "cold", "virus", "infection", "fever", "doping",
    "appendicitis", "tonsillitis", "gastroenteritis",
    "without a club", "vacation", "international duty",
    "fitness", "knock", "minor knock", "ill",
    "unknown injury",
]
CONTACT = [
    "shoulder", "collarbone", "clavicle",
    "fracture", "broken", "concussion", "head injury",
    "facial", "jaw", "skull", "teeth", "rib", "wrist", "elbow", "finger",
    "hand injury", "hand surgery", "dislocation", "cut", "laceration", "impact",
    "bruise", "contusion",
    "nose", "eye injury", "fractured", "metatarsal",
    "flesh wound", "broken nose", "bruised ribs",
]
OVERUSE = [
    "hamstring", "thigh", "calf", "groin", "adductor", "achilles", "patellar",
    "tendin", "muscle", "muscular", "abdominal", "stress fracture",
    "soreness", "strain", "overuse", "back pain", "hip", "iliopsoas",
    "soleus", "gastrocnemius", "quadriceps", "rectus femoris",
    "pubic", "pubalgia", "tendinitis", "tendinopathy",
    "bursitis", "fasciitis", "plantar", "shin splints", "stress reaction",
    "myositis", "fatigue fracture", "back problems", "lumbar",
    "tear", "torn",
    "thigh problems", "calf problems", "groin problems", "hip problems",
    "achilles tendon problems", "achilles tendon irritation",
    "patellar tendon problems", "torn muscle fiber", "torn muscle bundle",
    "muscle stiffness", "adductor pain", "hamstring strain", "calf strain",
    "muscular problems", "rupture",
]
KNEE = [
    "cruciate", "acl", "mcl", "lcl", "meniscus", "knee injury", "knee problems",
    "ligament", "torn ligament", "capsular", "knee surgery", "inner ligament",
    "cartilage damage",
]
ANKLE = ["ankle", "sprain", "ankle problems", "ankle injury"]

# Coarse body parts used to count recurrences in the injury history.
BODY_PARTS = ["hamstring", "calf", "thigh", "groin", "adductor",
              "knee", "ankle", "back", "shoulder", "hip", "foot", "achilles"]

SEVERE_DAYS = 29


def classify_mechanism(text):
    if pd.isna(text):
        return "unknown"
    t = str(text).lower()
    for label, words in (("not_injury", NOT_INJURY), ("contact", CONTACT), ("overuse", OVERUSE),
                         ("ambiguous_knee", KNEE), ("ambiguous_ankle", ANKLE)):
        if any(k in t for k in words):
            return label
    return "unknown"


def body_part(text):
    t = text.lower()
    return next((k for k in BODY_PARTS if k in t), "other")


def parse_date(s):
    if pd.isna(s) or not str(s).strip():
        return pd.NaT
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return pd.Timestamp(datetime.strptime(str(s).strip(), fmt))
        except ValueError:
            continue
    return pd.NaT


def parse_days(s):
    """First integer in a field such as '35 days'."""
    if pd.isna(s):
        return None
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else None


def age_at(dob, ref):
    try:
        return (datetime.strptime(ref, "%Y-%m-%d") - datetime.strptime(str(dob), "%Y-%m-%d")).days / 365.25
    except (TypeError, ValueError):
        return None


def load_squads(path):
    squads = pd.read_csv(path, low_memory=False)
    squads = squads.dropna(subset=["player_id"]).copy()
    squads["player_id"] = squads["player_id"].astype(int)
    return squads[squads["tournament_id"].isin(TOURNAMENTS)].copy()


def load_injuries(path):
    inj = pd.read_csv(path)
    inj["from_dt"] = inj["from_date"].map(parse_date)
    inj["days_missed"] = inj["days"].map(parse_days)
    inj["text"] = inj["injury"].astype(str).str.lower()
    return inj


def build_features(squads, tm_dir):
    """Age, position, market value and club minutes in the 12 months before kick-off."""
    tm_dir = Path(tm_dir)
    players = pd.read_csv(tm_dir / "players.csv.gz",
                          usecols=["player_id", "date_of_birth", "current_club_domestic_competition_id"],
                          dtype={"player_id": "Int64"})
    players["dob"] = pd.to_datetime(players["date_of_birth"], errors="coerce")
    players = players.set_index("player_id")
    apps = pd.read_csv(tm_dir / "appearances.csv.gz", usecols=["player_id", "date", "minutes_played"],
                       dtype={"player_id": "Int64"})
    apps["date"] = pd.to_datetime(apps["date"], errors="coerce")
    vals = pd.read_csv(tm_dir / "player_valuations.csv.gz",
                       usecols=["player_id", "date", "market_value_in_eur"], dtype={"player_id": "Int64"})
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce")

    apps_by_player = dict(tuple(apps.groupby("player_id")))
    vals_by_player = {pid: g.sort_values("date") for pid, g in vals.groupby("player_id")}

    rows = []
    for r in squads.itertuples(index=False):
        pid, t0 = r.player_id, start(r.tournament_id)
        ref = TOURNAMENTS[r.tournament_id][0]
        out = {"tournament_id": r.tournament_id, "player_id": pid, "position_broad": r.position_broad}

        if pid in players.index:
            p = players.loc[pid]
            if isinstance(p, pd.DataFrame):
                p = p.iloc[0]
            dob = p["dob"].strftime("%Y-%m-%d") if pd.notna(p["dob"]) else None
            out["current_club_competition_id"] = p["current_club_domestic_competition_id"]
        else:
            dob = r.dob
        out["age_at_tournament"] = age_at(dob, ref) if dob else None

        a = apps_by_player.get(pid)
        if a is None:
            out["club_minutes_12m"] = 0
        else:
            pre = a[(a["date"] >= t0 - pd.Timedelta(days=365)) & (a["date"] < t0)]
            out["club_minutes_12m"] = int(pd.to_numeric(pre["minutes_played"], errors="coerce").fillna(0).sum())

        # Latest valuation on or before the first match.
        out["market_value_at_tournament_eur"] = None
        v = vals_by_player.get(pid)
        if v is not None:
            before = v[v["date"] <= t0]
            if len(before):
                out["market_value_at_tournament_eur"] = int(before.iloc[-1]["market_value_in_eur"])
        rows.append(out)
    return pd.DataFrame(rows)


def build_history(squads, injuries):
    """Injury history counted strictly before the start of each observation window."""
    inj = injuries.dropna(subset=["from_dt"]).copy()
    inj["days_missed"] = inj["days_missed"].fillna(0)
    inj["is_severe"] = inj["days_missed"] >= SEVERE_DAYS
    inj["is_hamstring"] = inj["text"].str.contains("hamstring", na=False)
    inj["is_acl"] = inj["text"].apply(lambda s: "cruciate" in s or "acl" in s)
    inj["is_surgery"] = inj["text"].str.contains("surgery", na=False)
    inj["body_part"] = inj["text"].map(body_part)
    by_player = {pid: g.sort_values("from_dt") for pid, g in inj.groupby("player_id")}

    rows = []
    for r in squads.itertuples(index=False):
        w0 = window(r.tournament_id)[0]
        out = {"tournament_id": r.tournament_id, "player_id": r.player_id}
        a = by_player.get(r.player_id)
        before = a[a["from_dt"] < w0] if a is not None else inj.iloc[:0]
        last24 = before[before["from_dt"] >= w0 - pd.Timedelta(days=730)]
        out["prior_inj_count_24m"] = len(last24)
        out["prior_severe_count_24m"] = int(last24["is_severe"].sum())
        out["has_hamstring_history"] = int(last24["is_hamstring"].any())
        out["has_acl_history"] = int(before["is_acl"].any())  # any time, not just 24 months
        out["had_surgery_24m"] = int(last24["is_surgery"].any())
        # Number of body parts injured at least twice in 24 months.
        out["recurrence_history_count"] = int((last24["body_part"].value_counts() >= 2).sum())
        out["days_since_last_injury"] = (w0 - before["from_dt"].max()).days if len(before) else None
        rows.append(out)
    return pd.DataFrame(rows)


def build_outcomes(squads, injuries):
    """Injuries starting inside the observation window, with mechanism and severity."""
    inj = injuries.copy()
    inj["mechanism"] = inj["injury"].map(classify_mechanism)
    inj["is_severe"] = inj["days_missed"] >= SEVERE_DAYS
    inj = inj[inj["mechanism"] != "not_injury"]

    events = []
    for tid in TOURNAMENTS:
        w0, w1 = window(tid)
        squad = set(squads.loc[squads["tournament_id"] == tid, "player_id"])
        e = inj[inj["player_id"].isin(squad) & (inj["from_dt"] >= w0) & (inj["from_dt"] <= w1)].copy()
        e["tournament_id"] = tid
        e["days_into_window"] = (e["from_dt"] - w0).dt.days
        events.append(e)
    cols = ["player_id", "tournament_id", "injury", "mechanism", "from_dt", "days_missed",
            "is_severe", "days_into_window"]
    return pd.concat(events, ignore_index=True)[cols]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--squads", required=True)
    ap.add_argument("--injuries", required=True)
    ap.add_argument("--transfermarkt-datasets", required=True, dest="tm_dir",
                    help="folder with players, appearances and player_valuations .csv.gz")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    squads = load_squads(args.squads)
    injuries = load_injuries(args.injuries)
    print(f"{len(squads)} squad entries, {len(injuries)} injury records")

    build_features(squads, args.tm_dir).to_csv(out / "features.csv", index=False)
    build_history(squads, injuries).to_csv(out / "history.csv", index=False)
    outcomes = build_outcomes(squads, injuries)
    outcomes.to_csv(out / "outcomes.csv", index=False)
    print(outcomes["mechanism"].value_counts().to_string())
    print(f"wrote features.csv, history.csv, outcomes.csv to {out}")


if __name__ == "__main__":
    main()
