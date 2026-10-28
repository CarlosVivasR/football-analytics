# Football analytics

Two applied statistics projects on international football, by Carlos Vivas Rodriguez.

1. **`wc2026_model/`**: a match model for the 2026 World Cup. A time-weighted
   Dixon-Coles model gives a full scoreline distribution for any fixture, an Elo
   rating serves as a benchmark, and the model can be blended with de-vigged
   bookmaker odds. It is evaluated with a walk-forward backtest, proper scoring
   rules and paired bootstrap comparisons, including a head-to-head against the market.
2. **`injury_risk/`**: a survival analysis of severe overuse injuries around summer
   tournaments (Cox proportional hazards with `lifelines`), validated by leaving
   one tournament out at a time.

Both projects report negative and null results as well as positive ones. The
point is to show how far open data gets you and how I check that, not to sell a
model.

No data is included. The match model downloads its data (CC0). The injury
project needs input files that you provide yourself (see below).

## Repository layout

```
wc2026_model/
  data.py           download and load the martj42 results (CC0)
  dixon_coles.py    weighted Dixon-Coles fit and scoreline grid
  elo.py            Elo replay and ordered-logit map to 1X2
  scoring.py        RPS, Brier, log-loss, calibration, paired bootstrap
  backtest.py       walk-forward backtest: Dixon-Coles vs Elo vs climatology
  market.py         Shin de-vig, log opinion pool, odds join, weight tuning
  fit_ratings.py    CLI: final fit, ratings JSON and backtest
  head_to_head.py   CLI: model vs bookmaker on the same matches
injury_risk/
  tournaments.py    the 15 tournaments and observation windows
  features.py       CLI: covariates, injury history and outcome table
  model.py          CLI: primary Cox model (M1)
  loto.py           CLI: leave-one-tournament-out validation
tests/              small unit tests for both packages
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

Tested with Python 3.12, numpy 2.2, pandas 2.2, scipy 1.15, lifelines 0.30 and
matplotlib 3.10. Every command writes to `data/` and `outputs/`, both ignored by git.

---

## 1. World Cup 2026 match model

### Why

A prediction pool for the 2026 World Cup originally converted bookmaker
moneylines into scorelines. That gives no prediction when there are no odds (the
whole knockout bracket before teams are known), it cannot pin down the goal
level, and it had never been tested. I wanted a model that covers every fixture,
gives an honest scoreline distribution, and comes with a track record.

### Model

- **Dixon-Coles core.** Poisson goals with team attack and defence ratings, home
  advantage only at non-neutral venues, and the Dixon-Coles correction for low
  scores (rho is fitted, about -0.05). The likelihood is weighted by exponential
  time decay (half-life 1,460 days) with friendlies weighted 0.7. A small ridge
  penalty (0.03) keeps the ratings of teams with few matches near zero. After the fit, ratings are
  multiplied by 1.10. This corrects under-dispersion in heavy mismatches and was chosen on a
  pre-2022 validation slice.
- **Elo.** World Football Elo style updates (tournament weight, margin of victory,
  65-point home advantage) replayed over all history, mapped to home/draw/away
  probabilities with a two-parameter ordered logit fitted on 2018-2021.
- **Market blend.** When odds exist, they are de-vigged with Shin's method and
  combined with the model through a log opinion pool,
  `p ∝ p_model^w · p_market^(1-w)`.

Data: [martj42/international_results](https://github.com/martj42/international_results)
(CC0), about 49,000 men's internationals since 1872.

### Results

The figures come from the original analysis notes, which are not part of this
repository. The file names in brackets point to those notes. Rows marked
*this repo* are outputs of the code here on the same data snapshot (results up to
2026-06-12).

**Walk-forward backtest.** The model is refitted every 90 days on data before the
origin. The test set is every international from 2022-01-01 onwards (n = 4,571).
Lower is better for RPS.

| Model | RPS | Accuracy | Source |
|---|---|---|---|
| Dixon-Coles | 0.1688 | 0.601 | `bakeoff/00_VERDICT.md`, `calib/ship_confirm.json` |
| Elo + ordered logit | 0.1721 | 0.600 | `bakeoff/00_VERDICT.md` |
| Climatology (base rates) | 0.2283 | 0.478 | `00_FINAL_REPORT.md` |

- Dixon-Coles minus Elo, paired bootstrap on the same matches: ΔRPS -0.0034,
  95% CI [-0.0057, -0.0012] (*this repo*). On major tournaments (n = 2,745) the
  Dixon-Coles RPS is 0.1639 (`calib/ship_confirm.json`).
- The model is well calibrated overall: the ECE of the home-win probability is
  about 0.03 (`calib/ship_confirm.json`).
- More flexible challengers (ordered logit on Elo and Dixon-Coles features, gradient boosting,
  a small MLP) improved RPS by only about 0.002. The gain was significant
  overall but not on the priced slice where the market is available
  (`bakeoff/00_VERDICT.md`). They are not included here.

**Head-to-head with the bookmaker.** I used average pre-match 1X2 odds from
football-data.co.uk, mostly World Cup finals and qualifiers. There are no
friendlies in it.

| Slice | Market RPS | Model RPS | Source |
|---|---|---|---|
| 2014-2026, n = 1,587 (ratings not rescaled) | 0.1472 | 0.1534 | `calib/00_diagnosis.md` |
| Market favourite > 0.75, n = 505 | 0.0555 | 0.0615 | `calib/00_diagnosis.md` |
| 2022 onwards, n = 802 | 0.1408 | 0.1472 | `calib/ship_confirm.json` |
| 2022 onwards, n = 802 | 0.1408 | 0.1470 | *this repo* (orientation fix, below) |

- **The model does not beat the market.** On the 2022+ slice, model minus market
  is +0.0062, 95% CI [+0.0021, +0.0107]. Blending with w = 0.30 is
  indistinguishable from the market alone: +0.0004, CI [-0.0009, +0.0017]. A
  weight tuned out of sample comes out between 0.00 and 0.06, depending on the slice (*this repo*). The model is useful as
  an independent forecast when no odds exist, not as a way to beat the closing line.
- An earlier version of this comparison reported a model RPS of 0.268, worse
  than climatology. That number was wrong. The match table was re-sorted with an
  unstable sort, so forecasts for matches played on the same day were matched to
  the wrong results. The bug was found when the comparison was reproduced
  (`calib/00_diagnosis.md`). `market.py` sorts once with a stable sort and
  assigns forecasts by position.
- `market.py` also fixes a smaller problem. When the odds file listed the teams in
  the opposite order to the results file, home advantage went to the wrong team
  (20 of 1,587 matches). Forecasts are now made in the results file's orientation
  and then flipped. This moves the model RPS by about -0.0002.
- The 1.10 rating spread improved the priced heavy-favourite slice by
  ΔRPS -0.0020, CI [-0.0030, -0.0010]. Overall it was neutral: +0.0002, CI
  [-0.0001, +0.0005] (`calib/ship_confirm.json`).

**What did not make it in.** I re-checked the hyperparameters on a 2016-2021
validation slice in June 2026. Every frozen value was at or next to its optimum,
and no single change held up on the test era (`04_revalidation_jun2026.md`).
Altitude moved RPS on altitude matches by -0.0041, CI [-0.0110, +0.0025], so it
was left out even though the literature supports it. Rest-day differences were a
clean null (`improve/00_VERDICT.md`).

### How to run

```
# downloads data/results.csv if missing, writes outputs/ratings.json
python -m wc2026_model.fit_ratings

# ratings as of a past date, without the backtest
python -m wc2026_model.fit_ratings --cutoff 2026-06-11 --skip-backtest --out outputs/ratings_pre.json

# model vs market; the odds file is yours to provide
python -m wc2026_model.head_to_head --odds data/odds.csv --since 2022-01-01
```

The full backtest takes about a minute on a laptop. The odds CSV needs these columns:

| Column | Meaning |
|---|---|
| `date` | match date, YYYY-MM-DD |
| `home`, `away` | team names (matched to martj42 after normalising accents and common aliases) |
| `home_goals`, `away_goals` | final score, used for the outcome |
| `odds_H_avg`, `odds_D_avg`, `odds_A_avg` | decimal 1X2 odds |
| `competition` | optional label |

The football-data.co.uk international files are free for personal use but have
no open licence, so they are not redistributed here.

### Limitations

- The scoring rules measure 1X2 forecasts. The pool itself scores exact
  scorelines, and that was never evaluated directly.
- One scalar rho cannot reproduce the observed score dependence. Compared with
  independent Poisson, the data has more 0-0 draws and fewer 1-1 draws, and a
  single rho cannot raise one while lowering the other.
- The backtest test set is full of friendlies and lopsided qualifiers, which are
  easy to call. Balanced knockout games between strong teams are close to coin
  flips for any model of this kind.

### Exploratory work not included

A Poisson GLM variant, an earlier blend script with its own Dixon-Coles
implementation, gradient boosting, neural network and ordered-logit challengers,
calibration experiments (adaptive shrinkage, minnow floors, ridge tuning),
host-advantage and squad-value tilts, altitude and rest-day tests, and the web
app export. Those scripts tested ideas. The pieces here are the ones the
conclusions rest on.

---

## 2. Overuse injuries in summer tournaments

### Question

Can public pre-tournament information predict which players suffer a severe
overuse injury (at least 29 days out)? The predictors are workload, injury
history, age, position and market value. The window runs from 30 days before the
first match to 14 days after the final.

### Data

- **Squads:** 15 tournaments from 2008 to 2025 (3 World Cups, 5 Euros, 6 Copas
  América, Club World Cup 2025). That gives 7,499 player-tournament entries, of
  which 5,366 link to a player ID. The lists were assembled from
  [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup) (CC BY-SA 4.0),
  Wikipedia squad lists (CC BY-SA) and
  [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets)
  (CC0). The squad assembly code is not included.
- **Club minutes, market values, date of birth:** transfermarkt-datasets (CC0).
- **Injuries:** the records used in the analysis come from Transfermarkt player
  pages, which have no open licence. **Neither the injury data nor any code to
  collect it is included. You need to supply the injury table yourself.**

### Inputs you provide

`--squads` (CSV, one row per player per tournament):

| Column | Meaning |
|---|---|
| `tournament_id` | one of the ids in `injury_risk/tournaments.py`, e.g. `WC2018`, `EURO2024`, `KLUB2025` |
| `player_id` | integer player id as used in transfermarkt-datasets |
| `position_broad` | `GK`, `DEF`, `MID` or `FWD` |
| `dob` | YYYY-MM-DD; only used when the player is missing from `players.csv.gz` |

`--injuries` (CSV, one row per injury spell):

| Column | Meaning |
|---|---|
| `player_id` | same id as in the squads file |
| `injury` | free-text description in English, e.g. "Hamstring injury" |
| `from_date` | start of the spell, DD/MM/YYYY or YYYY-MM-DD |
| `days` | days out, as a number or text such as "35 days" |

`--transfermarkt-datasets`: a folder with `players.csv.gz`, `appearances.csv.gz`
and `player_valuations.csv.gz` from transfermarkt-datasets.

Injuries are classified as overuse or contact by keyword matching on the
description. Knee and ankle injuries without a clear mechanism, along with
unknown ones, are left out of the outcome (124 of 427 in-window injuries).

### Method

- Cox proportional hazards model (`lifelines`, ridge penalty 0.05). Time runs in
  days from the window start; the event is the first severe overuse injury.
- **Primary model M1, 14 covariates:** age, log market value, top-five league,
  club minutes in the previous 12 months, log prior injuries (24 months), prior
  severe injuries (24 months), hamstring history, ACL history, surgery (24
  months), number of body parts injured twice or more, days since the last injury
  (capped at 730), and position. n = 5,234 with 75 events.
- **Leave-one-tournament-out (LOTO):** fit on 14 tournaments, score the 15th
  with Harrell's C. The headline pools concordant pairs within tournaments, so
  each player is only compared with players exposed in the same window. 95% CIs
  come from 2,000 bootstrap resamples within tournaments.

M1 was fixed as the primary model after the exploratory runs, which is a
limitation in itself.

### Results

From `outputs/PORTFOLIO_SUMMARY.md`, `outputs/cox_v2_M1_full.csv`,
`outputs/cox_v2_summary.txt` and `outputs/validation_loto.txt` in the original
project. The code here reproduces all of them exactly.

| Covariate (M1) | HR (95% CI) | p |
|---|---|---|
| Body parts with recurrent injury | 1.17 (0.98-1.39) | 0.08 |
| Prior severe injuries, 24 months | 1.10 (0.97-1.25) | 0.13 |
| Club minutes, 12 months (per 1,000) | 1.00 (0.93-1.08) | 0.96 |
| Goalkeeper vs defender | 0.85 (0.60-1.20) | 0.35 |

- In-sample C-index is 0.65, but the global likelihood-ratio test is 13.92 on 14
  df (p ≈ 0.46). As a whole, the model is not distinguishable from no model.
- **LOTO C-index 0.632 (95% CI 0.554-0.708)** on within-tournament pairs,
  against 0.656 in-sample on the same pairs. Pooled over all pairs: 0.613
  (0.542-0.678) against 0.645, an optimism of about 0.03.
- Results by tournament range from 0.47 (WC 2014) to 0.97 (WC 2010). Nine of the
  15 tournaments have four or fewer events.

**Reading.** Injury history carries a weak signal, with recurrence the strongest
single covariate. Workload measured as club minutes carries none. Without the
history features, goalkeepers looked protected (HR 0.57, p = 0.08). With them,
the effect shrinks to 0.85, which suggests confounding by history.

### How to run

```
python -m injury_risk.features --squads path/to/squads.csv --injuries path/to/injuries.csv \
    --transfermarkt-datasets path/to/transfermarkt-datasets/csvs --out data/processed
python -m injury_risk.model --processed data/processed --out outputs
python -m injury_risk.loto --processed data/processed --out outputs
```

### Limitations

- 75 events for 14 covariates (about 5.4 per covariate), below the usual rule of
  thumb of 10. There is no correction for multiple comparisons across the
  exploratory runs.
- 24% of events (18 of 75) come from the Club World Cup 2025.
- There is no exposure measure for minutes played at the tournament, and club
  minutes are 0 for about half the sample (no coverage in the source).
- The keyword classifier is crude. For example, "cruciate ligament tear" counts
  as overuse because "tear" is matched before the knee terms. The 124 ambiguous
  or unknown injuries were excluded, and reclassifying them could change the
  result.
- Public injury records capture severe injuries and well-known players better
  than minor injuries and squad players.

### Exploratory work not included

A first model without injury history, M2 and M3 sensitivity models (players with
club minutes only, and without the Club World Cup), stratified and
position-specific variants, a 30-day outcome definition, player archetype
clustering, fixture-congestion features and a multicollinearity check. These
results are summarised in the original notes. Only M1 is presented as the primary result.

## License

Code under the MIT licence (see `LICENSE`). Data sources keep their own licences.
