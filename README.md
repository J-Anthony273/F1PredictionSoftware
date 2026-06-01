# F1 Race Finish Position Predictor

A machine learning project that predicts Formula 1 race finishing positions using session data, driver and team statistics, and circuit characteristics. The project is structured across three versions (V1 → V2 → V3), each adding more sophisticated feature engineering and modelling techniques.

---

## Project Structure

```
/
├── v1/                          # Data collection, cleaning, and baseline models
│   ├── datasetCollector.py
│   ├── datasetCombiner.py
│   ├── dataset-V1.py
│   ├── dataFilterer.py
│   ├── modelV1.py
│   ├── BASELINEMODEL.py
│   ├── datasets/
│   │   └── V1-datasets/
│   │       ├── datasetV1-{year}.csv
│   │       ├── datasetV1-combined.csv
│   │       └── datasetV1-filtered.csv
│   ├── graphs/
│   └── csvs/
│
├── v2/                          # Contextual feature engineering and ensemble model
│   ├── datasetCollector.py
│   ├── ENSEMBLEMODEL.py
│   ├── datasets/
│   │   ├── datasetV1-filtered.csv
│   │   └── datasetV2.csv
│   ├── graphs/
│   └── csvs/
│
└── v3/                          # Full feature set and three-model ensemble
    ├── constructorStandingsYearByYear.py
    ├── datasetV3Intial.py
    ├── driverRankingsDataset.py
    ├── driverExperience.py
    ├── historicalData.py
    ├── driverNameTest.py
    ├── circuitAdder.py
    ├── driverTrackRecord.py
    ├── trackOvertakeDifficulty.py
    ├── modelV3.py
    ├── modelV3-noDNF.py
    ├── FINALMODEL.py
    ├── test.py
    ├── datasets/
    │   ├── datasetV2.csv
    │   ├── f1ConstructorStandingsRaceByRace.csv
    │   ├── datasetV3Intial.csv
    │   ├── datasetV3RatingAddition.csv
    │   ├── datasetV3-3.csv
    │   └── historicalResults.csv
    ├── graphs/
    ├── csvs/
    └── model_v3_output.txt      # Logged terminal output from FINALMODEL
```

> **Before running any scripts**, create the `datasets/`, `graphs/`, and `csvs/` directories inside each version folder. For V1, also create `datasets/V1-datasets/` as a subdirectory.

---

## Dependencies

Install all required packages with:

```bash
pip install fastf1 pandas numpy scikit-learn scipy xgboost matplotlib seaborn tqdm requests
```

| Package | Purpose |
|---|---|
| `fastf1` | Fetching F1 session data and schedules |
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | Model training, hyperparameter search, evaluation |
| `xgboost` | XGBoost regressor |
| `scipy` | Spearman rank correlation |
| `matplotlib`, `seaborn` | Visualisations |
| `tqdm` | Progress bar during V2 dataset enrichment |
| `requests` | Fetching constructor standings from the Jolpi/Ergast API |

---

## Full Pipeline

The scripts must be run in the order described below. Each version builds on the outputs of the previous one.

---

### Version 1 — Data Collection and Baseline Models

**Step 1 — Collect raw session data**

Run `v1/datasetCombiner.py` to collect F1 session results for the years 2018–2025. This calls `datasetCollector.py` internally for each year and saves a per-year CSV and a combined CSV to `datasets/V1-datasets/`. A 20-minute wait is built in between years to respect FastF1 rate limits.

If any year fails or produces incomplete data, run `v1/dataset-V1.py` for that specific year as a fallback. Edit the `years` list at the top of the file to target the year(s) you need. Note that this file also contains debug print statements left over from development that can be removed.

**Step 2 — Filter and clean the combined dataset**

Run `v1/dataFilterer.py`. This reads `datasetV1-combined.csv`, applies filters, fills missing values, and saves `datasetV1-filtered.csv`.

**Step 3 — Train a baseline model**

- `v1/modelV1.py` — trains a Random Forest using a simple random 80/20 train/test split. Produces `graphs/rf_model_results.png`.
- `v1/BASELINEMODEL.py` — improved version using `GroupShuffleSplit` to keep all drivers from the same race in the same split, preventing data leakage. Produces `graphs/rf_race_model_results.png` and `csvs/rf_per_race_metrics.csv`.

---

### Version 2 — Contextual Features and RF + XGBoost Ensemble

**Step 4 — Enrich the dataset with contextual features**

Run `v2/datasetCollector.py`. This reads `datasetV1-filtered.csv`, fetches each race's results and weather data from FastF1, and appends four new columns. Output is saved as `datasets/datasetV2.csv`.

**Step 5 — Train the ensemble model**

Run `v2/ENSEMBLEMODEL.py`. This trains a Random Forest and an XGBoost model, then searches for the optimal blend weight between them by maximising median Spearman ρ across test races. Produces `graphs/rf_race_model_results.png`, `csvs/rf_per_race_metrics.csv`, and `csvs/ensemble_weight_search.csv`.

---

### Version 3 — Full Feature Set and RF + XGBoost + MLP Ensemble

The V3 dataset is built through a sequence of enrichment steps. Each script reads from and writes back to `datasets/datasetV3-3.csv` (or a predecessor file), so they must be run in the order listed.

**Step 6 — Fetch constructor standings**

Run `v3/constructorStandingsYearByYear.py`. This calls the Jolpi/Ergast API to retrieve constructor championship standings after every race from 2018–2025, saving the result as `datasets/f1ConstructorStandingsRaceByRace.csv`.

**Step 7 — Add car performance score**

Run `v3/datasetV3Intial.py`. This merges the constructor standings into `datasetV2.csv` and computes a `carPerformance` score for each driver/race entry based on their constructor's championship position and points. Output is saved as `datasets/datasetV3Intial.csv`.

**Step 8 — Add driver ratings**

Run `v3/driverRankingsDataset.py`. This applies pre-compiled driver ratings (sourced externally for each season from 2018–2025) to each row, normalising them to a 0–1 scale. Output is saved as `datasets/datasetV3RatingAddition.csv`.

**Step 9 — Add driver experience**

Run `v3/driverExperience.py`. This uses hardcoded career race counts per driver to assign a `driverExperience` score (normalised to 0–1) to each row, working backwards from the driver's total career races. Output is saved as `datasets/datasetV3-3.csv`.

**Step 10 — Fetch historical race results**

Run `v3/historicalData.py`. This fetches race results from 2010–2017 via FastF1 and saves them as `datasets/historicalResults.csv`. This historical data is needed for the track record calculation.

**Step 11 — Fix driver name inconsistencies**

Run `v3/driverNameTest.py`. This corrects encoding issues in driver names in `historicalResults.csv` (e.g. `Nico Hülkenberg` → `Nico Hulkenberg`) so they match the names used in the main dataset.

**Step 12 — Add circuit location**

Run `v3/circuitAdder.py`. This fetches the event schedule via FastF1 and adds a `Location` column to `datasetV3-3.csv`.

**Step 13 — Add driver track record**

Run `v3/driverTrackRecord.py`. For each driver/race entry, this computes the driver's average finishing position at that circuit using all prior race results (from 2010 onwards). If a driver has never raced at a circuit before, their overall career average is used as a fallback. Updates `datasetV3-3.csv`.

**Step 14 — Add track overtake difficulty**

Run `v3/trackOvertakeDifficulty.py`. This computes a `trackOvertakeDifficulty` score for each circuit based on historical pole-to-win conversion rates, smoothed with a global prior. The score is normalised to 0–1. Updates `datasetV3-3.csv`.

**Step 15 — Train the final model**

The following model scripts all use `datasetV3-3.csv`:

- `v3/modelV3.py` — trains the full RF + XGBoost + MLP ensemble. Searches all combinations of blend weights across the three models to find the combination that maximises median Spearman ρ on the test set.
- `v3/modelV3-noDNF.py` — same as `modelV3.py` but evaluates metrics excluding DNF drivers, giving a cleaner picture of predictive accuracy among classified finishers.
- `v3/FINALMODEL.py` — identical to `modelV3.py` with the addition of a logger that saves all terminal output to `model_v3_output.txt`.

---

## Dataset Reference

### Column Descriptions — Final Dataset (`datasetV3-3.csv`)

| Column | Description |
|---|---|
| `Season` | F1 season year |
| `RaceNumber` | Round number within the season |
| `Driver` | Driver full name |
| `Team` | Constructor name |
| `Location` | Circuit location name |
| `FP1`, `FP2`, `FP3` | Practice session finishing position (21 if the driver did not participate) |
| `Qualifying` | Qualifying grid position (21 if DNS) |
| `SprintShootout` | Sprint shootout position (0 if session not held) |
| `Sprint` | Sprint race position (0 if session not held) |
| `Race` | Race finishing position — the prediction target |
| `SprintHeld` | Binary flag — 1 if a sprint race was held that weekend |
| `SprintShootoutHeld` | Binary flag — 1 if a sprint shootout was held that weekend |
| `is_dnf` | Binary flag — 1 if the driver retired from the race |
| `dnf_rate` | Driver's DNF rate so far in the current season (rolling, resets each year) |
| `is_wet_race` | Binary flag — 1 if rainfall was recorded during the race |
| `driver_form` | Rolling form score over the last 5 races based on teammate head-to-head deltas, clipped to 0–1 |
| `team_form` | Rolling team form score over the last 5 races based on average team position rank vs other teams, clipped to 0–1 |
| `carPerformance` | Composite score derived from the constructor's championship position and points at that point in the season |
| `driverRating` | Normalised driver skill rating (0–1) sourced from external season-by-season ratings |
| `noOfRaces` | Driver's career race count at that point in time |
| `driverExperience` | `noOfRaces` normalised to 0–1 relative to the most experienced driver in the dataset |
| `driverTrackRecord` | Driver's average finishing position at this circuit in all prior races (career average used if no prior results at this circuit) |
| `trackOvertakeDifficulty` | Circuit overtaking difficulty score (0–1) based on historical pole-to-win conversion rates with Bayesian smoothing |

### Notes on missing/filled values

- `FP1`, `FP2`, `FP3`, `Qualifying` — filled with `21` when a driver did not participate.
- `Sprint`, `SprintShootout` — filled with `0` when the session was not held that weekend. The `SprintHeld`/`SprintShootoutHeld` binary columns allow the model to distinguish a genuine `0` result from a non-event.
- Sprint-related columns are set to `NaN` before filtering for seasons where the format didn't exist (Sprint introduced in 2021, Sprint Shootout replacing Sprint Qualifying in 2023).

---

## Model Details

### Features

All three versions use the following feature sets:

| Version | Features |
|---|---|
| V1 | `FP1`, `FP2`, `FP3`, `Qualifying`, `SprintShootout`, `Sprint` |
| V2 | V1 features + `driver_form`, `team_form`, `is_wet_race`, `dnf_rate` |
| V3 | V2 features + `carPerformance`, `driverRating`, `driverExperience`, `driverTrackRecord`, `trackOvertakeDifficulty` |

### Algorithm

All models use a **Random Forest Regressor** as the base, tuned via `RandomizedSearchCV` (50 iterations, 5-fold cross-validation, scored on R²). V2 and V3 add XGBoost, and V3 further adds a `MLPRegressor` (neural network) wrapped in a `StandardScaler` pipeline.

### Train / Test Split

An 80/20 split using `GroupShuffleSplit` with `RaceID` as the grouping key, ensuring all drivers from the same race are always in the same split. This prevents data leakage that would occur if drivers from the same race ended up in both train and test sets.

### DNF-Aware Sample Weighting

DNF rows are downweighted during training since retirements are largely unpredictable:

- DNF rows: weight `0.3`
- Prominent DNFs (qualified in the top 5, finished outside the top 10): weight `0.15`
- All other rows: weight `1.0`

### Ensemble Blending

Rather than a fixed split, the ensemble scripts perform a grid search over all valid weight combinations and select the blend that maximises median Spearman ρ across test races.

### Evaluation Metrics

Models are evaluated using:

- **MAE** (Mean Absolute Error) — average position error per driver
- **RMSE** (Root Mean Squared Error)
- **R²** — overall variance explained
- **Spearman ρ** — rank correlation per race (the primary metric), reported as median across all test races

---

## Outputs

| File | Script | Description |
|---|---|---|
| `v1/graphs/rf_model_results.png` | `modelV1.py` | Feature importances + actual vs predicted scatter |
| `v1/graphs/rf_race_model_results.png` | `BASELINEMODEL.py` | 4-panel evaluation plot with per-race MAE and Spearman ρ |
| `v1/csvs/rf_per_race_metrics.csv` | `BASELINEMODEL.py` | Per-race MAE, RMSE, and Spearman ρ |
| `v2/graphs/rf_race_model_results.png` | `ENSEMBLEMODEL.py` | 6-panel evaluation plot including ensemble weight search curve |
| `v2/csvs/rf_per_race_metrics.csv` | `ENSEMBLEMODEL.py` | Per-race MAE, RMSE, and Spearman ρ |
| `v2/csvs/ensemble_weight_search.csv` | `ENSEMBLEMODEL.py` | Full results of the RF/XGB blend weight search |
| `v3/graphs/rf_race_model_results.png` | `modelV3.py` / `FINALMODEL.py` | 6-panel evaluation plot for the three-model ensemble |
| `v3/csvs/rf_per_race_metrics.csv` | `modelV3.py` / `FINALMODEL.py` | Per-race MAE, RMSE, and Spearman ρ |
| `v3/csvs/ensemble_weight_search.csv` | `modelV3.py` / `FINALMODEL.py` | Full results of the RF/XGB/MLP blend weight grid search |
| `v3/model_v3_output.txt` | `FINALMODEL.py`, `modelV3-noDNF.py` | Full terminal output logged to file |

---

## External Data Sources

- **FastF1** — session lap times and results for FP1–FP3, Qualifying, Sprint Shootout, Sprint, and Race sessions, plus weather data and historical results (2010–2025)
- **Jolpi/Ergast API** (`https://api.jolpi.ca/ergast/`) — constructor championship standings after each race (2018–2025)
- **Driver ratings** — manually compiled season-by-season ratings from external sources, stored directly in `v3/driverRankingsDataset.py`

---

## Known Limitations and Notes

- **Data collection is slow.** FastF1 rate limits require a 20-minute wait between seasons in `datasetCombiner.py`. The V2 enrichment script also makes one API call per race.
- **Driver ratings are hardcoded.** The ratings in `driverRankingsDataset.py` are static and will need to be manually updated for future seasons.
- **Driver experience counts are hardcoded.** The career race totals in `driverExperience.py` are fixed as of a specific date and will need updating for new drivers or continued seasons.
- **`dataset-V1.py` contains debug statements** — the before/after assignment print statements inside the results loop are development leftovers and can be removed.
- **`test.py`** is a utility script that prints null value counts for `datasetV3RatingAddition.csv`. It is not part of the pipeline and can be ignored.
- **Sprint session history** — Sprint races were introduced in 2021 and the Sprint Shootout format replaced Sprint Qualifying in 2023. The filtering and filling logic in `dataFilterer.py` accounts for this automatically.