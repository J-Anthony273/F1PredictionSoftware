# CM3203-Project
A machine learning project that predicts Formula 1 race finishing positions using session data, driver and team statistics, and circuit characteristics. The project is structured across three versions (V1 → V2 → V3).

# Dependancies
Install all required packages with:
pip install fastf1 pandas numpy scikit-learn scipy xgboost matplotlib seaborn tqdm requests

fastf1 - Fetching F1 session data and schedules
pandas, numpy - Data manipulation
scikit-learn - Model training, hyperparameter search, evaluation
xgboost - XGBoost regressor
scipy - Spearman rank correlation
matplotlib, seaborn - Visualisations
tqdm - Progress bar during V2 dataset enrichment
requests - Fetching constructor standings from the Jolpi/Ergast API 

# Project Structure
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
    └── model_v3_output.txt  