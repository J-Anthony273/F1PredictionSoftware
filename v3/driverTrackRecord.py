#Imports
import pandas as pd
import numpy as np

df = pd.read_csv("datasets/datasetV3-3.csv")
history = pd.read_csv("datasets/historicalResults.csv")

combined = pd.concat([history, df[["Season", "RaceNumber", "Location", "Driver", "Race"]]],
                    ignore_index=True)

combined = combined.sort_values(["Season", "RaceNumber"]).reset_index(drop = True)

df = df.sort_values(["Season", "RaceNumber"]).reset_index(drop = True)

def computeDriverTrackRecord(combined, df):
    records = []
    for _, row in df.iterrows():
        driver = row["Driver"]
        location = row["Location"]
        season = row["Season"]
        race_no = row["RaceNumber"]

        prior = combined[
            (combined["Driver"] == driver)&
            (combined["Location"] == location)&
            (
                (combined["Season"] < season) |
                ((combined["Season"] == season) & (combined["RaceNumber"] < race_no))
            )
        ]

        if len(prior) == 0:
            overall = combined[
                    (combined['Driver'] == driver) &
                    (
                        (combined['Season'] < season) |
                        ((combined['Season'] == season) & (combined['RaceNumber'] < race_no))
                    )
                ]['Race'].mean()
            records.append(round(float(overall), 2) if not pd.isna(overall) else 10.0)
        else:
            records.append(round(float(prior['Race'].mean()), 2))
    
    return records
    
print("Computing driverTrackRecord")
df['driverTrackRecord'] = computeDriverTrackRecord(combined, df)
 
print(f"Done — sample output:")
print(df[['Season', 'RaceNumber', 'Location', 'Driver', 'driverTrackRecord']].head(20).to_string(index=False))
print(f"\nMissing values: {df['driverTrackRecord'].isna().sum()}")
 
df.to_csv('datasets/datasetV3-3.csv', index=False)
print("\nSaved updated dataset to datasets/datasetV3-3.csv")