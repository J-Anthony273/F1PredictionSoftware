#Imports
from datasetCollector import collectF1Data
import time
import pandas as pd
import os

#Years to be collected
years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
allDataframes = []

for i, year in enumerate(years):
    print(f"\n{'*'*70}")
    print(f"Collecting year {i+1}/{len(years)}: {year}")
    print(f"{'*'*70}")
    
    df = collectF1Data(year)

    if df is not None:
        allDataframes.append(df)
    
    # Wait between years to allow for rate limit to reset
    if i < len(years) - 1:
        wait_time = 1200
        print(f"\nWaiting {wait_time//60} minutes before next year...")
        time.sleep(wait_time)

if allDataframes:
    combined = pd.concat(allDataframes, ignore_index=True)
    combined = combined.sort_values(['Season', 'RaceNumber']).reset_index(drop=True)
    
    combined.to_csv('datasetV1-combined.csv', index=False)
    print(f"\n{'='*70}")
    print(f"Combined dataset saved: {combined.shape[0]} total records")
    print(f"Saved to: datasetV1-combined.csv")
    print(f"{'='*70}")
    
    print("\nData by season:")
    print(combined.groupby('Season')['Driver'].count())
else:
    print("\nNo data was collected!")