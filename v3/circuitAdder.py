#Imports
import fastf1
import pandas as pd

df = pd.read_csv('datasets/datasetV3-3.csv')

seasons = df['Season'].unique()
rows = []

for year in seasons:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for _, event in schedule.iterrows():
        rows.append({
            'Season'     : year,
            'RaceNumber' : int(event['RoundNumber']),
            'Location'   : event['Location']
        })

lookup = pd.DataFrame(rows)

df = df.merge(lookup, on=['Season', 'RaceNumber'], how='left')

print(df[['Season', 'RaceNumber', 'Location']].drop_duplicates().to_string(index=False))
print(f"\nMissing locations: {df['Location'].isna().sum()}")

df.to_csv('datasets/datasetV3-3.csv', index=False)
print("\nSaved updated dataset to datasets/datasetV3-3.csv")