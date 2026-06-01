#Imports
import fastf1
import pandas as pd

history_rows = []

for year in range(2010, 2018):
    
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for _, event in schedule.iterrows():
        try:
            session = fastf1.get_session(year, int(event['RoundNumber']), 'R')
            session.load()
            results = session.results[['FullName', 'Position']].copy()
            results.columns = ['Driver', 'Race']
            results['Season']     = year
            results['Location']   = event['Location']
            results['RaceNumber'] = int(event['RoundNumber'])
            history_rows.append(results)
            print(f"Loaded {year} R{int(event['RoundNumber'])} — {event['Location']}")
        except Exception as e:
            print(f"Skipping {year} R{int(event['RoundNumber'])}: {e}")

historyDf = pd.concat(history_rows, ignore_index=True)
historyDf = historyDf[['Season', 'RaceNumber', 'Location', 'Driver', 'Race']]

historyDf.to_csv('datasets/historicalResults.csv', index=False)
print(f"\nSaved {len(historyDf)} rows")
print(historyDf.head(10).to_string(index=False))