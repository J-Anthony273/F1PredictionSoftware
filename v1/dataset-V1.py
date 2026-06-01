#Imports
import fastf1
import pandas as pd
import numpy as np
import time
import os


#Used for specific years if the other script missed any.
years = [2023]
targetSessions = ['FP1', 'FP2', 'FP3', 'Q', 'Sprint Shootout', 'S', 'R']

sessionMap = {
    'FP1': 'FP1',
    'FP2': 'FP2',
    'FP3': 'FP3',
    'Q': 'Qualifying',
    'Sprint Shootout': 'SprintShootout',
    'S': 'Sprint',
    'R': 'Race'
}

dataset = []
skipped_rounds = []

for year in years:
    print(f"\n=== Processing season {year} ===")
    try:
        schedule = fastf1.get_event_schedule(year)
        print(f"Found {len(schedule)} events\n")
    except Exception as e:
        print(f"Could not fetch schedule for {year}: {e}")
        continue

    for idx, event in schedule.iterrows():
        roundNumber = event['RoundNumber']
        
        if roundNumber == 0:
            continue
            
        print(f"Round {roundNumber:2d} - {event['EventName']}")

        raceData = {}
        round_had_data = False

        for code in targetSessions:
            colName = sessionMap.get(code)
            
            try:
                session = fastf1.get_session(year, roundNumber, code)

                #Gets the FP session results using session.laps
                if code.startswith('FP'):
                    print(f"  {code}...", end=" ", flush=True)
                    session.load(laps=True, telemetry=False, weather=False)
                    
                    if not hasattr(session, 'drivers') or len(session.drivers) == 0:
                        print("no drivers")
                        time.sleep(1)
                        continue
                    
                    laps = session.laps
                    if laps is None or laps.empty:
                        print("no laps")
                        continue

                    fastest = (
                        laps.groupby('Driver')['LapTime']
                        .min()
                        .dropna()
                        .sort_values()
                    )

                    if fastest.empty:
                        print("no times")
                        continue

                    print(f"({len(fastest)})", end=" ")
                    round_had_data = True
                    
                    for pos, driver in enumerate(fastest.index, start=1):
                        drvInfo = session.get_driver(driver)
                        fullName = drvInfo["FullName"]
                        team = drvInfo["TeamName"]

                        key = (year, roundNumber, fullName)

                        if key not in raceData:
                            raceData[key] = {
                                'Season': year,
                                'RaceNumber': roundNumber,
                                'Driver': fullName,
                                'Team': team,
                                'FP1': np.nan,
                                'FP2': np.nan,
                                'FP3': np.nan,
                                'Qualifying': np.nan,
                                'SprintShootout': np.nan,
                                'Sprint': np.nan,
                                'Race': np.nan
                            }

                        raceData[key][colName] = pos
                    
                    time.sleep(2)
                        
                else:
                    print(f"{code}...", end=" ", flush=True)
                    if code == 'Sprint Shootout':
                        session.load(laps=True, telemetry=False, weather=False)
                    else:
                        session.load(laps=False, telemetry=False, weather=False)
                    
                    if not hasattr(session, 'drivers') or len(session.drivers) == 0:
                        print("X", end=" ")
                        time.sleep(1)
                        continue
                    
                    if not hasattr(session, 'results') or session.results is None or session.results.empty:
                        print("—", end=" ")
                        continue

                    print(f"{len(session.results)}", end=" ")
                    round_had_data = True

                    for _, row in session.results.iterrows():
                        if pd.isna(row["Position"]):
                            continue

                        driver = row['FullName']
                        team = row['TeamName']
                        position = int(row['Position'])

                        key = (year, roundNumber, driver)

                        if key not in raceData:
                            raceData[key] = {
                                'Season': year,
                                'RaceNumber': roundNumber,
                                'Driver': driver,
                                'Team': team,
                                'FP1': np.nan,
                                'FP2': np.nan,
                                'FP3': np.nan,
                                'Qualifying': np.nan,
                                'SprintShootout': np.nan,
                                'Sprint': np.nan,
                                'Race': np.nan
                            }
                        
                        print(f"\n    code={code}, colName={colName}, driver={driver}, position={position}")
                        print(f"    Before: raceData[key]['{colName}'] = {raceData[key][colName]}")
                        raceData[key][colName] = position
                        print(f"    After: raceData[key]['{colName}'] = {raceData[key][colName]}")
                        
                    time.sleep(1)

            except Exception as e:
                error_msg = str(e)[:50]
                if "does not exist" not in error_msg:
                    print(f"{type(e).__name__}", end=" ")
                time.sleep(1)
                continue

        print()
    
        if raceData:
            dataset.extend(raceData.values())
        elif round_had_data == False:
            skipped_rounds.append(f"Round {roundNumber} - {event['EventName']}")

print(f"\n{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"Total records: {len(dataset)}")

if skipped_rounds:
    print(f"\nSkipped rounds no data available:")
    for r in skipped_rounds:
        print(f"  - {r}")

if dataset:
    datasetFinal = pd.DataFrame(dataset)
    
    posCols = ['FP1', 'FP2', 'FP3', 'Qualifying', 'SprintShootout', 'Sprint', 'Race']
    for col in posCols:
        datasetFinal[col] = pd.to_numeric(datasetFinal[col], errors='coerce')

    datasetFinal = datasetFinal.sort_values(['Season', 'RaceNumber']).reset_index(drop=True)

    print(f"\nDataset shape: {datasetFinal.shape}")
    print("\nData coverage by race:")
    summary = datasetFinal.groupby('RaceNumber').agg({
        'Driver': 'count',
        'FP1': 'count',
        'FP2': 'count',
        'FP3': 'count',
        'Qualifying': 'count',
        'Race': 'count'
    })
    print(summary)

    print("\nSample data:")
    print(datasetFinal.head(10))
    filename = f"datasetV1-{year}.csv"
    filename = os.path.join('datasets', filename)
    datasetFinal.to_csv(filename, index=False)
    print("\nSaved to datasetV1.csv")
else:
    print("No data collected!")