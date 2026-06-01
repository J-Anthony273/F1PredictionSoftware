#Imports
import fastf1
import pandas as pd
import numpy as np
import time
import os

#Used to collect year by year
def collectF1Data(year):
    targetSessions = ['FP1', 'FP2', 'FP3', 'Q', 'SQ', 'S', 'R']
    
    sessionMap = {
        'FP1': 'FP1',
        'FP2': 'FP2',
        'FP3': 'FP3',
        'Q': 'Qualifying',
        'SQ': 'SprintShootout',
        'S': 'Sprint',
        'R': 'Race'
    }
    
    dataset = []
    
    print(f"\n{'='*70}")
    print(f"Processing season {year}")
    print(f"{'='*70}")
    
    try:
        schedule = fastf1.get_event_schedule(year)
        print(f"Loaded schedule: {len(schedule)} events\n")
        time.sleep(3)
    except Exception as e:
        print(f"Could not fetch schedule: {e}\n")
        return None

    for idx, event in schedule.iterrows():
        roundNumber = event['RoundNumber']
        
        if roundNumber == 0:
            continue
            
        print(f"Round {roundNumber:2d} - {event['EventName']:<30}")

        raceData = {}

        for code in targetSessions:
            colName = sessionMap.get(code)
            
            try:
                session = fastf1.get_session(year, roundNumber, code)

                #Uses session.laps for FP sessions
                if code.startswith('FP'):
                    print(f"  {code}...", end=" ", flush=True)
                    session.load(laps=True, telemetry=False, weather=False)
                    
                    if not hasattr(session, 'drivers') or len(session.drivers) == 0:
                        print("X")
                        time.sleep(2)
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
                        print("X (no times)")
                        continue

                    print(f"({len(fastest)} drivers)")
                    
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
                    
                    time.sleep(3)
                        
                else:
                    print(f"  {code}...", end=" ", flush=True)
                    
                    if code == 'SQ':
                        session.load(laps=True, telemetry=False, weather=False)
                    else:
                        session.load(laps=False, telemetry=False, weather=False)
                    
                    if not hasattr(session, 'drivers') or len(session.drivers) == 0:
                        print("X")
                        time.sleep(2)
                        continue
                    
                    if not hasattr(session, 'results') or session.results is None or session.results.empty:
                        print("—")
                        continue

                    print(f"{len(session.results)} drivers")

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
                        
                        raceData[key][colName] = position
                    
                    time.sleep(2)

            except Exception as e:
                error_msg = str(e)[:50]
                if "does not exist" not in error_msg:
                    print(f"{code}: {type(e).__name__}")
                time.sleep(2)
                continue

        if raceData:
            dataset.extend(raceData.values())

    print(f"\n{'='*70}")
    print(f"Total records collected: {len(dataset)}")
    print(f"{'='*70}\n")

    if not dataset:
        print(f"No data collected for {year}")
        return None
    
    datasetFinal = pd.DataFrame(dataset)
    
    posCols = ['FP1', 'FP2', 'FP3', 'Qualifying', 'SprintShootout', 'Sprint', 'Race']
    for col in posCols:
        datasetFinal[col] = pd.to_numeric(datasetFinal[col], errors='coerce')

    datasetFinal = datasetFinal.sort_values(['Season', 'RaceNumber']).reset_index(drop=True)

    filename = f"datasetV1-{year}.csv"
    filename = os.path.join('datasets/V1-datasets', filename)
    datasetFinal.to_csv(filename, index=False)
    print(f"Saved {datasetFinal.shape[0]} records to {filename}\n")
    
    return datasetFinal
