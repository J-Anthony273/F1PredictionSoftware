#Imports
import fastf1
import pandas as pd
import numpy as np
import time
import os
import warnings

from collections import defaultdict
from tqdm import tqdm

warnings.filterwarnings("ignore")

def isDNF(status: str) -> bool:
    if pd.isna(status):
        return True
    s = str(status).strip().upper()
    if s == "FINISHED":
        return False
    if s.startswith("+") and "LAP" in s:
        return False
    return True

def loadRaceData(season: int, raceNumber: int):
    session = fastf1.get_session(season, raceNumber, "R")
    session.load(laps = False, telemetry = False, weather = True, messages = False)
    return session.results, session.weather_data

def DNFLookup(season: int, raceNumber: int, results: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in results.iterrows():
        dnf = isDNF(row.get("Status", ""))
        lookup [row["FullName"]] = dnf
    return lookup

def getClassifiedPosition(results: pd.DataFrame, numDrivers: int) -> dict:
    posMap = {}
    for _, row in results.iterrows():
        if isDNF(row.get("Status", "")):
            posMap[row["FullName"]] = numDrivers + 1
        
        else:
            posMap[row["FullName"]] = int(row["Position"])

    return posMap

def driverDelta(driverName: str, teammateName: str, posMap: dict, classifiedCount: int) -> float:
    dnfPos = classifiedCount + 1

    driverPos = posMap.get(driverName, dnfPos)
    teammatePos = posMap.get(teammateName, dnfPos)

    driverDNFD = (driverPos == dnfPos)
    teammateDNFD = (teammatePos == dnfPos)

    if driverDNFD:
        return -0.1
    
    beatTeammate = (driverPos < teammatePos) or teammateDNFD

    if beatTeammate:
        if classifiedCount > 1:
            posBonus = 0.05 * (classifiedCount - driverPos)/(classifiedCount - 1)
        else: 
            posBonus = 0.05
        
        return 0.05 + posBonus
    
    else:
        if classifiedCount > 1:
            posPenalty = 0.05 * (driverPos - 1)/ (classifiedCount - 1)
        else: 
            posPenalty = 0.05
        return -posPenalty

def main(inputPath: str, outputPath: str):
    df = pd.read_csv(inputPath)

    df = df.sort_values(["Season", "RaceNumber"]).reset_index(drop = True)

    df["is_dnf"] = np.nan
    df["dnf_rate"] = np.nan
    df["is_wet_race"] = np.nan
    df["driver_form"] = np.nan
    df["team_form"] = np.nan

    currentSeason = None
    seasonDNFCounts = defaultdict(int)
    seasonRaceCounts = defaultdict(int)

    driverDeltaHistory = defaultdict(list)
    teamDeltaHistory = defaultdict(list)

    raceGroups = list(df.groupby(["Season", "RaceNumber"]))

    for (season, raceNumber), raceDF in tqdm(raceGroups, desc = "Races"):
        if season != currentSeason:
            currentSeason = season
            seasonDNFCounts = defaultdict(int)
            seasonRaceCounts = defaultdict(int)
            driverDeltaHistory = defaultdict(list)
            teamDeltaHistory = defaultdict(list)


        try:
            results, weather = loadRaceData(season, raceNumber)

        except Exception as e:
            print(f"\n Could not load {season} race {raceNumber}: {e}")
            continue

        numDrivers = len(results)
        posMap = getClassifiedPosition(results, numDrivers)
        dnfLookup = DNFLookup(season, raceNumber, results)
        classifiedCount = sum(1 for p in posMap.values() if p <= numDrivers)

        try:
            wet = int(bool(weather["Rainfall"].any()))
        except Exception:
            wet = 0

        teamToDrivers = defaultdict(list)
        for _, row in raceDF.iterrows():
            teamToDrivers[row["Team"]].append(row["Driver"])

        teamScores = {}

        for team, drivers in teamToDrivers.items():
            positions = [posMap.get(d, numDrivers + 1) for d in drivers]
            teamScores[team] = np.mean(positions)

        rankedTeams = sorted(teamScores.items(), key = lambda x: x[1])
        teamRank = {team: rank + 1 for rank, (team, _) in enumerate(rankedTeams)}

        numTeams = len(teamRank)
        teamDelta = {
                    team: round(0.1 * (1 - 2 * (rank - 1)/
                                (numTeams - 1)), 4)
                    if numTeams > 1 else 0.0
                    for team, rank in teamRank.items()
                }

        for idx, row in raceDF.iterrows():
            driver = row["Driver"]
            team = row["Team"]

            teammates = [d for d in teamToDrivers[team] if d != driver]
            teammate = teammates[0] if teammates else None

            driverDNF = int(dnfLookup.get(driver, False))

            df.at[idx, "is_dnf"] = driverDNF
            df.at[idx, "is_wet_race"] = wet

            racesSoFar = seasonRaceCounts[driver]
            if racesSoFar > 0:
                rate = seasonDNFCounts[driver] / racesSoFar
            else:
                rate = 0.0
                    
            df.at[idx, "dnf_rate"] = round(rate, 4)

            pastDeltas = driverDeltaHistory[driver][-5:]
            df.at[idx, "driver_form"] = round(
                        float(np.clip(0.5 + sum(pastDeltas), 0.0, 1.0)), 4
                        )

            pastTeamDeltas = teamDeltaHistory[team][-5:]
            df.at[idx, "team_form"] = round(
                        float(np.clip(0.5 + sum(pastTeamDeltas), 0.0, 1.0)), 4
                    )

        for _, row in raceDF.iterrows():
            driver = row["Driver"]
            team = row["Team"]

            seasonRaceCounts[driver] += 1
            if dnfLookup.get(driver, False):
                seasonDNFCounts[driver] += 1
                        
            teammates = [d for d in teamToDrivers[team] if d != driver]
            teammate = teammates[0] if teammates else None

            if teammate:
                delta = driverDelta(driver, teammate, posMap, classifiedCount)
            else:
                driverPos = posMap.get(driver, classifiedCount + 1)
                if driverPos <= classifiedCount and classifiedCount > 1:
                    delta = 0.05 * (classifiedCount - driverPos)/ (classifiedCount - 1)
                            
                else:
                    delta = -0.1

            driverDeltaHistory[driver].append(round(delta, 4))
            teamDeltaHistory[team].append(teamDelta.get(team, 0.0))

    df.to_csv(outputPath, index = False)
    print(f"\nDone. Output saved to: {outputPath}")
    print(f"Shape: {df.shape}")



inputPath = "datasets/datasetV1-filtered.csv"
outputPath = "datasets/datasetV2.csv"

main(inputPath, outputPath)