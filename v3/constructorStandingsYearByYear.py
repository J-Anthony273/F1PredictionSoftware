#Imports
import requests
import pandas as pd
import time

startYear = 2018
endYear = 2025

dataRows = []

for year in range(startYear, endYear + 1):
    roundNum = 1
    
    while True:
        print(f"Processing {year} round {roundNum}")
        url = f"https://api.jolpi.ca/ergast/f1/{year}/{roundNum}/constructorStandings.json"


        while True:
            try:
                response = requests.get(url, timeout=10)
                r = response.json()
                break
            except:
                print("Request failed, retrying in 2 seconds...")
                time.sleep(2)

        standingsLists = r["MRData"]["StandingsTable"]["StandingsLists"]

        if not standingsLists:
            break

        standings = standingsLists[0]["ConstructorStandings"]

        for s in standings:
            constructor = s.get("Constructor", {})

            dataRows.append({
                "season" : year,
                "round" : roundNum,
                "position": int(s.get("position", 0)),
                "constructorName": constructor.get("name", ""),
                "points": float(s.get("points", 0))
            })

        roundNum += 1
        time.sleep(0.5)

df = pd.DataFrame(dataRows)
df.to_csv("datasets/f1ConstructorStandingsRaceByRace.csv", index = False)

print(df.head())
    