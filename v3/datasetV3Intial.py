#Imports
import pandas as pd

df = pd.read_csv("datasets/datasetv2.csv")
standings = pd.read_csv("datasets/f1ConstructorStandingsRaceByRace.csv")

teamMap = {
    "Red Bull Racing": "Red Bull",
    "Alpine":          "Alpine F1 Team",
    "Alfa Romeo Racing": "Alfa Romeo",
    "Kick Sauber":     "Sauber",
    "RB":              "RB F1 Team",
    "Racing Bulls":    "RB F1 Team",
}

df["_constructor"] = df["Team"].replace(teamMap)

df.loc[(df["Team"] == "Racing Point") & (df["Season"] == 2018), "_constructor"] = "Force India"

maxPts = (
    standings.groupby(["season", "round"])["points"]
    .max()
    .reset_index()
    .rename(columns = {"season": "Season", "round": "RaceNumber", "points": "max_points"})
)

standingsRenamed = standings.rename(columns={
    "season":          "Season",
    "round":           "RaceNumber",
    "constructorName": "_constructor",
})

df = df.merge(standingsRenamed, on=["Season", "RaceNumber", "_constructor"], how="left")
df = df.merge(maxPts,           on=["Season", "RaceNumber"],                 how="left")

df["carPerformance"] = (
    0.5 - (df["position"] - 1) * 0.05 + 0.5 * (df["points"]/df["max_points"])
).round(2)

df = df.drop(columns = ["_constructor", "position", "points", "max_points"])

df.to_csv("datasets/datasetV3Intial.csv", index = False)