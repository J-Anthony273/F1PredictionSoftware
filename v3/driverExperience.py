#Imports
import pandas as pd

#Driver race numbers
driverRaces = {
    'Fernando Alonso': 425,
    'Lewis Hamilton': 380,
    'Kimi Räikkönen': 349,
    'Sebastian Vettel': 299,
    'Valtteri Bottas': 246,
    'Sergio Perez': 281,
    'Nico Hulkenberg': 250,
    'Daniel Ricciardo': 257,
    'Romain Grosjean': 179,
    'Robert Kubica': 99,
    'Kevin Magnussen': 185,
    'Daniil Kvyat': 110,
    'Marcus Ericsson': 97,
    'Stoffel Vandoorne': 41,
    'Max Verstappen': 233,
    'Carlos Sainz': 229,
    'Charles Leclerc': 171,
    'Pierre Gasly': 177,
    'Esteban Ocon': 180,
    'Lance Stroll': 189,
    'Alexander Albon': 128,
    'George Russell': 152,
    'Lando Norris': 152,
    'Yuki Tsunoda': 111,
    'Antonio Giovinazzi': 62,
    'Nicholas Latifi': 61,
    'Guanyu Zhou': 68,
    'Mick Schumacher': 43,
    'Sergey Sirotkin': 21,
    'Brendon Hartley': 25,
    'Nikita Mazepin': 21,
    'Nyck De Vries': 11,
    'Logan Sargeant': 36,
    'Jack Aitken': 1,
    'Pietro Fittipaldi': 2,
    'Liam Lawson': 35,
    'Oscar Piastri': 70,
    'Oliver Bearman': 27,
    'Franco Colapinto': 26,
    'Jack Doohan': 7,
    'Isack Hadjar': 23,
    'Gabriel Bortoleto': 24,
    'Kimi Antonelli': 24,
}

df = pd.read_csv("datasets/datasetV3RatingAddition.csv")
df = df.sort_values(["Season", "RaceNumber"]).copy()

for name, total in driverRaces.items():
    driverMask = df["Driver"] == name
    driverIndices = df[driverMask].index[::-1]

    for offset, idx in enumerate(driverIndices):
        df.loc[idx, "noOfRaces"] = total - offset

highest = df["noOfRaces"].max() 
df["driverExperience"] = round(df["noOfRaces"] / highest, 4)

df.to_csv("datasets/datasetV3-3.csv", index = False)
