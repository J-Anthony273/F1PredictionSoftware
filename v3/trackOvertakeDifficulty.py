#Imports
import numpy as np
import pandas as pd

df = pd.read_csv("datasets/datasetV3-3.csv")

circuits = {
    'Austin'      : {1:6, 2:5, 4:1, 6:1},
    'Baku'        : {1:3, 2:2, 3:2, 6:1, 10:1},
    'Barcelona'   : {1:9, 2:4, 4:1, 5:1},
    'Miami'       : {3:1, 4:1, 5:1, 9:1},
    'Miami Gardens': {3:1, 4:1, 5:1, 9:1},
    'Hockenheim'  : {1:2, 2:2, 14:1},
    'Budapest' : {1:5, 2:3, 3:4, 4:1, 8:1, 10:1},
    'Imola'       : {1:2, 2:2, 3:1},
    'São Paulo'  : {1:9, 2:3, 10:1, 17:1},
    'Istanbul'    : {1:2, 6:1},
    'Jeddah'      : {1:3, 2:1, 4:1},
    'Las Vegas'       : {1:1, 2:2},
    'Le Castellet'   : {1:3, 2:1},
    'Lusail'      : {1:2, 2:1, 3:1},
    'Singapore'   : {1:9, 2:1, 3:2, 5:1},
    'Marina Bay'        : {1:9,  2:1,  3:2,  5:1},
    'Melbourne'   : {1:5, 2:5, 3:2, 7:1},
    'Mexico City'      : {1:5, 2:2, 3:3},
    'Monaco'      : {1:9, 2:3, 3:2},
    'Monte Carlo'      : {1:9, 2:3, 3:2},
    'Montreal'    : {1:8, 2:3, 6:1, 7:1},
    'Montréal'          : {1:8,  2:3,  6:1,  7:1},
    'Monza'       : {1:8, 2:3, 3:1, 4:1, 7:1, 10:1},
    'Mugello'     : {1:1},
    'Nürburgring' : {2:3},
    'Portimão'    : {1:1, 2:1},
    'Sakhir'      : {1:8, 2:4, 3:2, 5:1},
    'Shanghai'    : {1:7, 2:1, 3:2, 6:1},
    'Silverstone' : {1:6, 2:6, 3:2, 4:1, 6:1},
    'Sochi'       : {1:2, 2:3, 3:2, 4:1},
    'Spa-Francorchamps'         : {1:8, 2:3, 3:1, 5:1, 6:1, 14:1},
    'Spielberg'   : {1:8, 2:3, 3:2, 4:1},
    'Suzuka'      : {1:8, 2:4, 3:1},
    'Yas Marina'  : {1:11, 2:3, 4:1},
    'Yas Island'        : {1:11, 2:3,  4:1},
    'Zandvoort'   : {1:5},
}

GLOBAL_POLE = 0.529 #pole to win percentage since 2011
PRIOR_RACES = 5   
K = 5             


def smoothed_pole_pct(data):
    total = sum(data.values())
    pole_wins = data.get(1, 0)
    return (pole_wins + PRIOR_RACES * GLOBAL_POLE) / (total + PRIOR_RACES)

def difficulty_score(data):
    total = sum(data.values())
    
    pole_pct = smoothed_pole_pct(data)
    avg_pos = sum(p * w for p, w in data.items()) / total
    
    top3_wins = sum(w for p, w in data.items() if p <= 3)
    top3_pct = top3_wins / total

    raw_score = 0.45 * pole_pct + 0.4 * top3_pct + 0.15 * (1 / avg_pos)
    
    reliability = total / (total + K)
    
    return raw_score * reliability

raw_scores = {}

for circuit, data in circuits.items():
    raw_scores[circuit] = difficulty_score(data)


min_val = min(raw_scores.values())
max_val = max(raw_scores.values())

difficulty = {}

for circuit in raw_scores:
    difficulty[circuit] = round(
        (raw_scores[circuit] - min_val) / (max_val - min_val),
        4
    )

df["trackOvertakeDifficulty"] = df["Location"].map(difficulty)

print(f"Missing values: {df['trackOvertakeDifficulty'].isna().sum()}")
print(df[['Location', 'trackOvertakeDifficulty']].drop_duplicates().sort_values('trackOvertakeDifficulty', ascending=False).to_string(index=False))
 
df.to_csv('datasets/datasetV3-3.csv', index=False)
print("\nSaved updated dataset to datasets/datasetV3-3.csv")