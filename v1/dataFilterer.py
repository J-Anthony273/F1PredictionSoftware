#Imports
import pandas as pd
import os
import numpy as np

#Function that filters the dataset and fills missing values
def filterF1Data(csv_path, requireRace, minFPSessions, requireAllFP, outputPath):
    df = pd.read_csv(csv_path)
    originalCount = len(df)

    #Gets rid of sprint-related columns for years where the sessions didn't exist
    df.loc[df['Season'].isin([2021, 2022]), 'SprintShootout'] = np.nan
    df.loc[df['Season'] < 2021, 'Sprint'] = np.nan

    #Creates masks for the filters.
    mask = pd.Series([True] * len(df))

    #Removes rows where there was not a race result
    if requireRace:
        mask &= df['Race'].notna()

    if requireAllFP:
        mask &= (df['FP1'].notna()) & (df['FP2'].notna()) & (df['FP3'].notna())
    
    elif minFPSessions > 0:
        fpCount = df[['FP1', 'FP2', 'FP3']].notna().sum(axis=1)
        mask &= fpCount >= minFPSessions

    dfFiltered = df[mask].copy()

    #Adds binary indicator for when sprints were held
    dfFiltered['SprintHeld'] = dfFiltered['Sprint'].notna().astype(int)
    dfFiltered['SprintShootoutHeld'] = dfFiltered["SprintShootout"].notna().astype(int)

    #Fills the sprint columns with 0 to represent them not happening
    dfFiltered['Sprint'] = dfFiltered['Sprint'].fillna(0)
    dfFiltered['SprintShootout'] = dfFiltered['SprintShootout'].fillna(0)

    #Fills the FP sessions and qualifying with 21 if they did not take part
    dfFiltered[['FP1', 'FP2', 'FP3', 'Qualifying']] = dfFiltered[['FP1', 'FP2', 'FP3', 'Qualifying']].fillna(21)

    numericCols = ['Season', 'RaceNumber', 'FP1', 'FP2', 'FP3', 'Qualifying',
                   'SprintShootout', 'Sprint', 'Race', 'SprintHeld', 'SprintShootoutHeld']
    dfFiltered[numericCols] = dfFiltered[numericCols].astype(int)

    fullOutputPath = os.path.join('datasets/V1-datasets', outputPath)
    dfFiltered.to_csv(fullOutputPath, index = False)

    print(f"Filtered dataset complete: {len(dfFiltered)}/{originalCount} rows kept, saved to {fullOutputPath}")

    return dfFiltered

df = filterF1Data("datasets/V1-datasets/datasetV1-combined.csv", requireRace = True, minFPSessions = 0, requireAllFP = False, outputPath = "datasetV1-filtered.csv")