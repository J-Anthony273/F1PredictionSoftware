import pandas as pd

df = pd.read_csv('datasets/datasetV3RatingAddition.csv')

print(df.isnull().sum())