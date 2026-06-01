import pandas as pd

history = pd.read_csv('datasets/historicalResults.csv')

history['Driver'] = history['Driver'].replace({
    'Nico Hülkenberg' : 'Nico Hulkenberg',
    'Sergio Pérez'    : 'Sergio Perez'
})

history.to_csv('datasets/historicalResults.csv', index=False)
print("Done — names aligned")