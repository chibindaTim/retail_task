import pandas as pd
import numpy as np

# Read the predictions CSV file
df = pd.read_csv('results/train_predictions.csv', header=None, names=['predictions'])

print(f'Total predictions: {len(df)}')
print(f'Unique predictions: {df["predictions"].nunique()}')
print(f'Are all predictions different? {df["predictions"].nunique() == len(df)}')

print(f'\nFirst 10 predictions:')
print(df.head(10))

print(f'\nLast 10 predictions:')
print(df.tail(10))

print(f'\nSummary statistics:')
print(df.describe())

# Check for duplicates
duplicates = df[df.duplicated()]
if len(duplicates) > 0:
    print(f'\nFound {len(duplicates)} duplicate predictions:')
    print(duplicates)
else:
    print('\nNo duplicate predictions found.')

# Check the range of predictions
print(f'\nPrediction range:')
print(f'Min: {df["predictions"].min():.6f}')
print(f'Max: {df["predictions"].max():.6f}')
print(f'Range: {df["predictions"].max() - df["predictions"].min():.6f}')
