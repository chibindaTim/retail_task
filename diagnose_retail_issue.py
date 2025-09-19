import pandas as pd
import numpy as np

# Load the retail data to understand what we're predicting
print("=== RETAIL TASK DIAGNOSIS ===")
df = pd.read_csv('data/train_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for potential target columns
potential_targets = ['churned', 'total_sales', 'avg_purchase_value', 'purchase_frequency']
print(f"\nPotential target columns:")
for col in potential_targets:
    if col in df.columns:
        print(f"  {col}: {df[col].dtype}, unique values: {df[col].nunique()}")
        if df[col].nunique() < 10:
            print(f"    Value counts: {df[col].value_counts().to_dict()}")
        else:
            print(f"    Range: {df[col].min()} to {df[col].max()}")

# Check what the model was actually trained on
print(f"\n=== CHECKING CURRENT MODEL PERFORMANCE ===")
try:
    metrics = open('results/train_metrics.txt', 'r').read()
    print("Current metrics:")
    print(metrics)
except:
    print("No metrics file found")

# The issue: R² = -0.23 suggests the model is worse than predicting the mean
print(f"\n=== DIAGNOSIS ===")
print("R² = -0.23 means the model is performing WORSE than just predicting the mean value!")
print("This typically happens when:")
print("1. Wrong model type (using regression for classification)")
print("2. Poor feature engineering")
print("3. Data leakage or preprocessing errors")
print("4. Incorrect target variable")
print("5. Model not converging properly")

# Check if 'churned' is binary (likely classification problem)
if 'churned' in df.columns:
    print(f"\n'churned' column analysis:")
    print(f"  Data type: {df['churned'].dtype}")
    print(f"  Unique values: {df['churned'].unique()}")
    print(f"  Value counts: {df['churned'].value_counts()}")
    
    if df['churned'].nunique() == 2:
        print("  *** ISSUE FOUND: 'churned' is BINARY - this is a CLASSIFICATION problem!")
        print("  *** But we're using REGRESSION models (Ridge/Lasso)!")
        print("  *** This explains the terrible R² score!")
