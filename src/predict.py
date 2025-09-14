import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pickle
import pandas as pd
import numpy as np
from data_preprocessing import Preprocessor
from train_model import LassoRegression, RidgeRegression

def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    with open('models/regression_model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    return model, preprocessor

def standardize_features(X, mean=None, std=None, chunk_size=10000):
    """Standardize features to have mean 0 and std 1"""
    X = np.array(X, dtype=np.float64)
    n_samples,n_features=X.shape
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    
    # Avoid division by zero for constant columns
    std_replaced = np.where(std == 0, 1, std)
    #Process in chunks to avoid memory issues
    X_scaled = np.empty_like(X)
    for i in range (0,n_samples,chunk_size):
        end=min(i+chunk_size , n_samples)
        chunk=X[i:end]
        X_scaled[i:end]=(chunk-mean)/std_replaced
    return X_scaled, mean, std_replaced

def generate_predictions(chunk_size=50000):
    """Generate predictions for the training data"""
    print("Loading data...")
    df = pd.read_csv(r'data\Retail.csv')
    
    print("Loading model and preprocessor...")
    model, preprocessor = load_model_and_preprocessor()
    
    print("Preprocessing data...")
    X, y = preprocessor.fit_transform(df, 'avg_purchase_value')
    
    print("Standardizing features in chunks...")
    X_scaled, mean, std = standardize_features(X, chunk_size=chunk_size)
    
    print("Generating predictions in chunks...")
    predictions = np.empty(len(X_scaled))
    for i in range(0, len(X_scaled), chunk_size):
        end=min(i + chunk_size, len(X_scaled))
        chunk = X_scaled[i:end]
        predictions[i:end]=model.predict(chunk)
        print(f'Processed {end}/{len(X_scaled)} samples')
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Actual_value': y.values,
        'Predicted_value': predictions,
        'Residual': y.values - predictions,
        'Absolute_Error': np.abs(y.values - predictions),
        'Percentage_Error': np.abs((y.values - predictions) / y.values) * 100
    })
    
    # Add some sample features for context
    results_df['income_bracket'] = df['income_bracket'].values
    results_df['total_transactions'] = df['total_transactions'].values
    results_df['customer_support_calls'] = df['customer_support_calls'].values
    
    
    # Save predictions
    print(f"Saving predictions in chunks")
    for i in range(0, len(results_df), chunk_size):
        end=min(i+chunk_size, len(results_df))
        chunk_df= results_df.iloc[i:end]
        if i==0:
            chunk_df.to_csv('results/train_predictions.csv', index=False)
        else:
            chunk_df.to_csv('results/train_predictions.csv', mode= 'a', header=False, index=False)
        print(f"Saved {end}/ {len(X_scaled)} predictions")
    
       
    print(f"Predictions saved to results/train_predictions.csv")
    
    # Print some statistics
    mse = np.mean((y.values - predictions)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y.values - predictions))
    r2 = 1 - (np.sum((y.values - predictions)**2) / np.sum((y.values - np.mean(y.values))**2))
    
    print(f"\nPrediction Statistics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Percentage Error: {results_df['Percentage_Error'].mean():.2f}%")
    
    return results_df

if __name__ == "__main__":
    results = generate_predictions(25000)
    print("\nFirst 10 predictions:")
    print(results[['customer_support_calls', 'total_transactions', 'income_bracket', 'Actual_value', 'Predicted_value', 'Percentage_Error']].head(10))