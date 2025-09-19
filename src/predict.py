#!/usr/bin/env python3

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from data_preprocessing import Preprocessor

# Metrics functions (same as train_model implementation)
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


def load_preprocessor_state(preproc_state_path: str) -> Preprocessor:
    if not os.path.exists(preproc_state_path):
        raise FileNotFoundError(f"Preprocessor state file not found: {preproc_state_path}")
    with open(preproc_state_path, 'rb') as f:
        state = pickle.load(f)
    preproc = Preprocessor()
    preproc.load_state(state)
    return preproc


def load_standardization_params(std_path: str):
    if not os.path.exists(std_path):
        raise FileNotFoundError(f"Standardization params file not found: {std_path}")
    with open(std_path, 'rb') as f:
        params = pickle.load(f)
    mean = np.array(params['mean'], dtype=float)
    std = np.array(params['std'], dtype=float)
    feature_columns = params.get('feature_columns', None)
    return mean, std, feature_columns


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved regression model on dataset.")
    parser.add_argument('--model_path', required=True, help='Path to saved model pickle (regression_model_final.pkl recommended).')
    parser.add_argument('--data_path', required=True, help='Path to CSV file containing features and true labels.')
    parser.add_argument('--metrics_output_path', required=True, help='Path to write metrics file (train_metrics.txt).')
    parser.add_argument('--predictions_output_path', required=True, help='Path to write predictions CSV (single column, no header).')
    parser.add_argument('--preprocessor_path', default='models/preprocessor.pkl', help='Path to saved preprocessor state (pickle).')
    parser.add_argument('--std_params_path', default='models/standardization_params.pkl', help='Path to standardization params (pickle).')
    parser.add_argument('--target_col', required=True, help='Name of target column in data_path (used to compute metrics).')
    args = parser.parse_args()

    # Basic checks
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    # Load model
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)

    # Load preprocessor state and instantiate preprocessor
    preproc = load_preprocessor_state(args.preprocessor_path)

    # Load standardization params
    mean, std, feature_columns = load_standardization_params(args.std_params_path)

    # Load data
    df = pd.read_csv(args.data_path)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in data file.")

    # Drop rows with missing target (consistent with training)
    df = df.dropna(subset=[args.target_col]).reset_index(drop=True)
    
    # Preprocess features (returns X and y)
    X, y_true = preproc.transform(df, args.target_col)

    # Align features with standardization params if feature_columns were saved
    if feature_columns is not None:
        # If standardization params had a specific ordering, reorder accordingly
        if list(feature_columns) != list(X.columns):
            # We expect preprocessor.feature_columns to match params' feature_columns
            # If mismatch, attempt to reindex (add missing cols as zeros)
            for col in feature_columns:
                if col not in X.columns:
                    X[col] = 0
            # Drop extra columns
            extra = [c for c in X.columns if c not in feature_columns]
            if extra:
                X = X.drop(columns=extra)
            X = X[feature_columns]

    X_values = X.values.astype(float)

    # Standardize with saved mean/std
    if mean.shape[0] != X_values.shape[1]:
        raise ValueError("Dimension mismatch between saved standardization params and input features.")
    X_scaled = (X_values - mean) / std

    # Predict
    y_pred = model.predict(X_scaled)

    # Save predictions CSV (single column, no header)
    pred_dir = os.path.dirname(args.predictions_output_path)
    if pred_dir and not os.path.exists(pred_dir):
        os.makedirs(pred_dir, exist_ok=True)
    pd.DataFrame(y_pred).to_csv(args.predictions_output_path, index=False, header=False)
    print(f"Saved predictions -> {args.predictions_output_path}")

    # Compute metrics and save in the strict required format
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics_text = (
        "Regression Metrics:\n"
        f"Mean Squared Error (MSE): {mse:.2f}\n"
        f"Root Mean Squared Error (RMSE): {rmse:.2f}\n"
        f"R-squared (R²) Score: {r2:.2f}\n"
    )

    # Ensure metrics directory exists
    metrics_dir = os.path.dirname(args.metrics_output_path)
    if metrics_dir and not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)

    with open(args.metrics_output_path, 'w') as f:
        f.write(metrics_text)
    print(f"Saved metrics -> {args.metrics_output_path}")

    # also print a small summary to stdout
    print("\nEvaluation Summary:")
    print(metrics_text)


if __name__ == "__main__":
    main()
