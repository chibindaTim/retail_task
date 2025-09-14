#!/usr/bin/env python3

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict


class Preprocessor:
    """Data preprocessor for retail customer data."""
    
    def __init__(self, top_k: int = 10):
        self.top_k = top_k
        self.feature_columns = None
        self.categorical_encodings = {}
        self.numerical_stats = {}
        
    def fit_transform(self, df: pd.DataFrame, target_col=None) -> pd.DataFrame:
        """Fit preprocessor and transform data."""
        df = df.copy()
        
        # Identify column types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Handle missing values
        for col in numerical_cols:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                self.numerical_stats[col] = {'median': median_val}
        
        for col in categorical_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna('Unknown')
        
        # Encode categorical variables
        for col in categorical_cols:
            # Get top categories
            top_cats = df[col].value_counts().head(self.top_k).index.tolist()
            self.categorical_encodings[col] = top_cats
            
            # Map rare categories to 'Other'
            df[col] = df[col].apply(lambda x: x if x in top_cats else 'Other')
            
            # One-hot encode
            dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_')
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        
        # Store feature columns
        self.feature_columns = df.columns.tolist()
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted parameters."""
        df = df.copy()
        
        # Handle missing values using stored stats
        for col, stats in self.numerical_stats.items():
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(stats['median'])
        
        # Handle categorical columns
        for col, top_cats in self.categorical_encodings.items():
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                df[col] = df[col].apply(lambda x: x if x in top_cats else 'Other')
                
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, prefix_sep='_')
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        
        # Ensure same columns as training
        if self.feature_columns:
            # Add missing columns
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Remove extra columns and reorder
            df = df[self.feature_columns]
        
        return df
    
    def save_state(self) -> dict:
        """Save preprocessor state for later loading."""
        return {
            'top_k': self.top_k,
            'feature_columns': self.feature_columns,
            'categorical_encodings': self.categorical_encodings,
            'numerical_stats': self.numerical_stats
        }
    
    def load_state(self, state: dict):
        """Load preprocessor state."""
        self.top_k = state['top_k']
        self.feature_columns = state['feature_columns']
        self.categorical_encodings = state['categorical_encodings']
        self.numerical_stats = state['numerical_stats']


class LinearRegressionNormal:
    """Ordinary Least Squares via normal equation (with intercept)."""
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None  # weights (without intercept)
        self.intercept_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        # X shape: (n_samples, n_features)
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_aug = X
        # Solve normal equation with small regularization for numerical stability
        xtx = X_aug.T.dot(X_aug)
        # add small identity to diagonal to avoid singularities
        reg = 1e-8 * np.eye(xtx.shape[0])
        w = np.linalg.solve(xtx + reg, X_aug.T.dot(y))
        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w
        return self

    def predict(self, X: np.ndarray):
        return X.dot(self.coef_) + self.intercept_


class RidgeRegression:
    """Ridge regression solved by closed-form solution."""
    def __init__(self, alpha: float = 1.0, fit_intercept: bool = True):
        self.alpha = float(alpha)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.fit_intercept:
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        else:
            X_aug = X
        n_features = X_aug.shape[1]
        I = np.eye(n_features)
        # Do not regularize intercept term
        if self.fit_intercept:
            I[0, 0] = 0.0
        xtx = X_aug.T.dot(X_aug)
        w = np.linalg.solve(xtx + self.alpha * I, X_aug.T.dot(y))
        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w
        return self

    def predict(self, X: np.ndarray):
        return X.dot(self.coef_) + self.intercept_


class LassoRegression:
    """
    Lasso via coordinate descent (standardization recommended for speed & convergence).
    Simple implementation; not highly optimized but works for small/medium feature counts.
    """
    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-6, fit_intercept: bool = True):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = 0.0

    @staticmethod
    def _soft_threshold(rho, lam):
        if rho < -lam:
            return rho + lam
        elif rho > lam:
            return rho - lam
        else:
            return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        Xc = X.copy()
        y_c = y.copy()

        # Initialize weights
        w = np.zeros(n_features)
        b = 0.0

        if self.fit_intercept:
            b = y_c.mean()
            y_c = y_c - b

        # Precompute column norms
        col_norms = (Xc ** 2).sum(axis=0)

        for it in range(self.max_iter):
            w_old = w.copy()
            for j in range(n_features):
                # residual excluding feature j
                r_j = y_c - Xc.dot(w) + Xc[:, j] * w[j]
                rho = (Xc[:, j] * r_j).sum()
                if col_norms[j] == 0:
                    w[j] = 0.0
                else:
                    w[j] = self._soft_threshold(rho, self.alpha / 2.0) / col_norms[j]
            # convergence check (L2 change)
            if np.linalg.norm(w - w_old, ord=2) < self.tol:
                break

        self.coef_ = w
        self.intercept_ = b if self.fit_intercept else 0.0
        return self

    def predict(self, X: np.ndarray):
        return X.dot(self.coef_) + self.intercept_


# ---------------------------
# Utilities
# ---------------------------
def custom_train_test_split(X: np.ndarray, y: np.ndarray, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_test = int(n * test_size)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0


# ---------------------------
# Training pipeline
# ---------------------------
def train_pipeline(data_path: str,
                   target_col: str,
                   top_k: int = 10,
                   sample_rows: int = None,
                   test_size: float = 0.2,
                   random_state: int = 42,
                   output_dir: str = "."):
    """
    Train pipeline:
     - load data_path (CSV)
     - fit Preprocessor
     - split, standardize
     - train Linear, Ridge, Lasso
     - evaluate on validation
     - save models and preprocessing artifacts
    """
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, 'models')
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("Loading data...")
    # Load data in chunks to avoid memory issues
    if sample_rows is not None and sample_rows > 0:
        # Read only the needed rows for sampling
        df = pd.read_csv(data_path, nrows=min(sample_rows * 5, 50000))  # Read 5x sample size or max 50k
        if len(df) > sample_rows:
            df = df.sample(n=sample_rows, random_state=random_state)
        df = df.reset_index(drop=True)
        print(f"Loaded and sampled {len(df)} rows for training.")
    else:
        # For full dataset, use chunking
        chunk_size = 10000
        chunks = []
        for chunk in pd.read_csv(data_path, chunksize=chunk_size):
            chunks.append(chunk)
            if len(chunks) * chunk_size >= 100000:  # Limit to 100k rows max
                break
        df = pd.concat(chunks, ignore_index=True)
        print(f"Loaded {len(df)} rows using chunked reading.")

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    
    # Convert target column to numeric (handle 'Yes'/'No' values)
    if df[target_col].dtype == 'object':
        # Map string values to numeric
        target_mapping = {'Yes': 1.0, 'No': 0.0, 'yes': 1.0, 'no': 0.0}
        df[target_col] = df[target_col].map(target_mapping)
        # Fill any unmapped values with 0
        df[target_col] = df[target_col].fillna(0.0)
    
    y = df[target_col].astype(float).values
    X_df = df.drop(columns=[target_col])

    # Preprocess features
    preproc = Preprocessor(top_k=top_k)
    X = preproc.fit_transform(X_df, target_col=None)  # returns DataFrame
    X_values = X.values.astype(float)

    # Train/validation split
    X_train, X_val, y_train, y_val = custom_train_test_split(X_values, y, test_size=test_size, random_state=random_state)
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    # Standardize using train stats
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std_replaced = np.where(std == 0, 1.0, std)
    X_train_scaled = (X_train - mean) / std_replaced
    X_val_scaled = (X_val - mean) / std_replaced

    # Train models
    models = {}
    print("Training LinearRegressionNormal...")
    lin = LinearRegressionNormal(fit_intercept=True).fit(X_train_scaled, y_train)
    models['linear'] = lin

    print("Training RidgeRegression (alpha=1.0)...")
    ridge = RidgeRegression(alpha=1.0, fit_intercept=True).fit(X_train_scaled, y_train)
    models['ridge'] = ridge

    print("Training LassoRegression (alpha=0.1, max_iter=1000)...")
    lasso = LassoRegression(alpha=0.1, max_iter=1000, fit_intercept=True).fit(X_train_scaled, y_train)
    models['lasso'] = lasso

    # Evaluate on validation
    evals = {}
    for name, model in models.items():
        y_pred = model.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        evals[name] = {'mse': mse, 'rmse': rmse, 'r2': r2}
        print(f"{name} -> MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # Choose best model by lowest RMSE
    best_name = min(evals.keys(), key=lambda n: evals[n]['rmse'])
    best_model = models[best_name]
    print(f"Best model: {best_name}")

    # Save models (ordered)
    model_files = ['regression_model1.pkl', 'regression_model2.pkl', 'regression_model3.pkl']
    for i, (name, model) in enumerate(models.items()):
        if i < len(model_files):
            path = os.path.join(models_dir, model_files[i])
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} -> {path}")

    # Save final best model
    final_path = os.path.join(models_dir, 'regression_model_final.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Saved final model -> {final_path}")

    # Save preprocessor state
    preproc_path = os.path.join(models_dir, 'preprocessor.pkl')
    with open(preproc_path, 'wb') as f:
        pickle.dump(preproc.save_state(), f)
    print(f"Saved preprocessor state -> {preproc_path}")

    # Save standardization params
    std_params = {'mean': mean.tolist(), 'std': std_replaced.tolist(), 'feature_columns': preproc.feature_columns}
    std_path = os.path.join(models_dir, 'standardization_params.pkl')
    with open(std_path, 'wb') as f:
        pickle.dump(std_params, f)
    print(f"Saved standardization params -> {std_path}")

    # Save validation metrics for best model in required format (to results/train_metrics.txt)
    best_metrics = evals[best_name]
    metrics_text = (
        "Regression Metrics:\n"
        f"Mean Squared Error (MSE): {best_metrics['mse']:.2f}\n"
        f"Root Mean Squared Error (RMSE): {best_metrics['rmse']:.2f}\n"
        f"R-squared (R²) Score: {best_metrics['r2']:.2f}\n"
    )
    metrics_path = os.path.join(results_dir, 'train_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(metrics_text)
    print(f"Saved metrics -> {metrics_path}")

    # Save predictions for the validation set (single column CSV, no header)
    val_preds = best_model.predict(X_val_scaled)
    preds_path = os.path.join(results_dir, 'train_predictions.csv')
    # Save as single column with no header
    pd.DataFrame(val_preds).to_csv(preds_path, index=False, header=False)
    print(f"Saved validation predictions -> {preds_path}")

    return {
        'models': models,
        'best_model_name': best_name,
        'best_metrics': best_metrics,
        'models_dir': models_dir,
        'results_dir': results_dir
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train regression models (no ML libs).")
    parser.add_argument('--data_path', required=True, help='Path to CSV training data (must include target column).')
    parser.add_argument('--target_col', required=True, help='Name of target column in CSV.')
    parser.add_argument('--sample_rows', type=int, default=None, help='If set, sample this many rows for faster experiments.')
    parser.add_argument('--output_dir', default='.', help='Output directory to store models/ and results/')
    args = parser.parse_args()

    train_pipeline(data_path=args.data_path,
                   target_col=args.target_col,
                   sample_rows=args.sample_rows,
                   output_dir=args.output_dir)
