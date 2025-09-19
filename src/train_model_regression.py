import pickle
import pandas as pd
import numpy as np
import os
from data_preprocessing import Preprocessor


class LinearRegression:
    """
    Custom Linear Regression implementation using normal equation.
    """
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if self.fit_intercept:
            # Add bias column
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_with_bias = X
        
        # Normal equation with regularization for numerical stability
        XtX = X_with_bias.T @ X_with_bias
        reg = 1e-8 * np.eye(XtX.shape[0])
        theta = np.linalg.solve(XtX + reg, X_with_bias.T @ y)
        
        if self.fit_intercept:
            self.bias = theta[0]
            self.weights = theta[1:]
        else:
            self.bias = 0
            self.weights = theta
        
        return self
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias


class RidgeRegression:
    """
    Ridge Regression with L2 regularization using gradient descent.
    """
    def __init__(self, alpha=1.0, learning_rate=0.01, max_iter=1000, fit_intercept=True):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0 if self.fit_intercept else 0.0
        
        # Gradient descent
        for i in range(self.max_iter):
            # Predictions
            y_pred = X @ self.weights + self.bias
            
            # Cost (MSE + L2 penalty)
            mse = np.mean((y - y_pred) ** 2)
            l2_penalty = self.alpha * np.sum(self.weights ** 2)
            cost = mse + l2_penalty
            self.cost_history.append(cost)
            
            # Gradients
            dw = (2/n_samples) * X.T @ (y_pred - y) + 2 * self.alpha * self.weights
            db = (2/n_samples) * np.sum(y_pred - y) if self.fit_intercept else 0
            
            # Updates
            self.weights -= self.learning_rate * dw
            if self.fit_intercept:
                self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias


class LassoRegression:
    """
    Lasso Regression with L1 regularization using coordinate descent.
    """
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-6, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = None
    
    def _soft_threshold(self, x, thresh):
        """Soft thresholding for L1 regularization."""
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = np.mean(y) if self.fit_intercept else 0.0
        
        # Center y if fitting intercept
        if self.fit_intercept:
            y_centered = y - self.bias
        else:
            y_centered = y
        
        # Coordinate descent
        for iteration in range(self.max_iter):
            weights_old = self.weights.copy()
            
            for j in range(n_features):
                # Compute residual excluding feature j
                residual = y_centered - X @ self.weights + X[:, j] * self.weights[j]
                
                # Coordinate update with soft thresholding
                rho = X[:, j] @ residual
                z = np.sum(X[:, j] ** 2)
                
                if z != 0:
                    self.weights[j] = self._soft_threshold(rho / z, self.alpha / z)
                else:
                    self.weights[j] = 0
            
            # Check convergence
            if np.linalg.norm(self.weights - weights_old) < self.tol:
                break
        
        return self
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        X = np.asarray(X, dtype=np.float64)
        return X @ self.weights + self.bias


def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.array(X)
    y = np.array(y)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def standardize_features(X):
    X = np.array(X, dtype=np.float64)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    std_replaced = np.where(std == 0, 1, std)
    
    X_scaled = (X - mean) / std_replaced
    return X_scaled, mean, std_replaced


def train_models(X_scaled, y):
    """
    Train multiple regression models for avg_purchase_value prediction.
    
    Args:
        X_scaled: Standardized feature matrix
        y: Continuous target values (avg_purchase_value)
        
    Returns:
        models: Dictionary of trained models
    """
    models = {}
    
    print("Training Linear Regression...")
    linear_reg = LinearRegression(fit_intercept=True)
    linear_reg.fit(X_scaled, y)
    models['linear'] = linear_reg
    
    print("Training Ridge Regression...")
    ridge_reg = RidgeRegression(
        alpha=1.0,
        learning_rate=0.01,
        max_iter=500,
        fit_intercept=True
    )
    ridge_reg.fit(X_scaled, y)
    models['ridge'] = ridge_reg
    
    print("Training Lasso Regression...")
    lasso_reg = LassoRegression(
        alpha=0.1,
        max_iter=500,
        fit_intercept=True
    )
    lasso_reg.fit(X_scaled, y)
    models['lasso'] = lasso_reg
    
    return models


def evaluate_models(models, X_scaled, y):
    """
    Evaluate regression models using appropriate metrics.
    
    Args:
        models: Dictionary of trained models
        X_scaled: Standardized validation features
        y: True continuous target values
        
    Returns:
        best_model: Model with lowest RMSE
        best_model_name: Name of the best model
    """
    best_model = None
    best_rmse = float('inf')
    best_model_name = None
    
    print("\nModel Evaluation:")
    print("-" * 50)
    
    for name, model in models.items():
        # Get predictions
        y_pred = model.predict(X_scaled)
        
        # Calculate regression metrics
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        print(f"\n{name.upper()} Regression Results:")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")
        
        # Select best model based on RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} (RMSE: {best_rmse:.4f})")
    return best_model, best_model_name


def save_models(models, best_model, preprocessor, mean, std):
    """
    Save all models and preprocessing components.
    """
    os.makedirs('models', exist_ok=True)
    
    # Save individual models
    model_names = ['regression_model1.pkl', 'regression_model2.pkl', 'regression_model3.pkl']
    for i, (name, model) in enumerate(models.items()):
        if i < len(model_names):
            filepath = f'models/{model_names[i]}'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {filepath}")
    
    # Save best model
    with open('models/regression_model_final.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Saved best model to models/regression_model_final.pkl")
    
    # Save the preprocessor
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("Preprocessor saved to models/preprocessor.pkl")
    
    # Save standardization parameters
    standardization_params = {
        'mean': mean, 
        'std': std, 
        'feature_columns': preprocessor.feature_columns
    }
    with open('models/standardization_params.pkl', 'wb') as f:
        pickle.dump(standardization_params, f)
    print("Standardization parameters saved to models/standardization_params.pkl")


def main():
    """
    Main training pipeline for retail avg_purchase_value regression.
    """
    print("Loading and preprocessing data for avg_purchase_value regression...")
    
    # Use sample for faster training
    sample_rows = 10000
    print(f"Using sample of {sample_rows} rows for faster training")
    
    # Load training data
    df = pd.read_csv('data/train_data.csv')
    if len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=42)
        print(f"Sampled {len(df)} rows from original dataset")
    
    # Preprocess data for regression
    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df, 'avg_purchase_value')
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target (avg_purchase_value) stats:")
    print(f"  Mean: {np.mean(y):.2f}")
    print(f"  Std:  {np.std(y):.2f}")
    print(f"  Min:  {np.min(y):.2f}")
    print(f"  Max:  {np.max(y):.2f}")
    
    # Split data for model validation
    X_train, X_val, y_train, y_val = custom_train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    X_train_scaled, mean, std = standardize_features(X_train)
    X_val_scaled = (X_val - mean) / std
    
    # Train multiple regression models
    models = train_models(X_train_scaled, y_train)
    
    # Evaluate models
    best_model, best_model_name = evaluate_models(models, X_val_scaled, y_val)
    
    # Save models and preprocessing components
    save_models(models, best_model, preprocessor, mean, std)
    
    # Generate final metrics in required format
    y_pred = best_model.predict(X_val_scaled)
    mse = np.mean((y_val - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_val - y_pred) ** 2)
    ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Save metrics in required format
    os.makedirs('results', exist_ok=True)
    with open('results/train_metrics.txt', 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")
        f.write(f"R-squared (R²) Score: {r2:.2f}\n")
    
    # Save predictions
    np.savetxt('results/train_predictions.csv', y_pred, delimiter=',', fmt='%.6f')
    
    print(f"\nTraining completed successfully!")
    print(f"Best model: {best_model_name}")
    print(f"Final metrics: MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")


if __name__ == "__main__":
    main()
