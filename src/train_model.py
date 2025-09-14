import pickle
import pandas as pd
import numpy as np
import os
from data_preprocessing import Preprocessor


class LogisticRegression:
    """
    Custom Logistic Regression implementation for binary classification.
    Uses gradient descent with optional L1/L2 regularization.
    """
    def __init__(self, learning_rate=0.01, max_iter=1000, regularization=None, alpha=0.01):
        # Learning parameters
        self.learning_rate = learning_rate  # Step size for gradient descent
        self.max_iter = max_iter  # Maximum training iterations
        self.regularization = regularization  # 'l1', 'l2', or None
        self.alpha = alpha  # Regularization strength
        
        # Model parameters (learned during training)
        self.weights = None  # Feature weights
        self.bias = None  # Bias term
        self.cost_history = []  # Track training progress
    
    def _sigmoid(self, z):
        """
        Sigmoid activation function for logistic regression.
        Converts linear output to probability between 0 and 1.
        """
        # Clip z to prevent overflow in exp function
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _soft_threshold(self, x, thresh):
        """
        Soft thresholding operator for L1 regularization (Lasso).
        """
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)
    
    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary target labels (0 or 1)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Initialize parameters with small random values
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
        # Gradient descent training loop
        for i in range(self.max_iter):
            # Forward pass: compute predictions
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_output)
            
            # Compute cost (log-likelihood with regularization)
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Add regularization to weight gradients
            if self.regularization == 'l2':
                dw += self.alpha * self.weights
            elif self.regularization == 'l1':
                # L1 regularization applied after gradient update
                pass
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Apply L1 regularization (soft thresholding)
            if self.regularization == 'l1':
                self.weights = self._soft_threshold(
                    self.weights, 
                    self.alpha * self.learning_rate
                )
        
        return self
    
    def _compute_cost(self, y_true, y_pred):
        """
        Compute logistic regression cost (negative log-likelihood) with regularization.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            
        Returns:
            cost: Scalar cost value
        """
        # Prevent log(0) by clipping predictions
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Binary cross-entropy loss
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Add regularization penalty
        if self.regularization == 'l2':
            cost += self.alpha * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            cost += self.alpha * np.sum(np.abs(self.weights))
        
        return cost
    
    def predict(self, X):
        """
        Make binary predictions (0 or 1) using 0.5 threshold.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        X = np.asarray(X, dtype=np.float64)
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            probabilities: Predicted probabilities for class 1
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        X = np.asarray(X, dtype=np.float64)
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)


class RidgeLogisticRegression(LogisticRegression):
    """
    Logistic Regression with L2 (Ridge) regularization.
    Helps prevent overfitting by penalizing large weights.
    """
    def __init__(self, alpha=1.0, learning_rate=0.01, max_iter=1000):
        super().__init__(learning_rate=learning_rate, max_iter=max_iter, regularization='l2', alpha=alpha)


class LassoLogisticRegression(LogisticRegression):
    """
    Logistic Regression with L1 (Lasso) regularization.
    Promotes feature selection by driving some weights to zero.
    """
    def __init__(self, alpha=1.0, learning_rate=0.01, max_iter=1000):
        super().__init__(learning_rate=learning_rate, max_iter=max_iter, regularization='l1', alpha=alpha)




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


def load_data(filepath):
    return pd.read_csv(filepath)


def standardize_features(X):
    X = np.array(X, dtype=np.float64)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    std_replaced = np.where(std == 0, 1, std)
    
    X_scaled = (X - mean) / std_replaced
    return X_scaled, mean, std_replaced


def train_models(X_scaled, y):
    """
    Train multiple classification models for churn prediction.
    
    Args:
        X_scaled: Standardized feature matrix
        y: Binary target labels (0 or 1)
        
    Returns:
        models: Dictionary of trained models
    """
    models = {}
    
    print("Training Logistic Regression...")
    # Basic logistic regression - optimized for speed
    logistic_reg = LogisticRegression(
        learning_rate=0.5,  # Higher learning rate for faster convergence
        max_iter=100  # Reduced iterations for speed
    )
    logistic_reg.fit(X_scaled, y)
    models['logistic'] = logistic_reg
    
    print("Training Ridge Logistic Regression...")
    # L2 regularized logistic regression - optimized for speed
    ridge_reg = RidgeLogisticRegression(
        alpha=0.01,  # Lower regularization for faster convergence
        learning_rate=0.5, 
        max_iter=100
    )
    ridge_reg.fit(X_scaled, y)
    models['ridge'] = ridge_reg
    
    print("Training Lasso Logistic Regression...")
    # L1 regularized logistic regression - optimized for speed
    lasso_reg = LassoLogisticRegression(
        alpha=0.01,  # Lower regularization for faster convergence
        learning_rate=0.5, 
        max_iter=100
    )
    lasso_reg.fit(X_scaled, y)
    models['lasso'] = lasso_reg
    
    return models


def evaluate_models(models, X_scaled, y):
    """
    Evaluate classification models using appropriate metrics.
    
    Args:
        models: Dictionary of trained models
        X_scaled: Standardized validation features
        y: True binary labels
        
    Returns:
        best_model: Model with highest accuracy
        best_model_name: Name of the best model
    """
    best_model = None
    best_accuracy = 0.0
    best_model_name = None
    
    
    for name, model in models.items():
        # Get binary predictions and probabilities
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)
        
        # Calculate classification metrics
        accuracy = np.mean(y == y_pred)
        
        # True Positives, False Positives, True Negatives, False Negatives
        tp = np.sum((y == 1) & (y_pred == 1))
        fp = np.sum((y == 0) & (y_pred == 1))
        tn = np.sum((y == 0) & (y_pred == 0))
        fn = np.sum((y == 1) & (y_pred == 0))
        
        # Precision, Recall, F1-Score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Log Loss (Cross-entropy)
        y_proba_clipped = np.clip(y_proba, 1e-15, 1 - 1e-15)
        log_loss = -np.mean(y * np.log(y_proba_clipped) + (1 - y) * np.log(1 - y_proba_clipped))
        
        print(f"\n{name.upper()} Classification Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Log Loss: {log_loss:.4f}")
        print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        # Select best model based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    print(f"\nBest model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
    return best_model, best_model_name


def save_models(models, best_model, preprocessor):
    os.makedirs('models', exist_ok=True)
    
    model_names = ['regression_model1.pkl', 'regression_model2.pkl', 'regression_model3.pkl']
    for i, (name, model) in enumerate(models.items()):
        if i < len(model_names):
            filepath = f'models/{model_names[i]}'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {filepath}")
    
    with open('models/regression_model_final.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Saved best model to models/regression_model_final.pkl")
    
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("Saved preprocessor to models/preprocessor.pkl")


def save_model_and_preprocessor(model, preprocessor, mean, std):
    """
    Save the best model and preprocessing components.
    
    Args:
        model: Trained classification model
        preprocessor: Fitted data preprocessor
        mean: Feature means for standardization
        std: Feature standard deviations for standardization
    """
    os.makedirs('models', exist_ok=True)
    
    # Save the best model
    with open('models/regression_model1.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Best model saved to models/regression_model1.pkl")
    
    # Save the preprocessor
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("Preprocessor saved to models/preprocessor.pkl")
    
    # Save standardization parameters
    standardization_params = {'mean': mean, 'std': std}
    with open('models/standardization_params.pkl', 'wb') as f:
        pickle.dump(standardization_params, f)
    print("Standardization parameters saved to models/standardization_params.pkl")


def load_model(filepath):
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def main():
    """
    Main training pipeline for retail churn classification.
    
    Steps:
    1. Load and preprocess data
    2. Split into train/validation sets
    3. Standardize features
    4. Train multiple classification models
    5. Evaluate and select best model
    6. Save best model and preprocessing components
    """
    print("Loading and preprocessing data...")
    
    # Use smaller sample for much faster training
    sample_rows = 5000  # Fixed sample size for speed
    print(f"Using sample of {sample_rows} rows for faster training")
    
    # Load training data
    df = pd.read_csv('data/train_data.csv')
    if len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=42)
        print(f"Sampled {len(df)} rows from {pd.read_csv('data/train_data.csv').shape[0]} total rows")
    
    # Preprocess data for classification
    preprocessor = Preprocessor()
    X, y = preprocessor.fit_transform(df, 'churned')
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: Churn=1: {np.sum(y)}, No Churn=0: {np.sum(1-y)}")
    print(f"Churn rate: {np.mean(y):.2%}")
    
    # Split data for model validation
    X_train, X_val, y_train, y_val = custom_train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features (important for gradient descent)
    X_train_scaled, mean, std = standardize_features(X_train)
    X_val_scaled = (X_val - mean) / std
    
    # Train multiple classification models
    models = train_models(X_train_scaled, y_train)
    
    print("\nEvaluating models on validation set...")
    best_model, best_model_name = evaluate_models(models, X_val_scaled, y_val)
    
    # Save the best model and preprocessing components
    save_model_and_preprocessor(best_model, preprocessor, mean, std)
    
    print(f"\nTraining completed successfully!")
    print(f"Best model: {best_model_name}")
    print(f"Model and preprocessor saved to models/ directory")

if __name__ == "__main__":
    main()