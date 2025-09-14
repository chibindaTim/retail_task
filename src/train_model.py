import pickle
import pandas as pd
import numpy as np
from data_preprocessing import Preprocessor


class LassoRegression:
    def __init__(self, learning_rate=0.01, alpha=2.0, max_iter=1000):
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
    
    def _soft_threshold(self, x, thresh):
        """Soft thresholding function for L1 regularization"""
        return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        # Initialize coefficients
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent Loop
        for i in range(self.max_iter):
            # Prediction
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute residuals
            residuals = y_pred - y
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, residuals)
            db = (1/n_samples) * np.sum(residuals)
            
            # Gradient descent update with soft thresholding
            self.weights = self._soft_threshold(
                self.weights - self.learning_rate * dw,
                self.alpha * self.learning_rate
            )
            self.bias -= self.learning_rate * db
        
        return self
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias


class RidgeRegression:
    def __init__(self, alpha=2.0, lr=0.01, epochs=1000):
        self.alpha = alpha
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Ridge regularization: add alpha * weights to gradient
            dw = (1/n_samples) * (np.dot(X.T, (y_pred - y)) + self.alpha * self.weights)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias


def custom_train_test_split(X, y, test_size=0.5, random_state=42):
    """Custom implementation of train-test split"""
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.array(X)
    y = np.array(y)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Generate random indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Split data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def load_data(filepath):
    """Load data from CSV file"""
    return pd.read_csv(filepath)


def standardize_features(X):
    """Standardize features to have mean 0 and std 1"""
    X = np.array(X, dtype=np.float32)  # ensure numeric array
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Avoid division by zero for constant columns
    std_replaced = np.where(std == 0, 1, std)
    
    X_scaled = (X - mean) / std_replaced
    return X_scaled, mean, std_replaced


def train_models(X_scaled, y):
    """Train both Ridge and Lasso regression models"""
    models = {}
    
    # Ridge Regression (L2)
    print("Training Ridge Regression...")
    ridge = RidgeRegression(alpha=100, lr=0.01, epochs=500)
    ridge.fit(X_scaled, y)
    models['ridge_regression'] = ridge
    
    # Lasso Regression (L1)
    print("Training Lasso Regression...")
    lasso = LassoRegression(learning_rate=0.01, alpha=100, max_iter=500)
    lasso.fit(X_scaled, y)
    models['lasso_regression'] = lasso

    return models


def evaluate_models(models, X_scaled, y):
    """Evaluate models and return the best one"""
    best_model = None
    best_mse = float('inf')
    best_model_name = None
    
    print("\nModel Evaluation:")
    print("-" * 30)
    
    for name, model in models.items():
        y_pred = model.predict(X_scaled)
        mse = np.mean((y_pred - y)**2)
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        print(f"{name}:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        print()
        
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_model_name = name
    
    print(f"Best model: {best_model_name} with MSE: {best_mse:.4f}")
    return best_model, best_model_name


def save_models(models, best_model, preprocessor):
    """Save all models and preprocessor"""
    import os
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Save individual models
    model_names = ['regression_model1.pkl', 'regression_model2.pkl']
    for i, (name, model) in enumerate(models.items()):
        filepath = f'models/{model_names[i]}'
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved {name} to {filepath}")
    
    # Save best model
    with open('models/regression_model_final.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("Saved best model to models/regression_model_final.pkl")
    
    # Save preprocessor
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("Saved preprocessor to models/preprocessor.pkl")


def load_model(filepath):
    """Load a saved model"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def main():
    """Main training pipeline"""
    print("Starting model training pipeline...")
    print("=" * 50)
    
    try:
        # Load data
        print("Loading data...")
        data_path = 'data\Retail.csv'
        df = load_data(data_path)
        print(f"Data loaded: {df.shape}")
        target= 'avg_purchase_value'
        # Preprocess data
        print("Preprocessing data...")
        preprocessor = Preprocessor()
        X, y= preprocessor.fit_transform(df, target)
        print(f"Preprocessed data shape: X={X.shape}, y={y.shape}")
        
        # Split data (80% train, 20% validation)
        print("Splitting data...")
        X_train, X_val, y_train, y_val = custom_train_test_split(
            X, y, test_size=0.5, random_state=42
        )
        print(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
        
        # Standardize features
        print("Standardizing features...")
        X_train_scaled, mean, std = standardize_features(X_train)
        X_val_scaled = (X_val - mean) / std  # Use training statistics
        
        # Train models
        print("Training models...")
        models = train_models(X_train_scaled, y_train)
        
        # Evaluate models on validation set
        best_model, best_model_name = evaluate_models(models, X_val_scaled, y_val)
        
        print(f"\nSkipping full dataset retraining due to memory constraints.")
        print(f"Using {best_model_name} trained on {len(X_train)} samples.")
        # Retrain best model on full dataset
        #print(f"Retraining {best_model_name} on full dataset...")
        #X_full_scaled, _, _ = standardize_features(X)
        #best_model.fit(X_full_scaled, y)
        
        # Evaluate final model on validation set for reporting
        final_pred = best_model.predict(X_val_scaled)
        final_mse = np.mean((y_val - final_pred)**2)
        final_rmse = np.sqrt(final_mse)
        ss_res = np.sum((y_val - final_pred) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        final_r2 = 1 - (ss_res / ss_tot)
        
        print(f"\nFinal Model Performance on Validation Set:")
        print(f"MSE: {final_mse:.2f}")
        print(f"RMSE: {final_rmse:.2f}")
        print(f"R² Score: {final_r2:.2f}")
        # Save models and preprocessor
        print("\nSaving models...")

        if preprocessor is not None:
            save_models(models, best_model, preprocessor)
        else:
            # Save models without preprocessor if it wasn't returned
            import os
            os.makedirs('models', exist_ok=True)
            
            model_names = ['regression_model1.pkl', 'regression_model2.pkl']
            for i, (name, model) in enumerate(models.items()):
                filepath = f'models/{model_names[i]}'
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Saved {name} to {filepath}")
            
            with open('models/regression_model_final.pkl', 'wb') as f:
                pickle.dump(best_model, f)
            print("Saved best model to models/regression_model_final.pkl")
            print("Note: No preprocessor object to save")
        
        print("\n" + "=" * 50)
        print("Training pipeline completed successfully!")
        print(f"Training completed successfully with {best_model_name} as the best performing model.")

    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        raise


def demo_loading_models():
    """Demonstrate loading and using saved models"""
    try:
        print("Loading saved models...")
        
        # Load the best model
        best_model = load_model('models/regression_model_final.pkl')
        
        # Load individual models
        lasso_model = load_model('models/regression_model1.pkl')
        ridge_model = load_model('models/regression_model2.pkl')
        
        print("\nAll models loaded successfully!")
        print("You can now use these models for predictions.")
        
    except FileNotFoundError as e:
        print(f"Model files not found. Please run main() first.")
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run the main training pipeline
    main()
    
    print("\n" + "="*50)
    
    # Demonstrate loading models
    demo_loading_models()