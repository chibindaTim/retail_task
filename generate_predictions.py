#!/usr/bin/env python3

import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predict import generate_predictions
# Import classification models for pickle loading
from train_model import LogisticRegression, RidgeLogisticRegression, LassoLogisticRegression


def main():
    """
    Main function to generate churn predictions.
    
    This function:
    1. Calls the prediction pipeline
    2. Displays results summary
    3. Shows sample predictions
    4. Provides file locations for outputs
    """
    print("=" * 60)
    print("RETAIL CUSTOMER CHURN PREDICTION")
    print("Classification Model - Prediction Generation")
    print("=" * 60)
    
    try:
        print("\nStarting prediction generation...")
        results = generate_predictions()
        
        print("\n" + "=" * 50)
        print("PREDICTION GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        # Display summary statistics
        total_customers = len(results)
        actual_churn = results['actual_churn'].sum()
        predicted_churn = results['predicted_churn'].sum()
        
        print(f"\nSUMMARY:")
        print(f"Total customers analyzed: {total_customers:,}")
        print(f"Actual churners: {actual_churn} ({actual_churn/total_customers:.1%})")
        print(f"Predicted churners: {predicted_churn} ({predicted_churn/total_customers:.1%})")
        
        # Show sample predictions
        print(f"\nSAMPLE PREDICTIONS (First 10 customers):")
        print("-" * 60)
        sample_df = results.head(10).copy()
        sample_df['churn_probability'] = sample_df['churn_probability'].round(3)
        print(sample_df.to_string(index=False))
        
        # Show high-risk customers
        high_risk = results[results['churn_probability'] >= 0.8]
        print(f"\nHIGH-RISK CUSTOMERS (Probability >= 80%): {len(high_risk)}")
        if len(high_risk) > 0:
            print("Top 5 highest risk:")
            top_risk = high_risk.nlargest(5, 'churn_probability')
            print(top_risk[['actual_churn', 'predicted_churn', 'churn_probability']].to_string(index=False))
        
        print(f"\nOUTPUT FILES:")
        print(f" Predictions: results/train_predictions.csv")
        print(f" Metrics: results/train_metrics.txt")
        
        print(f"\nTo view detailed metrics, check results/train_metrics.txt")
        print(f"To analyze predictions, open results/train_predictions.csv")
        
    except FileNotFoundError as e:
        print(f"\n ERROR: Required file not found - {e}")
        print("Make sure you have:")
        print("  - Trained model: models/regression_model1.pkl")
        print("  - Preprocessor: models/preprocessor.pkl")
        print("  - Training data: data/train_data.csv")
        print("\nRun 'python src/train_model.py' first to train the model.")
        
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        print("Please check your data and model files.")


if __name__ == "__main__":
    main()
