import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

class Preprocessor:
    def __init__(self, company_column='Company'):
        self.mean_values = None
        self.mode_values = None
        self.feature_columns = None
        self.categorical_columns = None
        self.numerical_columns = None
        self.company_column = company_column
        self.company_stats = defaultdict(dict)
        self.overall_stats = {}
        self.one_hot_encodings = {}
        
    def fit_transform(self, df, target_col):
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # Handle missing values in target
        df = df.dropna(subset=[target_col])
        
        # Convert target to numeric, coercing errors to NaN
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        
        # Drop rows with NaN in target
        df = df.dropna(subset=[target_col])
        
        # Store the original company column for imputation
        if self.company_column in df.columns:
            # Calculate company-specific statistics
            companies = df[self.company_column].dropna().unique()
            for company in companies:
                company_data = df[df[self.company_column] == company]
                self.company_stats[company] = {}
                
                # Calculate modes and medians for each company
                for col in df.columns:
                    if col != target_col and col != self.company_column:
                        if df[col].dtype == 'object':
                            mode_val = company_data[col].mode()
                            self.company_stats[company][col] = mode_val[0] if not mode_val.empty else 'Unknown'
                        else:
                            self.company_stats[company][col] = company_data[col].median()
        
        # Calculate overall statistics as fallback
        for col in df.columns:
            if col != target_col and col != self.company_column:
                if df[col].dtype == 'object':
                    mode_val = df[col].mode()
                    self.overall_stats[col] = mode_val[0] if not mode_val.empty else 'Unknown'
                else:
                    self.overall_stats[col] = df[col].median()
        
        # Apply company-based imputation
        if self.company_column in df.columns:
            for idx, row in df.iterrows():
                company = row[self.company_column]
                for col in df.columns:
                    if col != target_col and col != self.company_column and pd.isna(row[col]):
                        if pd.notna(company) and company in self.company_stats and col in self.company_stats[company]:
                            df.at[idx, col] = self.company_stats[company][col]
                        else:
                            df.at[idx, col] = self.overall_stats[col]
        
        # Fill any remaining missing values with overall statistics
        for col in df.columns:
            if col != target_col and df[col].isna().any():
                if df[col].dtype == 'object':
                    df[col] = df[col].fillna(self.overall_stats.get(col, 'Unknown'))
                else:
                    df[col] = df[col].fillna(self.overall_stats.get(col, 0))
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Identify numerical and categorical columns
        self.numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        # Store mean values for numerical columns
        if self.numerical_columns:
            self.mean_values = X[self.numerical_columns].mean()
        
        # Store mode values for categorical columns
        if self.categorical_columns:
            self.mode_values = X[self.categorical_columns].mode().iloc[0]
            
            # Apply one-hot encoding to categorical columns using get_dummies
            X = pd.get_dummies(X, columns=self.categorical_columns, prefix_sep='_')
            
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Ensure all columns are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
        # Fill any NaN values that might have been introduced
        X = X.fillna(0)
            
        return X, y
    
    def transform(self, df, target_col=None):
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        if target_col and target_col in df.columns:
            # Convert target to numeric, coercing errors to NaN
            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            df = df.dropna(subset=[target_col])
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            X = df
            y = None
        
        # Apply company-based imputation for new data
        if self.company_column in X.columns:
            for idx, row in X.iterrows():
                company = row[self.company_column]
                for col in X.columns:
                    if col != self.company_column and pd.isna(row[col]):
                        if pd.notna(company) and company in self.company_stats and col in self.company_stats[company]:
                            X.at[idx, col] = self.company_stats[company][col]
                        else:
                            X.at[idx, col] = self.overall_stats.get(col, 'Unknown' if X[col].dtype == 'object' else 0)
        
        # Fill any remaining missing values with stored statistics
        for col in X.columns:
            if X[col].isna().any():
                if X[col].dtype == 'object':
                    X[col] = X[col].fillna(self.mode_values.get(col, 'Unknown'))
                else:
                    X[col] = X[col].fillna(self.mean_values.get(col, 0))
        
        # Apply one-hot encoding to categorical columns
        if self.categorical_columns:
            X = pd.get_dummies(X, columns=self.categorical_columns, prefix_sep='_')
        
        # Ensure all columns from training are present (add missing columns with 0)
        missing_cols = set(self.feature_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
            
        # Drop any extra columns that weren't in training
        extra_cols = set(X.columns) - set(self.feature_columns)
        X = X.drop(columns=list(extra_cols))
        
        # Ensure column order is consistent with training
        X = X[self.feature_columns]
        
        # Ensure all columns are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
        # Fill any NaN values that might have been introduced
        X = X.fillna(0)
        
        return X, y
    
