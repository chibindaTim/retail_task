import pandas as pd
import numpy as np
import pickle
import re
from datetime import datetime

class Preprocessor:
    def __init__(self, categorical_columns_to_encode=None):
        # Imputation statistics
        self.imputation_stats = {}
        
        # Column information
        self.feature_columns = None
        self.missing_columns = None
        self.datetime_columns = None
        
        # Encoding information
        self.categorical_columns_to_encode = categorical_columns_to_encode or []
        
        # Imputation summary from training
        self.imputation_summary = {}
        
        # DateTime handling
        self.datetime_reference = None  # Reference datetime for calculations
        
    def _detect_datetime_columns(self, df):
        """
        Detect columns that contain datetime data in various formats
        """
        datetime_cols = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Get sample of non-null values
                sample = df[col].dropna().astype(str).head(100)
                
                if len(sample) == 0:
                    continue
                
                # Common datetime patterns
                datetime_patterns = [
                    r'^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}$',  # 2022-11-09 08:33:32
                    r'^\d{4}-\d{2}-\d{2}$',                      # 2022-11-09
                    r'^\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}$',   # MM/DD/YYYY HH:MM:SS
                    r'^\d{2}/\d{2}/\d{4}$',                      # MM/DD/YYYY
                    r'^\d{4}/\d{2}/\d{2}\s\d{2}:\d{2}:\d{2}$',   # YYYY/MM/DD HH:MM:SS
                ]
                
                # Check if majority of values match datetime patterns
                total_matches = 0
                for pattern in datetime_patterns:
                    matches = sample.str.match(pattern).sum()
                    total_matches += matches
                
                # If >80% of values match datetime patterns, consider it datetime
                if total_matches > len(sample) * 0.8:
                    datetime_cols.append(col)
                    print(f"  Detected datetime column: {col}")
                    print(f"  Sample value: {sample.iloc[0]}")
        
        return datetime_cols
    
    def _convert_datetime_features(self, df, datetime_cols, is_training=True):
        """
        Convert datetime columns to numerical features
        """
        df_converted = df.copy()
        
        for col in datetime_cols:
            print(f"  Converting datetime column: {col}")
            
            try:
                # Convert to datetime
                datetime_series = pd.to_datetime(df_converted[col], errors='coerce')
                
                if is_training and self.datetime_reference is None:
                    # Set reference datetime (earliest datetime in training data)
                    self.datetime_reference = datetime_series.min()
                    print(f"    Reference datetime set: {self.datetime_reference}")
                
                # Create multiple time-based features
                df_converted[f'{col}_year'] = datetime_series.dt.year
                df_converted[f'{col}_month'] = datetime_series.dt.month
                df_converted[f'{col}_day'] = datetime_series.dt.day
                df_converted[f'{col}_hour'] = datetime_series.dt.hour
                df_converted[f'{col}_minute'] = datetime_series.dt.minute
                #df_converted[f'{col}_weekday'] = datetime_series.dt.dayofweek
                #df_converted[f'{col}_quarter'] = datetime_series.dt.quarter
                
                # Days since reference (useful for trend analysis)
                if self.datetime_reference is not None:
                    days_since_ref = (datetime_series - self.datetime_reference).dt.days
                    df_converted[f'{col}_days_since_ref'] = days_since_ref
                
                
                # Drop original datetime column
                df_converted = df_converted.drop(columns=[col])
                
                print(f"    Created features: year, month, day, hour, minute, days_since_ref")
                
            except Exception as e:
                print(f"    Error converting {col}: {str(e)}")
                print(f"    Treating as categorical instead")
        
        return df_converted
    
    def extended_imputation(self, df, is_training=True):
        """Apply extended imputation with tracking columns"""
        # Create a copy
        df_imputed = df.copy()
        
        # Detect and convert datetime columns first
        if is_training:
            self.datetime_columns = self._detect_datetime_columns(df_imputed)
        
        if self.datetime_columns:
            df_imputed = self._convert_datetime_features(df_imputed, self.datetime_columns, is_training)
        
        # Get columns with missing values (after datetime conversion)
        missing_columns = df_imputed.columns[df_imputed.isnull().any()].tolist()
        
        if is_training:
            self.missing_columns = missing_columns
            if not hasattr(self, 'imputation_summary') or is_training:
                self.imputation_summary = {}
        
        for col in missing_columns:
            # Count original missing values
            original_missing = df_imputed[col].isnull().sum()
            missing_percentage = (original_missing / len(df_imputed)) * 100
            
            # Create tracking column (1 where values were missing, 0 otherwise)
            tracking_col_name = f"{col}_was_imputed"
            df_imputed[tracking_col_name] = df_imputed[col].isnull().astype(int)
            
            # Determine imputation strategy based on data type
            if df_imputed[col].dtype == 'object':
                # Categorical data - use mode
                if is_training:
                    mode_value = df_imputed[col].mode()
                    if len(mode_value) > 0:
                        fill_value = mode_value[0]
                        method = 'mode'
                    else:
                        fill_value = 'N/A'
                        method = 'default_string'
                    
                    # Store for future use
                    self.imputation_stats[col] = {
                        'method': method,
                        'fill_value': fill_value
                    }
                else:
                    # Use stored values for transform
                    if col in self.imputation_stats:
                        fill_value = self.imputation_stats[col]['fill_value']
                        method = self.imputation_stats[col]['method']
                    else:
                        # Fallback if column wasn't missing in training
                        fill_value = 'N/A'
                        method = 'default_string'
            
            else:
                # Numerical data - use median
                if is_training:
                    fill_value = df_imputed[col].median()
                    method = 'median'
                    
                    # Store for future use
                    self.imputation_stats[col] = {
                        'method': method,
                        'fill_value': fill_value
                    }
                else:
                    # Use stored values for transform
                    if col in self.imputation_stats:
                        fill_value = self.imputation_stats[col]['fill_value']
                        method = self.imputation_stats[col]['method']
                    else:
                        # Fallback if column wasn't missing in training
                        fill_value = df_imputed[col].median()
                        method = 'median'
            
            # Apply imputation
            df_imputed[col] = df_imputed[col].fillna(fill_value)
            
            # Verify imputation worked
            remaining_missing = df_imputed[col].isnull().sum()
            imputed_count = original_missing - remaining_missing
            
            # Store summary (for training or tracking)
            if is_training:
                self.imputation_summary[col] = {
                    'original_missing': original_missing,
                    'missing_percentage': missing_percentage,
                    'method': method,
                    'fill_value': fill_value,
                    'imputed_count': imputed_count,
                    'tracking_column': tracking_col_name
                }
        
        return df_imputed
    
    
    def fit_transform(self, df, target_col):
        """Fit the preprocessor and transform the training data"""
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        print("Detecting datetime columns...")
        
        # Handle missing values in target
        df = df.dropna(subset=[target_col])
        
        # Convert target to numeric, coercing errors to NaN
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        
        # Drop rows with NaN in target
        df = df.dropna(subset=[target_col])
        
        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        # Apply extended imputation (includes datetime detection and conversion)
        X = self.extended_imputation(X, is_training=True)
        
        # Apply one-hot encoding to specified categorical columns
        if self.categorical_columns_to_encode:
            # Only encode columns that exist in the data
            cols_to_encode = [col for col in self.categorical_columns_to_encode if col in X.columns]
            if cols_to_encode:
                print(f"One-hot encoding columns: {cols_to_encode}")
                X = pd.get_dummies(X, columns=cols_to_encode, drop_first=True)
        
        # Store feature columns (after all transformations)
        self.feature_columns = X.columns.tolist()
        
        # Ensure all columns are numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            
        # Fill any NaN values that might have been introduced
        X = X.fillna(0)
        
        print(f"Training completed:")
        print(f"  - Shape after preprocessing: {X.shape}")
        print(f"  - Datetime columns detected: {len(self.datetime_columns) if self.datetime_columns else 0}")
        print(f"  - Columns with imputation: {len(self.imputation_summary)}")
        print(f"  - Tracking columns added: {sum(1 for col in X.columns if '_was_imputed' in col)}")
        
        if self.datetime_columns:
            datetime_features = [col for col in X.columns if any(dt_col in col for dt_col in self.datetime_columns)]
            print(f"  - DateTime features created: {len(datetime_features)}")
        
        return X, y
    
    def get_imputation_summary(self):
        #Return the imputation summary from training
        return self.imputation_summary
    
    def get_datetime_info(self):
        #Return information about datetime columns and reference
        return {
            'datetime_columns': self.datetime_columns,
            'datetime_reference': self.datetime_reference
        }
    