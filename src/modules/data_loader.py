"""
Data Loading and Preprocessing Module
Handles loading, cleaning, and feature engineering for NYC taxi data
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the NYC taxi dataset
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed dataset
    """
    print("Loading dataset...")
    
    # Load data
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print("Dataset file not found. Please ensure the file path is correct.")
        return None
    
    # Display basic info
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # Convert datetime columns with error handling
    datetime_cols = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
    for col in datetime_cols:
        if col in df.columns:
            # Clean invalid datetime values first
            df[col] = df[col].astype(str)
            
            # Fix common datetime issues
            # Replace "24:" with "00:" (next day)
            df[col] = df[col].str.replace(' 24:', ' 00:', regex=False)
            
            # Convert to datetime with error handling
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Remove rows with invalid datetime
            invalid_dates = df[col].isna()
            if invalid_dates.sum() > 0:
                print(f"Removing {invalid_dates.sum()} rows with invalid {col}")
                df = df[~invalid_dates]
    
    # Calculate trip duration in minutes
    if 'tpep_pickup_datetime' in df.columns and 'tpep_dropoff_datetime' in df.columns:
        df['trip_duration_minutes'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Extract time features
    if 'tpep_pickup_datetime' in df.columns:
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df['pickup_day'] = df['tpep_pickup_datetime'].dt.day_name()
        df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
        df['pickup_weekday'] = df['tpep_pickup_datetime'].dt.weekday
    
    # Clean data
    print("\nCleaning data...")
    initial_rows = len(df)
    
    # Remove invalid trips
    if 'trip_duration_minutes' in df.columns:
        df = df[(df['trip_duration_minutes'] > 0) & (df['trip_duration_minutes'] < 300)]  # 0-5 hours
    
    if 'trip_distance' in df.columns:
        df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 100)]  # Reasonable distance limits
    
    if 'total_amount' in df.columns:
        df = df[(df['total_amount'] > 0) & (df['total_amount'] < 500)]  # Reasonable fare limits
    
    print(f"Removed {initial_rows - len(df)} invalid records. Final shape: {df.shape}")
    
    return df

def save_results(df, model_results, filename_prefix="taxi_analysis"):
    """
    Save analysis results to files
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        model_results (tuple): Model results from ML module
        filename_prefix (str): Prefix for saved files
    """
    # Save cleaned dataset
    df.to_csv(f"outputs/{filename_prefix}_cleaned_data.csv", index=False)
    
    # Save model if available
    if model_results and 'Random Forest' in model_results[0]:
        import joblib
        best_model = model_results[0]['Random Forest']['model']
        joblib.dump(best_model, f"outputs/{filename_prefix}_model.pkl")
    
    print(f"Results saved with prefix: {filename_prefix}") 