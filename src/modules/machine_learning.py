"""
Machine Learning Module
Handles model training, evaluation, and feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib

def build_prediction_model(df):
    """
    Build and evaluate trip duration prediction model
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        tuple: (results, feature_cols, scaler, le_dict)
    """
    print("\n" + "="*50)
    print("2. MACHINE LEARNING MODEL")
    print("="*50)
    
    if 'trip_duration_minutes' not in df.columns:
        print("Trip duration column not found. Skipping ML section.")
        return None
    
    # Feature selection and engineering
    print("\n2.1 Feature Selection and Engineering")
    print("-" * 40)
    
    # Select features
    feature_cols = []
    
    # Numerical features
    numerical_features = ['trip_distance', 'fare_amount', 'extra', 'mta_tax', 
                         'tip_amount', 'tolls_amount', 'pickup_hour', 'pickup_weekday',
                         'pickup_month', 'passenger_count']
    
    for col in numerical_features:
        if col in df.columns:
            feature_cols.append(col)
    
    # Categorical features
    categorical_features = ['PULocationID', 'DOLocationID', 'payment_type']
    
    for col in categorical_features:
        if col in df.columns:
            feature_cols.append(col)
    
    print(f"Selected features: {feature_cols}")
    
    # Prepare data
    df_model = df[feature_cols + ['trip_duration_minutes']].copy()
    df_model = df_model.dropna()
    
    # Encode categorical variables
    le_dict = {}
    for col in categorical_features:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le
    
    # Split features and target
    X = df_model[feature_cols]
    y = df_model['trip_duration_minutes']
    
    print(f"Model dataset shape: {X.shape}")
    print(f"Features used: {list(X.columns)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n2.2 Model Training and Evaluation")
    print("-" * 35)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for Linear Regression, original for Random Forest
        if name == 'Linear Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'model': model,
            'predictions': y_pred
        }
        
        print(f"RMSE: {rmse:.2f} minutes")
        print(f"MAE: {mae:.2f} minutes")
        print(f"R²: {r2:.3f}")
    
    # Feature importance for Random Forest
    if 'Random Forest' in results:
        print("\n2.3 Feature Importance (Random Forest)")
        print("-" * 40)
        
        rf_model = results['Random Forest']['model']
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance - Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Model comparison
    print("\n2.4 Model Comparison")
    print("-" * 20)
    
    comparison_df = pd.DataFrame(results).T[['RMSE', 'MAE', 'R²']].round(3)
    print(comparison_df)
    
    # Best model
    best_model_name = comparison_df['R²'].idxmax()
    print(f"\nBest performing model: {best_model_name}")
    
    return results, feature_cols, scaler, le_dict

def enhanced_model_analysis(df):
    """
    Enhanced machine learning analysis with cross-validation and hyperparameter tuning
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        
    Returns:
        dict: Best models for each algorithm
    """
    print("\n" + "="*50)
    print("ENHANCED MODEL ANALYSIS")
    print("="*50)
    
    if 'trip_duration_minutes' not in df.columns:
        return
    
    # Prepare enhanced feature set
    feature_cols = []
    
    # Core features
    core_features = ['trip_distance', 'fare_amount', 'pickup_hour', 'pickup_weekday']
    for col in core_features:
        if col in df.columns:
            feature_cols.append(col)
    
    # Engineered features
    if 'tpep_pickup_datetime' in df.columns:
        df['is_rush_hour'] = df['pickup_hour'].isin([7, 8, 17, 18, 19]).astype(int)
        df['is_weekend'] = df['pickup_weekday'].isin([5, 6]).astype(int)
        feature_cols.extend(['is_rush_hour', 'is_weekend'])
    
    if 'passenger_count' in df.columns:
        feature_cols.append('passenger_count')
    
    # Location features
    if 'PULocationID' in df.columns:
        # Create location popularity features
        location_popularity = df['PULocationID'].value_counts()
        df['pickup_popularity'] = df['PULocationID'].map(location_popularity)
        feature_cols.append('pickup_popularity')
    
    print(f"Enhanced features: {feature_cols}")
    
    # Prepare data
    df_model = df[feature_cols + ['trip_duration_minutes']].copy()
    df_model = df_model.dropna()
    
    X = df_model[feature_cols]
    y = df_model['trip_duration_minutes']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Advanced models
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    # Hyperparameter grids
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [10, 20],
            'min_samples_split': [5, 10]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 5]
        }
    }
    
    best_models = {}
    
    for name, model in models.items():
        print(f"\nOptimizing {name}...")
        
        # Grid search
        grid_search = GridSearchCV(
            model, param_grids[name], 
            cv=3, scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # Best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Predictions
        y_pred = best_model.predict(X_test)
        
        # Enhanced metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        r2 = r2_score(y_test, y_pred)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"RMSE: {rmse:.2f} minutes")
        print(f"MAE: {mae:.2f} minutes")
        print(f"MAPE: {mape:.1f}%")
        print(f"R²: {r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, 
                                   cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        print(f"CV RMSE: {cv_rmse:.2f} ± {np.sqrt(-cv_scores).std():.2f}")
    
    return best_models

def save_model(model, filename="taxi_duration_model.pkl"):
    """
    Save trained model to file
    
    Args:
        model: Trained model object
        filename (str): Output filename
    """
    joblib.dump(model, f"outputs/{filename}")
    print(f"Model saved as outputs/{filename}")

def load_model(filename="taxi_duration_model.pkl"):
    """
    Load trained model from file
    
    Args:
        filename (str): Model filename
        
    Returns:
        Trained model object
    """
    model = joblib.load(f"outputs/{filename}")
    print(f"Model loaded from outputs/{filename}")
    return model

def predict_duration(model, features, scaler=None):
    """
    Make trip duration predictions
    
    Args:
        model: Trained model
        features (pd.DataFrame): Feature values
        scaler: Optional scaler for features
        
    Returns:
        np.array: Predicted durations
    """
    if scaler:
        features_scaled = scaler.transform(features)
        predictions = model.predict(features_scaled)
    else:
        predictions = model.predict(features)
    
    return predictions 