"""
Advanced forecasting models for sales prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from config import VIZ_CONFIG

def create_features(df):
    """
    Create advanced features for time series forecasting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Monthly sales data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional features
    """
    # Create a copy to avoid modifying the original
    features_df = df.copy()
    
    # Add lag features (previous months' sales)
    for lag in range(1, 4):  # 1, 2, and 3 month lags
        features_df[f'lag_{lag}'] = features_df['Sales_Amount'].shift(lag)
    
    # Add rolling mean features
    for window in [2, 3, 6]:
        features_df[f'rolling_mean_{window}'] = features_df['Sales_Amount'].rolling(window=window).mean()
    
    # Add rolling std features
    for window in [2, 3, 6]:
        features_df[f'rolling_std_{window}'] = features_df['Sales_Amount'].rolling(window=window).std()
    
    # Add month of year (seasonality)
    features_df['month'] = pd.DatetimeIndex(features_df['Date']).month
    
    # Add quarter
    features_df['quarter'] = pd.DatetimeIndex(features_df['Date']).quarter
    
    # Add year
    features_df['year'] = pd.DatetimeIndex(features_df['Date']).year
    
    # Drop rows with NaN values (due to lag and rolling features)
    features_df = features_df.dropna()
    
    return features_df

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest model with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
        
    Returns:
    --------
    sklearn.ensemble.RandomForestRegressor
        Trained model
    """
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    # Create and train model with grid search
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {np.sqrt(-grid_search.best_score_):.2f}")
    
    return grid_search.best_estimator_

def feature_importance_plot(model, X):
    """
    Plot feature importance from the model.
    
    Parameters:
    -----------
    model : sklearn.ensemble.RandomForestRegressor
        Trained Random Forest model
    X : pandas.DataFrame
        Feature DataFrame
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]
    
    # Create plot
    plt.figure(figsize=VIZ_CONFIG['figsize_medium'])
    
    # Create plot title
    plt.title("Feature Importance")
    
    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])
    
    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=90)
    
    plt.tight_layout()
    plt.show()

def run_advanced_forecasting(monthly_sales):
    """
    Run advanced forecasting with feature engineering and Random Forest.
    
    Parameters:
    -----------
    monthly_sales : pandas.DataFrame
        Monthly aggregated sales data
        
    Returns:
    --------
    tuple
        Model, evaluation metrics, feature importance
    """
    # Create features
    features_df = create_features(monthly_sales)
    
    # Define features and target
    features = features_df.drop(['Date', 'Sales_Amount', 'Month_Year'], axis=1)
    target = features_df['Sales_Amount']
    
    # Split data into training and testing
    train_size = int(len(features) * 0.8)
    X_train = features[:train_size]
    y_train = target[:train_size]
    X_test = features[train_size:]
    y_test = target[train_size:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = train_random_forest(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    r2 = r2_score(y_test, test_pred)
    
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Testing RMSE: {test_rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Plot feature importance
    feature_importance_plot(model, X_train)
    
    # Plot actual vs predicted
    plt.figure(figsize=VIZ_CONFIG['figsize_large'])
    
    # Plot training data
    plt.plot(features_df['Date'][:train_size], y_train, 
             marker='o', color='blue', label='Training Data')
    plt.plot(features_df['Date'][:train_size], train_pred, 
             color='green', linestyle='--', label='Training Predictions')
    
    # Plot testing data
    plt.plot(features_df['Date'][train_size:], y_test, 
             marker='o', color='red', label='Testing Data')
    plt.plot(features_df['Date'][train_size:], test_pred, 
             color='orange', linestyle='--', label='Testing Predictions')
    
    plt.title('Advanced Sales Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales Amount', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return model, {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'r2': r2,
        'train_pred': train_pred,
        'test_pred': test_pred
    }