"""
Module for sales forecasting using various models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from config import FORECAST_CONFIG, VIZ_CONFIG

def prepare_forecast_data(monthly_sales, train_size=FORECAST_CONFIG['train_size']):
    """
    Prepare data for forecasting.
    
    Parameters:
    -----------
    monthly_sales : pandas.DataFrame
        Monthly aggregated sales data
    train_size : float
        Proportion of data to use for training
        
    Returns:
    --------
    tuple
        X_train, y_train, X_test, y_test, train_data, test_data
    """
    # Split data into training and testing
    train_size = int(len(monthly_sales) * train_size)
    train_data = monthly_sales[:train_size]
    test_data = monthly_sales[train_size:]
    
    # Create features and target
    X_train = train_data[['Month_Num']]
    y_train = train_data['Sales_Amount']
    X_test = test_data[['Month_Num']]
    y_test = test_data['Sales_Amount']
    
    return X_train, y_train, X_test, y_test, train_data, test_data

def train_linear_model(X_train, y_train):
    """
    Train a linear regression model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
        
    Returns:
    --------
    sklearn.linear_model.LinearRegression
        Trained model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the forecasting model.
    
    Parameters:
    -----------
    model : object
        Trained forecasting model
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    X_test : pandas.DataFrame
        Testing features
    y_test : pandas.Series
        Testing target
        
    Returns:
    --------
    dict
        Evaluation metrics and predictions
    """
    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    r2 = r2_score(y_test, test_pred)
    
    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'r2': r2,
        'train_pred': train_pred,
        'test_pred': test_pred
    }

def generate_future_forecast(model, monthly_sales, periods=FORECAST_CONFIG['forecast_periods']):
    """
    Generate forecast for future periods.
    
    Parameters:
    -----------
    model : object
        Trained forecasting model
    monthly_sales : pandas.DataFrame
        Monthly aggregated sales data
    periods : int
        Number of periods to forecast
        
    Returns:
    --------
    pandas.DataFrame
        Future forecast data
    """
    # Get last month number and date
    last_month_num = monthly_sales['Month_Num'].max()
    last_date = monthly_sales['Date'].max()
    
    # Create future month numbers
    future_month_nums = np.array(range(last_month_num + 1, last_month_num + periods + 1)).reshape(-1, 1)
    
    # Generate predictions
    future_predictions = model.predict(future_month_nums)
    
    # Create future dates
    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Month_Num': future_month_nums.flatten(),
        'Forecasted_Sales': future_predictions
    })
    
    return forecast_df

def plot_forecast(monthly_sales, test_data, test_pred, forecast_df):
    """
    Plot actual vs forecasted sales.
    
    Parameters:
    -----------
    monthly_sales : pandas.DataFrame
        Monthly aggregated sales data
    test_data : pandas.DataFrame
        Test data portion
    test_pred : numpy.ndarray
        Predictions for test data
    forecast_df : pandas.DataFrame
        Future forecast data
    """
    plt.figure(figsize=VIZ_CONFIG['figsize_large'])
    
    # Plot actual sales
    plt.plot(monthly_sales['Date'], monthly_sales['Sales_Amount'], 
             marker='o', color='blue', label='Actual Sales')
    
    # Plot test predictions
    plt.plot(test_data['Date'], test_pred, 
             color='green', linestyle='--', label='Model Prediction')
    
    # Plot future forecast
    plt.plot(forecast_df['Date'], forecast_df['Forecasted_Sales'], 
             color='red', linestyle='--', marker='x', label='Future Forecast')
    
    plt.title('Sales Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Sales Amount', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def run_forecasting(monthly_sales):
    """
    Run the complete forecasting process.
    
    Parameters:
    -----------
    monthly_sales : pandas.DataFrame
        Monthly aggregated sales data
        
    Returns:
    --------
    tuple
        Evaluation metrics, forecast DataFrame
    """
    # Prepare data
    X_train, y_train, X_test, y_test, train_data, test_data = prepare_forecast_data(monthly_sales)
    
    # Train model
    model = train_linear_model(X_train, y_train)
    
    # Evaluate model
    evaluation = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Generate future forecast
    forecast_df = generate_future_forecast(model, monthly_sales)
    
    # Plot forecast
    plot_forecast(monthly_sales, test_data, evaluation['test_pred'], forecast_df)
    
    return evaluation, forecast_df