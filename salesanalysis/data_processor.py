"""
Module for data preprocessing and feature engineering.
"""

import pandas as pd

def preprocess_data(df):
    """
    Clean and preprocess the sales data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw sales data
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed sales data with additional features
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values if any
    processed_df = processed_df.dropna()
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(processed_df['Date']):
        processed_df['Date'] = pd.to_datetime(processed_df['Date'])
    
    # Extract date features
    processed_df['Year'] = processed_df['Date'].dt.year
    processed_df['Month'] = processed_df['Date'].dt.month
    processed_df['Day'] = processed_df['Date'].dt.day
    processed_df['Weekday'] = processed_df['Date'].dt.day_name()
    processed_df['Month_Name'] = processed_df['Date'].dt.month_name()
    processed_df['Week_of_Year'] = processed_df['Date'].dt.isocalendar().week
    processed_df['Quarter'] = processed_df['Date'].dt.quarter
    
    # Create Month-Year string for plotting
    processed_df['Month_Year'] = processed_df['Date'].dt.strftime('%b %Y')
    
    # Remove duplicates if any
    processed_df = processed_df.drop_duplicates()
    
    return processed_df

def aggregate_monthly_sales(df):
    """
    Aggregate sales data to monthly level.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed sales data
        
    Returns:
    --------
    pandas.DataFrame
        Monthly aggregated sales data
    """
    # Group by month and sum sales
    monthly_sales = df.groupby(pd.Grouper(key='Date', freq='M'))['Sales_Amount'].sum().reset_index()
    
    # Add month-year string and month number
    monthly_sales['Month_Year'] = monthly_sales['Date'].dt.strftime('%b %Y')
    monthly_sales['Month_Num'] = range(1, len(monthly_sales) + 1)
    
    return monthly_sales