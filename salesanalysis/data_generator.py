"""
Module for generating synthetic sales data with realistic patterns.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from config import DATA_CONFIG, PRODUCT_CATEGORIES

def create_sample_data(
    n_customers=DATA_CONFIG['n_customers'],
    n_products=DATA_CONFIG['n_products'],
    n_records=DATA_CONFIG['n_records'],
    start_date=DATA_CONFIG['start_date'],
    end_date=DATA_CONFIG['end_date'],
    random_seed=DATA_CONFIG['random_seed']
):
    """
    Generate synthetic sales data with realistic patterns.
    
    Parameters:
    -----------
    n_customers : int
        Number of unique customers
    n_products : int
        Number of unique products
    n_records : int
        Number of sales records to generate
    start_date : str
        Start date for the data in 'YYYY-MM-DD' format
    end_date : str
        End date for the data in 'YYYY-MM-DD' format
    random_seed : int
        Seed for random number generation
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic sales data
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='D')
    
    # Products and categories
    categories = PRODUCT_CATEGORIES
    products = []
    for i, category in enumerate(categories):
        products.extend([f"{category} Item {j+1}" for j in range(n_products // len(categories))])
    
    # Create sales records
    records = []
    
    for _ in range(n_records):
        date = np.random.choice(dates)
        customer_id = f"CUST{np.random.randint(1, n_customers+1):03d}"
        product = np.random.choice(products)
        category = product.split(' ')[0]
        
        # Add seasonality and trends
        base_amount = np.random.lognormal(4, 0.5)  # Base sales amount
        
        # Seasonal effect (higher in December, lower in January)
        month = date.month
        if month == 12:  # December holiday season
            seasonal_factor = 1.5
        elif month == 1:  # January post-holiday slump
            seasonal_factor = 0.7
        elif month in [6, 7, 8]:  # Summer months
            seasonal_factor = 1.2
        else:
            seasonal_factor = 1.0
            
        # Weekday effect (higher on weekends)
        weekday = date.weekday()
        weekday_factor = 1.2 if weekday >= 5 else 1.0  # Weekend boost
        
        # Calculate final sales amount
        sales_amount = base_amount * seasonal_factor * weekday_factor
        
        records.append({
            'Date': date,
            'Customer_ID': customer_id,
            'Product': product,
            'Category': category,
            'Sales_Amount': round(sales_amount, 2)
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Sort by date
    df = df.sort_values('Date')
    
    return df