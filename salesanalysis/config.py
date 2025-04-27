"""
Configuration settings for the Sales Analysis project.
"""

# Data generation settings
DATA_CONFIG = {
    'n_customers': 100,
    'n_products': 20,
    'n_records': 5000,
    'start_date': '2022-01-01',
    'end_date': '2023-06-30',
    'random_seed': 42
}

# Categories for products
PRODUCT_CATEGORIES = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Sports']

# Visualization settings
VIZ_CONFIG = {
    'style': 'fivethirtyeight',
    'palette': 'Set2',
    'figsize_large': (14, 7),
    'figsize_medium': (12, 6),
    'figsize_small': (10, 6)
}

# RFM analysis settings
RFM_CONFIG = {
    'recency_quartiles': 4,
    'frequency_quartiles': 4,
    'monetary_quartiles': 4,
    'segment_thresholds': {
        'champions': 8,
        'loyal': 6,
        'potential': 4,
        'at_risk': 2
    }
}

# Forecasting settings
FORECAST_CONFIG = {
    'train_size': 0.8,
    'forecast_periods': 6,
    'model_type': 'linear'  # Options: 'linear', 'arima', 'prophet'
}