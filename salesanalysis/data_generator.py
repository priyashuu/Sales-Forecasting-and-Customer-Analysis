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
    """
    np.random.seed(random_seed)
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='D')
    
    categories = PRODUCT_CATEGORIES
    products = []
    for i, category in enumerate(categories):
        products.extend([f"{category} Item {j+1}" for j in range(n_products // len(categories))])
    
    records = []
    
    for _ in range(n_records):
        date = np.random.choice(dates)
        customer_id = f"CUST{np.random.randint(1, n_customers+1):03d}"
        product = np.random.choice(products)
        category = product.split(' ')[0]
        
        base_amount = np.random.lognormal(4, 0.5)
        
        month = pd.Timestamp(date).month

        if month == 12:
            seasonal_factor = 1.5
        elif month == 1:
            seasonal_factor = 0.7
        elif month in [6, 7, 8]:
            seasonal_factor = 1.2
        else:
            seasonal_factor = 1.0
        
        weekday = pd.Timestamp(date).weekday()
        weekday_factor = 1.2 if weekday >= 5 else 1.0
        
        sales_amount = base_amount * seasonal_factor * weekday_factor
        
        records.append({
            'Date': date,
            'Customer_ID': customer_id,
            'Product': product,
            'Category': category,
            'Sales_Amount': round(sales_amount, 2)
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values('Date')
    
    return df
