"""
Utility functions for the sales analysis project.
"""

def print_section_header(title):
    """
    Print a formatted section header.
    
    Parameters:
    -----------
    title : str
        Section title
    """
    print(f"\n\n{'='*80}")
    print(f"  {title.upper()}")
    print(f"{'='*80}\n")

def print_insights(df, monthly_sales, rfm, evaluation, forecast_df):
    """
    Print key insights and recommendations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed sales data
    monthly_sales : pandas.DataFrame
        Monthly aggregated sales data
    rfm : pandas.DataFrame
        RFM analysis results
    evaluation : dict
        Model evaluation metrics
    forecast_df : pandas.DataFrame
        Future forecast data
    """
    print_section_header("Key Insights and Recommendations")
    
    # Calculate average sales by category
    avg_category_sales = df.groupby('Category')['Sales_Amount'].mean().sort_values(ascending=False)
    print("\nAverage Sales by Category:")
    for category, avg_sales in avg_category_sales.items():
        print(f"{category}: ${avg_sales:.2f}")
    
    # Calculate monthly growth rate
    monthly_growth = monthly_sales.copy()
    monthly_growth['Growth_Rate'] = monthly_growth['Sales_Amount'].pct_change() * 100
    avg_growth = monthly_growth['Growth_Rate'].mean()
    print(f"\nAverage Monthly Growth Rate: {avg_growth:.2f}%")
    
    # Identify best selling months
    best_months = monthly_sales.nlargest(3, 'Sales_Amount')
    print("\nTop 3 Best Selling Months:")
    for _, row in best_months.iterrows():
        print(f"{row['Month_Year']}: ${row['Sales_Amount']:.2f}")
    
    # Identify top customers
    top_customers = df.groupby('Customer_ID')['Sales_Amount'].sum().nlargest(5)
    print("\nTop 5 Customers by Total Purchase:")
    for customer, amount in top_customers.items():
        print(f"{customer}: ${amount:.2f}")
    
    # Print model performance
    print(f"\nForecasting Model Performance:")
    print(f"Training RMSE: ${evaluation['train_rmse']:.2f}")
    print(f"Testing RMSE: ${evaluation['test_rmse']:.2f}")
    print(f"R² Score: {evaluation['r2']:.2f}")
    
    # Print customer segment insights
    segment_counts = rfm['Customer_Segment'].value_counts()
    print("\nCustomer Segment Distribution:")
    for segment, count in segment_counts.items():
        print(f"{segment}: {count} customers ({count/len(rfm)*100:.1f}%)")
    
    # Print future forecast
    print("\nSales Forecast for Next 6 Months:")
    for _, row in forecast_df.iterrows():
        print(f"{row['Date'].strftime('%b %Y')}: ${row['Forecasted_Sales']:.2f}")
    
    print("\nRecommendations:")
    print("1. Focus marketing efforts on top-performing categories which have the highest average sales.")
    print("2. Implement special promotions during peak months to capitalize on seasonal trends.")
    print("3. Target 'Champions' and 'Loyal Customers' segments with loyalty rewards to maintain their engagement.")
    print("4. Develop re-engagement campaigns for 'At Risk' customers to prevent churn.")
    print("5. Consider weekend-specific promotions as sales tend to be higher on weekends.")
    print("6. Prepare inventory and staffing based on the sales forecast to optimize operations.")