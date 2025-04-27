"""
Main module for the Sales Forecasting and Customer Analysis project.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from data_generator import create_sample_data
from data_processor import preprocess_data, aggregate_monthly_sales
from eda import run_eda
from customer_analysis import run_customer_analysis
from forecasting import run_forecasting
from utils import print_section_header, print_insights

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sales Forecasting and Customer Analysis')
    parser.add_argument('--generate-data', action='store_true', help='Only generate the dataset')
    parser.add_argument('--eda', action='store_true', help='Only run the EDA')
    parser.add_argument('--customer-analysis', action='store_true', help='Only run the customer analysis')
    parser.add_argument('--forecasting', action='store_true', help='Only run the forecasting')
    parser.add_argument('--no-plots', action='store_true', help='Disable plotting')
    return parser.parse_args()

def main():
    """Main function to run the sales analysis."""
    # Parse arguments
    args = parse_arguments()
    
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Disable plotting if requested
    if args.no_plots:
        plt.ioff()
    
    # Generate data
    print_section_header("Data Generation")
    sales_df = create_sample_data()
    print(f"Dataset created successfully!")
    print(f"Shape of the dataset: {sales_df.shape}")
    print("\nSample data:")
    print(sales_df.head())
    
    if args.generate_data:
        return sales_df
    
    # Preprocess data
    print_section_header("Data Preprocessing")
    processed_df = preprocess_data(sales_df)
    print("Data preprocessing completed.")
    print("\nProcessed data sample:")
    print(processed_df.head())
    
    # Aggregate to monthly
    monthly_sales = aggregate_monthly_sales(processed_df)
    
    # Run EDA
    if args.eda or not any([args.generate_data, args.customer_analysis, args.forecasting]):
        print_section_header("Exploratory Data Analysis")
        run_eda(processed_df, monthly_sales)
    
    # Run customer analysis
    if args.customer_analysis or not any([args.generate_data, args.eda, args.forecasting]):
        print_section_header("Customer Analysis: RFM Model")
        rfm = run_customer_analysis(processed_df)
    else:
        rfm = None
    
    # Run forecasting
    if args.forecasting or not any([args.generate_data, args.eda, args.customer_analysis]):
        print_section_header("Sales Forecasting")
        evaluation, forecast_df = run_forecasting(monthly_sales)
    else:
        evaluation, forecast_df = None, None
    
    # Print insights if running the full analysis
    if not any([args.generate_data, args.eda, args.customer_analysis, args.forecasting]):
        print_insights(processed_df, monthly_sales, rfm, evaluation, forecast_df)
    
    return sales_df, processed_df, monthly_sales, rfm, evaluation, forecast_df

if __name__ == "__main__":
    main()