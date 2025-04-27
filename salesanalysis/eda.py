"""
Module for exploratory data analysis and visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import VIZ_CONFIG

def setup_visualization():
    """Set up the visualization style and defaults."""
    plt.style.use(VIZ_CONFIG['style'])
    sns.set_palette(VIZ_CONFIG['palette'])

def plot_monthly_sales(monthly_sales):
    """
    Plot monthly sales trend.
    
    Parameters:
    -----------
    monthly_sales : pandas.DataFrame
        Monthly aggregated sales data
    """
    plt.figure(figsize=VIZ_CONFIG['figsize_medium'])
    plt.plot(monthly_sales['Date'], monthly_sales['Sales_Amount'], marker='o', linestyle='-')
    plt.title('Monthly Sales Trend', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Sales Amount', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_top_products(df, n=5):
    """
    Plot top selling products.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sales data
    n : int
        Number of top products to show
    """
    top_products = df.groupby('Product')['Sales_Amount'].sum().sort_values(ascending=False).head(n)
    
    plt.figure(figsize=VIZ_CONFIG['figsize_small'])
    sns.barplot(x=top_products.values, y=top_products.index)
    plt.title(f'Top {n} Selling Products', fontsize=16)
    plt.xlabel('Total Sales Amount', fontsize=12)
    plt.ylabel('Product', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_category_distribution(df):
    """
    Plot sales distribution by category.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sales data
    """
    category_sales = df.groupby('Category')['Sales_Amount'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=VIZ_CONFIG['figsize_small'])
    plt.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%', startangle=90, shadow=True)
    plt.title('Sales Distribution by Category', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    plt.show()

def plot_sales_heatmap(df):
    """
    Plot sales heatmap by month and weekday.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Sales data with date features
    """
    # Create pivot table
    sales_pivot = df.pivot_table(
        index='Weekday', 
        columns='Month_Name',
        values='Sales_Amount', 
        aggfunc='sum'
    )
    
    # Reorder weekdays
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sales_pivot = sales_pivot.reindex(weekday_order)
    
    # Reorder months
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    sales_pivot = sales_pivot[sales_pivot.columns.intersection(month_order)]
    
    plt.figure(figsize=VIZ_CONFIG['figsize_medium'])
    sns.heatmap(sales_pivot, annot=True, fmt='.0f', cmap='YlGnBu', linewidths=.5)
    plt.title('Sales Heatmap by Month and Weekday', fontsize=16)
    plt.tight_layout()
    plt.show()

def run_eda(df, monthly_sales):
    """
    Run all EDA visualizations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed sales data
    monthly_sales : pandas.DataFrame
        Monthly aggregated sales data
    """
    setup_visualization()
    plot_monthly_sales(monthly_sales)
    plot_top_products(df)
    plot_category_distribution(df)
    plot_sales_heatmap(df)