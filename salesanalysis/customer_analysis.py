"""
Module for customer analysis and segmentation using RFM.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from config import RFM_CONFIG, VIZ_CONFIG

def perform_rfm_analysis(df):
    """
    Perform RFM (Recency, Frequency, Monetary) analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed sales data
        
    Returns:
    --------
    pandas.DataFrame
        RFM analysis results with customer segments
    """
    # Calculate the latest date in the dataset and add one day
    today = df['Date'].max() + timedelta(days=1)
    
    # Group by customer and calculate RFM metrics
    rfm = df.groupby('Customer_ID').agg({
        'Date': lambda x: (today - x.max()).days,  # Recency
        'Sales_Amount': ['count', 'sum']  # Frequency, Monetary
    })
    
    # Rename columns
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # Create RFM quartiles
    rfm['R_Quartile'] = pd.qcut(
        rfm['Recency'], 
        RFM_CONFIG['recency_quartiles'], 
        labels=False, 
        duplicates='drop'
    )
    rfm['F_Quartile'] = pd.qcut(
        rfm['Frequency'], 
        RFM_CONFIG['frequency_quartiles'], 
        labels=False, 
        duplicates='drop'
    )
    rfm['M_Quartile'] = pd.qcut(
        rfm['Monetary'], 
        RFM_CONFIG['monetary_quartiles'], 
        labels=False, 
        duplicates='drop'
    )
    
    # Reverse recency (lower is better)
    rfm['R_Quartile'] = RFM_CONFIG['recency_quartiles'] - 1 - rfm['R_Quartile']
    
    # Calculate RFM Score
    rfm['RFM_Score'] = rfm['R_Quartile'] + rfm['F_Quartile'] + rfm['M_Quartile']
    
    # Create customer segments
    rfm['Customer_Segment'] = rfm.apply(segment_customer, axis=1)
    
    return rfm

def segment_customer(row):
    """
    Assign a customer segment based on RFM score.
    
    Parameters:
    -----------
    row : pandas.Series
        Row from RFM DataFrame
        
    Returns:
    --------
    str
        Customer segment name
    """
    thresholds = RFM_CONFIG['segment_thresholds']
    
    if row['RFM_Score'] >= thresholds['champions']:
        return 'Champions'
    elif row['RFM_Score'] >= thresholds['loyal']:
        return 'Loyal Customers'
    elif row['RFM_Score'] >= thresholds['potential']:
        return 'Potential Loyalists'
    elif row['RFM_Score'] >= thresholds['at_risk']:
        return 'At Risk'
    else:
        return 'Lost Customers'

def plot_customer_segments(rfm):
    """
    Visualize customer segments distribution.
    
    Parameters:
    -----------
    rfm : pandas.DataFrame
        RFM analysis results
    """
    segment_counts = rfm['Customer_Segment'].value_counts()
    
    plt.figure(figsize=VIZ_CONFIG['figsize_small'])
    sns.barplot(x=segment_counts.index, y=segment_counts.values)
    plt.title('Customer Segments Distribution', fontsize=16)
    plt.xlabel('Segment', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_rfm_heatmap(rfm):
    """
    Create a heatmap showing the relationship between RFM components.
    
    Parameters:
    -----------
    rfm : pandas.DataFrame
        RFM analysis results
    """
    # Calculate correlation
    corr = rfm[['Recency', 'Frequency', 'Monetary', 'RFM_Score']].corr()
    
    plt.figure(figsize=VIZ_CONFIG['figsize_small'])
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between RFM Components', fontsize=16)
    plt.tight_layout()
    plt.show()

def run_customer_analysis(df):
    """
    Run the complete customer analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed sales data
        
    Returns:
    --------
    pandas.DataFrame
        RFM analysis results
    """
    rfm = perform_rfm_analysis(df)
    plot_customer_segments(rfm)
    plot_rfm_heatmap(rfm)
    
    return rfm