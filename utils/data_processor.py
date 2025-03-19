import pandas as pd
import numpy as np

def calculate_metrics(df):
    """Calculate core metrics for the dashboard"""
    metrics = {}
    
    # Count total UGIDs and UPIDs
    metrics['total_ugids'] = len(df)
    metrics['total_upids'] = df['UPID'].nunique()
    
    # Categorize by measure type
    measure_counts = df['MedAdherenceMeasureCode'].value_counts().to_dict()
    metrics['mac_count'] = measure_counts.get('MAC', 0)
    metrics['mah_count'] = measure_counts.get('MAH', 0)
    metrics['mad_count'] = measure_counts.get('MAD', 0)
    
    # Count one-fills vs denominator gaps
    one_fill_df = df[df['OneFillCode'] == 'Yes']
    metrics['one_fill_count'] = len(one_fill_df)
    metrics['denominator_gap_count'] = len(df[df['OneFillCode'].isnull()])
    
    return metrics

def calculate_average_pdc(df):
    """Calculate average PDC by measure type, excluding one-fills and null PDC values"""
    # Filter out one-fills and null PDC values
    pdc_df = df[(df['OneFillCode'].isnull()) & (df['PDCNbr'].notnull())]
    
    # Calculate average PDC by measure type
    avg_pdc = pdc_df.groupby('MedAdherenceMeasureCode')['PDCNbr'].mean().to_dict()
    
    # Ensure all measure types have values (even if zero)
    for measure in ['MAC', 'MAH', 'MAD']:
        if measure not in avg_pdc:
            avg_pdc[measure] = 0
    
    return avg_pdc

def get_week_over_week_data(df):
    """Create week-over-week comparison data"""
    # Group by week and year
    weekly_data = df.groupby(['Year', 'WeekNumber']).agg(
        ugid_count=('UGID', 'count'),
        upid_count=('UPID', 'nunique'),
        file_load_date=('DataAsOfDate', 'max')
    ).reset_index().sort_values(['Year', 'WeekNumber'], ascending=[False, False])
    
    # Calculate week-over-week changes
    weekly_data['prev_ugid_count'] = weekly_data['ugid_count'].shift(-1)
    weekly_data['prev_upid_count'] = weekly_data['upid_count'].shift(-1)
    weekly_data['ugid_change_pct'] = ((weekly_data['ugid_count'] - weekly_data['prev_ugid_count']) / 
                                     weekly_data['prev_ugid_count'] * 100).round(1)
    weekly_data['upid_change_pct'] = ((weekly_data['upid_count'] - weekly_data['prev_upid_count']) / 
                                     weekly_data['prev_upid_count'] * 100).round(1)
    
    return weekly_data

def create_market_payer_summary(df):
    """Create summary data for market and payer analysis"""
    market_summary = df.groupby('MarketCode').agg(
        gap_count=('UGID', 'count'),
        patient_count=('UPID', 'nunique')
    ).reset_index().sort_values('gap_count', ascending=False)
    
    payer_summary = df.groupby('PayerCode').agg(
        gap_count=('UGID', 'count'),
        patient_count=('UPID', 'nunique')
    ).reset_index().sort_values('gap_count', ascending=False)
    
    return market_summary, payer_summary
