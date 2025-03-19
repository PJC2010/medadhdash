import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import toml
from google.cloud import bigquery
from google.oauth2 import service_account
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Medication Adherence Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load GCP credentials from TOML
def load_credentials():
    # Path to your TOML file
    toml_path = 'config.toml'  # Adjust path as needed
    
    try:
        # Load TOML file
        config = toml.load(toml_path)
        
        # Find credentials section
        if 'gcp' in config:
            return config['gcp']
        elif 'google_credentials' in config:
            return config['google_credentials']
        else:
            for section in config:
                if isinstance(config[section], dict) and 'type' in config[section]:
                    return config[section]
            
        st.error("No Google credentials found in TOML file")
        st.stop()
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        st.stop()

# Initialize BigQuery client
@st.cache_resource
def get_bigquery_client():
    credentials_dict = load_credentials()
    credentials = service_account.Credentials.from_service_account_info(credentials_dict)
    return bigquery.Client(credentials=credentials)

try:
    client = get_bigquery_client()
except Exception as e:
    st.error(f"Failed to initialize BigQuery client: {e}")
    st.stop()

# Function to run BigQuery queries
def run_query(query):
    try:
        return client.query(query).to_dataframe()
    except Exception as e:
        st.error(f"Query failed: {e}")
        st.stop()

# Get available weeks without relying on nested caching
def get_available_weeks(num_weeks=12):
    query = f"""
    SELECT DISTINCT
        EXTRACT(WEEK FROM DataAsOfDate) AS WeekNumber,
        EXTRACT(YEAR FROM DataAsOfDate) AS Year,
        MAX(DataAsOfDate) AS LastDataAsOfDate
    FROM `medadhdata2025.adherence_tracking.weekly_med_adherence_data`
    GROUP BY  WeekNumber, Year
    ORDER BY WeekNumber DESC
    LIMIT {num_weeks}
    """
    return run_query(query)

# Get medication adherence data
def get_med_adherence_data(start_date=None, end_date=None, measure_codes=None, market_codes=None, payer_codes=None):
    # Build query filters
    filters = []
    
    if start_date and end_date:
        filters.append(f"DataAsOfDate BETWEEN '{start_date}' AND '{end_date}'")
    
    if measure_codes and len(measure_codes) > 0:
        measure_list = "', '".join(measure_codes)
        filters.append(f"MedAdherenceMeasureCode IN ('{measure_list}')")
    
    if market_codes and len(market_codes) > 0:
        market_list = "', '".join(market_codes)
        filters.append(f"MarketCode IN ('{market_list}')")
    
    if payer_codes and len(payer_codes) > 0:
        payer_list = "', '".join(payer_codes)
        filters.append(f"PayerCode IN ('{payer_list}')")
    
    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    
    query = f"""
    SELECT 
        DataAsOfDate,
        EXTRACT(WEEK FROM DataAsOfDate) AS WeekNumber,
        EXTRACT(YEAR FROM DataAsOfDate) AS Year,
        UGID,
        UPID,
        MedAdherenceMeasureCode,
        MarketCode,
        PayerCode,
        OneFillCode,
        PDCNbr,
        NDCDesc
    FROM `medadhdata2025.adherence_tracking.weekly_med_adherence_data`
        {where_clause}
        ORDER BY DataAsOfDate DESC
    """
    
    return run_query(query)

# Get distinct values for filter options
def get_distinct_values(column_name):
    query = f"""
    SELECT DISTINCT {column_name}
    FROM `medadhdata2025.adherence_tracking.weekly_med_adherence_data`
    ORDER BY {column_name}
    """
    df = run_query(query)
    return df[column_name].tolist() if not df.empty else []

# Calculate metrics for dashboard
def calculate_metrics(df):
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

# Calculate average PDC excluding one-fills and null PDC values
def calculate_average_pdc(df):
    # Filter out one-fills and null PDC values
    pdc_df = df[(df['OneFillCode'].isnull()) & (df['PDCNbr'].notnull())]
    
    # Calculate average PDC by measure type
    if not pdc_df.empty:
        avg_pdc = pdc_df.groupby('MedAdherenceMeasureCode')['PDCNbr'].mean().to_dict()
    else:
        avg_pdc = {}
    
    # Ensure all measure types have values (even if zero)
    for measure in ['MAC', 'MAH', 'MAD']:
        if measure not in avg_pdc:
            avg_pdc[measure] = 0
    
    return avg_pdc

# Load data
st.title("Medication Adherence Dashboard")

# Sidebar filters
st.sidebar.title("Dashboard Filters")

# Load weeks data (without caching decorators to avoid issues)
weeks_df = get_available_weeks(num_weeks=12)

if weeks_df.empty:
    st.error("No weeks data available. Please check your BigQuery table.")
    st.stop()

# Week selection
weeks_options = [f"Week {row['WeekNumber']}, {row['Year']}" for _, row in weeks_df.iterrows()]
selected_week_index = st.sidebar.selectbox(
    "Select Week", 
    range(len(weeks_options)),
    format_func=lambda x: weeks_options[x]
)

# Get current and previous week
current_week = weeks_df.iloc[selected_week_index]
current_week_date = current_week['WeekNumber']

# Get previous week for comparison
prev_week_index = min(selected_week_index + 1, len(weeks_df) - 1) 
prev_week = weeks_df.iloc[prev_week_index]
prev_week_date = prev_week['WeekNumber']

# Measure type filter
measure_types = get_distinct_values("MedAdherenceMeasureCode")
selected_measures = st.sidebar.multiselect(
    "Measure Type", 
    measure_types,
    default=measure_types
)

# Market code filter
market_codes = get_distinct_values("MarketCode")
selected_markets = st.sidebar.multiselect(
    "Market Code", 
    market_codes,
    default=[]
)

# Payer code filter
payer_codes = get_distinct_values("PayerCode")
selected_payers = st.sidebar.multiselect(
    "Payer Code", 
    payer_codes,
    default=[]
)

# Load current week data
st.sidebar.write("Loading data...")
current_data = get_med_adherence_data(
    start_date=current_week_date - timedelta(days=7),
    end_date=current_week_date,
    measure_codes=selected_measures,
    market_codes=selected_markets,
    payer_codes=selected_payers
)

# Load previous week data
prev_data = get_med_adherence_data(
    start_date=prev_week_date - timedelta(days=7),
    end_date=prev_week_date,
    measure_codes=selected_measures,
    market_codes=selected_markets,
    payer_codes=selected_payers
)

# Check if data is available
if current_data.empty:
    st.error("No data available for the selected filters. Please adjust your selection.")
    st.stop()

# Process data
current_metrics = calculate_metrics(current_data)
prev_metrics = calculate_metrics(prev_data)
average_pdc = calculate_average_pdc(current_data)

# Dashboard Header
st.markdown(f"**Current Week: {current_week_date.strftime('%Y-%m-%d')}**")

# KPI Cards Row
col1, col2, col3, col4 = st.columns(4)

# UGIDs (Gaps) KPI
with col1:
    st.metric(
        "Total Gaps (UGIDs)",
        f"{current_metrics['total_ugids']:,}",
        f"{((current_metrics['total_ugids'] - prev_metrics['total_ugids']) / max(prev_metrics['total_ugids'], 1) * 100):.1f}%" 
    )

# UPIDs (Patients) KPI
with col2:
    st.metric(
        "Unique Patients (UPIDs)",
        f"{current_metrics['total_upids']:,}",
        f"{((current_metrics['total_upids'] - prev_metrics['total_upids']) / max(prev_metrics['total_upids'], 1) * 100):.1f}%" 
    )

# Fill Status KPI
with col3:
    st.metric(
        "One-Fill Gaps",
        f"{current_metrics['one_fill_count']:,}",
        f"{((current_metrics['one_fill_count'] - prev_metrics['one_fill_count']) / max(prev_metrics['one_fill_count'], 1) * 100):.1f}%" 
    )

# Denominator Gaps KPI
with col4:
    st.metric(
        "Denominator Gaps",
        f"{current_metrics['denominator_gap_count']:,}",
        f"{((current_metrics['denominator_gap_count'] - prev_metrics['denominator_gap_count']) / max(prev_metrics['denominator_gap_count'], 1) * 100):.1f}%" 
    )

# Measure Type Analysis
st.header("Measure Type Analysis")
col1, col2 = st.columns(2)

with col1:
    # Measure Type Donut Chart
    measure_data = pd.DataFrame({
        'Measure': ['MAC (Cholesterol)', 'MAH (Hypertension)', 'MAD (Diabetes)'],
        'Count': [current_metrics['mac_count'], current_metrics['mah_count'], current_metrics['mad_count']]
    })
    
    fig = px.pie(
        measure_data,
        values='Count',
        names='Measure',
        title="Distribution by Measure Type",
        hole=0.4
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # PDC Average by Measure Type
    pdc_data = pd.DataFrame({
        'Measure': ['MAC (Cholesterol)', 'MAH (Hypertension)', 'MAD (Diabetes)'],
        'Average PDC': [average_pdc.get('MAC', 0), average_pdc.get('MAH', 0), average_pdc.get('MAD', 0)]
    })
    
    fig = px.bar(
        pdc_data,
        x='Measure',
        y='Average PDC',
        title="Average PDC by Measure Type (Denominator Gaps Only)",
        color='Measure',
        text_auto='.2f'
    )
    
    # Add 80% threshold line
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=2.5,
        y0=0.8,
        y1=0.8,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.add_annotation(
        x=2.5,
        y=0.8,
        text="80% PDC Threshold",
        showarrow=False,
        yshift=10,
        xshift=-5,
        font=dict(color="red")
    )
    
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

# PDC Distribution Analysis
st.header("PDC Distribution")

# Filter for denominator gaps with valid PDC
pdc_analysis_df = current_data[(current_data['OneFillCode'].isnull()) & (current_data['PDCNbr'].notnull())]

if not pdc_analysis_df.empty:
    # Create PDC distribution histogram
    fig = px.histogram(
        pdc_analysis_df,
        x='PDCNbr',
        color='MedAdherenceMeasureCode',
        title="PDC Distribution for Denominator Gaps",
        nbins=20,
        histnorm='percent',
        barmode='overlay',
        opacity=0.7
    )
    
    # Add 80% threshold line
    fig.add_shape(
        type="line",
        x0=0.8,
        x1=0.8,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.add_annotation(
        x=0.8,
        y=1,
        text="80% PDC Threshold",
        showarrow=False,
        yshift=10,
        xshift=10,
        font=dict(color="red")
    )
    
    fig.update_layout(xaxis_title="PDC Value", yaxis_title="Percentage")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No valid PDC data available for analysis with current filters.")

# Footer
st.markdown("---")
st.markdown("Data last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))