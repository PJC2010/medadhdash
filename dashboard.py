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
@st.cache_data(ttl=600)  # Cache for 10 minutes
def run_query(query):
    try:
        df = client.query(query).to_dataframe()
        return df
    except Exception as e:
        st.error(f"Query failed: {e}")
        return pd.DataFrame()  # Return empty dataframe instead of stopping

# Get available weeks
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_available_weeks(num_weeks=12):
    query = f"""
    SELECT DISTINCT
        WeekOf,
        EXTRACT(WEEK FROM WeekOf) AS WeekNumber,
        EXTRACT(YEAR FROM WeekOf) AS Year
    FROM `medadhdata2025.adherence_tracking.weekly_med_adherence_data`
    WHERE WeekOf IS NOT NULL
    ORDER BY WeekOf DESC
    LIMIT {num_weeks}
    """
    return run_query(query)

# Get medication adherence data
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_med_adherence_data(start_date=None, end_date=None, measure_codes=None, market_codes=None, payer_codes=None):
    # Build query filters
    filters = []
    
    if start_date and end_date:
        filters.append(f"WeekOf = '{start_date}'")
    
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
        WeekOf,
        EXTRACT(WEEK FROM WeekOf) AS WeekNumber,
        EXTRACT(YEAR FROM WeekOf) AS Year,
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
    ORDER BY WeekOf DESC
    """
    
    return run_query(query)

# Get distinct values for filter options
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_distinct_values(column_name):
    query = f"""
    SELECT DISTINCT {column_name}
    FROM `medadhdata2025.adherence_tracking.weekly_med_adherence_data`
    WHERE {column_name} IS NOT NULL
    ORDER BY {column_name}
    """
    df = run_query(query)
    return df[column_name].tolist() if not df.empty else []

# Calculate metrics for dashboard
def calculate_metrics(df):
    if df.empty:
        return {
            'total_ugids': 0,
            'total_upids': 0,
            'mac_count': 0,
            'mah_count': 0,
            'mad_count': 0,
            'one_fill_count': 0,
            'denominator_gap_count': 0
        }
    
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
    if df.empty:
        return {'MAC': 0, 'MAH': 0, 'MAD': 0}
    
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

# Calculate percent change safely
def calculate_percent_change(current, previous):
    if previous == 0:
        return 0 if current == 0 else 100  # If previous is 0, return 0% if current is also 0, else 100%
    return ((current - previous) / previous) * 100

# Title and intro
st.title("Medication Adherence Dashboard")
st.markdown("This dashboard tracks gaps in medication adherence for patients with cholesterol, diabetes, and hypertension prescriptions.")

# Load weeks data
with st.spinner("Loading dashboard data..."):
    weeks_df = get_available_weeks(num_weeks=12)

    if weeks_df.empty:
        st.error("No weeks data available. Please check your BigQuery table.")
        st.stop()

# Sidebar filters
st.sidebar.title("Dashboard Filters")

# Week selection
weeks_options = [f"Week {int(row['WeekNumber'])}, {int(row['Year'])} (Week of {row['WeekOf'].strftime('%b %d, %Y')})" for _, row in weeks_df.iterrows()]
selected_week_index = st.sidebar.selectbox(
    "Select Week", 
    range(len(weeks_options)),
    format_func=lambda x: weeks_options[x]
)

# Get current and previous week
current_week = weeks_df.iloc[selected_week_index]
current_week_date = current_week['WeekOf']

# Get previous week for comparison
prev_week_index = min(selected_week_index + 1, len(weeks_df) - 1)
prev_week = weeks_df.iloc[prev_week_index]
prev_week_date = prev_week['WeekOf']

# Display loading indicator
with st.sidebar:
    with st.spinner("Loading data..."):
        # Measure type filter - load before applying filters to ensure options are available
        measure_types = get_distinct_values("MedAdherenceMeasureCode")
        selected_measures = st.multiselect(
            "Measure Type", 
            measure_types,
            default=measure_types
        )
        
        # Market code filter
        market_codes = get_distinct_values("MarketCode")
        selected_markets = st.multiselect(
            "Market Code", 
            market_codes,
            default=[]
        )
        
        # Payer code filter
        payer_codes = get_distinct_values("PayerCode")
        selected_payers = st.multiselect(
            "Payer Code", 
            payer_codes,
            default=[]
        )

# Set progress indicator
progress_bar = st.progress(0)
st.markdown("Loading data...")

# Load current week data
progress_bar.progress(25)
current_data = get_med_adherence_data(
    start_date=current_week_date,
    end_date=current_week_date,
    measure_codes=selected_measures,
    market_codes=selected_markets if selected_markets else None,
    payer_codes=selected_payers if selected_payers else None
)
progress_bar.progress(50)

# Load previous week data
prev_data = get_med_adherence_data(
    start_date=prev_week_date,
    end_date=prev_week_date,
    measure_codes=selected_measures,
    market_codes=selected_markets if selected_markets else None,
    payer_codes=selected_payers if selected_payers else None
)
progress_bar.progress(75)

# Process data
current_metrics = calculate_metrics(current_data)
prev_metrics = calculate_metrics(prev_data)
average_pdc = calculate_average_pdc(current_data)
progress_bar.progress(100)

# Remove progress indicators
progress_bar.empty()
st.markdown("")  # Clear loading message

# Check if data is available
if current_data.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
    # Continue anyway to show empty charts with proper formatting
else:
    st.success(f"Loaded {len(current_data):,} records for the selected week.")

# Dashboard Header
st.markdown(f"**Current Week: {current_week_date.strftime('%Y-%m-%d')}**")

# KPI Cards Row
col1, col2, col3, col4 = st.columns(4)

# UGIDs (Gaps) KPI
with col1:
    percent_change = calculate_percent_change(current_metrics['total_ugids'], prev_metrics['total_ugids'])
    st.metric(
        "Total Gaps (UGIDs)",
        f"{current_metrics['total_ugids']:,}",
        f"{percent_change:.1f}%",
        delta_color="inverse"  # Decreasing is good for gaps
    )

# UPIDs (Patients) KPI
with col2:
    percent_change = calculate_percent_change(current_metrics['total_upids'], prev_metrics['total_upids'])
    st.metric(
        "Unique Patients (UPIDs)",
        f"{current_metrics['total_upids']:,}",
        f"{percent_change:.1f}%",
        delta_color="inverse"  # Decreasing is good for gaps
    )

# Fill Status KPI
with col3:
    percent_change = calculate_percent_change(current_metrics['one_fill_count'], prev_metrics['one_fill_count'])
    st.metric(
        "One-Fill Gaps",
        f"{current_metrics['one_fill_count']:,}",
        f"{percent_change:.1f}%",
        delta_color="inverse"  # Decreasing is good for gaps
    )

# Denominator Gaps KPI
with col4:
    percent_change = calculate_percent_change(current_metrics['denominator_gap_count'], prev_metrics['denominator_gap_count'])
    st.metric(
        "Denominator Gaps",
        f"{current_metrics['denominator_gap_count']:,}",
        f"{percent_change:.1f}%",
        delta_color="inverse"  # Decreasing is good for gaps
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
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    
    # Only show percents if we have data
    if measure_data['Count'].sum() > 0:
        fig.update_traces(textinfo='percent+label')
    else:
        fig.update_traces(textinfo='none')
        
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
        text_auto='.2f',
        color_discrete_sequence=px.colors.qualitative.Safe
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
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Safe
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
    st.info("No valid PDC data available for analysis with current filters.")
    # Create an empty histogram to maintain layout
    empty_df = pd.DataFrame({'PDCNbr': [0], 'MedAdherenceMeasureCode': ['No Data']})
    fig = px.histogram(
        empty_df,
        x='PDCNbr',
        color='MedAdherenceMeasureCode',
        title="PDC Distribution for Denominator Gaps (No Data Available)",
        nbins=20
    )
    fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

# Market Analysis (if markets are selected)
if selected_markets:
    st.header("Market Analysis")
    
    if not current_data.empty:
        # Group by market
        market_analysis = current_data.groupby('MarketCode').agg(
            TotalGaps=('UGID', 'count'),
            UniquePatients=('UPID', 'nunique'),
            OneFillCount=('OneFillCode', lambda x: (x == 'Yes').sum()),
            AvgPDC=('PDCNbr', lambda x: x.mean() if x.count() > 0 else 0)
        ).reset_index()
        
        # Fill NaN values
        market_analysis['AvgPDC'] = market_analysis['AvgPDC'].fillna(0)
        
        # Create bar chart
        fig = px.bar(
            market_analysis,
            x='MarketCode',
            y='TotalGaps',
            color='MarketCode',
            title="Total Gaps by Market",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Create PDC by market chart
        fig = px.bar(
            market_analysis,
            x='MarketCode',
            y='AvgPDC',
            color='MarketCode',
            title="Average PDC by Market",
            text_auto='.2f'
        )
        # Add 80% threshold line
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(market_analysis)-0.5,
            y0=0.8,
            y1=0.8,
            line=dict(color="red", width=2, dash="dash"),
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected markets.")

# Payer Analysis (if payers are selected)
if selected_payers:
    st.header("Payer Analysis")
    
    if not current_data.empty:
        # Group by payer
        payer_analysis = current_data.groupby('PayerCode').agg(
            TotalGaps=('UGID', 'count'),
            UniquePatients=('UPID', 'nunique'),
            OneFillCount=('OneFillCode', lambda x: (x == 'Yes').sum()),
            AvgPDC=('PDCNbr', lambda x: x.mean() if x.count() > 0 else 0)
        ).reset_index()
        
        # Fill NaN values
        payer_analysis['AvgPDC'] = payer_analysis['AvgPDC'].fillna(0)
        
        # Create bar chart
        fig = px.bar(
            payer_analysis,
            x='PayerCode',
            y='TotalGaps',
            color='PayerCode',
            title="Total Gaps by Payer",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for the selected payers.")

# Footer with download option
st.markdown("---")
if not current_data.empty:
    st.download_button(
        label="Download Current Data as CSV",
        data=current_data.to_csv(index=False).encode('utf-8'),
        file_name=f'med_adherence_data_{current_week_date.strftime("%Y-%m-%d")}.csv',
        mime='text/csv',
    )

# Display data timestamp
st.markdown(f"Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")