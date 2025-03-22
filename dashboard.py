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
            'denominator_gap_count': 0,
            'measure_counts': {}
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
    
    # Store full measure counts dictionary
    metrics['measure_counts'] = measure_counts
    
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
        # Get average PDC for each measure type
        avg_pdc = pdc_df.groupby('MedAdherenceMeasureCode')['PDCNbr'].mean().to_dict()
        
        # Also calculate count and std deviation for each measure
        count_pdc = pdc_df.groupby('MedAdherenceMeasureCode')['PDCNbr'].count().to_dict()
        std_pdc = pdc_df.groupby('MedAdherenceMeasureCode')['PDCNbr'].std().to_dict()
        
        # Calculate compliance rate (PDC >= 0.8) for each measure
        compliant_counts = {}
        for measure in avg_pdc.keys():
            measure_df = pdc_df[pdc_df['MedAdherenceMeasureCode'] == measure]
            compliant_counts[measure] = (measure_df['PDCNbr'] >= 0.8).sum() / len(measure_df) if len(measure_df) > 0 else 0
    else:
        avg_pdc = {}
        count_pdc = {}
        std_pdc = {}
        compliant_counts = {}
    
    # Ensure standard measure types have values (even if zero)
    for measure in ['MAC', 'MAH', 'MAD']:
        if measure not in avg_pdc:
            avg_pdc[measure] = 0
            count_pdc[measure] = 0
            std_pdc[measure] = 0
            compliant_counts[measure] = 0
    
    # Return dictionary with all PDC statistics
    return {
        'avg': avg_pdc,
        'count': count_pdc,
        'std': std_pdc,
        'compliance_rate': compliant_counts
    }

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
pdc_statistics = calculate_average_pdc(current_data)
average_pdc = pdc_statistics['avg']  # For backward compatibility
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
st.markdown(f"**Current Week: Week of {current_week_date.strftime('%Y-%m-%d')}**")

# Add summary of applied filters
filters_applied = []
if selected_measures and len(selected_measures) < len(measure_types):
    filters_applied.append(f"Measures: {', '.join(selected_measures)}")
if selected_markets:
    filters_applied.append(f"Markets: {', '.join(selected_markets)}")
if selected_payers:
    filters_applied.append(f"Payers: {', '.join(selected_payers)}")

if filters_applied:
    st.markdown("**Applied Filters:** " + " | ".join(filters_applied))

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
    if 'measure_counts' in current_metrics and current_metrics['measure_counts']:
        # Create DataFrame from the full measure counts dictionary
        measure_counts = current_metrics['measure_counts']
        measure_labels = {
            'MAC': 'MAC (Cholesterol)',
            'MAH': 'MAH (Hypertension)',
            'MAD': 'MAD (Diabetes)'
        }
        
        measure_data = pd.DataFrame({
            'Measure': [measure_labels.get(key, key) for key in measure_counts.keys()],
            'Count': list(measure_counts.values())
        })
    else:
        # Fallback to standard measures if no counts available
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
    
    # Add count information to hover
    fig.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent:.1%}<extra></extra>')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display count table below the chart
    if measure_data['Count'].sum() > 0:
        st.markdown("#### Measure Counts")
        # Simplified dataframe display without custom column config
        st.dataframe(
            measure_data,
            hide_index=True
        )

with col2:
    # PDC Average by Measure Type - using all measures present in the data
    if 'measure_counts' in current_metrics and current_metrics['measure_counts']:
        # Get all measures present in the data
        measure_codes = list(current_metrics['measure_counts'].keys())
        measure_labels = {
            'MAC': 'MAC (Cholesterol)',
            'MAH': 'MAH (Hypertension)',
            'MAD': 'MAD (Diabetes)'
        }
        
        # Create data for the chart
        pdc_data = pd.DataFrame({
            'Measure': [measure_labels.get(code, code) for code in measure_codes],
            'Average PDC': [average_pdc.get(code, 0) for code in measure_codes]
        })
    else:
        # Fallback to standard measures
        pdc_data = pd.DataFrame({
            'Measure': ['MAC (Cholesterol)', 'MAH (Hypertension)', 'MAD (Diabetes)'],
            'Average PDC': [average_pdc.get('MAC', 0), average_pdc.get('MAH', 0), average_pdc.get('MAD', 0)]
        })
    
    # Add compliance rate to the data
    pdc_data['Compliance Rate'] = [pdc_statistics['compliance_rate'].get(code, 0) 
                                 for code in pdc_statistics['avg'].keys()]
    
    # Create a more informative bar chart with both average PDC and compliance rate
    fig = px.bar(
        pdc_data,
        x='Measure',
        y='Average PDC',
        title="PDC Statistics by Measure Type (Denominator Gaps Only)",
        color='Measure',
        text_auto='.2f',
        color_discrete_sequence=px.colors.qualitative.Safe,
        hover_data={
            'Measure': True,
            'Average PDC': ':.2f',
            'Compliance Rate': ':.1%'
        }
    )
    
    # Add 80% threshold line
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(pdc_data) - 0.5,
        y0=0.8,
        y1=0.8,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.add_annotation(
        x=len(pdc_data) - 0.5,
        y=0.8,
        text="80% PDC Threshold",
        showarrow=False,
        yshift=10,
        xshift=-5,
        font=dict(color="red")
    )
    
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    
    # Add a data table with detailed PDC statistics
    if not pdc_data.empty and pdc_data['Average PDC'].sum() > 0:
        st.markdown("#### PDC Statistics by Measure")
        
        # Create detailed statistics table
        stats_data = pd.DataFrame({
            'Measure': pdc_data['Measure'],
            'Count': [pdc_statistics['count'].get(code, 0) for code in pdc_statistics['avg'].keys()],
            'Average PDC': pdc_data['Average PDC'],
            'Std Deviation': [pdc_statistics['std'].get(code, 0) for code in pdc_statistics['avg'].keys()],
            'Compliance Rate': pdc_data['Compliance Rate']
        })
        
        # Format the compliance rate as percentage string for simpler display
        stats_data['Compliance Rate'] = stats_data['Compliance Rate'].apply(lambda x: f"{x:.1%}")
        
        # Simplified dataframe display without custom column config
        st.dataframe(
            stats_data,
            hide_index=True
        )

# PDC Distribution Analysis
st.header("PDC Distribution")

# Filter for denominator gaps with valid PDC
pdc_analysis_df = current_data[(current_data['OneFillCode'].isnull()) & (current_data['PDCNbr'].notnull())]

col1, col2 = st.columns(2)

with col1:
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

with col2:
    if not pdc_analysis_df.empty:
        # Calculate the count of members above/below 80% PDC
        pdc_threshold = 0.80
        compliant_count = (pdc_analysis_df['PDCNbr'] >= pdc_threshold).sum()
        non_compliant_count = (pdc_analysis_df['PDCNbr'] < pdc_threshold).sum()
        total_count = len(pdc_analysis_df)
        
        compliance_data = pd.DataFrame({
            'Status': ['Compliant (PDC ≥ 80%)', 'Non-Compliant (PDC < 80%)'],
            'Count': [compliant_count, non_compliant_count],
            'Percent': [compliant_count/total_count*100, non_compliant_count/total_count*100]
        })
        
        # Create compliance pie chart
        fig = px.pie(
            compliance_data,
            values='Count',
            names='Status',
            title="PDC Compliance Status",
            color_discrete_sequence=['#00CC96', '#EF553B'],
            hole=0.4,
        )
        
        # Add percentages to the labels
        fig.update_traces(
            texttemplate='%{label}<br>%{percent:.1%}',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent:.1%}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show compliance stats
        st.metric(
            "Compliance Rate", 
            f"{compliant_count/total_count:.1%}",
            help="Percentage of patients with PDC ≥ 80%"
        )
    else:
        st.info("No PDC data available to calculate compliance rates.")
        # Create empty pie chart
        empty_df = pd.DataFrame({
            'Status': ['No Data'],
            'Count': [1]
        })
        fig = px.pie(
            empty_df,
            values='Count',
            names='Status',
            title="PDC Compliance Status (No Data Available)",
            hole=0.4
        )
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