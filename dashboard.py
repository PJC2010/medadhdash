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
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_month_over_month_data(lookback_months=6, measure_codes=None, market_codes=None, payer_codes=None):
    """
    Get month-over-month trend data for medication adherence metrics
    
    Args:
        lookback_months: Number of months to look back from current date
        measure_codes: List of measure codes to filter by
        market_codes: List of market codes to filter by
        payer_codes: List of payer codes to filter by
    
    Returns:
        DataFrame with monthly trend data
    """
    # Build query filters
    filters = [f"WeekOf >= DATE_SUB(CURRENT_DATE(), INTERVAL {lookback_months} MONTH)"]
    
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
        EXTRACT(MONTH FROM WeekOf) AS Month,
        EXTRACT(YEAR FROM WeekOf) AS Year,
        FORMAT_DATE('%b %Y', DATE_TRUNC(WeekOf, MONTH)) AS MonthLabel,
        DATE_TRUNC(WeekOf, MONTH) AS MonthDate,
        MedAdherenceMeasureCode,
        COUNT(DISTINCT UGID) AS TotalGaps,
        COUNT(DISTINCT UPID) AS UniquePatients,
        COUNTIF(OneFillCode = 'Yes') AS OneFillCount,
        AVG(IF(OneFillCode IS NULL AND PDCNbr IS NOT NULL, PDCNbr, NULL)) AS AvgPDC,
        COUNTIF(PDCNbr >= 0.8 AND OneFillCode IS NULL AND PDCNbr IS NOT NULL) / 
        NULLIF(COUNTIF(OneFillCode IS NULL AND PDCNbr IS NOT NULL), 0) AS ComplianceRate
    FROM `medadhdata2025.adherence_tracking.weekly_med_adherence_data`
    {where_clause}
    GROUP BY Month, Year, MonthLabel, MonthDate, MedAdherenceMeasureCode
    ORDER BY Year, Month, MedAdherenceMeasureCode
    """
    
    return run_query(query)
def get_star_thresholds():
    """Return the CMS Star Rating thresholds for medication adherence measures"""
    return {
        "MAC": {  # Medication adherence for cholesterol (statins)
            "2_star": 0.84,  # 84%
            "3_star": 0.89,  # 89%
            "4_star": 0.91,  # 91%
            "5_star": 0.93   # 93%
        },
        "MAH": {  # Medication adherence for hypertension (RAS antagonists)
            "2_star": 0.84,  # 84%
            "3_star": 0.88,  # 88%
            "4_star": 0.91,  # 91%
            "5_star": 0.93   # 93%
        },
        "MAD": {  # Medication adherence for diabetes medication
            "2_star": 0.82,  # 82%
            "3_star": 0.86,  # 86%
            "4_star": 0.90,  # 90%
            "5_star": 0.92   # 92%
        }
    }
def create_monthly_trend_chart(df, metric="AvgPDC", include_thresholds=True):
    """
    Create a line chart showing month-over-month trends by measure type
    
    Args:
        df: DataFrame with monthly data
        metric: Which metric to plot ("AvgPDC" or "ComplianceRate")
        include_thresholds: Whether to include star rating thresholds
    
    Returns:
        Plotly figure object
    """
    if df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No monthly trend data available",
            xaxis_title="Month",
            yaxis_title="Value"
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Get the measure codes in the data
    measures = df["MedAdherenceMeasureCode"].unique()
    
    # Color mapping
    colors = {
        "MAC": "#636EFA",  # Blue
        "MAH": "#00CC96",  # Green
        "MAD": "#FFA15A"   # Orange
    }
    
    # Measure labels for legend
    measure_labels = {
        "MAC": "MAC (Cholesterol)",
        "MAH": "MAH (Hypertension)",
        "MAD": "MAD (Diabetes)"
    }
    
    # Add a line for each measure
    for measure in measures:
        measure_data = df[df["MedAdherenceMeasureCode"] == measure].copy()
        
        # Convert MonthDate to datetime for proper sorting
        if 'MonthDate' in measure_data.columns:
            measure_data['MonthDate'] = pd.to_datetime(measure_data['MonthDate'])
            measure_data = measure_data.sort_values("MonthDate")
        else:
            # Sort by MonthLabel if MonthDate is not available
            measure_data = measure_data.sort_values("MonthLabel")
        
        fig.add_trace(go.Scatter(
            x=measure_data["MonthLabel"],
            y=measure_data[metric],
            mode="lines+markers",
            name=measure_labels.get(measure, measure),
            line=dict(color=colors.get(measure, "#636EFA"), width=3),
            marker=dict(size=8)
        ))
    
    # Add threshold lines if requested and we're plotting AvgPDC
    if include_thresholds and metric == "AvgPDC":
        # Get star thresholds
        star_thresholds = get_star_thresholds()
        
        # Add threshold lines for each measure
        for measure in measures:
            if measure in star_thresholds:
                thresholds = star_thresholds[measure]
                
                # 3-star threshold (dotted line)
                fig.add_shape(
                    type="line",
                    x0=0,
                    x1=len(df["MonthLabel"].unique()) - 1,
                    y0=thresholds["3_star"],
                    y1=thresholds["3_star"],
                    line=dict(color=colors.get(measure, "#636EFA"), width=1.5, dash="dot"),
                    xref="x domain",
                    yref="y"
                )
                
                # 4-star threshold (dashed line)
                fig.add_shape(
                    type="line",
                    x0=0,
                    x1=len(df["MonthLabel"].unique()) - 1,
                    y0=thresholds["4_star"],
                    y1=thresholds["4_star"],
                    line=dict(color=colors.get(measure, "#636EFA"), width=1.5, dash="dash"),
                    xref="x domain",
                    yref="y"
                )
                
                # 5-star threshold (solid line)
                fig.add_shape(
                    type="line",
                    x0=0,
                    x1=len(df["MonthLabel"].unique()) - 1,
                    y0=thresholds["5_star"],
                    y1=thresholds["5_star"],
                    line=dict(color=colors.get(measure, "#636EFA"), width=1.5),
                    xref="x domain",
                    yref="y"
                )
                
                # Add annotations for the thresholds
                fig.add_annotation(
                    x=1,  # Right side of the chart
                    y=thresholds["3_star"],
                    text=f"{measure_labels.get(measure, measure)} 3★ ({thresholds['3_star']*100:.0f}%)",
                    showarrow=False,
                    xshift=10,
                    xref="paper",
                    yref="y",
                    font=dict(color=colors.get(measure, "#636EFA"), size=10)
                )
                
                fig.add_annotation(
                    x=1,  # Right side of the chart
                    y=thresholds["4_star"],
                    text=f"{measure_labels.get(measure, measure)} 4★ ({thresholds['4_star']*100:.0f}%)",
                    showarrow=False,
                    xshift=10,
                    xref="paper",
                    yref="y",
                    font=dict(color=colors.get(measure, "#636EFA"), size=10)
                )
                
                fig.add_annotation(
                    x=1,  # Right side of the chart
                    y=thresholds["5_star"],
                    text=f"{measure_labels.get(measure, measure)} 5★ ({thresholds['5_star']*100:.0f}%)",
                    showarrow=False,
                    xshift=10,
                    xref="paper",
                    yref="y",
                    font=dict(color=colors.get(measure, "#636EFA"), size=10)
                )
    
    # Set title based on metric
    title = "Average PDC by Month" if metric == "AvgPDC" else "Compliance Rate by Month"
    if include_thresholds and metric == "AvgPDC":
        title += " (with Star Rating Thresholds)"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="PDC Value" if metric == "AvgPDC" else "Compliance Rate",
        yaxis=dict(
            range=[0.75, 1.0],  # Set y-axis range to focus on relevant values
            tickformat=".0%"    # Format as percentage
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=60)
    )
    
    return fig

# Add this function to create the star rating comparison chart
def create_star_rating_comparison_chart(df, metric="AvgPDC"):
    """
    Create a bar chart comparing current PDC values to star rating thresholds
    
    Args:
        df: DataFrame with current performance data
        metric: Which metric to plot ("AvgPDC" or "ComplianceRate")
    
    Returns:
        Plotly figure object
    """
    if df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No data available for star rating comparison",
            xaxis_title="Measure",
            yaxis_title="Value"
        )
        return fig
    
    # Get star thresholds
    star_thresholds = get_star_thresholds()
    
    # Create figure
    fig = go.Figure()
    
    # Color mapping
    colors = {
        "MAC": "#636EFA",  # Blue
        "MAH": "#00CC96",  # Green
        "MAD": "#FFA15A"   # Orange
    }
    
    # Measure labels for display
    measure_labels = {
        "MAC": "MAC (Cholesterol)",
        "MAH": "MAH (Hypertension)",
        "MAD": "MAD (Diabetes)"
    }
    
    # Group by measure and get the latest data for each
    measure_data = df.groupby("MedAdherenceMeasureCode").agg({
        metric: "mean",  # Get the average PDC or compliance rate
        "MonthLabel": "first"  # Just need one month label for display
    }).reset_index()
    
    # X-axis categories for the chart
    measures = measure_data["MedAdherenceMeasureCode"].tolist()
    x_categories = [measure_labels.get(m, m) for m in measures]
    
    # Add bars for current values
    fig.add_trace(go.Bar(
        x=x_categories,
        y=measure_data[metric],
        name="Current Value",
        marker_color=[colors.get(m, "#636EFA") for m in measures],
        text=[f"{v*100:.1f}%" for v in measure_data[metric]],
        textposition="inside",
        textfont=dict(color="white"),
        width=0.5
    ))
    
    # Add a shape and annotation for each star threshold
    for star_level in ["3_star", "4_star", "5_star"]:
        star_num = star_level[0]  # Extract the number
        
        for i, measure in enumerate(measures):
            if measure in star_thresholds:
                threshold = star_thresholds[measure][star_level]
                
                # Line style based on star level
                line_style = "dot" if star_num == "3" else "dash" if star_num == "4" else "solid"
                
                # Line color based on star level
                line_color = "gray" if star_num == "3" else "orange" if star_num == "4" else "green"
                
                # Add horizontal line for this threshold for this measure
                fig.add_shape(
                    type="line",
                    x0=i-0.25,
                    x1=i+0.25,
                    y0=threshold,
                    y1=threshold,
                    line=dict(color=line_color, width=2, dash=line_style),
                )
                
                # Add annotation only once per threshold level
                if i == 0:
                    fig.add_trace(go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        name=f"{star_num}-Star Threshold",
                        line=dict(color=line_color, width=2, dash=line_style),
                    ))
                
                # Add text annotations for the thresholds
                fig.add_annotation(
                    x=i,
                    y=threshold,
                    text=f"{star_num}★: {threshold*100:.0f}%",
                    showarrow=False,
                    xshift=0,
                    yshift=10,
                    font=dict(size=10, color=line_color)
                )
    
    # Update layout
    fig.update_layout(
        title="Current Performance vs Star Rating Thresholds",
        xaxis_title="Measure",
        yaxis_title="PDC Value",
        yaxis=dict(
            range=[0.75, 1.0],  # Set y-axis range to focus on relevant values
            tickformat=".0%"    # Format as percentage
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=60)
    )
    
    return fig



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
st.markdown("This dashboard tracks gaps in medication adherence for patients with cholesterol, diabetes, and hypertension medications.")

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
    
    # Display count table below the chart as markdown
    if measure_data['Count'].sum() > 0:
        st.markdown("#### Measure Counts")
        
        # Create a markdown table instead of dataframe
        measure_table = "| Measure | Count |\n| --- | --- |\n"
        for _, row in measure_data.iterrows():
            measure_table += f"| {row['Measure']} | {row['Count']:,} |\n"
        
        st.markdown(measure_table)

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
    
    # Add a markdown table with detailed PDC statistics
    if not pdc_data.empty and pdc_data['Average PDC'].sum() > 0:
        st.markdown("#### PDC Statistics by Measure")
        
        # Create detailed statistics table as markdown
        stats_table = "| Measure | Count | Average PDC | Std Deviation | Compliance Rate |\n"
        stats_table += "| --- | --- | --- | --- | --- |\n"
        
        for i, row in pdc_data.iterrows():
            measure_code = list(pdc_statistics['avg'].keys())[i]
            stats_table += f"| {row['Measure']} | {pdc_statistics['count'].get(measure_code, 0):,} | "
            stats_table += f"{row['Average PDC']:.2f} | {pdc_statistics['std'].get(measure_code, 0):.3f} | "
            stats_table += f"{row['Compliance Rate']:.1%} |\n"
        
        st.markdown(stats_table)

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
        
        # Display market data as markdown
        st.markdown("#### Market Analysis Data")
        market_table = "| Market | Total Gaps | Unique Patients | One-Fill Count | Avg PDC |\n"
        market_table += "| --- | --- | --- | --- | --- |\n"
        
        for _, row in market_analysis.iterrows():
            market_table += f"| {row['MarketCode']} | {row['TotalGaps']:,} | {row['UniquePatients']:,} | "
            market_table += f"{row['OneFillCount']:,} | {row['AvgPDC']:.2f} |\n"
        
        st.markdown(market_table)
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
        
        # Display payer data as markdown
        st.markdown("#### Payer Analysis Data")
        payer_table = "| Payer | Total Gaps | Unique Patients | One-Fill Count | Avg PDC |\n"
        payer_table += "| --- | --- | --- | --- | --- |\n"
        
        for _, row in payer_analysis.iterrows():
            payer_table += f"| {row['PayerCode']} | {row['TotalGaps']:,} | {row['UniquePatients']:,} | "
            payer_table += f"{row['OneFillCount']:,} | {row['AvgPDC']:.2f} |\n"
        
        st.markdown(payer_table)
    else:
        st.info("No data available for the selected payers.")

st.header("Month-over-Month Trend Analysis")

# Load the monthly trend data
monthly_data = get_month_over_month_data(
    lookback_months=6,
    measure_codes=selected_measures,
    market_codes=selected_markets if selected_markets else None,
    payer_codes=selected_payers if selected_payers else None
)

col1, col2 = st.columns(2)

with col1:
    # Generate and display the monthly trend chart
    if not monthly_data.empty:
        trend_chart = create_monthly_trend_chart(monthly_data, "AvgPDC", include_thresholds=True)
        st.plotly_chart(trend_chart, use_container_width=True)
    else:
        st.info("No monthly trend data available for the selected filters.")

with col2:
    # Generate and display the star rating comparison chart
    if not monthly_data.empty:
        star_comparison_chart = create_star_rating_comparison_chart(monthly_data, "AvgPDC")
        st.plotly_chart(star_comparison_chart, use_container_width=True)
    else:
        st.info("No data available for star rating comparison.")

# Add a data table with month-over-month PDC averages if data is available
if not monthly_data.empty:
    st.markdown("### Month-over-Month PDC Averages")
    
    # Pivot the data to show months as columns
    pivot_df = monthly_data.pivot_table(
        index="MedAdherenceMeasureCode",
        columns="MonthLabel",
        values="AvgPDC",
        aggfunc="mean"
    ).reset_index()
    
    # Format the values as percentages
    for col in pivot_df.columns:
        if col != "MedAdherenceMeasureCode":
            pivot_df[col] = pivot_df[col].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "-")
    
    # Add measure labels
    measure_labels = {
        "MAC": "MAC (Cholesterol)",
        "MAH": "MAH (Hypertension)",
        "MAD": "MAD (Diabetes)"
    }
    pivot_df["Measure"] = pivot_df["MedAdherenceMeasureCode"].map(lambda x: measure_labels.get(x, x))
    
    # Reorder columns to show Measure first
    cols = pivot_df.columns.tolist()
    cols.remove("MedAdherenceMeasureCode")
    cols.remove("Measure")
    cols = ["Measure"] + cols
    
    # Show the table
    st.dataframe(pivot_df[cols], use_container_width=True)
    
    # Also add a note about star rating thresholds
    st.markdown("### Star Rating Thresholds")
    
    # Create a DataFrame with the thresholds
    star_thresholds = get_star_thresholds()
    thresholds_data = []
    
    for measure, thresholds in star_thresholds.items():
        thresholds_data.append({
            "Measure": measure_labels.get(measure, measure),
            "3-Star": f"{thresholds['3_star']:.0%}",
            "4-Star": f"{thresholds['4_star']:.0%}",
            "5-Star": f"{thresholds['5_star']:.0%}"
        })
    
    thresholds_df = pd.DataFrame(thresholds_data)
    st.dataframe(thresholds_df, use_container_width=True)

# Footer with download 