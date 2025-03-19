import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import os
from utils.data_loader import BigQueryConnector
from utils.data_processor import (
    calculate_metrics, 
    calculate_average_pdc, 
    get_week_over_week_data,
    create_market_payer_summary
)

# Page config
st.set_page_config(
    page_title="Medication Adherence Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Connect to BigQuery
@st.cache_resource
def get_bigquery_connector():
    credentials_path = os.path.join(os.path.dirname(__file__), "auth", "medadhdata2025-47d6f2bb49b8.json")
    return BigQueryConnector(credentials_path)

bq = get_bigquery_connector()

# Sidebar filters
st.sidebar.title("Dashboard Filters")

# Get available weeks for selection
@st.cache_data(ttl=3600)
def load_available_weeks():
    return bq.get_latest_weeks(num_weeks=12)

weeks_df = load_available_weeks()
weeks_options = [f"Week {row['WeekNumber']}, {row['Year']}" for _, row in weeks_df.iterrows()]

selected_week_index = st.sidebar.selectbox(
    "Select Week", 
    range(len(weeks_options)),
    format_func=lambda x: weeks_options[x]
)

current_week = weeks_df.iloc[selected_week_index]
current_week_date = current_week['LastDataAsOfDate']

# Get previous week for comparison
prev_week_index = min(selected_week_index + 1, len(weeks_df) - 1)
prev_week = weeks_df.iloc[prev_week_index]
prev_week_date = prev_week['LastDataAsOfDate']

# Measure type filter
@st.cache_data(ttl=3600)
def load_measure_types():
    return bq.get_distinct_values("MedAdherenceMeasureCode")

measure_types = load_measure_types()
selected_measures = st.sidebar.multiselect(
    "Measure Type", 
    measure_types,
    default=measure_types
)

# Market code filter
@st.cache_data(ttl=3600)
def load_market_codes():
    return bq.get_distinct_values("MarketCode")

market_codes = load_market_codes()
selected_markets = st.sidebar.multiselect(
    "Market Code", 
    market_codes,
    default=[]
)

# Payer code filter
@st.cache_data(ttl=3600)
def load_payer_codes():
    return bq.get_distinct_values("PayerCode")

payer_codes = load_payer_codes()
selected_payers = st.sidebar.multiselect(
    "Payer Code", 
    payer_codes,
    default=[]
)

# Load data based on filters
@st.cache_data(ttl=3600)
def load_data(date, measure_codes=None, market_codes=None, payer_codes=None):
    # Set date range to cover the entire week
    start_date = date - timedelta(days=7)
    end_date = date
    
    return bq.get_med_adherence_data(
        start_date=start_date,
        end_date=end_date,
        measure_codes=measure_codes if measure_codes else None,
        market_codes=market_codes if market_codes else None,
        payer_codes=payer_codes if payer_codes else None
    )

# Load current and previous week data
current_data = load_data(
    current_week_date, 
    selected_measures if selected_measures else None,
    selected_markets if selected_markets else None,
    selected_payers if selected_payers else None
)

prev_data = load_data(
    prev_week_date,
    selected_measures if selected_measures else None,
    selected_markets if selected_markets else None,
    selected_payers if selected_payers else None
)

# Check if data is available
if current_data.empty:
    st.error("No data available for the selected filters. Please adjust your selection.")
    st.stop()

# Process data
current_metrics = calculate_metrics(current_data)
prev_metrics = calculate_metrics(prev_data)
average_pdc = calculate_average_pdc(current_data)
weekly_trend = get_week_over_week_data(pd.concat([current_data, prev_data]))
market_summary, payer_summary = create_market_payer_summary(current_data)

# Dashboard Header
st.title("Medication Adherence Gap Analysis")
st.markdown(f"**Current Week: {current_week_date.strftime('%Y-%m-%d')}**")

# KPI Cards Row
col1, col2, col3, col4 = st.columns(4)

# UGIDs (Gaps) KPI
with col1:
    st.metric(
        "Total Gaps (UGIDs)",
        f"{current_metrics['total_ugids']:,}",
        f"{((current_metrics['total_ugids'] - prev_metrics['total_ugids']) / prev_metrics['total_ugids'] * 100):.1f}%" if prev_metrics['total_ugids'] > 0 else "N/A"
    )

# UPIDs (Patients) KPI
with col2:
    st.metric(
        "Unique Patients (UPIDs)",
        f"{current_metrics['total_upids']:,}",
        f"{((current_metrics['total_upids'] - prev_metrics['total_upids']) / prev_metrics['total_upids'] * 100):.1f}%" if prev_metrics['total_upids'] > 0 else "N/A"
    )

# Fill Status KPI
with col3:
    st.metric(
        "One-Fill Gaps",
        f"{current_metrics['one_fill_count']:,}",
        f"{((current_metrics['one_fill_count'] - prev_metrics['one_fill_count']) / prev_metrics['one_fill_count'] * 100):.1f}%" if prev_metrics['one_fill_count'] > 0 else "N/A"
    )

# Denominator Gaps KPI
with col4:
    st.metric(
        "Denominator Gaps",
        f"{current_metrics['denominator_gap_count']:,}",
        f"{((current_metrics['denominator_gap_count'] - prev_metrics['denominator_gap_count']) / prev_metrics['denominator_gap_count'] * 100):.1f}%" if prev_metrics['denominator_gap_count'] > 0 else "N/A"
    )

# Weekly Trends
st.header("Week over Week Trends")
col1, col2 = st.columns(2)

with col1:
    # UGIDs Trend
    fig = px.line(
        weekly_trend.head(8),
        x="file_load_date",
        y="ugid_count",
        title="Weekly Gaps (UGIDs)",
        markers=True
    )
    fig.update_layout(xaxis_title="Week", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # UPIDs Trend
    fig = px.line(
        weekly_trend.head(8),
        x="file_load_date",
        y="upid_count",
        title="Weekly Unique Patients (UPIDs)",
        markers=True
    )
    fig.update_layout(xaxis_title="Week", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

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

# Fill Status Analysis
st.header("Fill Status Analysis")

# One-Fill vs Denominator by Measure Type
measure_fill_data = pd.DataFrame({
    'Measure': ['MAC', 'MAH', 'MAD'],
    'One Fill': [
        len(current_data[(current_data['MedAdherenceMeasureCode'] == 'MAC') & (current_data['OneFillCode'] == 'Yes')]),
        len(current_data[(current_data['MedAdherenceMeasureCode'] == 'MAH') & (current_data['OneFillCode'] == 'Yes')]),
        len(current_data[(current_data['MedAdherenceMeasureCode'] == 'MAD') & (current_data['OneFillCode'] == 'Yes')])
    ],
    'Denominator Gap': [
        len(current_data[(current_data['MedAdherenceMeasureCode'] == 'MAC') & (current_data['OneFillCode'].isnull())]),
        len(current_data[(current_data['MedAdherenceMeasureCode'] == 'MAH') & (current_data['OneFillCode'].isnull())]),
        len(current_data[(current_data['MedAdherenceMeasureCode'] == 'MAD') & (current_data['OneFillCode'].isnull())])
    ]
})

# Reshape for stacked bar chart
fill_status_long = pd.melt(
    measure_fill_data,
    id_vars=['Measure'],
    value_vars=['One Fill', 'Denominator Gap'],
    var_name='Fill Status',
    value_name='Count'
)

fig = px.bar(
    fill_status_long,
    x='Measure',
    y='Count',
    color='Fill Status',
    title="Fill Status by Measure Type",
    barmode='stack',
    text_auto=True
)
st.plotly_chart(fig, use_container_width=True)

# Geographic and Payer Analysis
st.header("Geographic and Payer Analysis")
col1, col2 = st.columns(2)

with col1:
    # Top Markets Bar Chart
    fig = px.bar(
        market_summary.head(10),
        x='gap_count',
        y='MarketCode',
        title="Top 10 Markets by Gap Count",
        orientation='h',
        text_auto=True
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Top Payers Bar Chart
    fig = px.bar(
        payer_summary.head(10),
        x='gap_count',
        y='PayerCode',
        title="Top 10 Payers by Gap Count",
        orientation='h',
        text_auto=True
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

# PDC Distribution
st.header("PDC Distribution Analysis")

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