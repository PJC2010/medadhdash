import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Import BigQuery libraries
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas_gbq

# Set page configuration
st.set_page_config(
    page_title="Medicare Medication Adherence Dashboard",
    page_icon="💊",
    layout="wide"
)

# Create a connection to BigQuery
def create_bigquery_connection():
    # There are two options for authentication:
    
    # OPTION 1: Service Account (recommended for production)
    # Load service account credentials from secrets
    # (For Streamlit Cloud, use st.secrets)
    # For local development, you can use a JSON key file
    
    try:
        # Try to get credentials from Streamlit secrets (for production)
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
    except:
        # For local development, specify the path to your service account key file
        credentials_path = "medadhdata2025-6e124598af28.json"  # Replace with your file path
        try:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
        except:
            # OPTION 2: User account authentication (will open browser window)
            # If no service account is available, use user authentication
            credentials = None
            st.warning("Using user authentication. You may need to authenticate in a browser window.")
    
    # Create a BigQuery client
    if credentials:
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    else:
        # This will use user authentication
        client = bigquery.Client()
    
    return client

# Function to load data from BigQuery
def load_data_from_bigquery():
    client = create_bigquery_connection()
    
    # Dictionary to store all our data
    data = {}
    
    # Get Market Codes
    market_codes_query = """
    SELECT DISTINCT MarketCode 
    FROM `your-project.your_dataset.market_dimension`
    ORDER BY MarketCode
    """
    market_codes_df = client.query(market_codes_query).to_dataframe()
    data["MarketCodes"] = market_codes_df['MarketCode'].tolist()
    
    # Get Payer Codes
    payer_codes_query = """
    SELECT DISTINCT payer_code 
    FROM `your-project.your_dataset.payer_dimension`
    ORDER BY payer_code
    """
    payer_codes_df = client.query(payer_codes_query).to_dataframe()
    data["payer_codes"] = payer_codes_df['payer_code'].tolist()
    
    # Get Medication Adherence Data - Triple-Weighted Measures
    # Based on the CMS Star Ratings document
    adherence_query = """
    SELECT 
        CASE 
            WHEN metric_id = 'D08' THEN 'Diabetes Medications (MAD)'
            WHEN metric_id = 'D09' THEN 'Hypertension (RAS antagonists) (MAH)'
            WHEN metric_id = 'D10' THEN 'Cholesterol (Statins) (MAC)'
        END AS metric,
        current_rate,
        four_star_threshold AS "4_star_threshold",
        five_star_threshold AS "5_star_threshold",
        previous_year_rate AS last_year_rate,
        3 AS weight  -- Triple-weighted
    FROM `your-project.your_dataset.star_ratings_metrics`
    WHERE metric_id IN ('D08', 'D09', 'D10')
    AND measurement_year = 2024  -- Update to your current measurement year
    """
    data["adherence_df"] = client.query(adherence_query).to_dataframe()
    
    # Calculate gap to next threshold
    for i in range(len(data["adherence_df"])):
        if data["adherence_df"].loc[i, "current_rate"] < data["adherence_df"].loc[i, "4_star_threshold"]:
            data["adherence_df"].loc[i, "gap_to_next"] = data["adherence_df"].loc[i, "4_star_threshold"] - data["adherence_df"].loc[i, "current_rate"]
        else:
            data["adherence_df"].loc[i, "gap_to_next"] = data["adherence_df"].loc[i, "5_star_threshold"] - data["adherence_df"].loc[i, "current_rate"]
    
    # Get Monthly Trend Data
    trend_query = """
    SELECT 
        report_date,
        metric_id,
        CASE 
            WHEN metric_id = 'D08' THEN 'Diabetes Medications (MAD)'
            WHEN metric_id = 'D09' THEN 'Hypertension (RAS antagonists) (MAH)'
            WHEN metric_id = 'D10' THEN 'Cholesterol (Statins) (MAC)'
        END AS metric,
        pdc_rate AS rate
    FROM `your-project.your_dataset.medication_adherence_monthly`
    WHERE metric_id IN ('D08', 'D09', 'D10')
    AND report_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
    ORDER BY metric_id, report_date
    """
    trend_df = client.query(trend_query).to_dataframe()
    
    # Transform trend data into the format needed for our dashboard
    trend_df['month'] = trend_df['report_date'].dt.strftime('%b %Y')
    months = sorted(trend_df['month'].unique().tolist())
    
    rates = {}
    for metric in data["adherence_df"]["metric"]:
        metric_data = trend_df[trend_df["metric"] == metric]
        rates[metric] = metric_data["rate"].tolist()
    
    data["trend_data"] = {"months": months, "rates": rates}
    
    # Get At-Risk Member Data
    risk_query = """
    SELECT 
        CASE 
            WHEN category = 'MAD' THEN 'Diabetes Medications'
            WHEN category = 'MAH' THEN 'Hypertension (RAS antagonists)'
            WHEN category = 'MAC' THEN 'Cholesterol (Statins)'
        END AS category,
        SUM(CASE WHEN pdc BETWEEN 80 AND 85 THEN 1 ELSE 0 END) AS at_risk_members,
        SUM(CASE WHEN pdc < 80 THEN 1 ELSE 0 END) AS non_adherent_members,
        SUM(CASE WHEN adr BETWEEN 0 AND 10 THEN 1 ELSE 0 END) AS critical_adr
    FROM `your-project.your_dataset.member_adherence_detail`
    WHERE report_date = (SELECT MAX(report_date) FROM `your-project.your_dataset.member_adherence_detail`)
    GROUP BY category
    """
    data["risk_df"] = client.query(risk_query).to_dataframe()
    
    # Get ADR Distribution
    adr_query = """
    SELECT 
        CASE 
            WHEN adr < 0 THEN '<0 days (failed)'
            WHEN adr BETWEEN 0 AND 10 THEN '1-10 days'
            WHEN adr BETWEEN 11 AND 20 THEN '11-20 days'
            WHEN adr BETWEEN 21 AND 30 THEN '21-30 days'
            WHEN adr BETWEEN 31 AND 60 THEN '31-60 days'
            WHEN adr > 60 THEN '>60 days'
        END AS adr_range,
        COUNT(*) AS count
    FROM `your-project.your_dataset.member_adherence_detail`
    WHERE report_date = (SELECT MAX(report_date) FROM `your-project.your_dataset.member_adherence_detail`)
    GROUP BY adr_range
    ORDER BY CASE 
        WHEN adr_range = '<0 days (failed)' THEN 1
        WHEN adr_range = '1-10 days' THEN 2
        WHEN adr_range = '11-20 days' THEN 3
        WHEN adr_range = '21-30 days' THEN 4
        WHEN adr_range = '31-60 days' THEN 5
        WHEN adr_range = '>60 days' THEN 6
    END
    """
    data["adr_df"] = client.query(adr_query).to_dataframe()
    
    # Get Member-Level Adherence Data
    member_query = """
    SELECT 
        md.member_id,
        md.member_name AS name,
        md.age,
        md.MarketCode,
        md.payer_code,
        MAX(CASE WHEN ma.category = 'MAD' THEN ma.pdc END) AS diabetes_pdc,
        MAX(CASE WHEN ma.category = 'MAH' THEN ma.pdc END) AS hypertension_pdc,
        MAX(CASE WHEN ma.category = 'MAC' THEN ma.pdc END) AS statin_pdc,
        MAX(CASE WHEN ma.category = 'MAD' THEN ma.adr END) AS diabetes_adr,
        MAX(CASE WHEN ma.category = 'MAH' THEN ma.adr END) AS hypertension_adr,
        MAX(CASE WHEN ma.category = 'MAC' THEN ma.adr END) AS statin_adr,
        md.last_fill_date
    FROM `your-project.your_dataset.member_dimension` md
    LEFT JOIN `your-project.your_dataset.member_adherence_detail` ma 
        ON md.member_id = ma.member_id 
        AND ma.report_date = (SELECT MAX(report_date) FROM `your-project.your_dataset.member_adherence_detail`)
    WHERE md.is_active = TRUE
    GROUP BY md.member_id, md.member_name, md.age, md.MarketCode, md.payer_code, md.last_fill_date
    """
    data["member_df"] = client.query(member_query).to_dataframe()
    
    # Get Market/Payer Level Metrics
    market_payer_query = """
    SELECT 
        MarketCode,
        payer_code,
        CASE 
            WHEN metric_id = 'D08' THEN 'Diabetes Medications (MAD)'
            WHEN metric_id = 'D09' THEN 'Hypertension (RAS antagonists) (MAH)'
            WHEN metric_id = 'D10' THEN 'Cholesterol (Statins) (MAC)'
        END AS metric,
        pdc_rate AS rate,
        member_count
    FROM `your-project.your_dataset.market_payer_metrics`
    WHERE report_date = (SELECT MAX(report_date) FROM `your-project.your_dataset.market_payer_metrics`)
    AND metric_id IN ('D08', 'D09', 'D10')
    """
    data["market_payer_df"] = client.query(market_payer_query).to_dataframe()
    
    # Get SUPD Data
    supd_query = """
    SELECT 
        current_rate,
        four_star_threshold AS "4_star_threshold",
        five_star_threshold AS "5_star_threshold",
        previous_year_rate AS last_year_rate
    FROM `your-project.your_dataset.star_ratings_metrics`
    WHERE metric_id = 'D12'  -- SUPD metric ID from CMS document
    AND measurement_year = 2024  -- Update to your current measurement year
    """
    supd_df = client.query(supd_query).to_dataframe()
    data["supd_data"] = {
        "metric": "Statin Use in Persons with Diabetes (SUPD)",
        "current_rate": supd_df.iloc[0]["current_rate"],
        "4_star_threshold": supd_df.iloc[0]["4_star_threshold"],
        "5_star_threshold": supd_df.iloc[0]["5_star_threshold"],
        "last_year_rate": supd_df.iloc[0]["last_year_rate"],
    }
    
    # Get MTM Data
    mtm_query = """
    SELECT 
        current_rate,
        previous_year_rate AS last_year_rate,
        target_rate AS target
    FROM `your-project.your_dataset.star_ratings_metrics`
    WHERE metric_id = 'D11'  -- MTM CMR Completion Rate from CMS document
    AND measurement_year = 2024  -- Update to your current measurement year
    """
    mtm_df = client.query(mtm_query).to_dataframe()
    data["mtm_data"] = {
        "metric": "MTM Program Completion Rate for CMR",
        "current_rate": mtm_df.iloc[0]["current_rate"],
        "last_year_rate": mtm_df.iloc[0]["last_year_rate"],
        "target": mtm_df.iloc[0]["target"]
    }
    
    return data

# Load data - either from BigQuery or use sample data for development
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_data():
    try:
        # Try to load from BigQuery
        return load_data_from_bigquery()
    except Exception as e:
        st.error(f"Error loading data from BigQuery: {e}")
        # Fall back to sample data for development
        return generate_sample_data()

# Sample data generation for development (keep your existing function)
def generate_sample_data():
    # Your existing sample data generation code
    # ...
    # This is just a stub - keep your full implementation
    
    # Current date for reference
    today = datetime.now()
    
    # Define sample market codes and payer codes
    market_codes = ["East", "West", "Central", "South", "Northeast"]
    payer_codes = ["Medicare", "MMP", "DSNP", "CSNP", "ISNP"]
    
    # Sample data for the three main medication adherence metrics
    adherence_data = {
        "metric": ["Diabetes Medications (MAD)", "Hypertension (RAS antagonists) (MAH)", "Cholesterol (Statins) (MAC)"],
        "current_rate": [88.5, 89.2, 90.1],
        "4_star_threshold": [90, 91, 91],
        "5_star_threshold": [92, 93, 93],
        "last_year_rate": [86.2, 88.0, 89.5],
        "weight": [3, 3, 3],
    }
    
    # Continue with the rest of your sample data generation...
    
    return {
        "adherence_df": pd.DataFrame(adherence_data),
        # Include all your other sample data structure
    }

# Load the data
data = get_data()

# Now continue with your existing dashboard code, but using the data loaded from BigQuery
# ...

# Dashboard title
st.title("Medicare Medication Adherence Dashboard")

# Add filters in the sidebar
st.sidebar.header("Filters")

# Market code filter
selected_markets = st.sidebar.multiselect(
    "Market Code",
    options=data["MarketCode"],
    default=data["MarketCode"]
)

# Payer code filter
selected_payers = st.sidebar.multiselect(
    "Payer Code",
    options=data["payer_codes"],
    default=data["payer_codes"]
)

# Filter the data based on selections
filtered_members = data["member_df"]
if selected_markets:
    filtered_members = filtered_members[filtered_members["MarketCode"].isin(selected_markets)]
if selected_payers:
    filtered_members = filtered_members[filtered_members["payer_code"].isin(selected_payers)]

# Filter market/payer data
filtered_market_payer = data["market_payer_df"]
if selected_markets:
    filtered_market_payer = filtered_market_payer[filtered_market_payer["MarketCode"].isin(selected_markets)]
if selected_payers:
    filtered_market_payer = filtered_market_payer[filtered_market_payer["payer_code"].isin(selected_payers)]

# Continue with the rest of your dashboard implementation
# ...

# The remaining code is the same as your original dashboard implementation