import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.oauth2 import service_account
from google.cloud import bigquery
import datetime
import json

# Set page configuration
st.set_page_config(
    page_title="Medicare Advantage Medication Adherence Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load BigQuery credentials from JSON file
@st.cache_resource
def get_bigquery_client():
   # Specify the path to your downloaded JSON credentials file
   # You can use an absolute path or a relative path to your script
   credentials_path = "medadhdata2025-ce0f2b2ff824.json"
   # Load credentials from the JSON file
   credentials = service_account.Credentials.from_service_account_file(
       credentials_path
   )
   # Create BigQuery client
   client = bigquery.Client(credentials=credentials, project=credentials.project_id)
   return client
# Function to query data from BigQuery
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data(query):
    client = get_bigquery_client()
    query_job = client.query(query)
    return query_job.to_dataframe()

# Get the current timestamp
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Load the data
@st.cache_data(ttl=3600)
def load_medication_adherence_data():
    query = """
    SELECT 
        MarketCode, 
        PayerCode, 
        MedAdherenceMeasureCode, 
        NDCDesc, 
    FROM `medadhdata2025.adherence_tracking.weekly_med_adherence_data`
    """
    return load_data(query)

# Load the star rating thresholds data
@st.cache_data
def load_star_thresholds():
    # This could come from BigQuery or be hardcoded based on CMS data
    # Using the values from your documentation for 2024
    thresholds = {
        "MAD": {  # Medication adherence for diabetes medication
            "2-Stars": 82,
            "3-Stars": 86,
            "4-Stars": 90,
            "5-Stars": 92
        },
        "MAC": {  # Medication adherence for cholesterol (statins)
            "2-Stars": 84,
            "3-Stars": 89,
            "4-Stars": 91,
            "5-Stars": 93
        },
        "MAH": {  # Medication adherence for hypertension (RAS antagonists)
            "2-Stars": 84,
            "3-Stars": 88,
            "4-Stars": 91,
            "5-Stars": 93
        }
    }
    return thresholds

# Load data
try:
    df = load_medication_adherence_data()
    thresholds = load_star_thresholds()
    
    # Calculate last data update time from the data
    if 'Last Activity Date' in df.columns and not df['Last Activity Date'].empty:
        last_update = df['Last Activity Date'].max()
        last_update_text = f"Data Last Updated: {last_update}"
    else:
        last_update_text = f"Dashboard Last Refreshed: {current_time}"
    
    # Process data for dashboard
    # For demonstration, generating random adherence rates 
    # In reality, you would calculate these from your actual data
    adherence_data = {
        "MAD": np.random.uniform(85, 95),
        "MAC": np.random.uniform(85, 95),
        "MAH": np.random.uniform(85, 95)
    }
    
    # Create unique lists for filter options
    market_codes = sorted(df['MarketCode'].unique().tolist())
    payer_codes = sorted(df['PayerCode'].unique().tolist())
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    adherence_data = {"MAD": 0, "MAC": 0, "MAH": 0}
    market_codes = []
    payer_codes = []
    last_update_text = f"Dashboard Last Refreshed: {current_time} (Error loading data)"

# Dashboard title and structure
st.title("Medicare Advantage Medication Adherence Dashboard")
st.write(last_update_text)

# Create tabs
tab1, tab2 = st.tabs(["Overall Performance", "Adherence Trends"])

# Create sidebar with filters
st.sidebar.title("Filters")

# Add filter for MarketCode
selected_markets = st.sidebar.multiselect(
    "Select Markets",
    options=market_codes,
    default=market_codes[:5] if len(market_codes) > 5 else market_codes
)

# Add filter for PayerCode
selected_payers = st.sidebar.multiselect(
    "Select Payers",
    options=payer_codes,
    default=payer_codes[:3] if len(payer_codes) > 3 else payer_codes
)

# Date range filter
date_range = st.sidebar.selectbox(
    "Select Time Period",
    ["Year to Date", "Last Quarter", "Last Month", "Last Week"]
)
def get_star_rating(value, measure, thresholds):
    """Determine star rating based on value and thresholds"""
    if value >= thresholds[measure]["5-Stars"]:
        return "5-Stars"
    elif value >= thresholds[measure]["4-Stars"]:
        return "4-Stars"
    elif value >= thresholds[measure]["3-Stars"]:
        return "3-Stars"
    elif value >= thresholds[measure]["2-Stars"]:
        return "2-Stars"
    else:
        return "1-Star"

def get_color_for_rating(value, measure):
    """Get color based on star rating"""
    rating = get_star_rating(value, measure)
    if rating == "5-Stars" or rating == "4-Stars":
        return "green"
    elif rating == "3-Stars":
        return "yellow"
    else:
        return "red"

def create_gauge_chart(value, measure, thresholds):
    """Create a gauge chart for displaying measure performance against thresholds"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={"text": f"{measure} Performance"},
        delta={"reference": thresholds[measure]["4-Stars"], "increasing": {"color": "green"}},
        gauge={
            "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "darkblue"},
            "bar": {"color": "darkblue"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, thresholds[measure]["2-Stars"]], "color": "red"},
                {"range": [thresholds[measure]["2-Stars"], thresholds[measure]["3-Stars"]], "color": "orange"},
                {"range": [thresholds[measure]["3-Stars"], thresholds[measure]["4-Stars"]], "color": "yellow"},
                {"range": [thresholds[measure]["4-Stars"], thresholds[measure]["5-Stars"]], "color": "lightgreen"},
                {"range": [thresholds[measure]["5-Stars"], 100], "color": "green"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": thresholds[measure]["4-Stars"]}
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig

# Function to create performance by market heatmap
def create_market_performance_heatmap(df, selected_markets, selected_payers):
    """Create a heatmap showing performance by market"""
    # In a real scenario, you would calculate these from your real data
    # For demonstration, generating random data
    markets = selected_markets if selected_markets else market_codes[:5]
    
    data = []
    for market in markets:
        row = {
            "Market": market,
            "MAD": np.random.uniform(82, 95),
            "MAC": np.random.uniform(82, 95),
            "MAH": np.random.uniform(82, 95)
        }
        data.append(row)
    
    heatmap_df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    for measure in ["MAD", "MAC", "MAH"]:
        # Create a heatmap-like visualization
        colors = [get_color_for_rating(val, measure) for val in heatmap_df[measure]]
        
        fig.add_trace(go.Bar(
            y=heatmap_df["Market"],
            x=heatmap_df[measure],
            orientation='h',
            name=measure,
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0)', width=1)
            ),
            text=heatmap_df[measure].round(1),
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Performance by Market",
        barmode='group',
        height=400,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

with tab1:
    st.header("Overall Performance")
    
    # Top KPI cards row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("MAD Performance")
        st.metric(
            label=f"Diabetes Medication Adherence ({get_star_rating(adherence_data['MAD'], 'MAD', thresholds)})",
            value=f"{adherence_data['MAD']:.1f}%", 
            delta=f"{np.random.uniform(-2, 5):.1f}%"
        )
        st.plotly_chart(create_gauge_chart(adherence_data['MAD'], 'MAD', thresholds), use_container_width=True)
        
    with col2:
        st.subheader("MAC Performance")
        st.metric(
            label=f"Cholesterol Medication Adherence ({get_star_rating(adherence_data['MAC'], 'MAC', thresholds)})",
            value=f"{adherence_data['MAC']:.1f}%", 
            delta=f"{np.random.uniform(-2, 5):.1f}%"
        )
        st.plotly_chart(create_gauge_chart(adherence_data['MAC'], 'MAC', thresholds), use_container_width=True)
        
    with col3:
        st.subheader("MAH Performance")
        st.metric(
            label=f"Hypertension Medication Adherence ({get_star_rating(adherence_data['MAH'], 'MAH', thresholds)})",
            value=f"{adherence_data['MAH']:.1f}%", 
            delta=f"{np.random.uniform(-2, 5):.1f}%"
        )
        st.plotly_chart(create_gauge_chart(adherence_data['MAH'], 'MAH', thresholds), use_container_width=True)
    
    # Performance by market
    st.plotly_chart(create_market_performance_heatmap(df, selected_markets, selected_payers), use_container_width=True)
    
    # Gap closure summary
    st.subheader("Gap Closure Summary")
    
    gap_col1, gap_col2, gap_col3 = st.columns(3)
    
    # Calculate gap metrics (in a real scenario, you would calculate from your data)
    # For demonstration, using random data
    for i, (col, measure) in enumerate(zip([gap_col1, gap_col2, gap_col3], ["MAD", "MAC", "MAH"])):
        with col:
            total_members = np.random.randint(5000, 10000)
            compliant_members = int(total_members * adherence_data[measure] / 100)
            non_compliant_members = total_members - compliant_members
            
            # Calculate gap to next star threshold
            current_rating = get_star_rating(adherence_data[measure], measure)
            next_rating_map = {
                "1-Star": "2-Stars",
                "2-Stars": "3-Stars",
                "3-Stars": "4-Stars",
                "4-Stars": "5-Stars",
                "5-Stars": "5-Stars"  # Already at max
            }
            next_rating = next_rating_map[current_rating]
            
            if next_rating != current_rating:
                next_threshold = thresholds[measure][next_rating]
                members_needed = int((next_threshold - adherence_data[measure]) / 100 * total_members) + 1
            else:
                members_needed = 0
            
            st.write(f"### {measure}")
            st.write(f"**Total Members:** {total_members}")
            st.write(f"**Compliant Members:** {compliant_members} ({adherence_data[measure]:.1f}%)")
            st.write(f"**Non-Compliant Members:** {non_compliant_members}")
            
            if members_needed > 0:
                st.write(f"**Gap to {next_rating}:** {members_needed} members")
            else:
                st.write("**Congratulations!** Already achieved 5-Star rating.")
def generate_monthly_trend_data(measures, months=12):
    """Generate mock monthly trend data for demonstration"""
    base_date = datetime.datetime.now() - datetime.timedelta(days=30*months)
    dates = [base_date + datetime.timedelta(days=30*i) for i in range(months+1)]
    
    data = []
    for measure in measures:
        # Start with base value and add some random walk
        base_value = np.random.uniform(80, 85)
        values = [base_value]
        
        for i in range(1, months+1):
            # Add some randomness with slight upward trend
            next_value = values[-1] + np.random.uniform(-1, 2)
            # Ensure value stays within reasonable bounds
            next_value = max(min(next_value, 98), 75)
            values.append(next_value)
        
        for i, date in enumerate(dates):
            data.append({
                "Date": date,
                "Measure": measure,
                "Adherence": values[i]
            })
    
    return pd.DataFrame(data)

def create_trend_chart(trend_df, thresholds):
    """Create a line chart showing trends over time"""
    fig = go.Figure()
    
    for measure in ["MAD", "MAC", "MAH"]:
        measure_data = trend_df[trend_df["Measure"] == measure]
        
        fig.add_trace(go.Scatter(
            x=measure_data["Date"],
            y=measure_data["Adherence"],
            mode='lines+markers',
            name=measure,
            line=dict(width=3),
            marker=dict(size=8),
        ))
    
    # Add threshold reference lines
    for measure in ["MAD", "MAC", "MAH"]:
        for star, value in thresholds[measure].items():
            if star in ["4-Stars", "5-Stars"]:  # Only show 4 and 5 star thresholds to avoid clutter
                fig.add_trace(go.Scatter(
                    x=[trend_df["Date"].min(), trend_df["Date"].max()],
                    y=[value, value],
                    mode='lines',
                    line=dict(dash='dash', width=1),
                    name=f"{measure} {star} ({value}%)",
                    opacity=0.7,
                ))
    
    fig.update_layout(
        title="Medication Adherence Trends Over Time",
        xaxis_title="Month",
        yaxis_title="Adherence Rate (%)",
        yaxis=dict(range=[75, 100]),
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


with tab2:
    st.header("Adherence Trends")
    
    # Generate mock trend data for demonstration
    trend_df = generate_monthly_trend_data(["MAD", "MAC", "MAH"])
    
    # Time series trends
    st.plotly_chart(create_trend_chart(trend_df, thresholds), use_container_width=True)
    
    
    
    # Adherence by demographics section
    st.subheader("Adherence by Demographics")
    
    # Create mock data for demographics
    def create_demographic_chart(df, selected_markets, selected_payers):
        """Create chart showing adherence by demographics"""
        # For demonstration using random data
        # In a real application, you would calculate this from your actual data
        
        if selected_markets:
            markets = selected_markets
        else:
            markets = market_codes[:5] if market_codes else ["Market A", "Market B", "Market C", "Market D", "Market E"]
        
        data = []
        for market in markets:
            for measure in ["MAD", "MAC", "MAH"]:
                data.append({
                    "Market": market,
                    "Measure": measure,
                    "Adherence": np.random.uniform(80, 95)
                })
        
        demo_df = pd.DataFrame(data)
        
        fig = px.bar(
            demo_df,
            x="Market",
            y="Adherence",
            color="Measure",
            barmode="group",
            title="Medication Adherence by Market",
            color_discrete_sequence=px.colors.qualitative.G10,
        )
        
        fig.update_layout(
            xaxis_title="Market",
            yaxis_title="Adherence Rate (%)",
            yaxis=dict(range=[75, 100]),
            height=400,
        )
        
        return fig
    
    st.plotly_chart(create_demographic_chart(df, selected_markets, selected_payers), use_container_width=True)
def predict_eoy_performance(trend_df):
    """Create a simple prediction of end-of-year performance"""
    # Get the latest data point for each measure
    latest_data = trend_df.sort_values('Date').groupby('Measure').last().reset_index()
    
    # Create a simple projection (in reality, you would use more sophisticated methods)
    projections = []
    for _, row in latest_data.iterrows():
        measure = row['Measure']
        current_value = row['Adherence']
        
        # Simple projection - assume slight improvement
        projected_value = min(current_value + np.random.uniform(0.5, 2.0), 98)
        
        current_rating = get_star_rating(current_value, measure)
        projected_rating = get_star_rating(projected_value, measure)
        
        projections.append({
            'Measure': measure,
            'Current Value': current_value,
            'Current Rating': current_rating,
            'Projected EOY Value': projected_value,
            'Projected Rating': projected_rating,
            'Will Improve': projected_rating > current_rating
        })
    
    return pd.DataFrame(projections)

# Add to sidebar - Advanced Options
st.sidebar.markdown("---")
st.sidebar.subheader("Advanced Options")

show_projections = st.sidebar.checkbox("Show EOY Projections", value=True)
show_thresholds = st.sidebar.checkbox("Show Star Rating Thresholds", value=True)

# If projections are enabled, add them to the dashboard
if show_projections:
    # Add to Overall Performance tab
    with tab1:
        st.markdown("---")
        st.subheader("End of Year Projections")
        
        # Calculate projections
        projections_df = predict_eoy_performance(trend_df)
        
        # Display projections in a formatted table
        for _, row in projections_df.iterrows():
            measure = row['Measure']
            current_value = row['Current Value']
            projected_value = row['Projected EOY Value']
            current_rating = row['Current Rating']
            projected_rating = row['Projected Rating']
            
            proj_col1, proj_col2 = st.columns([1, 4])
            
            with proj_col1:
                st.subheader(f"{measure}")
            
            with proj_col2:
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    st.metric(
                        label=f"Current ({current_rating})",
                        value=f"{current_value:.1f}%"
                    )
                
                with metric_col2:
                    st.metric(
                        label=f"Projected EOY ({projected_rating})",
                        value=f"{projected_value:.1f}%",
                        delta=f"{projected_value - current_value:.1f}%"
                    )

# If thresholds are enabled, add a threshold reference table
if show_thresholds:
    with st.sidebar:
        st.markdown("---")
        st.subheader("Star Rating Thresholds")
        
        # Convert thresholds to a DataFrame for display
        threshold_data = []
        for measure, ratings in thresholds.items():
            for rating, value in ratings.items():
                threshold_data.append({
                    "Measure": measure,
                    "Rating": rating,
                    "Threshold": value
                })
        
        threshold_df = pd.DataFrame(threshold_data)
        
        # Display as a styled table
        st.dataframe(
            threshold_df.pivot(index="Rating", columns="Measure", values="Threshold"),
            use_container_width=True
        )

# Add a data refresh button
st.sidebar.markdown("---")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.experimental_rerun()

# Add information about the dashboard
with st.sidebar:
    st.markdown("---")
    st.markdown("### About this Dashboard")
    st.markdown("""
    This dashboard tracks medication adherence performance for Medicare Advantage members across three key measures:
    
    - **MAD**: Medication Adherence for Diabetes
    - **MAC**: Medication Adherence for Cholesterol
    - **MAH**: Medication Adherence for Hypertension
    
    All measures are triple-weighted in CMS Star Ratings.
    
    Data is sourced from Google BigQuery and updated daily.
    """)
    
    st.markdown("---")
    

# Add a footer
st.markdown("---")
st.markdown("*Note: All metrics are based on Proportion of Days Covered (PDC) rates. Members are considered adherent with PDC ≥ 80%.*")

# Add a download button for the data
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

with st.sidebar:
    st.markdown("---")
    st.subheader("Export Data")
    
    csv = convert_df_to_csv(df)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name='medication_adherence_data.csv',
        mime='text/csv',
    )