import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Set page configuration
st.set_page_config(
    page_title="Medicare Medication Adherence Dashboard",
    page_icon="💊",
    layout="wide"
)

# Create sample data (in a real implementation, this would be loaded from a database)
def generate_sample_data():
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
    
    # Calculate gap to next threshold
    for i in range(len(adherence_data["metric"])):
        if adherence_data["current_rate"][i] < adherence_data["4_star_threshold"][i]:
            adherence_data[f"gap_to_next"] = [adherence_data["4_star_threshold"][i] - adherence_data["current_rate"][i] for i in range(len(adherence_data["metric"]))]
        else:
            adherence_data[f"gap_to_next"] = [adherence_data["5_star_threshold"][i] - adherence_data["current_rate"][i] for i in range(len(adherence_data["metric"]))]
    
    # Create monthly trend data for each metric (past 12 months)
    months = [(today - timedelta(days=30*i)).strftime('%b %Y') for i in range(12, 0, -1)]
    
    trend_data = {}
    for i, metric in enumerate(adherence_data["metric"]):
        # Create a slightly increasing trend with some random variation
        base_rate = adherence_data["last_year_rate"][i]
        target_rate = adherence_data["current_rate"][i]
        
        monthly_rates = []
        for j in range(12):
            progress = j / 11  # 0 to 1 over the year
            expected_rate = base_rate + progress * (target_rate - base_rate)
            rate = expected_rate + random.uniform(-0.5, 0.5)  # Add some noise
            monthly_rates.append(round(rate, 1))
        
        trend_data[metric] = monthly_rates
    
    # Risk member counts
    risk_data = {
        "category": ["Diabetes Medications", "Hypertension (RAS antagonists)", "Cholesterol (Statins)"],
        "at_risk_members": [156, 203, 175],  # PDC between 80-85%
        "non_adherent_members": [342, 418, 387],  # PDC < 80%
        "critical_adr": [78, 93, 85]  # Members with ADR < 10 days
    }
    
    # Generate ADR distribution
    adr_distribution = {
        "adr_range": ["<0 days (failed)", "1-10 days", "11-20 days", "21-30 days", "31-60 days", ">60 days"],
        "count": [215, 182, 275, 312, 567, 824]
    }
    
    # Generate member-level sample data (would be much larger in reality)
    member_data = []
    for i in range(100):
        # Randomly assign market and payer codes
        market_code = random.choice(market_codes)
        payer_code = random.choice(payer_codes)
        
        member = {
            "member_id": f"M{10000+i}",
            "name": f"Patient {i+1}",
            "age": random.randint(65, 90),
            "market_code": market_code,
            "payer_code": payer_code,
            "diabetes_pdc": round(random.uniform(60, 100), 1) if random.random() > 0.3 else None,
            "hypertension_pdc": round(random.uniform(60, 100), 1) if random.random() > 0.3 else None,
            "statin_pdc": round(random.uniform(60, 100), 1) if random.random() > 0.3 else None,
            "diabetes_adr": random.randint(-20, 60) if random.random() > 0.3 else None,
            "hypertension_adr": random.randint(-20, 60) if random.random() > 0.3 else None,
            "statin_adr": random.randint(-20, 60) if random.random() > 0.3 else None,
            "last_fill_date": (today - timedelta(days=random.randint(1, 45))).strftime('%Y-%m-%d'),
        }
        member_data.append(member)
    
    # Generate market/payer level data
    market_payer_data = []
    for market in market_codes:
        for payer in payer_codes:
            for metric in ["Diabetes Medications (MAD)", "Hypertension (RAS antagonists) (MAH)", "Cholesterol (Statins) (MAC)"]:
                # Generate random rates for each market/payer combination
                base_rate = adherence_data["current_rate"][adherence_data["metric"].index(metric)]
                # Add some variation by market and payer
                variation = random.uniform(-3.0, 3.0)
                
                record = {
                    "market_code": market,
                    "payer_code": payer,
                    "metric": metric,
                    "rate": round(base_rate + variation, 1),
                    "member_count": random.randint(50, 500)
                }
                market_payer_data.append(record)
    
    # Placeholder for SUPD data
    supd_data = {
        "metric": "Statin Use in Persons with Diabetes (SUPD)",
        "current_rate": 83.6,
        "4_star_threshold": 90,
        "5_star_threshold": 94,
        "last_year_rate": 82.2,
    }
    
    # Placeholder for MTM data
    mtm_data = {
        "metric": "MTM Program Completion Rate for CMR",
        "current_rate": 78.5,
        "last_year_rate": 75.8,
        "target": 85.0
    }
    
    return {
        "adherence_df": pd.DataFrame(adherence_data),
        "trend_data": {"months": months, "rates": trend_data},
        "risk_df": pd.DataFrame(risk_data),
        "adr_df": pd.DataFrame(adr_distribution),
        "member_df": pd.DataFrame(member_data),
        "supd_data": supd_data,
        "mtm_data": mtm_data,
        "market_codes": market_codes,
        "payer_codes": payer_codes,
        "market_payer_df": pd.DataFrame(market_payer_data)
    }

# Load the sample data
data = generate_sample_data()

# Dashboard title
st.title("Medicare Medication Adherence Dashboard")

# Add filters in the sidebar
st.sidebar.header("Filters")

# Market code filter
selected_markets = st.sidebar.multiselect(
    "Market Code",
    options=data["market_codes"],
    default=data["market_codes"]
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
    filtered_members = filtered_members[filtered_members["market_code"].isin(selected_markets)]
if selected_payers:
    filtered_members = filtered_members[filtered_members["payer_code"].isin(selected_payers)]

# Filter market/payer data
filtered_market_payer = data["market_payer_df"]
if selected_markets:
    filtered_market_payer = filtered_market_payer[filtered_market_payer["market_code"].isin(selected_markets)]
if selected_payers:
    filtered_market_payer = filtered_market_payer[filtered_market_payer["payer_code"].isin(selected_payers)]

# If no data after filtering, show a warning
if len(filtered_members) == 0:
    st.sidebar.warning("No data available for the selected filters. Please adjust your selection.")

# Date information
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.subheader("Medication Adherence Performance")
    st.caption(f"Last updated: {datetime.now().strftime('%B %d, %Y')}")
with col2:
    st.metric("Days Left in Measurement Year", "288", "-1")
with col3:
    # Show how many members match the filter
    st.metric("Members Selected", f"{len(filtered_members):,}", help="Number of members matching the selected filters")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overall Performance", 
    "Adherence Trends", 
    "At-Risk Members", 
    "Member Lookup",
    "Related Metrics"
])

with tab1:
    st.header("Triple-Weighted Medication Adherence Measures")
    
    # Add a summary of the filters applied
    if len(selected_markets) < len(data["market_codes"]) or len(selected_payers) < len(data["payer_codes"]):
        filter_text = "Filtered by: "
        if len(selected_markets) < len(data["market_codes"]):
            filter_text += f"Markets ({', '.join(selected_markets)})"
        if len(selected_payers) < len(data["payer_codes"]):
            if len(selected_markets) < len(data["market_codes"]):
                filter_text += ", "
            filter_text += f"Payers ({', '.join(selected_payers)})"
        st.markdown(f"*{filter_text}*")
    
    # Calculate metrics based on filtered data (in a real implementation, this would query a database)
    filtered_metrics = {}
    for metric in data["adherence_df"]["metric"]:
        metric_data = filtered_market_payer[filtered_market_payer["metric"] == metric]
        if not metric_data.empty:
            # Calculate weighted average based on member count
            weighted_rate = (metric_data["rate"] * metric_data["member_count"]).sum() / metric_data["member_count"].sum()
            filtered_metrics[metric] = round(weighted_rate, 1)
        else:
            # If no data for this metric, use the overall average
            idx = data["adherence_df"][data["adherence_df"]["metric"] == metric].index[0]
            filtered_metrics[metric] = data["adherence_df"]["current_rate"][idx]
    
    # Create metrics cards
    cols = st.columns(3)
    
    for i, col in enumerate(cols):
        metric = data["adherence_df"]["metric"][i]
        current = filtered_metrics[metric]
        last_year = data["adherence_df"]["last_year_rate"][i]
        delta = current - last_year
        
        # Determine color based on threshold
        if current >= data["adherence_df"]["5_star_threshold"][i]:
            tier = "5 ⭐ (Excellent)"
            color = "green"
        elif current >= data["adherence_df"]["4_star_threshold"][i]:
            tier = "4 ⭐ (Good)"
            color = "blue"
        else:
            tier = "3 ⭐ or below (Needs Improvement)"
            color = "red"
        
        col.metric(
            label=metric,
            value=f"{current}%",
            delta=f"{delta:.1f}% from last year",
            delta_color="normal"
        )
        
        # Add threshold and gap information
        col.markdown(f"**Current Tier:** <span style='color:{color}'>{tier}</span>", unsafe_allow_html=True)
        
        if current < data["adherence_df"]["5_star_threshold"][i]:
            if current < data["adherence_df"]["4_star_threshold"][i]:
                next_threshold = data["adherence_df"]["4_star_threshold"][i]
                next_tier = "4 ⭐"
            else:
                next_threshold = data["adherence_df"]["5_star_threshold"][i]
                next_tier = "5 ⭐"
            
            gap = next_threshold - current
            col.markdown(f"**Gap to {next_tier}:** {gap:.1f}%")
        else:
            col.markdown("**Status:** Exceeding 5-star threshold")
    
    # Add a comparison chart
    st.subheader("Comparison to Star Rating Thresholds")
    
    fig = go.Figure()
    
    # Add bars for current rates
    fig.add_trace(go.Bar(
        x=data["adherence_df"]["metric"],
        y=[filtered_metrics[m] for m in data["adherence_df"]["metric"]],
        name='Current Rate',
        marker_color='royalblue'
    ))
    
    # Add lines for thresholds
    for i, row in data["adherence_df"].iterrows():
        fig.add_trace(go.Scatter(
            x=[row["metric"], row["metric"]],
            y=[row["4_star_threshold"], row["4_star_threshold"]],
            mode='markers',
            marker=dict(symbol='line-ns', line_width=3, size=10, line_color='orange'),
            name='4-Star Threshold' if i == 0 else None,
            showlegend=i == 0,
            hovertemplate=f"4-Star: {row['4_star_threshold']}%"
        ))
        
        fig.add_trace(go.Scatter(
            x=[row["metric"], row["metric"]],
            y=[row["5_star_threshold"], row["5_star_threshold"]],
            mode='markers',
            marker=dict(symbol='line-ns', line_width=3, size=10, line_color='green'),
            name='5-Star Threshold' if i == 0 else None,
            showlegend=i == 0,
            hovertemplate=f"5-Star: {row['5_star_threshold']}%"
        ))
    
    fig.update_layout(
        title='Current Performance vs. Star Rating Thresholds',
        xaxis_title='Medication Category',
        yaxis_title='Proportion of Days Covered (%)',
        yaxis=dict(range=[75, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display data table
    st.subheader("Detailed Performance Data")
    
    # Create a display dataframe with the filtered metrics
    display_df = data["adherence_df"][["metric", "4_star_threshold", "5_star_threshold", "last_year_rate"]].copy()
    # Add the filtered current rates
    display_df["current_rate"] = [filtered_metrics[m] for m in display_df["metric"]]
    # Reorder columns
    display_df = display_df[["metric", "current_rate", "4_star_threshold", "5_star_threshold", "last_year_rate"]]
    display_df.columns = ["Medication Category", "Current PDC Rate (%)", "4-Star Threshold (%)", "5-Star Threshold (%)", "Last Year PDC Rate (%)"]
    
    st.dataframe(display_df, use_container_width=True)

with tab2:
    st.header("Medication Adherence Trends")
    
    # Show the applied filters
    if len(selected_markets) < len(data["market_codes"]) or len(selected_payers) < len(data["payer_codes"]):
        filter_text = "Filtered by: "
        if len(selected_markets) < len(data["market_codes"]):
            filter_text += f"Markets ({', '.join(selected_markets)})"
        if len(selected_payers) < len(data["payer_codes"]):
            if len(selected_markets) < len(data["market_codes"]):
                filter_text += ", "
            filter_text += f"Payers ({', '.join(selected_payers)})"
        st.markdown(f"*{filter_text}*")
    
    # Create a selection for which metrics to display
    selected_metrics = st.multiselect(
        "Select medication categories to display:",
        options=data["adherence_df"]["metric"].tolist(),
        default=data["adherence_df"]["metric"].tolist()
    )
    
    # Create the trend chart
    trend_fig = go.Figure()
    
    # Add a line for each selected metric
    colors = ['royalblue', 'green', 'orange']
    
    for i, metric in enumerate(selected_metrics):
        trend_fig.add_trace(go.Scatter(
            x=data["trend_data"]["months"],
            y=data["trend_data"]["rates"][metric],
            mode='lines+markers',
            name=metric,
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    
    # Add 4-star threshold reference lines
    for i, metric in enumerate(selected_metrics):
        idx = data["adherence_df"]["metric"].tolist().index(metric)
        threshold = data["adherence_df"]["4_star_threshold"][idx]
        
        trend_fig.add_trace(go.Scatter(
            x=[data["trend_data"]["months"][0], data["trend_data"]["months"][-1]],
            y=[threshold, threshold],
            mode='lines',
            line=dict(color=colors[i % len(colors)], width=1, dash='dash'),
            name=f'{metric} 4-Star Threshold',
            showlegend=False
        ))
    
    trend_fig.update_layout(
        title='Monthly Medication Adherence Rates (PDC)',
        xaxis_title='Month',
        yaxis_title='PDC Rate (%)',
        yaxis=dict(range=[80, 95]),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    st.plotly_chart(trend_fig, use_container_width=True)
    
    # Add breakdown by market or payer
    st.subheader("Performance by Market and Payer")
    
    # Choose dimension to break down by
    breakdown_dimension = st.radio(
        "Break down by:",
        ["Market Code", "Payer Code"]
    )
    
    # Choose metric to view
    breakdown_metric = st.selectbox(
        "Select medication category:",
        options=data["adherence_df"]["metric"].tolist()
    )
    
    # Filter data for the selected metric
    metric_data = filtered_market_payer[filtered_market_payer["metric"] == breakdown_metric]
    
    if breakdown_dimension == "Market Code":
        # Group by market code
        grouped_data = metric_data.groupby("market_code").apply(
            lambda x: (x["rate"] * x["member_count"]).sum() / x["member_count"].sum()
        ).reset_index()
        grouped_data.columns = ["Market Code", "PDC Rate (%)"]
        
        # Create chart
        breakdown_fig = px.bar(
            grouped_data,
            x="Market Code",
            y="PDC Rate (%)",
            color="PDC Rate (%)",
            color_continuous_scale=["red", "yellow", "green"],
            range_color=[80, 95],
            title=f"{breakdown_metric} by Market Code"
        )
    else:
        # Group by payer code
        grouped_data = metric_data.groupby("payer_code").apply(
            lambda x: (x["rate"] * x["member_count"]).sum() / x["member_count"].sum()
        ).reset_index()
        grouped_data.columns = ["Payer Code", "PDC Rate (%)"]
        
        # Create chart
        breakdown_fig = px.bar(
            grouped_data,
            x="Payer Code",
            y="PDC Rate (%)",
            color="PDC Rate (%)",
            color_continuous_scale=["red", "yellow", "green"],
            range_color=[80, 95],
            title=f"{breakdown_metric} by Payer Code"
        )
    
    breakdown_fig.update_layout(height=400)
    st.plotly_chart(breakdown_fig, use_container_width=True)
    
    # Add some commentary
    st.subheader("Observations")
    st.markdown("""
    - **Diabetes Medications:** Showing steady improvement over the past 6 months
    - **Hypertension Medications:** Slight decline in the last month that requires attention
    - **Cholesterol Medications:** Consistently closest to the 4-star threshold
    """)

with tab3:
    st.header("Members at Risk of Non-Adherence")
    
    # Show the applied filters
    if len(selected_markets) < len(data["market_codes"]) or len(selected_payers) < len(data["payer_codes"]):
        filter_text = "Filtered by: "
        if len(selected_markets) < len(data["market_codes"]):
            filter_text += f"Markets ({', '.join(selected_markets)})"
        if len(selected_payers) < len(data["payer_codes"]):
            if len(selected_markets) < len(data["market_codes"]):
                filter_text += ", "
            filter_text += f"Payers ({', '.join(selected_payers)})"
        st.markdown(f"*{filter_text}*")
    
    # Display summary metrics
    risk_cols = st.columns(3)
    
    # Apply scaling factor based on filtered data size
    scaling_factor = len(filtered_members) / len(data["member_df"]) if len(data["member_df"]) > 0 else 1
    
    for i, col in enumerate(risk_cols):
        category = data["risk_df"]["category"][i]
        # Scale the counts based on the filter
        at_risk = int(data["risk_df"]["at_risk_members"][i] * scaling_factor)
        non_adherent = int(data["risk_df"]["non_adherent_members"][i] * scaling_factor)
        critical = int(data["risk_df"]["critical_adr"][i] * scaling_factor)
        
        col.subheader(category)
        col.metric("Members at Risk (PDC 80-85%)", at_risk, help="Members with PDC between 80-85% who are at risk of falling below the adherence threshold")
        col.metric("Currently Non-Adherent", non_adherent, delta=f"+{int(non_adherent * 0.05)} in last 30 days", delta_color="inverse")
        col.metric("Critical ADR (<10 days)", critical, help="Members with less than 10 Allowable Days Remaining before becoming non-adherent")
    
    # ADR Distribution
    st.subheader("Allowable Days Remaining (ADR) Distribution")
    
    # Scale the ADR distribution based on the filter
    scaled_adr_df = data["adr_df"].copy()
    scaled_adr_df["count"] = [int(count * scaling_factor) for count in data["adr_df"]["count"]]
    
    adr_fig = px.bar(
        scaled_adr_df,
        x="adr_range",
        y="count",
        color="adr_range",
        color_discrete_map={
            "<0 days (failed)": "red",
            "1-10 days": "orange",
            "11-20 days": "yellow",
            "21-30 days": "lightgreen",
            "31-60 days": "green",
            ">60 days": "darkgreen"
        },
        labels={"adr_range": "Allowable Days Remaining", "count": "Number of Members"}
    )
    
    adr_fig.update_layout(
        title="Distribution of Members by Allowable Days Remaining",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(adr_fig, use_container_width=True)
    
    # Intervention priority list
    st.subheader("Priority Intervention List")
    
    # Filter and sort member data to show highest priority cases
    priority_df = filtered_members.copy()
    
    # Create a function to determine priority
    def calculate_priority(row):
        priority = 0
        if row["diabetes_adr"] is not None and row["diabetes_adr"] < 10:
            priority += 3
        if row["hypertension_adr"] is not None and row["hypertension_adr"] < 10:
            priority += 3
        if row["statin_adr"] is not None and row["statin_adr"] < 10:
            priority += 3
        
        # Add higher priority for multiple medications
        meds_count = sum(1 for x in [row["diabetes_pdc"], row["hypertension_pdc"], row["statin_pdc"]] if x is not None)
        if meds_count > 1:
            priority += meds_count
            
        return priority
    
    priority_df["priority"] = priority_df.apply(calculate_priority, axis=1)
    priority_df = priority_df.sort_values("priority", ascending=False).head(10)
    
    # Prepare display columns
    display_cols = ["member_id", "name", "age", "market_code", "payer_code", "diabetes_adr", "hypertension_adr", "statin_adr", "last_fill_date"]
    display_names = ["Member ID", "Name", "Age", "Market", "Payer", "Diabetes ADR", "Hypertension ADR", "Statin ADR", "Last Fill Date"]
    
    # Rename columns for display
    display_priority_df = priority_df[display_cols].copy()
    display_priority_df.columns = display_names
    
    st.dataframe(display_priority_df, use_container_width=True)
    
    st.caption("ADR = Allowable Days Remaining before member becomes non-adherent. Negative values indicate member is already non-adherent.")

with tab4:
    st.header("Member Lookup")
    
    # Show the applied filters
    if len(selected_markets) < len(data["market_codes"]) or len(selected_payers) < len(data["payer_codes"]):
        filter_text = "Filtered by: "
        if len(selected_markets) < len(data["market_codes"]):
            filter_text += f"Markets ({', '.join(selected_markets)})"
        if len(selected_payers) < len(data["payer_codes"]):
            if len(selected_markets) < len(data["market_codes"]):
                filter_text += ", "
            filter_text += f"Payers ({', '.join(selected_payers)})"
        st.markdown(f"*{filter_text}*")
    
    # Add a search box for member ID or name
    search_term = st.text_input("Search by Member ID or Name")
    
    if search_term:
        # Filter the data based on the search term and existing filters
        filtered_df = filtered_members[
            filtered_members["member_id"].str.contains(search_term, case=False) | 
            filtered_members["name"].str.contains(search_term, case=False)
        ]
        
        if len(filtered_df) > 0:
            st.subheader(f"Results for '{search_term}'")
            st.dataframe(filtered_df, use_container_width=True)
            
            # If only one member is found, show detailed information
            if len(filtered_df) == 1:
                member = filtered_df.iloc[0]
                
                st.subheader(f"Detailed Information for {member['name']} ({member['member_id']})")
                
                # Member demographics and information
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Age", member["age"])
                col2.metric("Market", member["market_code"])
                col3.metric("Payer", member["payer_code"])
                col4.metric("Last Fill Date", member["last_fill_date"])
                
                # Create medication adherence metrics display
                st.markdown("### Medication Adherence Metrics")
                
                med_cols = st.columns(3)
                
                # Diabetes
                if not pd.isna(member["diabetes_pdc"]):
                    med_cols[0].subheader("Diabetes Medications")
                    
                    # Show PDC with color-coded status
                    pdc = member["diabetes_pdc"]
                    adr = member["diabetes_adr"]
                    
                    if pdc >= 90:
                        med_cols[0].markdown(f"**PDC Rate:** <span style='color:green'>{pdc}%</span>", unsafe_allow_html=True)
                    elif pdc >= 80:
                        med_cols[0].markdown(f"**PDC Rate:** <span style='color:orange'>{pdc}%</span>", unsafe_allow_html=True)
                    else:
                        med_cols[0].markdown(f"**PDC Rate:** <span style='color:red'>{pdc}%</span>", unsafe_allow_html=True)
                    
                    # Show ADR with appropriate color
                    if adr < 0:
                        med_cols[0].markdown(f"**ADR:** <span style='color:red'>{adr} days (non-adherent)</span>", unsafe_allow_html=True)
                    elif adr < 10:
                        med_cols[0].markdown(f"**ADR:** <span style='color:orange'>{adr} days (critical)</span>", unsafe_allow_html=True)
                    else:
                        med_cols[0].markdown(f"**ADR:** <span style='color:green'>{adr} days</span>", unsafe_allow_html=True)
                if not pd.isna(member["hypertension_pdc"]):
                        med_cols[1].subheader("Hypertension Medications")
                        pdc = member["hypertension_pdc"]
                        adr = member["hypertension_adr"]
                