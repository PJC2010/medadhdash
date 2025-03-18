
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import bigquery
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set the environment variable for Google credentials if needed
# (This isn't necessary if you're already using Application Default Credentials)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'medadhdata2025-6e124598af28.json'

# Define your project and dataset constants
PROJECT_ID = 'medadhdata2025'  # Replace with your actual project ID
DATASET = "adherence_tracking"       # Replace with your actual dataset

# Initialize BigQuery client without explicitly specifying credentials
# This will use Application Default Credentials automatically
client = bigquery.Client(project=PROJECT_ID)

# Function to execute a query and return results as a DataFrame
def execute_query(query):
    """Execute a BigQuery SQL query and return results as a pandas DataFrame."""
    try:
        return client.query(query).to_dataframe()
    except Exception as e:
        print(f"Error executing query: {e}")
        return pd.DataFrame()

# 1. Query to get basic information about the dataset
def explore_data_structure():
    query = f"""
    SELECT 
        column_name, 
        data_type 
    FROM 
        `{PROJECT_ID}.{DATASET}`.INFORMATION_SCHEMA.COLUMNS 
    WHERE 
        table_name = 'weekly_med_adherence_data'
    """
    return execute_query(query)

# 2. Query to get a sample of data
def get_sample_data(limit=1000):
    query = f"""
    SELECT * 
    FROM `{PROJECT_ID}.{DATASET}.weekly_med_adherence_data`
    LIMIT {limit}
    """
    return execute_query(query)

# 3. Identify refill events
def identify_refill_events():
    # Assuming your table has UPID, LastFillDate, NDCDesc, DrugDispensedDaysSupplyNbr columns
    query = f"""
    WITH numbered_fills AS (
        SELECT 
            UPID,
            NDCDesc,
            LastFillDate,
            DrugDispensedDaysSupplyNbr,
            ROW_NUMBER() OVER(PARTITION BY UPID, NDCDesc ORDER BY LastFillDate) as fill_number
        FROM 
            `{PROJECT_ID}.{DATASET}.weekly_med_adherence_data`
    )
    
    SELECT 
        UPID,
        NDCDesc,
        LastFillDate,
        DrugDispensedDaysSupplyNbr,
        fill_number,
        CASE 
            WHEN fill_number > 1 THEN 'Refill'
            ELSE 'Initial Fill'
        END as fill_type
    FROM 
        numbered_fills
    ORDER BY 
        UPID, 
        NDCDesc, 
        LastFillDate
    """
    return execute_query(query)

# 4. Calculate PDC (Proportion of Days Covered) per member, NDCDesc class
def calculate_pdc():
    # This query calculates PDC for each member and NDCDesc combination
    query = f"""
    WITH fill_data AS (
        SELECT 
            UPID,
            MedAdherenceMeasureCode,  -- Assuming you have this field or can create it
            MIN(LastFillDate) as first_LastFillDate,
            MAX(DATE_ADD(LastFillDate, INTERVAL DrugDispensedDaysSupplyNbr DAY)) as last_coverage_date,
            SUM(DrugDispensedDaysSupplyNbr) as total_DrugDispensedDaysSupplyNbr
        FROM 
            `{PROJECT_ID}.{DATASET}.weekly_med_adherence_data`
        GROUP BY 
            UPID, MedAdherenceMeasureCode
    ),
    
    date_range AS (
        SELECT 
            UPID,
            MedAdherenceMeasureCode,
            first_LastFillDate,
            LEAST(last_coverage_date, DATE('{datetime.now().strftime('%Y-%m-%d')}')) as end_date,  -- Use current date or your study end date
            DATE_DIFF(LEAST(last_coverage_date, DATE('{datetime.now().strftime('%Y-%m-%d')}')), first_LastFillDate, DAY) as period_days
        FROM 
            fill_data
    )
    
    SELECT 
        f.UPID,
        f.MedAdherenceMeasureCode,
        f.first_LastFillDate,
        dr.end_date,
        dr.period_days,
        f.total_DrugDispensedDaysSupplyNbr,
        SAFE_DIVIDE(f.total_DrugDispensedDaysSupplyNbr, dr.period_days) as raw_pdc,
        LEAST(SAFE_DIVIDE(f.total_DrugDispensedDaysSupplyNbr, dr.period_days), 1.0) as capped_pdc,
        CASE 
            WHEN LEAST(SAFE_DIVIDE(f.total_DrugDispensedDaysSupplyNbr, dr.period_days), 1.0) >= 0.8 THEN TRUE
            ELSE FALSE
        END as is_adherent
    FROM 
        fill_data f
    JOIN 
        date_range dr ON f.UPID = dr.UPID AND f.MedAdherenceMeasureCode = dr.MedAdherenceMeasureCode
    WHERE 
        dr.period_days >= 30  -- Only include members with at least 30 days in period
    """
    return execute_query(query)

# 5. Identify gaps in therapy
def identify_therapy_gaps(min_gap_days=7):
    query = f"""
    WITH fill_periods AS (
        SELECT 
            UPID,
            NDCDesc,
            LastFillDate as start_date,
            DATE_ADD(LastFillDate, INTERVAL DrugDispensedDaysSupplyNbr DAY) as end_date
        FROM 
            `{PROJECT_ID}.{DATASET}.weekly_med_adherence_data`
    ),
    
    ordered_fills AS (
        SELECT 
            UPID,
            NDCDesc,
            start_date,
            end_date,
            LEAD(start_date) OVER(PARTITION BY UPID, NDCDesc ORDER BY start_date) as next_LastFillDate
        FROM 
            fill_periods
    )
    
    SELECT 
        UPID,
        NDCDesc,
        end_date as gap_start_date,
        next_LastFillDate as gap_end_date,
        DATE_DIFF(next_LastFillDate, end_date, DAY) as gap_days
    FROM 
        ordered_fills
    WHERE 
        next_LastFillDate IS NOT NULL
        AND DATE_DIFF(next_LastFillDate, end_date, DAY) > {min_gap_days}
    ORDER BY 
        gap_days DESC
    """
    return execute_query(query)

# 6. Track intervention outcomes - requires intervention table to exist
def track_intervention_outcomes(days_before=30, days_after=60):
    try:
        # Check if interventions table exists
        metadata_query = f"""
        SELECT table_name 
        FROM `{PROJECT_ID}.{DATASET}`.INFORMATION_SCHEMA.TABLES 
        WHERE table_name = 'interventions'
        """
        result = execute_query(metadata_query)
        
        if result.empty:
            print("Interventions table does not exist. Please create it first.")
            return pd.DataFrame()
    
        query = f"""
        WITH member_adherence_before AS (
            SELECT 
                i.UPID,
                i.intervention_id,
                i.intervention_date,
                i.intervention_type,
                COUNT(DISTINCT a.LastFillDate) as fills_before,
                SUM(a.DrugDispensedDaysSupplyNbr) as DrugDispensedDaysSupplyNbr_before,
                {days_before} as days_measured_before
            FROM 
                `{PROJECT_ID}.{DATASET}.interventions` i
            LEFT JOIN 
                `{PROJECT_ID}.{DATASET}.weekly_med_adherence_data` a
                ON i.UPID = a.UPID
                AND a.LastFillDate BETWEEN DATE_SUB(i.intervention_date, INTERVAL {days_before} DAY) AND i.intervention_date
            GROUP BY 
                i.UPID, i.intervention_id, i.intervention_date, i.intervention_type
        ),
        
        member_adherence_after AS (
            SELECT 
                i.UPID,
                i.intervention_id,
                COUNT(DISTINCT a.LastFillDate) as fills_after,
                SUM(a.DrugDispensedDaysSupplyNbr) as DrugDispensedDaysSupplyNbr_after,
                {days_after} as days_measured_after
            FROM 
                `{PROJECT_ID}.{DATASET}.interventions` i
            LEFT JOIN 
                `{PROJECT_ID}.{DATASET}.weekly_med_adherence_data` a
                ON i.UPID = a.UPID
                AND a.LastFillDate BETWEEN i.intervention_date AND DATE_ADD(i.intervention_date, INTERVAL {days_after} DAY)
            GROUP BY 
                i.UPID, i.intervention_id
        )
        
        SELECT 
            b.UPID,
            b.intervention_id,
            b.intervention_date,
            b.intervention_type,
            b.fills_before,
            a.fills_after,
            b.DrugDispensedDaysSupplyNbr_before,
            a.DrugDispensedDaysSupplyNbr_after,
            SAFE_DIVIDE(b.DrugDispensedDaysSupplyNbr_before, b.days_measured_before) as daily_supply_rate_before,
            SAFE_DIVIDE(a.DrugDispensedDaysSupplyNbr_after, a.days_measured_after) as daily_supply_rate_after,
            CASE 
                WHEN SAFE_DIVIDE(a.DrugDispensedDaysSupplyNbr_after, a.days_measured_after) > 
                     SAFE_DIVIDE(b.DrugDispensedDaysSupplyNbr_before, b.days_measured_before) THEN 'Improved'
                WHEN SAFE_DIVIDE(a.DrugDispensedDaysSupplyNbr_after, a.days_measured_after) = 
                     SAFE_DIVIDE(b.DrugDispensedDaysSupplyNbr_before, b.days_measured_before) THEN 'Maintained'
                ELSE 'Declined'
            END as outcome_status
        FROM 
            member_adherence_before b
        JOIN 
            member_adherence_after a ON b.intervention_id = a.intervention_id
        """
        return execute_query(query)
    except Exception as e:
        print(f"Error tracking intervention outcomes: {e}")
        return pd.DataFrame()

# 7. Identify common patterns/themes in adherence
def identify_adherence_patterns():
    query = f"""
    WITH adherence_stats AS (
        SELECT 
            UPID,
            MedAdherenceMeasureCode,
            COUNT(DISTINCT LastFillDate) as total_fills,
            MIN(LastFillDate) as first_LastFillDate,
            MAX(LastFillDate) as last_LastFillDate,
            DATE_DIFF(MAX(LastFillDate), MIN(LastFillDate), DAY) as therapy_length_days,
            AVG(DrugDispensedDaysSupplyNbr) as avg_DrugDispensedDaysSupplyNbr
        FROM 
            `{PROJECT_ID}.{DATASET}.weekly_med_adherence_data`
        GROUP BY 
            UPID, MedAdherenceMeasureCode
        HAVING 
            COUNT(DISTINCT LastFillDate) > 1
    )
    
    SELECT 
        MedAdherenceMeasureCode,
        COUNT(DISTINCT UPID) as member_count,
        AVG(total_fills) as avg_fills_per_member,
        AVG(therapy_length_days) as avg_therapy_length_days,
        AVG(avg_DrugDispensedDaysSupplyNbr) as avg_DrugDispensedDaysSupplyNbr,
        STDDEV(avg_DrugDispensedDaysSupplyNbr) as stddev_DrugDispensedDaysSupplyNbr,
        -- Categorize refill patterns
        COUNTIF(total_fills >= 3 AND avg_DrugDispensedDaysSupplyNbr >= 60) as consistent_90day_members,
        COUNTIF(total_fills >= 6 AND avg_DrugDispensedDaysSupplyNbr <= 35) as consistent_30day_members,
        COUNTIF(total_fills < 3 AND therapy_length_days > 90) as potential_nonadherent_members
    FROM 
        adherence_stats
    GROUP BY 
        MedAdherenceMeasureCode
    ORDER BY 
        member_count DESC
    """
    return execute_query(query)

# 8. Create adherence timeline visualization
def plot_adherence_timeline(UPID):
    query = f"""
    SELECT 
        UPID,
        NDCDesc,
        LastFillDate,
        DrugDispensedDaysSupplyNbr,
        DATE_ADD(LastFillDate, INTERVAL DrugDispensedDaysSupplyNbr DAY) as end_coverage_date
    FROM 
        `{PROJECT_ID}.{DATASET}.weekly_med_adherence_data`
    WHERE 
        UPID = '{UPID}'
    ORDER BY 
        NDCDesc, LastFillDate
    """
    df = execute_query(query)
    
    if df.empty:
        print(f"No data found for member {UPID}")
        return None
    
    # Plot setup
    plt.figure(figsize=(15, 8))
    NDCDescs = df['NDCDesc'].unique()
    
    for i, med in enumerate(NDCDescs):
        med_df = df[df['NDCDesc'] == med]
        
        for _, row in med_df.iterrows():
            plt.plot([row['LastFillDate'], row['end_coverage_date']], 
                      [i, i], 
                      linewidth=6, 
                      solid_capstyle='butt')
            
            # Mark fill dates with a dot
            plt.scatter(row['LastFillDate'], i, color='red', s=50, zorder=5)
        
        # Add gaps visualization
        for j in range(len(med_df) - 1):
            current_end = med_df.iloc[j]['end_coverage_date']
            next_start = med_df.iloc[j+1]['LastFillDate']
            
            if current_end < next_start:  # There's a gap
                plt.plot([current_end, next_start], 
                         [i, i], 
                         'r--', 
                         linewidth=2)
    
    plt.yticks(range(len(NDCDescs)), NDCDescs)
    plt.xlabel('Date')
    plt.ylabel('NDCDesc')
    plt.title(f'Adherence Timeline for Member {UPID}')
    plt.tight_layout()
    
    # Calculate overall PDC
    total_days = (df['end_coverage_date'].max() - df['LastFillDate'].min()).days
    covered_days = sum((row['end_coverage_date'] - row['LastFillDate']).days for _, row in df.iterrows())
    pdc = min(covered_days / total_days, 1.0) if total_days > 0 else 0
    
    plt.figtext(0.5, 0.01, f'Overall PDC: {pdc:.2%}', 
                ha='center', fontsize=12, bbox=dict(facecolor='lightgray', alpha=0.5))
    
    return plt

# 9. Create an intervention outcomes dashboard
def create_intervention_dashboard():
    # Get intervention outcomes data
    outcomes_df = track_intervention_outcomes()
    
    if outcomes_df.empty:
        print("No intervention data available")
        return None
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Outcomes by Intervention Type
    outcomes_by_type = outcomes_df.groupby('intervention_type')['outcome_status'].value_counts().unstack()
    outcomes_by_type.plot(kind='bar', stacked=True, ax=axs[0, 0])
    axs[0, 0].set_title('Intervention Outcomes by Type')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].legend(title='Outcome')
    
    # Plot 2: Average Daily Supply Rate Before vs After
    outcomes_df.plot.scatter(x='daily_supply_rate_before', 
                            y='daily_supply_rate_after', 
                            c=outcomes_df['outcome_status'].map({'Improved': 'green', 
                                                                'Maintained': 'blue', 
                                                                'Declined': 'red'}),
                            ax=axs[0, 1])
    axs[0, 1].plot([0, 1], [0, 1], 'k--')  # Diagonal line
    axs[0, 1].set_title('Daily Supply Rate: Before vs After')
    axs[0, 1].set_xlabel('Before Intervention')
    axs[0, 1].set_ylabel('After Intervention')
    
    # Plot 3: Outcome Distribution
    outcomes_df['outcome_status'].value_counts().plot(kind='pie', 
                                                     autopct='%1.1f%%', 
                                                     ax=axs[1, 0],
                                                     colors=['green', 'blue', 'red'])
    axs[1, 0].set_title('Overall Outcome Distribution')
    
    # Plot 4: Time Series of Interventions and Outcomes
    outcomes_df['intervention_date'] = pd.to_datetime(outcomes_df['intervention_date'])
    outcome_time = outcomes_df.groupby([pd.Grouper(key='intervention_date', freq='M'), 'outcome_status']).size().unstack()
    outcome_time.plot(kind='line', marker='o', ax=axs[1, 1])
    axs[1, 1].set_title('Intervention Outcomes Over Time')
    axs[1, 1].set_xlabel('Intervention Month')
    axs[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    return fig

# 10. Create a function to identify members needing intervention
def identify_members_needing_intervention(pdc_threshold=0.8, gap_days=15):
    query = f"""
    WITH current_status AS (
        SELECT 
            UPID,
            MedAdherenceMeasureCode,
            MAX(DATE_ADD(LastFillDate, INTERVAL DrugDispensedDaysSupplyNbr DAY)) as last_coverage_date,
            MAX(LastFillDate) as last_LastFillDate,
            SUM(DrugDispensedDaysSupplyNbr) as total_DrugDispensedDaysSupplyNbr,
            MIN(LastFillDate) as first_LastFillDate
        FROM 
            `{PROJECT_ID}.{DATASET}.weekly_med_adherence_data`
        GROUP BY 
            UPID, MedAdherenceMeasureCode
    ),
    
    member_status AS (
        SELECT
            cs.*,
            DATE_DIFF(CURRENT_DATE(), first_LastFillDate, DAY) as days_in_therapy,
            DATE_DIFF(CURRENT_DATE(), last_coverage_date, DAY) as days_since_coverage_end,
            SAFE_DIVIDE(total_DrugDispensedDaysSupplyNbr, DATE_DIFF(CURRENT_DATE(), first_LastFillDate, DAY)) as current_pdc
        FROM 
            current_status cs
    )
    
    SELECT
        UPID,
        MedAdherenceMeasureCode,
        last_LastFillDate,
        last_coverage_date,
        days_since_coverage_end,
        days_in_therapy,
        current_pdc,
        CASE 
            WHEN days_since_coverage_end > {gap_days} THEN 'Currently Lapsed'
            WHEN days_since_coverage_end <= 0 THEN 'Currently Covered'
            ELSE 'Coverage Gap < {gap_days} days'
        END as coverage_status,
        CASE
            WHEN current_pdc < {pdc_threshold} AND days_since_coverage_end > 0 THEN 'High Priority'
            WHEN current_pdc < {pdc_threshold} AND days_since_coverage_end <= 0 THEN 'Medium Priority'
            WHEN current_pdc >= {pdc_threshold} AND days_since_coverage_end > {gap_days} THEN 'Medium Priority'
            ELSE 'Low Priority'
        END as intervention_priority
    FROM
        member_status
    WHERE
        days_in_therapy >= 90  -- Only include members with at least 90 days in therapy
    ORDER BY
        CASE 
            WHEN current_pdc < {pdc_threshold} AND days_since_coverage_end > 0 THEN 1
            WHEN current_pdc < {pdc_threshold} AND days_since_coverage_end <= 0 THEN 2
            WHEN current_pdc >= {pdc_threshold} AND days_since_coverage_end > {gap_days} THEN 2
            ELSE 3
        END,
        days_since_coverage_end DESC
    """
    return execute_query(query)

# Example CLI interface to run various analyses
def main():
    print("Medicare Adherence Analysis Tool")
    print("=" * 30)
    
    # Get sample data to confirm connection
    print("Testing connection to BigQuery...")
    structure = explore_data_structure()
    
    if structure.empty:
        print("❌ Error connecting to BigQuery or table not found.")
        return
    
    print("✅ Connected successfully!")
    print(f"Found these columns in weekly_med_adherence_data table:")
    print(structure)
    
    while True:
        print("\nAvailable analyses:")
        print("1. Get sample data")
        print("2. Calculate PDC rates")
        print("3. Identify refill events")
        print("4. Identify therapy gaps")
        print("5. Plot adherence timeline for a member")
        print("6. Identify members needing intervention")
        print("7. Show adherence patterns")
        print("8. Track intervention outcomes")
        print("9. Exit")
        
        choice = input("\nSelect an analysis (1-9): ")
        
        if choice == '1':
            n = int(input("Number of rows to sample: "))
            df = get_sample_data(n)
            print(df.head())
            print(f"Retrieved {len(df)} rows")
            
        elif choice == '2':
            print("Calculating PDC rates...")
            pdc_df = calculate_pdc()
            print(f"PDC calculated for {len(pdc_df)} member-NDCDesc combinations")
            adherent = pdc_df[pdc_df['is_adherent']].shape[0]
            print(f"Adherent: {adherent} ({adherent/len(pdc_df):.2%})")
            print(f"Average PDC: {pdc_df['capped_pdc'].mean():.2%}")
            
        elif choice == '3':
            print("Identifying refill events...")
            refills_df = identify_refill_events()
            initial = refills_df[refills_df['fill_type'] == 'Initial Fill'].shape[0]
            refill = refills_df[refills_df['fill_type'] == 'Refill'].shape[0]
            print(f"Found {initial} initial fills and {refill} refills")
            
        elif choice == '4':
            gap_days = int(input("Minimum gap days to identify: "))
            gaps_df = identify_therapy_gaps(gap_days)
            print(f"Found {len(gaps_df)} therapy gaps of {gap_days}+ days")
            if not gaps_df.empty:
                print(f"Average gap: {gaps_df['gap_days'].mean():.1f} days")
                print(f"Longest gap: {gaps_df['gap_days'].max()} days")
            
        elif choice == '5':
            UPID = input("Enter member ID: ")
            plt = plot_adherence_timeline(UPID)
            if plt:
                plt.show()
                
        elif choice == '6':
            print("Identifying members needing intervention...")
            int_df = identify_members_needing_intervention()
            high = int_df[int_df['intervention_priority'] == 'High Priority'].shape[0]
            medium = int_df[int_df['intervention_priority'] == 'Medium Priority'].shape[0]
            low = int_df[int_df['intervention_priority'] == 'Low Priority'].shape[0]
            print(f"High Priority: {high}")
            print(f"Medium Priority: {medium}")
            print(f"Low Priority: {low}")
            
        elif choice == '7':
            print("Analyzing adherence patterns...")
            patterns_df = identify_adherence_patterns()
            print(patterns_df)
            
        elif choice == '8':
            print("Tracking intervention outcomes...")
            outcomes_df = track_intervention_outcomes()
            if not outcomes_df.empty:
                print(f"Analyzed {len(outcomes_df)} interventions")
                improved = outcomes_df[outcomes_df['outcome_status'] == 'Improved'].shape[0]
                maintained = outcomes_df[outcomes_df['outcome_status'] == 'Maintained'].shape[0]
                declined = outcomes_df[outcomes_df['outcome_status'] == 'Declined'].shape[0]
                print(f"Improved: {improved} ({improved/len(outcomes_df):.2%})")
                print(f"Maintained: {maintained} ({maintained/len(outcomes_df):.2%})")
                print(f"Declined: {declined} ({declined/len(outcomes_df):.2%})")
            
        elif choice == '9':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please select 1-9.")

if __name__ == "__main__":
    main()