from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import toml
import json
import os
import streamlit as st

class BigQueryConnector:
    def __init__(self, toml_path=None, credentials_path=None):
        """
        Initialize BigQuery connector with credentials from TOML, JSON file,
        or Streamlit secrets
        """
        # Check if we have credentials in Streamlit secrets
        
        if hasattr(st, "secrets") and "gcp" in st.secrets:
            credentials_dict = st.secrets["gcp"]
            self.credentials = service_account.Credentials.from_service_account_info(
                credentials_dict
            )
        elif toml_path:
            config = toml.load(toml_path)
            ...
        elif credentials_path:
            # Load from JSON file (code from previous example)
            ...
        else:
            raise ValueError("No credentials provided")
        
        # Initialize BigQuery client
        self.client = bigquery.Client(credentials=self.credentials)
    
    def run_query(self, query):
        """Execute a query and return results as pandas DataFrame"""
        return self.client.query(query).to_dataframe()

    def get_med_adherence_data(self, start_date=None, end_date=None, 
                               measure_codes=None, market_codes=None, payer_codes=None):
        """
        Fetch medication adherence data based on filters
        Returns a DataFrame with all necessary fields
        """
        # Build query filters based on parameters
        filters = []
        if start_date and end_date:
            filters.append(f"DataAsOfDate BETWEEN '{start_date}' AND '{end_date}'")
        if measure_codes:
            measure_list = "', '".join(measure_codes)
            filters.append(f"MedAdherenceMeasureCode IN ('{measure_list}')")
        if market_codes:
            market_list = "', '".join(market_codes)
            filters.append(f"MarketCode IN ('{market_list}')")
        if payer_codes:
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
        
        return self.run_query(query)

    def get_distinct_values(self, column_name):
        """Get distinct values for a given column for filter options"""
        query = f"""
        SELECT DISTINCT {column_name}
        FROM `medadhdata2025.adherence_tracking.weekly_med_adherence_data`
        ORDER BY {column_name}
        """
        return self.run_query(query)[column_name].tolist()
    
    def get_latest_weeks(self, num_weeks=12):
        """Get the latest file load weeks"""
        query = f"""
        SELECT DISTINCT
            EXTRACT(WEEK FROM DataAsOfDate) AS WeekNumber,
            EXTRACT(YEAR FROM DataAsOfDate) AS Year,
            MAX(DataAsOfDate) AS LastDataAsOfDate
        FROM `medadhdata2025.adherence_tracking.weekly_med_adherence_data`
        GROUP BY WeekNumber, Year
        ORDER BY Year DESC, WeekNumber DESC
        LIMIT {num_weeks}
        """
        return self.run_query(query)