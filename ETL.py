import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from google.cloud import bigquery
from google.oauth2 import service_account
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("excel_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ExcelProcessor")

WATCH_DIRECTORY = r"C:/Users/pcastillo/OneDrive - VillageMD\Documents - VMD- Quality Leadership- PHI/Data Updates/MedAdhData Dropzone/Input"  # Directory to watch for new Excel files
PROCESSED_DIRECTORY = r"C:/Users/pcastillo/OneDrive - VillageMD\Documents - VMD- Quality Leadership- PHI/Data Updates/MedAdhData Dropzone/Processed"   # Directory to move processed files
CSV_OUTPUT_DIRECTORY = r"C:/Users/pcastillo/OneDrive - VillageMD\Documents - VMD- Quality Leadership- PHI/Data Updates/MedAdhData Dropzone/Output"   # Directory to save CSV files
ARCHIVE_DIRECTORY = r"C:/Users/pcastillo/OneDrive - VillageMD\Documents - VMD- Quality Leadership- PHI/Data Updates/MedAdhData Dropzone/Archive"   # Directory to archive original files
ERROR_DIRECTORY = r"C:/Users/pcastillo/OneDrive - VillageMD\Documents - VMD- Quality Leadership- PHI/Data Updates/MedAdhData Dropzone/Errors"   # Directory for files that failed processin

CREDENTIALS_PATH = 'medadhdata2025-6e124598af28.json'
PROJECT_ID = 'medadhdata2025'
DATASET_NAME = "adherence_tracking"
TABLE_NAME = "weekly_med_adherence_data"

for directory in [WATCH_DIRECTORY, PROCESSED_DIRECTORY, CSV_OUTPUT_DIRECTORY, ARCHIVE_DIRECTORY, ERROR_DIRECTORY]:
    os.makedirs(directory, exist_ok=True)

class ExcelProcessor:
    def __init__(self):
        if os.path.exists(CREDENTIALS_PATH):
            self.credentials = service_account.Credentials.from_service_account_file(
                CREDENTIALS_PATH)
            self.client = bigquery.Client(credentials=self.credentials, project=PROJECT_ID)
            logger.info("BigQuery client initialized successfully")
        else:
            self.client = None
            logger.warning(f"BigQuery credentials file not found at {CREDENTIALS_PATH}")
        
        if self.client:
            self.create_bigquery_resources()
    
    def create_bigquery_resources(self):
        try:
            dataset_id = f"{PROJECT_ID}.{DATASET_NAME}"
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = "US"
            try:
                self.client.get_dataset(dataset_id)
                logger.info(f"Dataset {dataset_id} already exists")
            except Exception:
                dataset = self.client.create_dataset(dataset, exists_ok=True)
                logger.info(f"Dataset {dataset_id} created")
            
            schema = [
                bigquery.SchemaField("data_week", "STRING"),
                bigquery.SchemaField("DataAsOfDate", "DATE"),
                bigquery.SchemaField("PayerMemberId", "STRING"),
                bigquery.SchemaField("DateOfBirth", "DATE"),
                bigquery.SchemaField("PayerCode", "STRING"),
                bigquery.SchemaField("MarketCode", "STRING"),
                bigquery.SchemaField("PDCNbr", "FLOAT"),
                bigquery.SchemaField("ADRNbr", "INTEGER"),
                bigquery.SchemaField("InitialFillDate", "DATE"),
                bigquery.SchemaField("LastFillDate", "DATE"),
                bigquery.SchemaField("NextFillDate", "DATE"),
                bigquery.SchemaField("LastImpactableDate", "DATE"),
                bigquery.SchemaField("OneFillCode", "STRING"),
                bigquery.SchemaField("DrugDispensedQuantityNbr", "INTEGER"),
                bigquery.SchemaField("DrugDispensedDaysSupplyNbr", "INTEGER"),
                bigquery.SchemaField("MedAdherenceMeasureCode", "STRING"),
                bigquery.SchemaField("NDCDesc", "STRING"),
                bigquery.SchemaField("UPID", "STRING"),
                bigquery.SchemaField("loaded_timestamp", "TIMESTAMP")
            ]
            
            table_id = f"{dataset_id}.{TABLE_NAME}"
            table = bigquery.Table(table_id, schema=schema)
            
            try:
                self.client.get_table(table_id)
                logger.info(f"Table {table_id} already exists")
            except Exception:
                table = self.client.create_table(table, exists_ok=True)
                logger.info(f"Table {table_id} created")
                
        except Exception as e:
            logger.error(f"Error creating BigQuery resources: {e}")
    
    def process_excel_file(self, file_path):
        logger.info(f"Processing file: {file_path}")
        
        try:
            file_name = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(file_name)[0]
            
            try:
                logger.info(f"Reading MASTER sheet from {file_path}")
                df = pd.read_excel(file_path, sheet_name="MASTER")
                logger.info(f"Successfully read MASTER sheet with {len(df)} rows and {len(df.columns)} columns")
                logger.info(f"Columns in Excel file: {list(df.columns)}")
            except Exception as e:
                logger.error(f"Error reading MASTER sheet: {e}")
                shutil.move(file_path, os.path.join(ERROR_DIRECTORY, file_name))
                return False
            
            if df.empty:
                logger.warning(f"MASTER sheet is empty in {file_path}")
                shutil.move(file_path, os.path.join(ERROR_DIRECTORY, file_name))
                return False
            
            df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
            
            required_columns = [
                'DataAsOfDate', 'PayerMemberId', 'DateOfBirth', 
                'PayerCode', 'MarketCode', 'PDCNbr', 'ADRNbr',
                'MedAdherenceMeasureCode', 'NDCDesc'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                shutil.move(file_path, os.path.join(ERROR_DIRECTORY, file_name))
                return False
            
            processed_df = df.copy()
            
            sensitive_columns = ['PatientName', 'PatientPhoneNumber', 'PatientAddress']
            for col in sensitive_columns:
                if col in processed_df.columns:
                    logger.info(f"Removing sensitive column: {col}")
                    processed_df = processed_df.drop(columns=[col])
            
            logger.info("Creating UPID field")
            processed_df['UPID'] = processed_df.apply(
                lambda row: str(row['MarketCode']) + str(row['PayerCode']) + 
                             str(row['PayerMemberId']) + 
                             str((row['DateOfBirth']) if pd.notnull(row['DateOfBirth']) else ''),
                axis=1
            )
            
            date_columns = [
                'DataAsOfDate', 'DateOfBirth', 'InitialFillDate', 
                'LastFillDate', 'NextFillDate', 'LastImpactableDate'
            ]
            
            for col in date_columns:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_datetime(processed_df[col], errors='coerce')
            
            processed_df['data_week'] = datetime.now().date()
            
            processed_df['loaded_timestamp'] = datetime.now()
            
            logger.info(f"DataFrame has {len(processed_df.columns)} columns after processing: {list(processed_df.columns)}")
            
            csv_filename = f"{file_name_without_ext}_processed.csv"
            csv_path = os.path.join(CSV_OUTPUT_DIRECTORY, csv_filename)
            
            logger.info(f"Saving processed data to CSV: {csv_path}")
            processed_df.to_csv(csv_path, index=False)
            
            if self.client:
                try:
                    self.upload_to_bigquery(processed_df, csv_path)
                except Exception as e:
                    logger.error(f"Error uploading to BigQuery: {e}")
            
            archive_path = os.path.join(ARCHIVE_DIRECTORY, file_name)
            logger.info(f"Moving original file to archive: {archive_path}")
            shutil.move(file_path, archive_path)
            
            logger.info(f"Successfully processed file: {file_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            error_path = os.path.join(ERROR_DIRECTORY, os.path.basename(file_path))
            shutil.move(file_path, error_path)
            return False
    
    def upload_to_bigquery(self, df, csv_path):
        logger.info(f"Uploading {csv_path} to BigQuery")
        
        table_id = f"{PROJECT_ID}.{DATASET_NAME}.{TABLE_NAME}"
        
        logger.info(f"DataFrame has {len(df.columns)} columns: {list(df.columns)}")
        
        expected_columns = [
            "data_week", "DataAsOfDate", "PayerMemberId", "DateOfBirth", 
            "PayerCode", "MarketCode", "PDCNbr", "ADRNbr", 
            "InitialFillDate", "LastFillDate", "NextFillDate", 
            "LastImpactableDate", "OneFillCode", "DrugDispensedQuantityNbr", 
            "DrugDispensedDaysSupplyNbr", "MedAdherenceMeasureCode", 
            "NDCDesc", "UPID", "loaded_timestamp"
        ]
        
        upload_df = pd.DataFrame()
        
        for col in expected_columns:
            if col in df.columns:
                upload_df[col] = df[col]
            else:
                logger.warning(f"Column {col} not found in dataframe, adding as empty column")
                upload_df[col] = np.nan
        
        # Convert ADRNbr from float to integer
        if 'ADRNbr' in upload_df.columns:
            logger.info(f"Converting ADRNbr from {upload_df['ADRNbr'].dtype} to integer")
            upload_df['ADRNbr'] = pd.to_numeric(upload_df['ADRNbr'], errors='coerce').fillna(0).astype(np.int64)
            logger.info(f"Sample ADRNbr values after conversion: {upload_df['ADRNbr'].head()}")
        
        # Convert other numeric columns that might have similar issues
        if 'DrugDispensedQuantityNbr' in upload_df.columns:
            upload_df['DrugDispensedQuantityNbr'] = pd.to_numeric(upload_df['DrugDispensedQuantityNbr'], errors='coerce').fillna(0).astype(np.int64)
        
        if 'DrugDispensedDaysSupplyNbr' in upload_df.columns:
            upload_df['DrugDispensedDaysSupplyNbr'] = pd.to_numeric(upload_df['DrugDispensedDaysSupplyNbr'], errors='coerce').fillna(0).astype(np.int64)
        
        # Ensure date columns are in proper format for BigQuery
        date_columns = ['data_week', 'DataAsOfDate', 'DateOfBirth', 'InitialFillDate', 'LastFillDate', 'NextFillDate', 'LastImpactableDate']
        for col in date_columns:
            if col in upload_df.columns:
                # Handle date conversion properly - convert to string in YYYY-MM-DD format
                # First, make sure NaN values are handled
                mask = pd.isna(upload_df[col])
                upload_df.loc[~mask, col] = pd.to_datetime(upload_df.loc[~mask, col]).dt.strftime('%Y-%m-%d')
                logger.info(f"Converted {col} to date string format. Sample: {upload_df[col].head()}")
        
        # Create dictionary of data types for upload
        dtypes = {
            "data_week": "STRING",
            "DataAsOfDate": "DATE",
            "PayerMemberId": "STRING",
            "DateOfBirth": "DATE",
            "PayerCode": "STRING",
            "MarketCode": "STRING",
            "PDCNbr": "FLOAT64",
            "ADRNbr": "INT64",
            "InitialFillDate": "DATE",
            "LastFillDate": "DATE",
            "NextFillDate": "DATE",
            "LastImpactableDate": "DATE",
            "OneFillCode": "STRING",
            "DrugDispensedQuantityNbr": "INT64",
            "DrugDispensedDaysSupplyNbr": "INT64",
            "MedAdherenceMeasureCode": "STRING",
            "NDCDesc": "STRING",
            "UPID": "STRING",
            "loaded_timestamp": "TIMESTAMP"
        }
        
        # Create schema from the dtypes dictionary
        schema = [bigquery.SchemaField(col, dtype) for col, dtype in dtypes.items()]
        
        # Configure the job
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition='WRITE_APPEND',
        )
        
        # Replace NaN values with None for proper BigQuery handling
        upload_df = upload_df.replace({np.nan: None})
        
        try:
            # Load the dataframe to BigQuery
            job = self.client.load_table_from_dataframe(
                upload_df, table_id, job_config=job_config
            )
            
            # Wait for the job to complete
            job_result = job.result()
            
            logger.info(f"Loaded {job.output_rows} rows to {table_id}")
            return True
        except Exception as e:
            logger.error(f"BigQuery job failed: {e}")
            
            if 'job' in locals() and hasattr(job, 'errors') and job.errors:
                logger.error(f"Detailed errors: {job.errors}")
            
            return False


class ExcelFileEventHandler(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor
        self.processing_files = set()
    
    def on_created(self, event):
        if not event.is_directory and self._is_excel_file(event.src_path):
            self._process_after_delay(event.src_path)
    
    def _is_excel_file(self, file_path):
        return file_path.lower().endswith(('.xlsx', '.xlsm', '.xls'))
    
    def _process_after_delay(self, file_path):
        logger.info(f"New Excel file detected: {file_path}")
        
        if file_path in self.processing_files:
            logger.info(f"File {file_path} is already in the processing queue")
            return
        
        self.processing_files.add(file_path)
        
        try:
            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                try:
                    with open(file_path, 'rb') as f:
                        pass
                    break
                except PermissionError:
                    logger.info(f"File {file_path} is still being written, waiting...")
                    attempt += 1
                    time.sleep(2)
            
            self.processor.process_excel_file(file_path)
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True)
        finally:
            self.processing_files.remove(file_path)


def start_file_monitoring():
    processor = ExcelProcessor()
    event_handler = ExcelFileEventHandler(processor)
    observer = Observer()
    
    for filename in os.listdir(WATCH_DIRECTORY):
        if filename.lower().endswith(('.xlsx', '.xlsm', '.xls')):
            file_path = os.path.join(WATCH_DIRECTORY, filename)
            logger.info(f"Processing existing file: {file_path}")
            processor.process_excel_file(file_path)
    
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False)
    observer.start()
    
    logger.info(f"Started monitoring {WATCH_DIRECTORY} for Excel files")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()


if __name__ == "__main__":
    start_file_monitoring()