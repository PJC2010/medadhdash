# medadhdash

Med Adherence Dashboard
This project is a Streamlit web application that visualizes medication adherence data from a Google BigQuery database. It's designed to help track and analyze gaps in medication adherence for patients with specific health conditions.

📜 Description
The Medication Adherence Dashboard provides an interactive interface to explore medication adherence metrics. It tracks gaps in medication adherence for patients taking medications for cholesterol, diabetes, and hypertension. The dashboard allows users to filter data by week, measure type, market, and payer to gain insights into adherence trends and identify areas for improvement.

✨ Features
Interactive Dashboard: A user-friendly interface built with Streamlit.

KPI Metrics: Key Performance Indicators such as:

Total Gaps (UGIDs)

Unique Patients (UPIDs)

One-Fill Gaps

Denominator Gaps

Data Filtering: Filter data by:

Week

Measure Type (MAC, MAH, MAD)

Market Code

Payer Code

Visualizations:

Distribution of adherence gaps by measure type

PDC (Proportion of Days Covered) statistics and distribution

Month-over-month trend analysis with CMS Star Rating thresholds

Analysis by market and payer

📊 Data Source
The application connects to a Google BigQuery table named medadhdata2025.adherence_tracking.weekly_med_adherence_data. You will need access to this BigQuery table and the appropriate GCP credentials to run the application.

🚀 Getting Started
Prerequisites
Python 3.11 or later

Access to the Google Cloud Platform (GCP) and the BigQuery dataset.

A GCP service account with BigQuery read access.

Installation
Clone the repository:

Bash

git clone <repository-url>
cd medadhdash
Install the required Python packages:

Bash

pip install -r requirements.txt
Configuration
GCP Credentials:

The application uses a config.toml file to load GCP credentials. You will need the JSON key file for your GCP service account.

Open the config.toml file.

Copy the contents of your GCP service account JSON key file and paste them into a [gcp] section in config.toml. It should look something like this:

Ini, TOML

[gcp]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account-email"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
Running the Application
Using Streamlit:

Once you have configured your credentials, you can run the application using Streamlit:

Bash

streamlit run dashboard.py
The application will be available at http://localhost:8501.

Using the Dev Container:

This project includes a development container, which simplifies setup. If you have Docker and the "Dev Containers" extension for VS Code installed, you can use it.

Open the project folder in VS Code.

When prompted, click "Reopen in Container".

The dev container will automatically install all dependencies and start the Streamlit application.

📁 Project Structure
.
├── .devcontainer
│   └── devcontainer.json  # Dev container configuration
├── .gitignore             # Files to ignore in git
├── config.toml            # Configuration for GCP credentials
├── dashboard.py           # The main Streamlit application script
├── README.md              # This file
└── requirements.txt       # Python package dependencies
