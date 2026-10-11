




from datetime import datetime
from pathlib import Path
import sys
import subprocess
import sqlite3
import hashlib
from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
SOURCE_FILE = DATA_DIR / "owid_covid_data.csv"
RAW_FILE = DATA_DIR / "nigeria_covid_raw.csv"
CLEAN_FILE = DATA_DIR / "nigeria_covid_clean.csv"
QUALITY_REPORT = DATA_DIR / "data_quality_report.csv"
DATABASE_FILE = DATA_DIR / "surveillance.db"


def extract_nigeria():
    df = pd.read_csv(SOURCE_FILE)
    nigeria_df = df[df["country"] == "Nigeria"].copy()
    nigeria_df.to_csv(RAW_FILE, index=False)
    print(f"Nigeria records extracted: {len(nigeria_df)}")


def validate_data():
    df = pd.read_csv(RAW_FILE)

    required_columns = [
        "country",
        "date",
        "total_cases",
        "new_cases",
        "total_deaths",
        "new_deaths",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Required column missing: {column}")

    duplicate_dates = df["date"].duplicated().sum()

    if duplicate_dates > 0:
        raise ValueError(f"Duplicate dates detected: {duplicate_dates}")

    print("Required columns: OK")
    print(f"Duplicate dates: {duplicate_dates}")

    for column in ["total_cases", "new_cases", "total_deaths", "new_deaths"]:
        print(f"{column} missing values: {df[column].isna().sum()}")

    print("Data validation completed successfully.")


def clean_data():
    df = pd.read_csv(RAW_FILE)

    df["date"] = pd.to_datetime(df["date"])

    required_fields = [
        "total_cases",
        "new_cases",
        "total_deaths",
        "new_deaths",
    ]

    clean_df = df.dropna(subset=required_fields).copy()

    clean_df.to_csv(CLEAN_FILE, index=False)

    print(f"Raw records: {len(df)}")
    print(f"Clean records: {len(clean_df)}")
    print(f"Records removed: {len(df) - len(clean_df)}")
    print(f"Clean dataset saved to: {CLEAN_FILE}")


def load_to_database():
    df = pd.read_csv(CLEAN_FILE)

    connection = sqlite3.connect(DATABASE_FILE)

    df.to_sql(
        "nigeria_covid_surveillance",
        connection,
        if_exists="replace",
        index=False,
    )

    connection.close()
def run_analysis():
    subprocess.run(
        [sys.executable, str(PROJECT_DIR / "analysis.py")],
        check=True,
    )
  
    print(f"Database saved to: {DATABASE_FILE}")
    print("Table created: nigeria_covid_surveillance")
    
    
def run_feature_engineering():
    """Create the engineered feature dataset required by the time-series experiment."""
    subprocess.run(
        [sys.executable, str(PROJECT_DIR / "src" / "feature_engineering.py")],
        check=True,
    )


def predictive_modeling():
    """
    Run the recovered five-fold time-series tuning experiment and report its output.

    This task creates rf_tuning_grid.csv and rf_tuning_final.csv. The experiment
    uses TimeSeriesSplit(n_splits=5, gap=7), selecting parameters on folds 1-3
    and reporting folds 4-5 as holdout folds.
    """
    tuning_script = PROJECT_DIR / "scripts" / "tune_rf_timeseries.py"
    subprocess.run([sys.executable, str(tuning_script)], check=True)

    model_results = DATA_DIR / "rf_tuning_final.csv"
    if not model_results.exists():
        raise FileNotFoundError(
            f"Expected modelling output was not created: {model_results}"
        )

    df = pd.read_csv(model_results)
    print("=" * 80)
    print("TIME-SERIES PREDICTIVE MODELLING RESULTS")
    print(f"Results file: {model_results}")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

def data_quality_assessment():
    df = pd.read_csv(CLEAN_FILE)

    required_columns = [
        "country",
        "date",
        "total_cases",
        "new_cases",
        "total_deaths",
        "new_deaths",
    ]

    missing_required_columns = [
        column for column in required_columns if column not in df.columns
    ]

    missing_values = df[
        ["total_cases", "new_cases", "total_deaths", "new_deaths"]
    ].isna().sum().sum()

    duplicate_dates = df["date"].duplicated().sum()
    nigeria_records = (df["country"] == "Nigeria").sum()

    report = pd.DataFrame(
        {
            "quality_check": [
                "Total records",
                "Missing critical values",
                "Duplicate dates",
                "Missing required columns",
                "Nigeria records",
                "Earliest date",
                "Latest date",
            ],
            "result": [
                len(df),
                missing_values,
                duplicate_dates,
                len(missing_required_columns),
                nigeria_records,
                df["date"].min(),
                df["date"].max(),
            ],
        }
    )

    report.to_csv(QUALITY_REPORT, index=False)

    with open(CLEAN_FILE, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()

    print(f"Final dataset MD5: {md5}")
    print("Data Quality Assessment completed.")
    print(report.to_string(index=False))
    print(f"Quality report saved to: {QUALITY_REPORT}")


with DAG(
    dag_id="group3_surveillance_pipeline",
    start_date=datetime(2026, 9, 25),
    schedule=None,
    catchup=False,
) as dag:

    extract = PythonOperator(
        task_id="extract_nigeria",
        python_callable=extract_nigeria,
    )

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    clean = PythonOperator(
        task_id="clean_data",
        python_callable=clean_data,
    )

    load = PythonOperator(
        task_id="load_to_database",
        python_callable=load_to_database,
    )
    quality = PythonOperator(
        task_id="data_quality_assessment",
        python_callable=data_quality_assessment,
    )

    analysis = PythonOperator(
        task_id="analysis",
        python_callable=run_analysis,
    )
    
    features = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_feature_engineering,
    )

    predictive = PythonOperator(
        task_id="predictive_modeling",
        python_callable=predictive_modeling,
    )
    extract >> validate >> clean >> load >> quality >> analysis >> features >> predictive
