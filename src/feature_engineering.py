import pandas as pd
from pathlib import Path

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

INPUT_FILE = DATA_DIR / "nigeria_covid_clean.csv"
OUTPUT_FILE = DATA_DIR / "nigeria_covid_features.csv"


# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------

def create_features():

    # Load cleaned surveillance data
    df = pd.read_csv(INPUT_FILE)

    # Convert date and ensure chronological order
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # -----------------------------------------------------
    # Lag features
    # -----------------------------------------------------

    df["new_cases_lag_1"] = df["new_cases"].shift(1)
    df["new_cases_lag_7"] = df["new_cases"].shift(7)

    df["new_deaths_lag_1"] = df["new_deaths"].shift(1)
    df["new_deaths_lag_7"] = df["new_deaths"].shift(7)

    # -----------------------------------------------------
    # Rolling features
    # -----------------------------------------------------

    df["new_cases_rolling_7"] = (
        df["new_cases"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    df["new_deaths_rolling_7"] = (
        df["new_deaths"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    # -----------------------------------------------------
    # Change features
    # -----------------------------------------------------

    df["cases_change_1d"] = (
        df["new_cases"] - df["new_cases"].shift(1)
    )

    df["deaths_change_1d"] = (
        df["new_deaths"] - df["new_deaths"].shift(1)
    )

    # -----------------------------------------------------
    # Calendar features
    # -----------------------------------------------------

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    # -----------------------------------------------------
    # Prediction target
    #
    # Target = next day's reported new cases
    # -----------------------------------------------------

    df["target_next_day_cases"] = df["new_cases"].shift(-1)

    # -----------------------------------------------------
    # Remove rows that cannot have complete features/target
    # -----------------------------------------------------

    feature_columns = [
        "new_cases_lag_1",
        "new_cases_lag_7",
        "new_deaths_lag_1",
        "new_deaths_lag_7",
        "new_cases_rolling_7",
        "new_deaths_rolling_7",
        "cases_change_1d",
        "deaths_change_1d",
        "day_of_week",
        "month",
        "target_next_day_cases",
    ]

    before = len(df)

    df = df.dropna(subset=feature_columns).copy()

    after = len(df)

    # Save engineered dataset
    df.to_csv(OUTPUT_FILE, index=False)

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print("========================================")
    print("FEATURE ENGINEERING COMPLETED")
    print("========================================")
    print(f"Input records: {before}")
    print(f"Output records: {after}")
    print(f"Records removed: {before - after}")
    print(f"Features created: {len(feature_columns) - 1}")
    print("Target: target_next_day_cases")
    print(f"Output file: {OUTPUT_FILE}")
    print()
    print("Engineered columns:")
    for column in feature_columns:
        print(f"- {column}")
    print("========================================")


if __name__ == "__main__":
    create_features()
