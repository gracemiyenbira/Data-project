import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "surveillance.db"

# Connect to database
conn = sqlite3.connect(DB_PATH)

# Load surveillance data
df = pd.read_sql_query(
    """
    SELECT date, total_cases, new_cases, total_deaths, new_deaths
    FROM nigeria_covid_surveillance
    ORDER BY date
    """,
    conn
)

conn.close()

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# ---------------------------------------------------------
# 1. Overall surveillance summary
# ---------------------------------------------------------

summary = pd.DataFrame({
    "metric": [
        "Total clean records",
        "Earliest date",
        "Latest date",
        "Cumulative confirmed cases",
        "Cumulative reported deaths",
        "Peak daily new cases",
        "Date of peak daily cases",
        "Peak daily new deaths",
        "Date of peak daily deaths",
        "Case fatality ratio (%)"
    ],
    "value": [
        len(df),
        df["date"].min().date(),
        df["date"].max().date(),
        df["total_cases"].max(),
        df["total_deaths"].max(),
        df["new_cases"].max(),
        df.loc[df["new_cases"].idxmax(), "date"].date(),
        df["new_deaths"].max(),
        df.loc[df["new_deaths"].idxmax(), "date"].date(),
        round(
            df["total_deaths"].max()
            / df["total_cases"].max()
            * 100,
            2
        )
    ]
})

summary.to_csv(
    DATA_DIR / "surveillance_summary.csv",
    index=False
)

# ---------------------------------------------------------
# 2. Yearly surveillance summary
# ---------------------------------------------------------

df["year"] = df["date"].dt.year

yearly = (
    df.groupby("year")
    .agg(
        total_new_cases=("new_cases", "sum"),
        total_new_deaths=("new_deaths", "sum")
    )
    .reset_index()
)

yearly.to_csv(
    DATA_DIR / "yearly_surveillance_summary.csv",
    index=False
)

# ---------------------------------------------------------
# 3. Daily new cases visualization
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["new_cases"])
plt.title("Daily Reported COVID-19 Cases in Nigeria")
plt.xlabel("Date")
plt.ylabel("New Cases")
plt.tight_layout()

plt.savefig(
    DATA_DIR / "daily_new_cases.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# 4. Daily new deaths visualization
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["new_deaths"])
plt.title("Daily Reported COVID-19 Deaths in Nigeria")
plt.xlabel("Date")
plt.ylabel("New Deaths")
plt.tight_layout()

plt.savefig(
    DATA_DIR / "daily_new_deaths.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# 5. Annual cases visualization
# ---------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.bar(
    yearly["year"].astype(str),
    yearly["total_new_cases"]
)

plt.title("Annual Reported COVID-19 Cases in Nigeria")
plt.xlabel("Year")
plt.ylabel("New Cases")
plt.tight_layout()

plt.savefig(
    DATA_DIR / "annual_new_cases.png",
    dpi=300
)

plt.close()

# ---------------------------------------------------------
# Completion message
# ---------------------------------------------------------

print("========================================")
print("SURVEILLANCE ANALYSIS COMPLETED")
print("========================================")
print(f"Records analysed: {len(df)}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Cumulative cases: {df['total_cases'].max():,.0f}")
print(f"Cumulative deaths: {df['total_deaths'].max():,.0f}")
print(f"Peak daily cases: {df['new_cases'].max():,.0f}")
print(f"Peak daily deaths: {df['new_deaths'].max():,.0f}")
print()
print("Files created:")
print("- surveillance_summary.csv")
print("- yearly_surveillance_summary.csv")
print("- daily_new_cases.png")
print("- daily_new_deaths.png")
print("- annual_new_cases.png")
print("========================================")
