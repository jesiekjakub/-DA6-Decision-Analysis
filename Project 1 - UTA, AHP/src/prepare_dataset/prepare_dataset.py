"""Prepare MCDA dataset from OECD Better Life Index raw data.

Transforms the long-format OECD BLI CSV into a wide-format dataset
suitable for UTA and AHP analysis. Filters to European countries,
selects specific indicators as criteria, and adds distance from Poznan.
"""

import json
import math
import sys

import pandas as pd

from common.config import RAW_DIR, CAPITALS_FILE, DATASET_FILE


POZNAN_LAT = 52.4064
POZNAN_LON = 16.9252

SELECTED_INDICATORS = [
    "Employment rate",
    "Long-term unemployment rate",
    "Personal earnings",
    "Life expectancy",
    "Life satisfaction",
    "Employees working very long hours",
    "Air pollution",
]

EUROPEAN_COUNTRIES = [
    "Austria", "Belgium", "Czech Republic", "Denmark", "Estonia",
    "Finland", "France", "Germany", "Greece", "Hungary", "Iceland",
    "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
    "Netherlands", "Norway", "Poland", "Portugal", "Slovak Republic",
    "Slovenia", "Spain", "Sweden", "Switzerland",
    "United Kingdom",
]


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in km between two points on Earth."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_raw_csv():
    """Find the raw OECD BLI CSV file in data/raw/."""
    csvs = list(RAW_DIR.glob("*.csv"))
    if not csvs:
        print("ERROR: No CSV file found in data/raw/")
        print("Please download the dataset from:")
        print("  https://www.kaggle.com/datasets/joebeachcapital/oecd-better-life-index")
        print(f"and place the CSV file in: {RAW_DIR}")
        sys.exit(1)
    if len(csvs) > 1:
        print(f"WARNING: Multiple CSV files found in data/raw/: {[c.name for c in csvs]}")
        print(f"Using: {csvs[0].name}")
    return csvs[0]


def explore_dataset(df: pd.DataFrame) -> None:
    """Print diagnostic information about the raw dataset."""
    print(f"\n{'='*60}")
    print("RAW DATASET EXPLORATION")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if "Indicator" in df.columns:
        indicators = sorted(df["Indicator"].unique())
        print(f"\nAll indicators ({len(indicators)}):")
        for i, ind in enumerate(indicators, 1):
            marker = " <-- SELECTED" if ind in SELECTED_INDICATORS else ""
            print(f"  {i:2d}. {ind}{marker}")

    if "Country" in df.columns:
        countries = sorted(df["Country"].unique())
        print(f"\nAll countries ({len(countries)}):")
        for c in countries:
            marker = " <-- EUROPEAN" if c in EUROPEAN_COUNTRIES else ""
            print(f"  - {c}{marker}")

    # Show columns that might be used for filtering aggregation level
    for col in ["Inequality", "INEQUALITY", "Measure", "MEASURE"]:
        if col in df.columns:
            print(f"\nUnique values in '{col}': {sorted(df[col].unique())}")


def main() -> None:
    raw_csv = find_raw_csv()
    print(f"Reading: {raw_csv.name}")
    df = pd.read_csv(raw_csv)

    explore_dataset(df)

    # --- Filter European countries ---
    if "Country" not in df.columns:
        print("ERROR: 'Country' column not found. Available columns:", list(df.columns))
        sys.exit(1)

    df_europe = df[df["Country"].isin(EUROPEAN_COUNTRIES)]
    found_countries = set(df_europe["Country"].unique())
    missing = set(EUROPEAN_COUNTRIES) - found_countries
    if missing:
        print(f"\nWARNING: These European countries are NOT in the dataset: {sorted(missing)}")
    print(f"\nEuropean countries found: {len(found_countries)}")

    # --- Filter selected indicators ---
    if "Indicator" not in df.columns:
        print("ERROR: 'Indicator' column not found.")
        sys.exit(1)

    available_indicators = set(df["Indicator"].unique())
    missing_indicators = set(SELECTED_INDICATORS) - available_indicators
    if missing_indicators:
        print(f"\nWARNING: These indicators are NOT in the dataset: {sorted(missing_indicators)}")
        print("Available indicators containing similar keywords:")
        for mi in missing_indicators:
            keyword = mi.split()[0].lower()
            matches = [i for i in available_indicators if keyword in i.lower()]
            print(f"  '{mi}' -> possible matches: {matches}")

    df_filtered = df_europe[df_europe["Indicator"].isin(SELECTED_INDICATORS)]

    # --- Filter to "Total" aggregation (avoid gender/inequality splits) ---
    inequality_col = None
    for col in ["Inequality", "INEQUALITY"]:
        if col in df_filtered.columns:
            inequality_col = col
            break

    if inequality_col:
        unique_vals = df_filtered[inequality_col].unique()
        print(f"\n'{inequality_col}' values in filtered data: {sorted(unique_vals)}")
        if "Total" in unique_vals:
            df_filtered = df_filtered[df_filtered[inequality_col] == "Total"]
        elif "TOT" in unique_vals:
            df_filtered = df_filtered[df_filtered[inequality_col] == "TOT"]
        else:
            print(f"WARNING: No 'Total' or 'TOT' value found in '{inequality_col}'. Using all rows.")

    # --- Pivot long to wide ---
    df_wide = df_filtered.pivot_table(
        index="Country",
        columns="Indicator",
        values="Value",
        aggfunc="first",
    ).reset_index()

    print(f"\n{'='*60}")
    print("PIVOTED DATASET")
    print(f"{'='*60}")
    print(f"Shape: {df_wide.shape}")
    print(f"Columns: {list(df_wide.columns)}")

    # --- Handle missing values ---
    nan_counts = df_wide[SELECTED_INDICATORS].isna().sum()
    if nan_counts.any():
        print(f"\nMissing values per indicator:\n{nan_counts[nan_counts > 0]}")
        rows_with_nan = df_wide[SELECTED_INDICATORS].isna().any(axis=1)
        countries_with_nan = df_wide.loc[rows_with_nan, "Country"].tolist()
        print(f"Countries with missing data: {countries_with_nan}")

        # Drop rows with any NaN first
        df_clean = df_wide.dropna(subset=SELECTED_INDICATORS)
        if len(df_clean) >= 12:
            print(f"After dropping NaN rows: {len(df_clean)} countries remain (>= 12, OK)")
            df_wide = df_clean
        else:
            print(f"Only {len(df_clean)} countries after dropna (< 12). Imputing with median.")
            for col in SELECTED_INDICATORS:
                median_val = df_wide[col].median()
                filled = df_wide[col].isna().sum()
                if filled > 0:
                    df_wide[col] = df_wide[col].fillna(median_val)
                    print(f"  Filled {filled} NaN in '{col}' with median={median_val:.2f}")

    # --- Compute distances from Poznan ---
    with open(CAPITALS_FILE) as f:
        capitals = json.load(f)

    distances = []
    for country in df_wide["Country"]:
        if country not in capitals:
            print(f"ERROR: '{country}' not found in european_capitals.json")
            sys.exit(1)
        cap = capitals[country]
        dist = haversine(POZNAN_LAT, POZNAN_LON, cap["lat"], cap["lon"])
        distances.append(round(dist, 1))

    df_wide["Distance from Poznan (km)"] = distances

    # --- Reorder columns ---
    final_columns = ["Country"] + SELECTED_INDICATORS + ["Distance from Poznan (km)"]
    df_wide = df_wide[final_columns]
    df_wide = df_wide.sort_values("Country").reset_index(drop=True)

    # --- Validation ---
    n_alternatives = len(df_wide)
    n_criteria = len(final_columns) - 1  # exclude Country
    assert 12 <= n_alternatives <= 50, f"Alternatives count {n_alternatives} outside 12-50 range"
    assert 4 <= n_criteria <= 9, f"Criteria count {n_criteria} outside 4-9 range"
    assert df_wide[final_columns[1:]].isna().sum().sum() == 0, "NaN values remain in dataset"

    print(f"\n{'='*60}")
    print("FINAL DATASET")
    print(f"{'='*60}")
    print(f"Alternatives: {n_alternatives}")
    print(f"Criteria: {n_criteria}")
    print(f"\n{df_wide.to_string(index=False)}")

    # --- Save ---
    df_wide.to_csv(DATASET_FILE, index=False)
    print(f"\nDataset saved to: {DATASET_FILE}")


if __name__ == "__main__":
    main()
