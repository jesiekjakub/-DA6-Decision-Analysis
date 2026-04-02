"""Project-wide constants and file paths."""

import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREFERENCES_DIR = DATA_DIR / "preferences"
OUTPUT_DIR = DATA_DIR / "output"

DATASET_FILE = PROCESSED_DIR / "dataset.csv"
METADATA_FILE = PROCESSED_DIR / "criteria_metadata.csv"
PREFERENCES_FILE = PREFERENCES_DIR / "preferences.csv"
SELECTED_SUBSET_FILE = PREFERENCES_DIR / "selected_consistent_subset.csv"
CAPITALS_FILE = PROCESSED_DIR / "european_capitals.json"

GAMMA = 4  # segments per criterion, so gamma+1 characteristic points

# weight bounds from project spec
WEIGHT_UB = 0.5      # u_i(beta_i) <= 0.5
WEIGHT_LB = 0.0625   # u_i(beta_i) >= 1/(2n)

DELTA = 0.001  # strict preference margin: U(a) >= U(b) + delta

MIN_SEGMENT_SHARE = 0.15  # anti-flatness: each segment >= 15% of criterion weight
NON_LINEARITY_THRESHOLD = 0.25  # min diff between first and last segment (fraction of weight)
