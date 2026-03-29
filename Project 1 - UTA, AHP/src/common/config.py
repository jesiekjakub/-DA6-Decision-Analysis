"""Shared constants and paths for the UTA project."""

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

# Number of linear segments per criterion (gamma_i). gamma+1 characteristic points.
GAMMA = 4

# Weight constraints (from project requirements)
WEIGHT_UB = 0.5      # u_i(beta_i) <= 0.5 — no single criterion dominates
WEIGHT_LB = 0.0625   # u_i(beta_i) >= 1/(2*n) — no criterion ignored

# Small constant to enforce strict inequality U(a) > U(b) as U(a) >= U(b) + delta
DELTA = 0.001

# Minimum share of each segment in criterion weight (anti-flatness)
MIN_SEGMENT_SHARE = 0.15

# Minimum difference between first and last segment to prevent linearity.
# Expressed as fraction of criterion weight u_i(beta_i).
NON_LINEARITY_THRESHOLD = 0.25
