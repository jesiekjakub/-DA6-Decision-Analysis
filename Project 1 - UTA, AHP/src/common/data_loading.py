"""Loading functions for dataset, preferences, etc."""

import pandas as pd

from .config import DATASET_FILE, METADATA_FILE, PREFERENCES_FILE, SELECTED_SUBSET_FILE


def load_data() -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(DATASET_FILE)
    meta = pd.read_csv(METADATA_FILE)
    directions = {
        row["criterion"]: 1 if row["nature"] == "gain" else -1
        for _, row in meta.iterrows()
    }
    return df, directions


def load_preferences() -> list[tuple[str, str]]:
    pref_df = pd.read_csv(PREFERENCES_FILE)
    return [(row["preferred"], row["over"]) for _, row in pref_df.iterrows()]


def load_removal_indices() -> set[int]:
    """Reads which preference indices were selected for removal (task 2.1 output)."""
    df = pd.read_csv(SELECTED_SUBSET_FILE)
    return set(df["removed_index"].tolist())
