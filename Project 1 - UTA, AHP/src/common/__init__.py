"""Re-exports for the common package."""

from .config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    PREFERENCES_DIR,
    OUTPUT_DIR,
    DATASET_FILE,
    METADATA_FILE,
    PREFERENCES_FILE,
    SELECTED_SUBSET_FILE,
    GAMMA,
    WEIGHT_UB,
    WEIGHT_LB,
    DELTA,
    MIN_SEGMENT_SHARE,
    NON_LINEARITY_THRESHOLD,
)
from .data_loading import load_data, load_preferences, load_removal_indices
from .uta_core import (
    compute_characteristic_points,
    interpolate_value,
    compute_utility,
    create_marginal_value_variables,
    add_normalization_constraints,
    add_monotonicity_constraints,
    add_weight_bound_constraints,
    add_anti_flatness_constraints,
    solve_model,
)
