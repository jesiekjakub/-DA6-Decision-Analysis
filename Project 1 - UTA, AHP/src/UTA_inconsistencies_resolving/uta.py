"""UTA method with inconsistency resolution.

Implements the UTA preference disaggregation method with piecewise-linear
marginal value functions. Resolves inconsistencies by finding all minimal
subsets of pairwise comparisons that need to be removed.

Formalization follows the lecture by Prof. Kadziński (slides 17-31).
"""

import pathlib
import pandas as pd
import pulp

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_FILE = DATA_DIR / "dataset.csv"
METADATA_FILE = DATA_DIR / "criteria_metadata.csv"
PREFERENCES_FILE = DATA_DIR / "preferences.csv"

# Number of linear segments per criterion (γ_i). γ+1 characteristic points.
GAMMA = 4

# Weight constraints (from project requirements)
WEIGHT_UB = 0.5      # u_i(β_i) <= 0.5 — no single criterion dominates
WEIGHT_LB = 0.0625   # u_i(β_i) >= 1/(2*n) — no criterion ignored

# Small constant to enforce strict inequality U(a) > U(b) as U(a) >= U(b) + δ
DELTA = 0.001

# Minimum share of each segment in criterion weight (anti-flatness)
MIN_SEGMENT_SHARE = 0.15

# Minimum difference between first and last segment to prevent linearity.
# Expressed as fraction of criterion weight u_i(β_i).
NON_LINEARITY_THRESHOLD = 0.25


def load_data() -> tuple[pd.DataFrame, dict[str, int]]:
    """Load dataset and criteria metadata."""
    df = pd.read_csv(DATASET_FILE)
    meta = pd.read_csv(METADATA_FILE)
    directions = {
        row["criterion"]: 1 if row["nature"] == "gain" else -1
        for _, row in meta.iterrows()
    }
    return df, directions


def load_preferences() -> list[tuple[str, str]]:
    """Load pairwise comparisons from preferences.csv."""
    pref_df = pd.read_csv(PREFERENCES_FILE)
    return [(row["preferred"], row["over"]) for _, row in pref_df.iterrows()]


def compute_characteristic_points(
    df: pd.DataFrame, directions: dict[str, int]
) -> dict[str, list[float]]:
    """Compute γ+1 equally-spaced characteristic points per criterion.

    Points are ordered from worst (α_i) to best (β_i) performance:
      - Gain criteria: [min, ..., max]
      - Cost criteria: [max, ..., min]

    Ranges are based on the FULL set of alternatives (all 26 countries).
    """
    criteria = [c for c in directions]
    char_points = {}
    for c in criteria:
        col_min = df[c].min()
        col_max = df[c].max()
        if directions[c] == 1:  # gain: worst=min, best=max
            alpha_i = col_min
            beta_i = col_max
        else:  # cost: worst=max, best=min
            alpha_i = col_max
            beta_i = col_min
        points = [
            alpha_i + j / GAMMA * (beta_i - alpha_i) for j in range(GAMMA + 1)
        ]
        char_points[c] = points
    return char_points


def interpolate_value(
    raw_val: float,
    char_points: list[float],
    u_vars: list[pulp.LpVariable],
) -> pulp.LpAffineExpression:
    """Linear interpolation of marginal value (slide 17).

    Given raw performance value, finds which segment it falls in and returns
    a PuLP expression: u_i(x_i^j) + t * [u_i(x_i^{j+1}) - u_i(x_i^j)]
    where t is the interpolation coefficient.
    """
    # Determine if points go ascending or descending
    ascending = char_points[-1] > char_points[0]

    for k in range(len(char_points) - 1):
        lo = char_points[k]
        hi = char_points[k + 1]
        lo_raw, hi_raw = (lo, hi) if ascending else (hi, lo)

        if lo_raw - 1e-9 <= raw_val <= hi_raw + 1e-9:
            span = abs(hi - lo)
            if span < 1e-12:
                return u_vars[k]
            t = abs(raw_val - lo) / span
            return u_vars[k] + t * (u_vars[k + 1] - u_vars[k])

    # Edge case: value at boundary
    if abs(raw_val - char_points[0]) < 1e-9:
        return u_vars[0]
    if abs(raw_val - char_points[-1]) < 1e-9:
        return u_vars[-1]

    raise ValueError(
        f"Value {raw_val} outside characteristic points range "
        f"[{min(char_points)}, {max(char_points)}]"
    )


def compute_utility(
    country: str,
    df: pd.DataFrame,
    criteria: list[str],
    char_points: dict[str, list[float]],
    u_vars: dict[str, list[pulp.LpVariable]],
) -> pulp.LpAffineExpression:
    """Compute U(a) = Σ_i u_i(g_i(a)) for a given country."""
    row = df[df["Country"] == country].iloc[0]
    expr = pulp.LpAffineExpression()
    for c in criteria:
        raw_val = row[c]
        expr += interpolate_value(raw_val, char_points[c], u_vars[c])
    return expr


def build_inconsistency_milp(
    df: pd.DataFrame,
    directions: dict[str, int],
    preferences: list[tuple[str, str]],
    cuts: list[frozenset],
) -> tuple[pulp.LpProblem, dict[str, list[pulp.LpVariable]], list[pulp.LpVariable]]:
    """Build MILP for inconsistency resolution (slides 29-31).

    Variables:
      u[criterion][j] — marginal value at j-th characteristic point
      v[k] ∈ {0,1}    — binary: 1 if k-th comparison is removed

    Objective: Min Σ v[k]
    """
    criteria = list(directions.keys())
    char_points = compute_characteristic_points(df, directions)

    model = pulp.LpProblem("UTA_inconsistency", pulp.LpMinimize)

    # --- Decision variables: u_i(x_i^j) ---
    u = {}
    for c in criteria:
        u[c] = [
            pulp.LpVariable(f"u_{c}_x{j}", lowBound=0)
            for j in range(GAMMA + 1)
        ]

    # --- Binary variables: v_k for each pairwise comparison ---
    v = [pulp.LpVariable(f"v_{k}", cat=pulp.LpBinary) for k in range(len(preferences))]

    # --- Objective: Min Σ v[k] ---
    model += pulp.lpSum(v), "minimize_removals"

    # --- C1: Normalization ---
    # u_i(α_i) = 0 (worst performance gets 0)
    for c in criteria:
        model += u[c][0] == 0, f"norm_worst_{c}"

    # Σ u_i(β_i) = 1 (sum of best marginal values = 1)
    model += (
        pulp.lpSum(u[c][GAMMA] for c in criteria) == 1,
        "norm_sum_best",
    )

    # --- C2: Monotonicity ---
    # Points ordered worst→best, so u[c][j+1] >= u[c][j]
    for c in criteria:
        for j in range(GAMMA):
            model += (
                u[c][j + 1] - u[c][j] >= 0,
                f"mono_{c}_seg{j}",
            )

    # --- C3: Weight bounds (project requirement) ---
    for c in criteria:
        model += u[c][GAMMA] <= WEIGHT_UB, f"weight_ub_{c}"
        model += u[c][GAMMA] >= WEIGHT_LB, f"weight_lb_{c}"

    # --- C4: Pairwise comparisons with inconsistency resolution ---
    # U(a) >= U(b) + δ - v_{a,b}
    for k, (preferred, over) in enumerate(preferences):
        u_preferred = compute_utility(preferred, df, criteria, char_points, u)
        u_over = compute_utility(over, df, criteria, char_points, u)
        model += (
            u_preferred - u_over >= DELTA - v[k],
            f"pref_{k}_{preferred}_over_{over}",
        )

    # --- C5: Anti-flatness (project requirement) ---
    # Minimum segment share
    for c in criteria:
        for j in range(GAMMA):
            model += (
                u[c][j + 1] - u[c][j] >= MIN_SEGMENT_SHARE * u[c][GAMMA],
                f"min_seg_{c}_seg{j}",
            )

    # Non-linearity: for each criterion, at least one pair of consecutive
    # segments must differ by at least NON_LINEARITY_THRESHOLD * u_i(β_i).
    # We use a binary variable d[c]: d=0 means first segment > last,
    # d=1 means last segment > first. Either way, the function is non-linear.
    BIG = 1.0  # upper bound on segment difference (utilities in [0,1])
    for c in criteria:
        seg_first = u[c][1] - u[c][0]
        seg_last = u[c][GAMMA] - u[c][GAMMA - 1]
        threshold = NON_LINEARITY_THRESHOLD * u[c][GAMMA]
        d = pulp.LpVariable(f"d_{c}", cat=pulp.LpBinary)
        # If d=0: seg_first - seg_last >= threshold
        # If d=1: seg_last - seg_first >= threshold
        model += (
            seg_first - seg_last >= threshold - BIG * d,
            f"nonlin_{c}_concave",
        )
        model += (
            seg_last - seg_first >= threshold - BIG * (1 - d),
            f"nonlin_{c}_convex",
        )

    # --- Cuts from previous iterations ---
    for cut_idx, removal_set in enumerate(cuts):
        # Σ_{k ∈ V_k} v[k] <= |V_k| - 1
        model += (
            pulp.lpSum(v[k] for k in removal_set) <= len(removal_set) - 1,
            f"cut_{cut_idx}",
        )

    return model, u, v


def find_all_minimal_removals(
    df: pd.DataFrame,
    directions: dict[str, int],
    preferences: list[tuple[str, str]],
) -> list[frozenset]:
    """Find all minimal subsets of comparisons to remove (slide 31).

    Iteratively solves MILP, adding cuts to prevent finding the same
    removal set again. Stops when MILP becomes infeasible.
    """
    cuts: list[frozenset] = []
    removals: list[frozenset] = []
    iteration = 0

    while True:
        iteration += 1
        model, u, v = build_inconsistency_milp(df, directions, preferences, cuts)

        # Try GLPK first, fall back to CBC
        try:
            solver = pulp.GLPK_CMD(msg=0)
            model.solve(solver)
        except pulp.PulpSolverError:
            solver = pulp.PULP_CBC_CMD(msg=0)
            model.solve(solver)

        if model.status != pulp.constants.LpStatusOptimal:
            print(f"\nIteration {iteration}: Infeasible — all minimal removal sets found.")
            break

        # Extract removal set V_k = {k : v[k] = 1}
        removal_set = frozenset(
            k for k in range(len(preferences)) if v[k].varValue > 0.5
        )
        obj_val = pulp.value(model.objective)

        print(f"Iteration {iteration}: V* = {int(obj_val)}, "
              f"removed = {sorted(removal_set)}")

        removals.append(removal_set)
        cuts.append(removal_set)

    return removals


def print_results(
    removals: list[frozenset],
    preferences: list[tuple[str, str]],
) -> None:
    """Print all minimal removal sets and corresponding consistent subsets."""
    print(f"\n{'='*70}")
    print(f"RESULTS: Found {len(removals)} minimal removal set(s)")
    print(f"{'='*70}")

    for idx, removal_set in enumerate(removals):
        print(f"\n--- Removal set {idx + 1} ---")
        print("Removed comparisons:")
        for k in sorted(removal_set):
            pref, over = preferences[k]
            print(f"  [{k}] {pref} ≻ {over}")

        print("Consistent subset (kept comparisons):")
        for k in range(len(preferences)):
            if k not in removal_set:
                pref, over = preferences[k]
                print(f"  [{k}] {pref} ≻ {over}")


def main() -> None:
    print("Loading data...")
    df, directions = load_data()
    preferences = load_preferences()
    criteria = list(directions.keys())

    print(f"Dataset: {len(df)} alternatives, {len(criteria)} criteria")
    print(f"Preferences: {len(preferences)} pairwise comparisons")
    print()

    # Print preferences
    print("Pairwise comparisons:")
    for k, (pref, over) in enumerate(preferences):
        print(f"  [{k}] {pref} ≻ {over}")

    # Print characteristic points
    char_points = compute_characteristic_points(df, directions)
    print(f"\nCharacteristic points (γ={GAMMA}, {GAMMA+1} points per criterion):")
    for c in criteria:
        nature = "gain" if directions[c] == 1 else "cost"
        pts = [f"{p:.1f}" for p in char_points[c]]
        print(f"  {c} ({nature}): {' → '.join(pts)}")

    # Find all minimal removal sets
    print(f"\n{'='*70}")
    print("FINDING ALL MINIMAL REMOVAL SETS")
    print(f"{'='*70}")

    removals = find_all_minimal_removals(df, directions, preferences)
    print_results(removals, preferences)


if __name__ == "__main__":
    main()
