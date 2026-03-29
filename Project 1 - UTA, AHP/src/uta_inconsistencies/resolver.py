"""UTA method with inconsistency resolution.

Implements the UTA preference disaggregation method with piecewise-linear
marginal value functions. Resolves inconsistencies by finding all minimal
subsets of pairwise comparisons that need to be removed.

Formalization follows the lecture by Prof. Kadzinski (slides 17-31).
"""

import pandas as pd
import pulp

from common.config import GAMMA, DELTA
from common.data_loading import load_data, load_preferences
from common.uta_core import (
    compute_characteristic_points,
    compute_utility,
    create_marginal_value_variables,
    add_normalization_constraints,
    add_monotonicity_constraints,
    add_weight_bound_constraints,
    add_anti_flatness_constraints,
    solve_model,
)


def build_inconsistency_milp(
    df: pd.DataFrame,
    directions: dict[str, int],
    preferences: list[tuple[str, str]],
    cuts: list[frozenset],
) -> tuple[pulp.LpProblem, dict[str, list[pulp.LpVariable]], list[pulp.LpVariable]]:
    """Build MILP for inconsistency resolution (slides 29-31).

    Variables:
      u[criterion][j] -- marginal value at j-th characteristic point
      v[k] in {0,1}   -- binary: 1 if k-th comparison is removed

    Objective: Min sum v[k]
    """
    criteria = list(directions.keys())
    char_points = compute_characteristic_points(df, directions)

    model = pulp.LpProblem("UTA_inconsistency", pulp.LpMinimize)

    # --- Decision variables ---
    u = create_marginal_value_variables(criteria)

    # --- Binary variables: v_k for each pairwise comparison ---
    v = [pulp.LpVariable(f"v_{k}", cat=pulp.LpBinary) for k in range(len(preferences))]

    # --- Objective: Min sum v[k] ---
    model += pulp.lpSum(v), "minimize_removals"

    # --- Constraints C1-C3, C5 (shared) ---
    add_normalization_constraints(model, u, criteria)
    add_monotonicity_constraints(model, u, criteria)
    add_weight_bound_constraints(model, u, criteria)
    add_anti_flatness_constraints(model, u, criteria)

    # --- C4: Pairwise comparisons with inconsistency resolution ---
    # U(a) >= U(b) + delta - v_{a,b}
    for k, (preferred, over) in enumerate(preferences):
        u_preferred = compute_utility(preferred, df, criteria, char_points, u)
        u_over = compute_utility(over, df, criteria, char_points, u)
        model += (
            u_preferred - u_over >= DELTA - v[k],
            f"pref_{k}_{preferred}_over_{over}",
        )

    # --- Cuts from previous iterations ---
    for cut_idx, removal_set in enumerate(cuts):
        # sum_{k in V_k} v[k] <= |V_k| - 1
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

        solve_model(model)

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
    print(f"\nCharacteristic points (gamma={GAMMA}, {GAMMA+1} points per criterion):")
    for c in criteria:
        nature = "gain" if directions[c] == 1 else "cost"
        pts = [f"{p:.1f}" for p in char_points[c]]
        print(f"  {c} ({nature}): {' -> '.join(pts)}")

    # Find all minimal removal sets
    print(f"\n{'='*70}")
    print("FINDING ALL MINIMAL REMOVAL SETS")
    print(f"{'='*70}")

    removals = find_all_minimal_removals(df, directions, preferences)
    print_results(removals, preferences)


if __name__ == "__main__":
    main()
