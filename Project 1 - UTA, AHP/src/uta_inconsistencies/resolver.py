"""Task 2.1 -- inconsistency resolution.

Finds all minimal subsets of preferences to remove so that the remaining
ones are consistent with an additive value function.
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
    """MILP: minimize number of removed preferences (sum v[k])."""
    criteria = list(directions.keys())
    char_points = compute_characteristic_points(df, directions)

    model = pulp.LpProblem("UTA_inconsistency", pulp.LpMinimize)

    u = create_marginal_value_variables(criteria)

    # v[k] = 1 means k-th preference is removed
    v = [pulp.LpVariable(f"v_{k}", cat=pulp.LpBinary) for k in range(len(preferences))]

    model += pulp.lpSum(v), "minimize_removals"

    # C1-C3, C5
    add_normalization_constraints(model, u, criteria)
    add_monotonicity_constraints(model, u, criteria)
    add_weight_bound_constraints(model, u, criteria)
    add_anti_flatness_constraints(model, u, criteria)

    # C4: U(a) >= U(b) + delta - v[k]  (relaxed when v[k]=1)
    for k, (preferred, over) in enumerate(preferences):
        u_preferred = compute_utility(preferred, df, criteria, char_points, u)
        u_over = compute_utility(over, df, criteria, char_points, u)
        model += (
            u_preferred - u_over >= DELTA - v[k],
            f"pref_{k}_{preferred}_over_{over}",
        )

    # cuts: prevent re-finding the same removal sets
    for cut_idx, removal_set in enumerate(cuts):
        # exclude this exact set: sum v[k] <= |set| - 1
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
    """Iteratively solve MILP + add cuts until infeasible."""
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

        # which preferences got removed?
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
    """Show each removal set and which preferences remain."""
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

    # show preferences
    print("Pairwise comparisons:")
    for k, (pref, over) in enumerate(preferences):
        print(f"  [{k}] {pref} ≻ {over}")

    # characteristic points
    char_points = compute_characteristic_points(df, directions)
    print(f"\nCharacteristic points (gamma={GAMMA}, {GAMMA+1} points per criterion):")
    for c in criteria:
        nature = "gain" if directions[c] == 1 else "cost"
        pts = [f"{p:.1f}" for p in char_points[c]]
        print(f"  {c} ({nature}): {' -> '.join(pts)}")

    # find all minimal removal sets
    print(f"\n{'='*70}")
    print("FINDING ALL MINIMAL REMOVAL SETS")
    print(f"{'='*70}")

    removals = find_all_minimal_removals(df, directions, preferences)
    print_results(removals, preferences)


if __name__ == "__main__":
    main()
