"""UTA 2.2 -- Most discriminant value function.

For the largest consistent subset of preferences (identified in task 2.1),
finds the additive value function that maximizes the minimum utility gap
between alternatives in a preference relationship.
"""

import pandas as pd
import pulp
import matplotlib.pyplot as plt

from common.config import GAMMA, OUTPUT_DIR
from common.data_loading import load_data, load_preferences, load_removal_indices
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


def get_consistent_preferences(
    preferences: list[tuple[str, str]],
    removal_indices: set[int],
) -> list[tuple[int, tuple[str, str]]]:
    """Return preferences with removed indices excluded.

    Returns list of (original_index, (preferred, over)) tuples.
    """
    return [
        (i, pref) for i, pref in enumerate(preferences) if i not in removal_indices
    ]


def build_discrimination_model(
    df: pd.DataFrame,
    directions: dict[str, int],
    consistent_prefs: list[tuple[int, tuple[str, str]]],
) -> tuple[pulp.LpProblem, dict[str, list[pulp.LpVariable]], pulp.LpVariable]:
    """Build LP/MILP maximizing discrimination threshold epsilon.

    Objective: maximize epsilon
    C4: U(a) - U(b) >= epsilon for each kept preference (no binary v variables)
    C1-C3, C5: same as task 2.1
    """
    criteria = list(directions.keys())
    char_points = compute_characteristic_points(df, directions)

    model = pulp.LpProblem("UTA_discrimination", pulp.LpMaximize)

    # --- Decision variables ---
    u = create_marginal_value_variables(criteria)

    # --- Epsilon: discrimination threshold to maximize ---
    epsilon = pulp.LpVariable("epsilon", lowBound=0)

    # --- Objective: maximize epsilon ---
    model += epsilon, "maximize_discrimination"

    # --- Constraints C1-C3, C5 (shared) ---
    add_normalization_constraints(model, u, criteria)
    add_monotonicity_constraints(model, u, criteria)
    add_weight_bound_constraints(model, u, criteria)
    add_anti_flatness_constraints(model, u, criteria)

    # --- C4: Preference constraints (hard, with epsilon) ---
    for orig_k, (preferred, over) in consistent_prefs:
        u_pref = compute_utility(preferred, df, criteria, char_points, u)
        u_over = compute_utility(over, df, criteria, char_points, u)
        model += (
            u_pref - u_over >= epsilon,
            f"pref_{orig_k}_{preferred}_over_{over}",
        )

    return model, u, epsilon


def print_model_details(
    model: pulp.LpProblem,
    u: dict[str, list[pulp.LpVariable]],
    epsilon: pulp.LpVariable,
    criteria: list[str],
    char_points: dict[str, list[float]],
) -> None:
    """Print all model equations, variable values, and objective value."""
    print(f"\n{'='*70}")
    print("MODEL EQUATIONS")
    print(f"{'='*70}")
    print(f"\nObjective: Maximize epsilon")
    print(f"\nConstraints ({len(model.constraints)}):")
    for name, constraint in model.constraints.items():
        print(f"  {name}: {constraint}")

    print(f"\n{'='*70}")
    print("VARIABLE VALUES")
    print(f"{'='*70}")

    print(f"\nepsilon* = {epsilon.varValue:.6f}")

    print(f"\nMarginal values u_i(x_i^j):")
    for c in criteria:
        pts = char_points[c]
        vals = [u[c][j].varValue for j in range(GAMMA + 1)]
        print(f"\n  {c}:")
        for j in range(GAMMA + 1):
            label = "worst" if j == 0 else ("best" if j == GAMMA else "")
            print(f"    u({pts[j]:.1f}) = {vals[j]:.6f}  {label}")

    print(f"\nCriterion weights (u_i(beta_i)):")
    for c in criteria:
        w = u[c][GAMMA].varValue
        print(f"  {c}: {w:.6f}")

    print(f"\n{'='*70}")
    print(f"OBJECTIVE FUNCTION VALUE: epsilon* = {epsilon.varValue:.6f}")
    print(f"{'='*70}")


def rank_alternatives(
    df: pd.DataFrame,
    directions: dict[str, int],
    u: dict[str, list[pulp.LpVariable]],
    criteria: list[str],
    char_points: dict[str, list[float]],
) -> pd.DataFrame:
    """Compute U(a) for all countries and return ranked DataFrame."""
    results = []
    for _, row in df.iterrows():
        country = row["Country"]
        utility_expr = compute_utility(country, df, criteria, char_points, u)
        utility_val = pulp.value(utility_expr)
        results.append({"Country": country, "U(a)": round(utility_val, 6)})

    ranking = pd.DataFrame(results).sort_values("U(a)", ascending=False).reset_index(drop=True)
    ranking.index += 1
    ranking.index.name = "Rank"
    return ranking


def plot_marginal_value_functions(
    u: dict[str, list[pulp.LpVariable]],
    criteria: list[str],
    char_points: dict[str, list[float]],
    directions: dict[str, int],
) -> plt.Figure:
    """Plot piecewise-linear marginal value functions for all criteria.

    All plots scaled to the same maximum Y value. Returns the figure.
    """
    max_weight = max(u[c][GAMMA].varValue for c in criteria)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, c in enumerate(criteria):
        ax = axes[idx]
        pts = char_points[c]
        vals = [u[c][j].varValue for j in range(GAMMA + 1)]

        # For cost criteria, characteristic points go from worst (high) to best (low).
        # Plot in natural raw value order (ascending) for readability.
        if directions[c] == -1:
            x_plot = list(reversed(pts))
            y_plot = list(reversed(vals))
        else:
            x_plot = pts
            y_plot = vals

        ax.plot(x_plot, y_plot, "o-", linewidth=2, markersize=6, color="steelblue")
        ax.set_ylim(0, max_weight * 1.1)
        ax.set_title(c, fontsize=10, fontweight="bold")
        ax.set_xlabel("Raw value", fontsize=9)
        ax.set_ylabel("Marginal value", fontsize=9)
        ax.grid(True, alpha=0.3)

        nature = "gain" if directions[c] == 1 else "cost"
        ax.text(0.02, 0.95, f"({nature})", transform=ax.transAxes,
                fontsize=8, verticalalignment="top", color="gray")

    plt.tight_layout()
    return fig


def main() -> None:
    print("Loading data...")
    df, directions = load_data()
    preferences = load_preferences()
    criteria = list(directions.keys())

    print(f"Dataset: {len(df)} alternatives, {len(criteria)} criteria")
    print(f"Total preferences: {len(preferences)}")

    # Load removal choice and build consistent subset
    removal_indices = load_removal_indices()
    print(f"Removed preference indices: {sorted(removal_indices)}")
    for idx in sorted(removal_indices):
        print(f"  [{idx}] {preferences[idx][0]} ≻ {preferences[idx][1]}")

    consistent_prefs = get_consistent_preferences(preferences, removal_indices)
    print(f"Consistent subset: {len(consistent_prefs)} preferences")

    # Build and solve
    char_points = compute_characteristic_points(df, directions)
    model, u, epsilon = build_discrimination_model(df, directions, consistent_prefs)

    solve_model(model)

    print(f"\nSolver status: {pulp.LpStatus[model.status]}")

    if model.status != pulp.constants.LpStatusOptimal:
        print("ERROR: Model is not optimal. Check constraints.")
        return

    # Print full model details
    print_model_details(model, u, epsilon, criteria, char_points)

    # Rank all alternatives
    ranking = rank_alternatives(df, directions, u, criteria, char_points)
    print(f"\n{'='*70}")
    print("RANKING OF ALL ALTERNATIVES")
    print(f"{'='*70}")
    print(ranking.to_string())

    # Plot (saved to file when run as script)
    fig = plot_marginal_value_functions(u, criteria, char_points, directions)
    output_path = OUTPUT_DIR / "marginal_value_functions.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
