"""Shared UTA model building functions.

Contains characteristic points computation, interpolation, utility calculation,
and common constraint builders used by both the inconsistency resolver and
the discrimination solver.
"""

import pandas as pd
import pulp

from .config import GAMMA, WEIGHT_UB, WEIGHT_LB, MIN_SEGMENT_SHARE, NON_LINEARITY_THRESHOLD


def compute_characteristic_points(
    df: pd.DataFrame, directions: dict[str, int]
) -> dict[str, list[float]]:
    """Compute gamma+1 equally-spaced characteristic points per criterion.

    Points are ordered from worst (alpha_i) to best (beta_i) performance:
      - Gain criteria: [min, ..., max]
      - Cost criteria: [max, ..., min]

    Ranges are based on the FULL set of alternatives (all 26 countries).
    """
    char_points = {}
    for c in directions:
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
    """Linear interpolation of marginal value (lecture slide 17).

    Given raw performance value, finds which segment it falls in and returns
    a PuLP expression: u_i(x_i^j) + t * [u_i(x_i^{j+1}) - u_i(x_i^j)]
    where t is the interpolation coefficient.
    """
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
    """Compute U(a) = sum_i u_i(g_i(a)) for a given country."""
    row = df[df["Country"] == country].iloc[0]
    expr = pulp.LpAffineExpression()
    for c in criteria:
        raw_val = row[c]
        expr += interpolate_value(raw_val, char_points[c], u_vars[c])
    return expr


def create_marginal_value_variables(
    criteria: list[str],
) -> dict[str, list[pulp.LpVariable]]:
    """Create u_i(x_i^j) decision variables for all criteria."""
    u = {}
    for c in criteria:
        u[c] = [
            pulp.LpVariable(f"u_{c}_x{j}", lowBound=0)
            for j in range(GAMMA + 1)
        ]
    return u


def add_normalization_constraints(
    model: pulp.LpProblem,
    u: dict[str, list[pulp.LpVariable]],
    criteria: list[str],
) -> None:
    """C1: Normalization — u_i(alpha_i)=0, sum u_i(beta_i)=1."""
    for c in criteria:
        model += u[c][0] == 0, f"norm_worst_{c}"
    model += (
        pulp.lpSum(u[c][GAMMA] for c in criteria) == 1,
        "norm_sum_best",
    )


def add_monotonicity_constraints(
    model: pulp.LpProblem,
    u: dict[str, list[pulp.LpVariable]],
    criteria: list[str],
) -> None:
    """C2: Monotonicity — u[c][j+1] >= u[c][j]."""
    for c in criteria:
        for j in range(GAMMA):
            model += (
                u[c][j + 1] - u[c][j] >= 0,
                f"mono_{c}_seg{j}",
            )


def add_weight_bound_constraints(
    model: pulp.LpProblem,
    u: dict[str, list[pulp.LpVariable]],
    criteria: list[str],
) -> None:
    """C3: Weight bounds — u_i(beta_i) in [WEIGHT_LB, WEIGHT_UB]."""
    for c in criteria:
        model += u[c][GAMMA] <= WEIGHT_UB, f"weight_ub_{c}"
        model += u[c][GAMMA] >= WEIGHT_LB, f"weight_lb_{c}"


def add_anti_flatness_constraints(
    model: pulp.LpProblem,
    u: dict[str, list[pulp.LpVariable]],
    criteria: list[str],
) -> None:
    """C5: Anti-flatness — minimum segment share + non-linearity."""
    # Minimum segment share
    for c in criteria:
        for j in range(GAMMA):
            model += (
                u[c][j + 1] - u[c][j] >= MIN_SEGMENT_SHARE * u[c][GAMMA],
                f"min_seg_{c}_seg{j}",
            )

    # Non-linearity: at least one pair of consecutive segments must differ
    BIG = 1.0
    for c in criteria:
        seg_first = u[c][1] - u[c][0]
        seg_last = u[c][GAMMA] - u[c][GAMMA - 1]
        threshold = NON_LINEARITY_THRESHOLD * u[c][GAMMA]
        d = pulp.LpVariable(f"d_{c}", cat=pulp.LpBinary)
        model += (
            seg_first - seg_last >= threshold - BIG * d,
            f"nonlin_{c}_concave",
        )
        model += (
            seg_last - seg_first >= threshold - BIG * (1 - d),
            f"nonlin_{c}_convex",
        )


def solve_model(model: pulp.LpProblem) -> int:
    """Solve model with GLPK, falling back to CBC. Returns status code."""
    try:
        solver = pulp.GLPK_CMD(msg=0)
        model.solve(solver)
    except pulp.PulpSolverError:
        solver = pulp.PULP_CBC_CMD(msg=0)
        model.solve(solver)
    return model.status
