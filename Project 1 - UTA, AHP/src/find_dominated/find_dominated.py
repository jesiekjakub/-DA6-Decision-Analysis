"""Pareto dominance check."""

import pandas as pd

from common.data_loading import load_data


def find_dominated_pairs(df: pd.DataFrame, directions: dict[str, int]) -> list[tuple[str, str]]:
    criteria = list(directions.keys())
    pairs = []
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            dominates = True
            strictly_better = False
            for c in criteria:
                vi = df.loc[i, c] * directions[c]
                vj = df.loc[j, c] * directions[c]
                if vi < vj:
                    dominates = False
                    break
                if vi > vj:
                    strictly_better = True
            if dominates and strictly_better:
                pairs.append((df.loc[i, "Country"], df.loc[j, "Country"]))
    return pairs


def main() -> None:
    df, directions = load_data()
    pairs = find_dominated_pairs(df, directions)

    if not pairs:
        print("No dominated alternatives found.")
        return

    print(f"Found {len(pairs)} dominance relationships:\n")
    for dominator, dominated in sorted(pairs):
        print(f"  {dominator} dominates {dominated}")

    # per-country summary
    dominated_counts: dict[str, list[str]] = {}
    for dominator, dominated in pairs:
        dominated_counts.setdefault(dominated, []).append(dominator)

    print(f"\nDominated alternatives ({len(dominated_counts)}):")
    for country, dominators in sorted(dominated_counts.items(), key=lambda x: -len(x[1])):
        print(f"  {country} — dominated by {len(dominators)}: {', '.join(sorted(dominators))}")


if __name__ == "__main__":
    main()
