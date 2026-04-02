# src/ahp/global_weights.py
from ahp.hierarchy_setup import CATEGORIES, CATEGORY_CRITERIA

def compute_global_weights(w_goal, w_categories):
    """
    w_goal       — weight vector for the 3 categories (from Goal matrix)
    w_categories — dict: category name -> weight vector for its criteria
    """
    global_w = {}
    for cat_idx, cat in enumerate(CATEGORIES):
        for crit_idx, crit in enumerate(CATEGORY_CRITERIA[cat]):
            global_w[crit] = w_goal[cat_idx] * w_categories[cat][crit_idx]

    # renormalize to sum exactly to 1
    total = sum(global_w.values())
    return {c: v / total for c, v in global_w.items()}