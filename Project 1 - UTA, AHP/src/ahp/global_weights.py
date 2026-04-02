from ahp.hierarchy_setup import CATEGORIES, CATEGORY_CRITERIA

def compute_global_weights(w_goal, w_categories):
    # Compute global criterion weights by multiplying category weights with criterion weights within each category
    global_w = {}
    for cat_idx, cat in enumerate(CATEGORIES):
        for crit_idx, crit in enumerate(CATEGORY_CRITERIA[cat]):
            global_w[crit] = w_goal[cat_idx] * w_categories[cat][crit_idx]

    # Normalize to sum to 1
    total = sum(global_w.values())
    return {c: v / total for c, v in global_w.items()}