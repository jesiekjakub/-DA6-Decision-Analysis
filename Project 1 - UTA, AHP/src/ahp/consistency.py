import numpy as np

def reconstruct_matrix(weights):
    # Reconstruct the pairwise comparison matrix from the weight vector
    return np.outer(weights, 1.0 / weights)

def max_discrepancy(A_orig, A_rec, labels):
    # Find the largest absolute difference between A_orig and A_rec for discovering the most inconsistent pair
    n = A_orig.shape[0]
    best = {"diff": -1.0, "i": 0, "j": 0}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = abs(A_orig[i,j] - A_rec[i,j])
            if d > best["diff"]:
                best = {"diff": d, "i": i, "j": j}
    i, j = best["i"], best["j"]
    return {
        "row_label": labels[i],
        "col_label": labels[j],
        "dm_value":  A_orig[i, j],
        "rec_value": A_rec[i, j],
        "diff":      best["diff"],
    }