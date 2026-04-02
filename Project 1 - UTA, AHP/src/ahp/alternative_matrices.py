# src/ahp/alternative_matrices.py
import numpy as np
from ahp.hierarchy_setup import DIRECTIONS

THRESHOLDS = {
    "Employment rate":                   [(2,1),(4,2),(6,3),(8,4),(10,5),(12,6),(16,7),(20,8)],
    "Long-term unemployment rate":       [(0.5,1),(1.0,2),(2.0,3),(3.0,4),(4.0,5),(5.0,6),(6.0,7),(8.0,8)],
    "Personal earnings":                 [(2000,1),(5000,2),(10000,3),(15000,4),(20000,5),(25000,6),(30000,7),(35000,8)],
    "Life expectancy":                   [(0.5,1),(1.0,2),(2.0,3),(3.0,4),(4.0,5),(5.5,6),(6.5,7),(7.5,8)],
    "Life satisfaction":                 [(0.1,1),(0.2,2),(0.4,3),(0.6,4),(0.8,5),(1.0,6),(1.2,7),(1.6,8)],
    "Employees working very long hours": [(0.5,1),(1.5,2),(2.5,3),(3.5,4),(5.0,5),(6.5,6),(8.0,7),(9.5,8)],
    "Air pollution":                     [(1.0,1),(2.0,2),(4.0,3),(6.0,4),(8.0,5),(10.0,6),(13.0,7),(16.0,8)],
    "Distance from Poznan (km)":         [(100,1),(250,2),(450,3),(650,4),(900,5),(1200,6),(1600,7),(2000,8)],
}

def diff_to_score(abs_diff, thresholds):
    for upper, score in thresholds:
        if abs_diff < upper:
            return score
    return 9

def build_alternative_matrix(criterion, df):
    vals = df[criterion].values
    direction = DIRECTIONS[criterion]
    thresh = THRESHOLDS[criterion]
    n = len(vals)
    A = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            abs_diff = abs(vals[i] - vals[j])
            score = diff_to_score(abs_diff, thresh)
            signed_diff = (vals[i] - vals[j]) * direction
            if signed_diff > 0:
                A[i, j] = score
                A[j, i] = 1.0 / score
            elif signed_diff < 0:
                A[i, j] = 1.0 / score
                A[j, i] = score
    return A