import numpy as np
from ahp.hierarchy_setup import CRITERIA
from ahp.weights import ahp_weights
from ahp.alternative_matrices import build_alternative_matrix

def compute_ahp_scores(df, global_weights):
    # Aggregated scores for each alternative by computing weighted sums of criterion scores
    n = len(df)
    scores = np.zeros(n)

    for crit in CRITERIA:
        A = build_alternative_matrix(crit, df)
        w = ahp_weights(A)["weights"]
        scores += global_weights[crit] * w

    return scores