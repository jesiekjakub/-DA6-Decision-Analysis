import numpy as np

_RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
       6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

def ahp_weights(A: np.ndarray) -> dict:
    n = A.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argmax(eigenvalues.real)
    lambda_max = float(eigenvalues[idx].real)

    w = eigenvectors[:, idx].real
    w = np.abs(w)
    w = w / w.sum()

    CI = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    RI = _RI.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0

    return {
        "weights":    w,
        "lambda_max": lambda_max,
        "CI":         CI,
        "RI":         RI,
        "CR":         CR,
        "consistent": CR < 0.10,
    }