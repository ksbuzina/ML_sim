import numpy as np


def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    """metric Lifetime Value"""

    diff = y_true - y_pred
    mask = diff >= 0
    error_1 = np.mean(np.abs(diff[mask])) if sum(mask) > 0 else 0
    error_2 = np.mean(np.abs(100*diff[~mask])) if sum(~mask) > 0 else 0
    error = 0.5*(error_1 + error_2)

    return error
