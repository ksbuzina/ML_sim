import numpy as np


def smape(y_true: np.array, y_pred: np.array) -> float:
    """
    The symmetric Mean Absolute Percentage Error (sMAPE) metric using only NumPy

    :param y_true: The ground truth labels given in the dataset
    :param y_pred: Our predictions
    :return: The sMAPE score
    """
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator == 0
    denominator[mask] = 1
    return np.mean(2 * np.abs(np.array(y_true) - np.array(y_pred)) / denominator)
