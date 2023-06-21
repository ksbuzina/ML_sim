import numpy as np


def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    """
    The Root Mean Squared Log Error (RMSLE) metric using only NumPy

    :param y_true: The ground truth labels given in the dataset
    :param y_pred: Our predictions
    :return: The RMSLE score
    """
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))
