from typing import List

import numpy as np


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = None) -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values​​
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if method == "standard":
        return sum([relevance[i]/np.log2(i+2) for i in range(len(relevance[:k]))])

    if method == "industry":
        return sum([(2**relevance[i] - 1)/np.log2(i+2) for i in range(len(relevance[:k]))])

    else:
        raise ValueError
    