from typing import List

import numpy as np


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    sort_rel = sorted(relevance, reverse=True)

    if method == "standard":
        return (np.sum([relevance[i]/np.log2(i+2) for i in range(len(relevance[:k]))]))/(np.sum([sort_rel[i]/np.log2(i+2) for i in range(len(sort_rel[:k]))]))

    if method == "industry":
        return (np.sum([(2**relevance[i] - 1)/np.log2(i+2) for i in range(len(relevance[:k]))]))/(np.sum([(2**sort_rel[i] - 1)/np.log2(i+2) for i in range(len(sort_rel[:k]))]))

    else:
        raise ValueError
    