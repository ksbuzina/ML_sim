import numpy as np
from typing import Tuple
from sklearn.neighbors import KernelDensity


def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(embeddings)

    return 1/np.exp(kde.score_samples(embeddings))


def group_diversity(embeddings: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """Calculate group diversity based on kde uniqueness.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    threshold: float :
       group diversity threshold for rejecting the group

    Returns
    -------
    Tuple[bool, float]
        reject: bool
            True if the group should be rejected, False otherwise
        group_diversity: float
            The calculated group diversity

    """
    group_diversity = np.mean(kde_uniqueness(embeddings))

    if group_diversity < threshold:
        reject = True
    else:
        reject = False

    return reject, group_diversity
