from sklearn.neighbors import KernelDensity
import numpy as np


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
