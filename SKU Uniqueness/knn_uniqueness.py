import numpy as np


def knn_uniqueness(embeddings: np.ndarray, num_neighbors: int) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on knn.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group 
    num_neighbors: int :
        number of neighbors to estimate uniqueness    

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    n = embeddings.shape[0]
    distances = np.zeros((n, n))

    # Calculate pairwise distances between embeddings
    for i in range(n):
        for j in range(n):
            distances[i, j] = np.sqrt(np.sum((embeddings[j] - embeddings[i])**2))

    # Sort distances and select the nearest neighbors
    sorted_indices = np.argsort(distances, axis=1)
    nearest_neighbors = sorted_indices[:, 1:num_neighbors+1]

    # Calculate uniqueness estimates
    uniqueness = np.zeros(n)
    for i in range(n):
        neighbor_distances = distances[i, nearest_neighbors[i]]
        uniqueness[i] = np.mean(neighbor_distances)

    return uniqueness
