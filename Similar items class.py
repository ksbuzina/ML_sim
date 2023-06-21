from typing import Dict
from typing import List
from typing import Tuple
import itertools

import numpy as np
from scipy.spatial.distance import cosine


class SimilarItems:
    """Similar items class"""

    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """
        pair_sims = {}
        for k1, k2 in list(map(dict, itertools.combinations(embeddings.items(), 2))):
            pair_sims[(k1, k2)] = round(
                1 - cosine(embeddings[k1], embeddings[k2]), 8)
        return pair_sims

    @staticmethod
    def knn(
        sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.
        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.
        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        knn_dict = {}
        sim = dict(sorted(sim.items(), key=lambda elem: elem[1], reverse=True))

        for (node1, node2), weight in sim.items():
            if node1 not in knn_dict:
                knn_dict[node1] = []
            if node2 not in knn_dict:
                knn_dict[node2] = []
            if len(knn_dict[node1]) < top:
                knn_dict[node1].append((node2, weight))
            if len(knn_dict[node2]) < top:
                knn_dict[node2].append((node1, weight))

        knn_dict = dict(sorted(knn_dict.items(), key=lambda elem: elem[0]))

        return knn_dict

    @staticmethod
    def knn_price(
        knn_dict: Dict[int, List[Tuple[int, float]]],
        prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """

        knn_price_dict = {}
        for key, value in knn_dict.items():
            weighted_sum = 0.0
            weight_sum = 0.0
            for tuple_item in value:
                product_id, distance = tuple_item
                weighted_sum += prices[product_id] * (distance + 1)
                weight_sum += (distance + 1)
            new_price = weighted_sum / weight_sum
            knn_price_dict[key] = round(new_price, 2)

        return knn_price_dict

    @staticmethod
    def transform(
        embeddings: Dict[int, np.ndarray],
        prices: Dict[int, float],
        top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        knn_price_dict = SimilarItems.knn_price(SimilarItems.knn(
            SimilarItems.similarity(embeddings), top), prices)
        return knn_price_dict
