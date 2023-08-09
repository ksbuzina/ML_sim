from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseSelector(ABC):
    threshold: float = 0.5

    @classmethod
    @abstractmethod
    def fit(cls, X, y) -> BaseSelector:
        # Correlation between features and target
        pass

    def transform(self, X):
        return X[self.selected_features_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    @property
    def n_features_(self):
        return len(self.selected_features_)

    @property
    def original_features_(self):
        return self.original_features

    @property
    def selected_features_(self):
        return self.selected_features


@dataclass
class VarianceSelector(BaseSelector):
    min_var: float = 0.4
    original_features: list[str] = None
    selected_features: list[str] = None

    def fit(self, X, y=None) -> VarianceSelector:
        variances = np.var(X, axis=0)
        self.original_features = X.columns.tolist()
        self.selected_features = X.columns[variances > self.min_var].tolist()
        return self


@dataclass
class SpearmanSelector(BaseSelector):
    threshold: float = 0.5
    original_features: list[str] = None
    selected_features: list[str] = None

    def fit(self, X, y) -> SpearmanSelector:
        corr = pd.concat([X, y], axis=1).corr(method="spearman")
        corr_target = corr.iloc[:-1, -1]
        self.original_features = X.columns.tolist()
        self.selected_features = corr_target[
            abs(corr_target) >= self.threshold
        ].index.tolist()
        return self


@dataclass
class PearsonSelector(BaseSelector):
    threshold: float = 0.5
    original_features: list[str] = None
    selected_features: list[str] = None

    def fit(self, X, y) -> PearsonSelector:
        # Correlation between features and target
        corr = pd.concat([X, y], axis=1).corr(method="pearson")
        corr_target = corr.iloc[:-1, -1]
        self.original_features = X.columns.tolist()
        self.selected_features = corr_target[
            abs(corr_target) >= self.threshold
        ].index.tolist()
        return self
