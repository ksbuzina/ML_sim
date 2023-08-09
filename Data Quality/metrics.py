"""Metrics."""

from typing import Any, Dict, Union, List
from dataclasses import dataclass
import datetime
from datetime import datetime
from scipy.stats import norm

import pandas as pd


@dataclass
class Metric:
    """Base class for Metric"""

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}


@dataclass
class CountTotal(Metric):
    """Total number of rows in DataFrame"""

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"total": len(df)}


@dataclass
class CountZeros(Metric):
    """Number of zeros in choosen column"""

    column: str

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == 0)
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountNull(Metric):
    """Number of empty values in chosen columns"""

    columns: List[str]
    aggregation: str = "any"  # either "all", or "any"

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.aggregation == "all":
            k = sum(df[self.columns].isnull().all(axis=1))
        elif self.aggregation == "any":
            k = sum(df[self.columns].isnull().any(axis=1))
        else:
            raise ValueError(
                "Invalid aggregation type. Must be either 'all' or 'any'.")
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountDuplicates(Metric):
    """Number of duplicates in chosen columns"""

    columns: List[str]

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df.duplicated(subset=self.columns))
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountValue(Metric):
    """Number of values in chosen column"""

    column: str
    value: Union[str, int, float]

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == self.value)
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowValue(Metric):
    """Number of values below threshold"""

    column: str
    value: float
    strict: bool = False

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = sum(df[self.column] < self.value)
        else:
            k = sum(df[self.column] <= self.value)
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowColumn(Metric):
    """Count how often column X below Y"""

    column_x: str
    column_y: str
    strict: bool = False

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = sum(df[self.column_x] < df[self.column_y])
        else:
            k = sum(df[self.column_x] <= df[self.column_y])
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountRatioBelow(Metric):
    """Count how often X / Y below Z"""

    column_x: str
    column_y: str
    column_z: str
    strict: bool = False

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        if self.strict:
            k = sum(df[self.column_x] / df[self.column_y] < df[self.column_z])
        else:
            k = sum(df[self.column_x] / df[self.column_y] <= df[self.column_z])
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountCB(Metric):
    """Calculate lower/upper bounds for N%-confidence interval"""

    column: str
    conf: float = 0.95

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        values = df[self.column] 
        # Вычисление нижней границы доверительного интервала
        lcb = values.quantile((1 - self.conf) / 2)
        # Вычисление верхней границы доверительного интервала
        ucb = values.quantile(1 - (1 - self.conf) / 2)
    
        return {"lcb": lcb, "ucb": ucb}


@dataclass
class CountLag(Metric):
    """A lag between latest date and today"""

    column: str
    fmt: str = "%Y-%m-%d"

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        
        dates = pd.to_datetime(df[self.column], format=self.fmt)
        today = pd.to_datetime("today")  # Текущая дата

        # Находим последнюю дату
        last_day = dates.max()

        # Вычисляем разницу между последней датой и текущим днем
        lag = (today - last_day).days

        today = today.strftime(self.fmt)
        last_day = last_day.strftime(self.fmt)

        return {"today": today, "last_day": last_day, "lag": lag}
