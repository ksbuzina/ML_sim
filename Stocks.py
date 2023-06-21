import pandas as pd
import numpy as np


def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # проверяем, что gmv/price <= stock, если нет, то ограничиваем gmv
    mask1 = df.gmv > df.stock * df.price
    df.loc[mask1, 'gmv'] = df.loc[mask1, 'stock'] * df.loc[mask1, 'price']

    # проверяем, что gmv/price > gmv/price (целочисленное деление)
    mask2 = df.gmv % df.price != 0
    df.loc[mask2, 'gmv'] = (df.loc[mask2, 'gmv'] //
                            df.loc[mask2, 'price']) * df.loc[mask2, 'price']

    return df
