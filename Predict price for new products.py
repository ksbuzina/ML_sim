import pandas as pd


def fillna_with_mean(
    df: pd.DataFrame, target: str, group: str
) -> pd.DataFrame:
    ''' Predict price for new products'''

    df = df.copy()
    df[target] = df.groupby(group)[target] \
        .apply(lambda x: x.fillna(int(x.mean())))
    return df
