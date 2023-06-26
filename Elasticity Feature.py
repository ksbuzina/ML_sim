import pandas as pd
from scipy import stats as st
from sklearn.metrics import r2_score


def elasticity_df(df: pd.DataFrame) -> pd.DataFrame:
    ''' Функция на основе исторических данных оценивает эластичность (спроса по цене) '''

    df = df.copy()
    df['qty_log'] = np.log1p(df.qty)
    result = pd.DataFrame()
    result['elasticity'] = df.groupby('sku')[['price', 'qty_log']] \
        .apply(lambda x: (st.linregress(x['price'], x['qty_log']).rvalue**2))

    return result.reset_index(inplace=False)
