from typing import List
import pandas as pd
from sklearn.preprocessing import TargetEncoder

def feature_engineering(df: pd.DataFrame, cols2encode: List[str]) -> pd.DataFrame:
    """Simple feature engineering."""

    cols2larger_zero = ["mta_tax", "tolls_amount", "airport_fee",]
    df[cols2larger_zero] = (df[cols2larger_zero] > 0) * 1

    df[cols2encode] = TargetEncoder().fit_transform(df[cols2encode], df["total_amount"])

    return df