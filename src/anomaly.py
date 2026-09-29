from typing import Union

import pandas as pd
from sklearn.ensemble import IsolationForest


def fit_anomaly_model(
    features: pd.DataFrame, contamination: Union[str, float] = "auto"
) -> tuple[IsolationForest, pd.Series]:
    """Fit a reproducible Isolation Forest to complete numeric features."""
    numeric = features.select_dtypes(include="number").dropna()
    if numeric.empty:
        raise ValueError("At least one complete numeric feature row is required.")
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(numeric)
    labels = pd.Series(model.predict(numeric), index=numeric.index, name="anomaly_label")
    return model, labels