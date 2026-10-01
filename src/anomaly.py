from pathlib import Path
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


def build_provider_anomaly_table(
    derived_dir: Path | str = Path("data/derived"),
) -> pd.DataFrame:
    """Score provider-procedure combinations against their peer group using utilization ratios."""
    path = Path(derived_dir) / "provider_procedure_stats.parquet"
    if not path.is_file():
        raise FileNotFoundError("Missing provider procedure table: provider_procedure_stats.parquet")

    frame = pd.read_parquet(path).copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "provider_id",
                "provider_id_type",
                "procedure_code",
                "claim_count",
                "beneficiary_count",
                "provider_rate",
                "peer_median",
                "provider_to_peer_ratio",
                "peer_percentile",
                "anomaly_score",
                "anomaly_flag",
            ]
        )

    rows: list[dict] = []
    for _, row in frame.iterrows():
        procedure_code = row["procedure_code"]
        provider_id = row["provider_id"]
        procedure_frame = frame[frame["procedure_code"] == procedure_code].copy()
        peer_rates = pd.to_numeric(procedure_frame["procedure_rate"], errors="coerce").dropna()
        if provider_id in procedure_frame["provider_id"].values:
            peer_rates = peer_rates[procedure_frame["provider_id"].values != provider_id]
        peer_rates = peer_rates.dropna()

        provider_rate = pd.to_numeric(row["procedure_rate"], errors="coerce")
        peer_median = (
            float(peer_rates.median())
            if not peer_rates.empty
            else float(provider_rate)
            if pd.notna(provider_rate)
            else None
        )
        if pd.notna(provider_rate) and peer_median not in (None, 0):
            ratio = provider_rate / peer_median
        else:
            ratio = None

        if pd.notna(provider_rate) and not peer_rates.empty:
            percentile = (peer_rates <= provider_rate).mean() * 100.0
        elif pd.notna(provider_rate):
            percentile = 100.0
        else:
            percentile = None

        rows.append(
            {
                "provider_id": provider_id,
                "provider_id_type": row.get("provider_id_type"),
                "procedure_code": procedure_code,
                "claim_count": row.get("claim_count"),
                "beneficiary_count": row.get("beneficiary_count"),
                "provider_rate": provider_rate,
                "peer_median": peer_median,
                "provider_to_peer_ratio": ratio,
                "peer_percentile": percentile,
                "anomaly_score": None,
                "anomaly_flag": False,
            }
        )

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    feature_columns = [
        "provider_rate",
        "peer_median",
        "provider_to_peer_ratio",
        "peer_percentile",
        "claim_count",
    ]
    features = result[feature_columns].copy().fillna(0)
    if len(features) >= 2:
        model, labels = fit_anomaly_model(features)
        result["anomaly_score"] = -model.score_samples(features)
        result["anomaly_flag"] = labels.to_numpy() == -1
    else:
        result["anomaly_score"] = 0.0
        result["anomaly_flag"] = False

    result = result.sort_values(
        ["anomaly_score", "provider_to_peer_ratio"], ascending=[False, False]
    ).reset_index(drop=True)
    return result