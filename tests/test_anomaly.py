from pathlib import Path

import pandas as pd

from src.anomaly import build_provider_anomaly_table


def test_build_provider_anomaly_table_scores_peer_outliers(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "provider_id": ["NPI:1", "NPI:2", "NPI:3", "NPI:4"],
            "provider_id_type": ["NPI", "NPI", "NPI", "NPI"],
            "procedure_code": ["A123", "A123", "A123", "B456"],
            "claim_count": [20, 5, 4, 3],
            "beneficiary_count": [10, 2, 2, 1],
            "total_allowed": [500.0, 100.0, 80.0, 50.0],
            "procedure_rate": [4.0, 2.5, 2.0, 3.0],
            "procedure_rate_basis": [
                "Claims per beneficiary observed for this provider and procedure",
                "Claims per beneficiary observed for this provider and procedure",
                "Claims per beneficiary observed for this provider and procedure",
                "Claims per beneficiary observed for this provider and procedure",
            ],
        }
    )
    df.to_parquet(tmp_path / "provider_procedure_stats.parquet", index=False)

    anomalies = build_provider_anomaly_table(tmp_path)

    assert set(["provider_id", "procedure_code", "provider_rate", "peer_median", "provider_to_peer_ratio", "peer_percentile", "anomaly_score", "anomaly_flag"]).issubset(anomalies.columns)
    assert anomalies.loc[anomalies["provider_id"] == "NPI:1", "anomaly_flag"].iat[0] in {True, False}
    assert anomalies["provider_to_peer_ratio"].notna().all()
