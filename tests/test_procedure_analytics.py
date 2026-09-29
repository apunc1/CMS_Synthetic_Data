from pathlib import Path

import pandas as pd

from src.procedure_analytics import (
    build_procedure_analytics,
    get_procedure_codes,
    get_procedure_details,
)


def test_build_procedure_analytics_uses_distinct_claims_and_source_scopes(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    derived = tmp_path / "derived"
    processed.mkdir()
    pd.DataFrame(
        {
            "claim_id": ["C1", "C1", "C2"],
            "bene_id": ["b1", "b1", "b2"],
            "hcpcs_code": ["A123", "A123", "A123"],
            "claim_from_date": pd.to_datetime(
                ["2015-01-01", "2015-01-01", "2015-02-01"]
            ),
            "rendering_provider_npi": ["n1", "n1", "n2"],
            "organization_npi": ["", "", ""],
            "place_of_service_code": ["11", "11", "22"],
            "line_allowed_charge": [10.0, 10.0, 30.0],
            "diagnosis_code_1": ["D1", "D1", "D2"],
        }
    ).to_parquet(processed / "carrier.parquet", index=False)
    pd.DataFrame(
        {
            "claim_id": ["O1"],
            "bene_id": ["b3"],
            "hcpcs_code": ["A123"],
            "claim_from_date": pd.to_datetime(["2015-03-01"]),
            "rendering_provider_npi": ["n1"],
            "attending_provider_npi": [""],
            "organization_npi": [""],
            "facility_provider_id": [""],
            "revenue_center_rate": [999.0],
            "diagnosis_code_1": ["D2"],
        }
    ).to_parquet(processed / "outpatient.parquet", index=False)
    pd.DataFrame(
        {
            "claim_id": ["I1"],
            "bene_id": ["b3"],
            "hcpcs_code": ["A123"],
            "claim_from_date": pd.to_datetime(["2015-04-01"]),
            "attending_provider_npi": [""],
            "operating_provider_npi": [""],
            "organization_npi": [""],
            "facility_provider_id": ["H1"],
            "diagnosis_code_1": ["D1"],
        }
    ).to_parquet(processed / "inpatient.parquet", index=False)

    row_counts = build_procedure_analytics(processed, derived)

    stats = pd.read_parquet(derived / "procedure_stats.parquet").iloc[0]
    providers = pd.read_parquet(derived / "provider_procedure_stats.parquet")
    places = pd.read_parquet(derived / "pos_procedure_stats.parquet")
    diagnoses = pd.read_parquet(derived / "diagnosis_procedure_stats.parquet")
    trends = pd.read_parquet(derived / "procedure_trends.parquet")

    assert row_counts["procedure_stats.parquet"] == 1
    assert stats.claim_count == 4
    assert stats.unique_beneficiaries == 3
    assert stats.unique_providers == 3
    assert stats.total_allowed == 50.0
    assert stats.allowed_amount_claims == 3
    assert stats.unique_diagnoses == 2
    assert stats.unique_places_of_service == 2
    assert "CMS_PROVIDER_NUMBER:H1" in set(providers.provider_id)
    assert set(places.place_of_service_code) == {"11", "22"}
    assert set(diagnoses.diagnosis_code) == {"D1", "D2"}
    assert trends.claim_count.sum() == 4

    codes = get_procedure_codes(derived)
    details = get_procedure_details("A123", derived)

    assert codes is not None
    assert codes.iloc[0]["procedure_code"] == "A123"
    assert details["summary"]["claim_count"] == 4
    assert len(details["diagnoses"]) == 2