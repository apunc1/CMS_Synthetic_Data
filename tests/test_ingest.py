from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from src.ingest import load_source, source_type_for, standardize_columns


def test_source_type_for_beneficiary_years_and_archives() -> None:
    assert source_type_for(Path("beneficiary_2019.csv")) == "beneficiary"
    assert source_type_for(Path("Carrier.zip")) == "carrier"
    assert source_type_for(Path("Outpatient.zip")) == "outpatient"


def test_standardize_carrier_hcpcs_fields() -> None:
    frame = pd.DataFrame(
        {
            "BENE_ID": ["b1"],
            "CLM_ID": ["c1"],
            "HCPCS_CD": ["A1234"],
            "LINE_PLACE_OF_SRVC_CD": ["11"],
        }
    )

    result = standardize_columns(frame, "carrier")

    assert result.loc[0, "bene_id"] == "b1"
    assert result.loc[0, "claim_id"] == "c1"
    assert result.loc[0, "hcpcs_code"] == "A1234"
    assert result.loc[0, "place_of_service_code"] == "11"


def test_standardize_columns_rejects_missing_required_source_fields() -> None:
    with pytest.raises(ValueError, match="Missing required carrier columns: CLM_ID"):
        standardize_columns(pd.DataFrame({"BENE_ID": ["b1"]}), "carrier")


def test_standardize_inpatient_code_families() -> None:
    frame = pd.DataFrame(
        {
            "BENE_ID": ["b1"],
            "CLM_ID": ["c1"],
            "HCPCS_CD": ["A1234"],
            "ICD_DGNS_CD25": ["Z9999"],
            "ICD_PRCDR_CD25": ["0W3P8ZZ"],
        }
    )

    result = standardize_columns(frame, "inpatient")

    assert result.loc[0, "hcpcs_code"] == "A1234"
    assert result.loc[0, "diagnosis_code_25"] == "Z9999"
    assert result.loc[0, "icd10_pcs_procedure_code_25"] == "0W3P8ZZ"


def test_load_source_reads_verified_carrier_archive(tmp_path: Path) -> None:
    path = tmp_path / "Carrier.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "carrier.csv",
            "BENE_ID|CLM_ID|HCPCS_CD|LINE_PLACE_OF_SRVC_CD\n"
            "b1|c1|A1234|01\n",
        )

    result = load_source(path)

    assert result.loc[0, "hcpcs_code"] == "A1234"
    assert result.loc[0, "place_of_service_code"] == "01"