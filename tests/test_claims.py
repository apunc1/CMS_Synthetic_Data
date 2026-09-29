import pandas as pd
import pytest

from src.claims import validate_claim_columns


def test_validate_claim_columns_accepts_required_columns() -> None:
    validate_claim_columns(pd.DataFrame(columns=["procedure", "date"]), ["procedure"])


def test_validate_claim_columns_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing required claim columns: procedure"):
        validate_claim_columns(pd.DataFrame(columns=["date"]), ["procedure"])