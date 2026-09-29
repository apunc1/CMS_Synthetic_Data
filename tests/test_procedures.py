import pandas as pd
import pytest

from src.procedures import summarize_procedures


def test_summarize_procedures_counts_codes() -> None:
    frame = pd.DataFrame({"code": ["A", "B", "A"]})
    result = summarize_procedures(frame, "code")
    assert result.to_dict("records") == [
        {"code": "A", "claim_count": 2},
        {"code": "B", "claim_count": 1},
    ]


def test_summarize_procedures_requires_column() -> None:
    with pytest.raises(KeyError, match="Missing procedure column: code"):
        summarize_procedures(pd.DataFrame(), "code")