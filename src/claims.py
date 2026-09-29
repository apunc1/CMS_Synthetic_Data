from collections.abc import Iterable

import pandas as pd


def validate_claim_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    """Raise a helpful error when expected source columns are absent."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required claim columns: {', '.join(missing)}")