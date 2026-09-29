import pandas as pd


def summarize_procedures(frame: pd.DataFrame, procedure_column: str) -> pd.DataFrame:
    """Count claims by procedure code, preserving missing-code rows."""
    if procedure_column not in frame.columns:
        raise KeyError(f"Missing procedure column: {procedure_column}")
    return (
        frame.groupby(procedure_column, dropna=False)
        .size()
        .rename("claim_count")
        .reset_index()
        .sort_values("claim_count", ascending=False)
        .reset_index(drop=True)
    )