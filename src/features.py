import pandas as pd


def build_numeric_features(
    frame: pd.DataFrame, group_column: str, measure_columns: list[str]
) -> pd.DataFrame:
    """Aggregate selected numeric measures by a caller-mapped group column."""
    missing = {group_column, *measure_columns} - set(frame.columns)
    if missing:
        raise KeyError(f"Missing feature columns: {', '.join(sorted(missing))}")
    return frame.groupby(group_column)[measure_columns].mean().reset_index()