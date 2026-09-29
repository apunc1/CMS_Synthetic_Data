from pathlib import Path
from typing import Iterator, Optional
from zipfile import ZipFile

import pandas as pd

from src.config import RAW_DATA_DIR


SUPPORTED_SUFFIXES = {".csv", ".parquet", ".zip"}


def discover_data_files(directory: Path = RAW_DATA_DIR) -> list[Path]:
    """Return supported local data files without reading their contents."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def load_table(
    path: Path, *, delimiter: str = "|", zip_member: Optional[str] = None
) -> pd.DataFrame:
    """Load an explicitly selected CMS CSV, ZIP member, or Parquet table."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, sep=delimiter, dtype=str)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".zip":
        with ZipFile(path) as archive:
            csv_members = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]
            if zip_member is None:
                if len(csv_members) != 1:
                    raise ValueError(
                        f"Expected one CSV member in {path.name}; found {len(csv_members)}."
                    )
                zip_member = csv_members[0]
            elif zip_member not in csv_members:
                raise ValueError(f"CSV member {zip_member!r} not found in {path.name}.")
            with archive.open(zip_member) as member:
                return pd.read_csv(member, sep=delimiter, dtype=str)
    raise ValueError(f"Unsupported data format: {suffix}")


def iter_table_chunks(
    path: Path,
    *,
    chunksize: int = 50_000,
    delimiter: str = "|",
    zip_member: Optional[str] = None,
) -> Iterator[pd.DataFrame]:
    """Yield text-preserving CSV chunks from a CSV or a ZIP archive."""
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")
    if path.suffix.lower() == ".csv":
        yield from pd.read_csv(
            path,
            sep=delimiter,
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
        )
        return
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            csv_members = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]
            if zip_member is None:
                if len(csv_members) != 1:
                    raise ValueError(
                        f"Expected one CSV member in {path.name}; found {len(csv_members)}."
                    )
                zip_member = csv_members[0]
            elif zip_member not in csv_members:
                raise ValueError(f"CSV member {zip_member!r} not found in {path.name}.")
            with archive.open(zip_member) as member:
                yield from pd.read_csv(
                    member,
                    sep=delimiter,
                    dtype=str,
                    keep_default_na=False,
                    chunksize=chunksize,
                )
        return
    raise ValueError(f"Chunked input must be CSV or ZIP: {path.suffix}")