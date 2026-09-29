from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest

from src.data_loader import discover_data_files, load_table


def test_load_table_reads_pipe_delimited_csv(tmp_path: Path) -> None:
    path = tmp_path / "claims.csv"
    path.write_text("BENE_ID|HCPCS_CD\n1|A1234\n")

    result = load_table(path)

    assert result.to_dict("records") == [{"BENE_ID": "1", "HCPCS_CD": "A1234"}]


def test_load_table_reads_single_csv_from_zip(tmp_path: Path) -> None:
    path = tmp_path / "Carrier.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("carrier.csv", "BENE_ID|HCPCS_CD\n1|A1234\n")

    result = load_table(path, zip_member="carrier.csv")

    assert result.to_dict("records") == [{"BENE_ID": "1", "HCPCS_CD": "A1234"}]


def test_load_table_rejects_ambiguous_zip(tmp_path: Path) -> None:
    path = tmp_path / "multiple.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("one.csv", "value\n1\n")
        archive.writestr("two.csv", "value\n2\n")

    with pytest.raises(ValueError, match="Expected one CSV member"):
        load_table(path)


def test_discover_data_files_includes_zip_archives(tmp_path: Path) -> None:
    csv_path = tmp_path / "claims.csv"
    zip_path = tmp_path / "carrier.zip"
    csv_path.touch()
    zip_path.touch()

    assert discover_data_files(tmp_path) == [zip_path, csv_path]