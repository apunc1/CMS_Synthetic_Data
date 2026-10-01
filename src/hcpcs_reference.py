from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd


def normalize_hcpcs_code(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    code = str(value).strip()
    if not code or code.lower() in {"nan", "none"}:
        return None
    code = code.replace(" ", "")
    if code.endswith(".0") and code[:-2].replace(".", "", 1).isdigit():
        code = code[:-2]
    return code.upper()


def iter_anweb_archives(download_dir: Path | str | None = None) -> list[Path]:
    root = Path(download_dir) if download_dir is not None else Path.home() / "Downloads"
    if not root.exists():
        return []

    patterns = [
        "*ANWEB*.zip",
        "*anweb*.zip",
        "*HCPC*ANWEB*.zip",
        "*HCPC*anweb*.zip",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(root.glob(pattern), key=lambda path: path.name.lower()))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for match in matches:
        if match not in seen:
            deduped.append(match)
            seen.add(match)
    return sorted(deduped, key=lambda path: (path.stat().st_mtime, path.name.lower()), reverse=True)


def _read_long_description_mapping(archive_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(archive_path) as archive:
        excel_names = [
            name for name in archive.namelist() if name.lower().endswith(".xlsx") and "anweb" in name.lower()
        ]
        if not excel_names:
            return {}

        for excel_name in sorted(excel_names, key=lambda name: name.lower()):
            excel_bytes = archive.read(excel_name)
            try:
                workbook = pd.ExcelFile(io.BytesIO(excel_bytes))
            except Exception:
                continue

            for sheet_name in workbook.sheet_names:
                frame = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=sheet_name)
                if frame.empty:
                    continue

                columns = [str(column).strip() for column in frame.columns]
                hcpc_column = next(
                    (
                        column
                        for column in frame.columns
                        if str(column).strip().lower() in {"hcpc", "hcpcs", "hcpcs code", "hcpcs_cd"}
                    ),
                    None,
                )
                description_column = next(
                    (
                        column
                        for column in frame.columns
                        if "long description" in str(column).lower()
                    ),
                    None,
                )

                if hcpc_column is None or description_column is None:
                    continue

                mapping: dict[str, str] = {}
                for _, row in frame[[hcpc_column, description_column]].dropna().iterrows():
                    code = normalize_hcpcs_code(row[hcpc_column])
                    description = str(row[description_column]).strip()
                    if code and description:
                        mapping[code] = description
                if mapping:
                    return mapping
    return {}


def load_anweb_descriptions(download_dir: Path | str | None = None) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for archive_path in iter_anweb_archives(download_dir):
        descriptions.update(_read_long_description_mapping(archive_path))
    return descriptions


def iter_cpt_reference_files(download_dir: Path | str | None = None) -> list[Path]:
    root = Path(download_dir) if download_dir is not None else Path.home() / "Downloads"
    if not root.exists():
        return []

    patterns = [
        "*CPT*.csv",
        "*cpt*.csv",
        "*CPT*.xlsx",
        "*cpt*.xlsx",
        "*CPT*.xls",
        "*cpt*.xls",
        "*CPT*.zip",
        "*cpt*.zip",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(root.glob(pattern), key=lambda path: path.name.lower()))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for match in matches:
        if match not in seen:
            deduped.append(match)
            seen.add(match)
    return sorted(deduped, key=lambda path: (path.stat().st_mtime, path.name.lower()), reverse=True)


def _read_cpt_mapping(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                entries = [
                    name for name in archive.namelist() if name.lower().endswith((".csv", ".xlsx", ".xls"))
                ]
                mapping: dict[str, str] = {}
                for entry_name in sorted(entries, key=lambda name: name.lower()):
                    file_bytes = archive.read(entry_name)
                    if entry_name.lower().endswith(".csv"):
                        frame = pd.read_csv(io.BytesIO(file_bytes))
                    else:
                        try:
                            frame = pd.read_excel(io.BytesIO(file_bytes))
                        except Exception:
                            continue
                    mapping.update(_read_long_description_mapping_from_frame(frame))
                return mapping
        except Exception:
            return {}

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        return {}
    return _read_long_description_mapping_from_frame(frame)


def _read_long_description_mapping_from_frame(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {}

    columns = [str(column).strip() for column in frame.columns]
    code_column = next(
        (
            column
            for column in frame.columns
            if str(column).strip().lower() in {"cpt", "hcpcs", "hcpcs code", "hcpcs_cd", "procedure_code", "code"}
        ),
        None,
    )
    description_column = next(
        (
            column
            for column in frame.columns
            if "long description" in str(column).lower()
        ),
        next(
            (
                column
                for column in frame.columns
                if "description" in str(column).lower()
            ),
            None,
        ),
    )

    if code_column is None or description_column is None:
        return {}

    mapping: dict[str, str] = {}
    for _, row in frame[[code_column, description_column]].dropna().iterrows():
        code = normalize_hcpcs_code(row[code_column])
        description = str(row[description_column]).strip()
        if code and description:
            mapping[code] = description
    return mapping


def load_cpt_descriptions(download_dir: Path | str | None = None) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    for reference_path in iter_cpt_reference_files(download_dir):
        descriptions.update(_read_cpt_mapping(reference_path))
    return descriptions


def get_long_description(code: str | object, download_dir: Path | str | None = None) -> str | None:
    normalized_code = normalize_hcpcs_code(code)
    if not normalized_code:
        return None

    anweb = load_anweb_descriptions(download_dir)
    description = anweb.get(normalized_code)
    if description:
        return description

    cpt = load_cpt_descriptions(download_dir)
    return cpt.get(normalized_code)
