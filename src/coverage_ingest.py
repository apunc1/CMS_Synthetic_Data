"""Build policy/code relationships from current local CMS MCD ZIP exports."""

import argparse
import io
from pathlib import Path, PurePosixPath
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlencode
from zipfile import ZipFile

import pandas as pd

from src.config import DERIVED_DATA_DIR, RAW_DATA_DIR


ARCHIVES = {
    "LCD": ("current_lcd.zip", "current_lcd_csv.zip"),
    "Article": ("current_article.zip", "current_article_csv.zip"),
    "NCD": ("ncd.zip", "ncd_csv.zip"),
}
POLICY_OUTPUT_COLUMNS = [
    "procedure_code",
    "code_system",
    "policy_type",
    "policy_id",
    "display_id",
    "policy_version",
    "title",
    "status",
    "effective_date",
    "retirement_date",
    "contractor",
    "jurisdiction",
    "code_description",
    "relationship_type",
    "source_url",
]
ICD_OUTPUT_COLUMNS = [
    "procedure_code",
    "policy_id",
    "policy_version",
    "policy_title",
    "icd10_code",
    "icd10_description",
    "code_relationship",
    "source_url",
]
DOCUMENT_OUTPUT_COLUMNS = [
    "procedure_code",
    "source_policy_type",
    "source_policy_id",
    "source_policy_version",
    "source_policy_title",
    "related_policy_type",
    "related_policy_id",
    "related_policy_version",
    "related_policy_title",
    "effective_date",
    "relationship_type",
    "source_url",
]
LOCAL_COVERAGE_OUTPUTS = {
    "policy_matches": "coverage_policy_matches.parquet",
    "icd10_relationships": "coverage_icd10_relationships.parquet",
    "ncd_relationships": "coverage_ncd_relationships.parquet",
}


def _archive_path(coverage_dir: Path, policy_type: str) -> Path:
    filename, _ = ARCHIVES[policy_type]
    path = coverage_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing CMS coverage archive: {path}")
    return path


def _iter_csv_chunks(
    archive_path: Path, csv_bundle_name: str, table_name: str, chunksize: int
) -> Iterator[pd.DataFrame]:
    with ZipFile(archive_path) as outer:
        nested_name = next(
            (
                name
                for name in outer.namelist()
                if PurePosixPath(name).name.lower() == csv_bundle_name.lower()
            ),
            None,
        )
        if nested_name is None:
            raise FileNotFoundError(
                f"{csv_bundle_name} is missing from {archive_path.name}."
            )
        nested_bytes = outer.read(nested_name)
    with ZipFile(io.BytesIO(nested_bytes)) as inner:
        table_member = next(
            (
                name
                for name in inner.namelist()
                if PurePosixPath(name).name.lower() == table_name.lower()
            ),
            None,
        )
        if table_member is None:
            raise FileNotFoundError(
                f"{table_name} is missing from {csv_bundle_name}."
            )
        with inner.open(table_member) as member:
            reader = pd.read_csv(
                member,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
                chunksize=chunksize,
            )
            yield from reader


def _read_table(
    archive_path: Path,
    csv_bundle_name: str,
    table_name: str,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    chunks = list(
        _iter_csv_chunks(archive_path, csv_bundle_name, table_name, chunksize)
    )
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _pair_key(frame: pd.DataFrame, id_column: str, version_column: str) -> pd.Series:
    return (
        frame[id_column].astype(str).str.strip()
        + "\x1f"
        + frame[version_column].astype(str).str.strip()
    )


def _pair_keys(frame: pd.DataFrame, id_column: str, version_column: str) -> Set[str]:
    if frame.empty:
        return set()
    return set(_pair_key(frame, id_column, version_column))


def _filter_pair_chunks(
    archive_path: Path,
    csv_bundle_name: str,
    table_name: str,
    id_column: str,
    version_column: str,
    keys: Set[str],
    chunksize: int,
) -> pd.DataFrame:
    matching = []
    for chunk in _iter_csv_chunks(
        archive_path, csv_bundle_name, table_name, chunksize
    ):
        mask = _pair_key(chunk, id_column, version_column).isin(keys)
        if mask.any():
            matching.append(chunk.loc[mask])
    return pd.concat(matching, ignore_index=True) if matching else pd.DataFrame()


def _filter_code_chunks(
    archive_path: Path,
    csv_bundle_name: str,
    table_name: str,
    code_column: str,
    procedure_codes: Set[str],
    chunksize: int,
) -> pd.DataFrame:
    matching = []
    for chunk in _iter_csv_chunks(
        archive_path, csv_bundle_name, table_name, chunksize
    ):
        if code_column not in chunk.columns:
            raise ValueError(f"{table_name} has no {code_column} column.")
        codes = chunk[code_column].astype(str).str.strip().str.upper()
        mask = codes.isin(procedure_codes)
        if mask.any():
            matched = chunk.loc[mask].copy()
            matched["procedure_code"] = codes.loc[mask].values
            matching.append(matched)
    return pd.concat(matching, ignore_index=True) if matching else pd.DataFrame()


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _source_url(policy_type: str, policy_id: str, version: str) -> str:
    if policy_type == "LCD":
        path, id_key = "lcd.aspx", "lcdid"
    elif policy_type == "Article":
        path, id_key = "article.aspx", "articleid"
    else:
        path, id_key = "ncd.aspx", "ncdid"
    params = {id_key: policy_id}
    if version:
        params["ver" if policy_type != "NCD" else "ncdver"] = version
    return (
        "https://www.cms.gov/medicare-coverage-database/view/"
        + path
        + "?"
        + urlencode(params)
    )


def _metadata_by_pair(
    archive_path: Path,
    csv_bundle_name: str,
    table_name: str,
    id_column: str,
    version_column: str,
    keys: Set[str],
    chunksize: int,
) -> Dict[str, dict]:
    metadata = _filter_pair_chunks(
        archive_path,
        csv_bundle_name,
        table_name,
        id_column,
        version_column,
        keys,
        chunksize,
    )
    if metadata.empty:
        return {}
    return {
        key: row
        for key, row in zip(
            _pair_key(metadata, id_column, version_column),
            metadata.to_dict("records"),
        )
    }


def _policy_context(
    policy_type: str,
    archive_path: Path,
    csv_bundle_name: str,
    policy_keys: Set[str],
    chunksize: int,
) -> Dict[str, dict]:
    prefix = policy_type.lower()
    id_column = prefix + "_id"
    version_column = prefix + "_version"
    context = {key: {"contractor": "", "jurisdiction": ""} for key in policy_keys}
    contractor_links = _filter_pair_chunks(
        archive_path,
        csv_bundle_name,
        f"{prefix}_x_contractor.csv",
        id_column,
        version_column,
        policy_keys,
        chunksize,
    )
    if not contractor_links.empty:
        contractors = _read_table(
            archive_path, csv_bundle_name, "contractor.csv", chunksize
        )
        contractor_key_columns = [
            "contractor_id",
            "contractor_type_id",
            "contractor_version",
        ]
        if set(contractor_key_columns + ["contractor_bus_name"]).issubset(
            contractor_links.columns.union(contractors.columns)
        ):
            joined = contractor_links.merge(
                contractors[contractor_key_columns + ["contractor_bus_name"]],
                on=contractor_key_columns,
                how="left",
            )
            for key, frame in joined.groupby(_pair_key(joined, id_column, version_column)):
                names = sorted(
                    {
                        value.strip()
                        for value in frame["contractor_bus_name"].astype(str)
                        if value.strip()
                    }
                )
                context[key]["contractor"] = "; ".join(names)

    contractor_jurisdictions = _read_table(
        archive_path, csv_bundle_name, "contractor_jurisdiction.csv", chunksize
    )
    if (
        not contractor_links.empty
        and not contractor_jurisdictions.empty
        and "state_id" in contractor_jurisdictions.columns
    ):
        contractor_key_columns = [
            "contractor_id",
            "contractor_type_id",
            "contractor_version",
        ]
        active_jurisdictions = contractor_links.merge(
            contractor_jurisdictions,
            on=contractor_key_columns,
            how="inner",
        )
        if not active_jurisdictions.empty:
            today = pd.Timestamp.now().normalize()
            active_dates = pd.to_datetime(
                active_jurisdictions.get(
                    "active_date", pd.Series("", index=active_jurisdictions.index)
                ),
                errors="coerce",
            )
            term_dates = pd.to_datetime(
                active_jurisdictions.get(
                    "term_date", pd.Series("", index=active_jurisdictions.index)
                ),
                errors="coerce",
            )
            active_mask = (active_dates.isna() | (active_dates <= today)) & (
                term_dates.isna() | (term_dates >= today)
            )
            active_jurisdictions = active_jurisdictions.loc[active_mask]
            states = _read_table(
                archive_path, csv_bundle_name, "state_lookup.csv", chunksize
            )
            state_names = dict(
                zip(states.get("state_id", []), states.get("state_abbrev", []))
            )
            for key, frame in active_jurisdictions.groupby(
                _pair_key(active_jurisdictions, id_column, version_column)
            ):
                abbreviations = sorted(
                    {
                        state_names.get(str(state_id), "")
                        for state_id in frame["state_id"].astype(str)
                        if state_names.get(str(state_id), "")
                    }
                )
                context[key]["jurisdiction"] = ", ".join(abbreviations)

    jurisdiction_table = f"{prefix}_x_primary_jurisdiction.csv"
    jurisdiction_links = _filter_pair_chunks(
        archive_path,
        csv_bundle_name,
        jurisdiction_table,
        id_column,
        version_column,
        policy_keys,
        chunksize,
    )
    if not jurisdiction_links.empty:
        states = _read_table(archive_path, csv_bundle_name, "state_lookup.csv", chunksize)
        state_names = dict(
            zip(states.get("state_id", []), states.get("state_abbrev", []))
        )
        for key, frame in jurisdiction_links.groupby(
            _pair_key(jurisdiction_links, id_column, version_column)
        ):
            if context[key]["jurisdiction"]:
                continue
            abbreviations = sorted(
                {
                    state_names.get(state_id, "")
                    for state_id in frame["state_id"].astype(str)
                    if state_names.get(state_id, "")
                }
            )
            context[key]["jurisdiction"] = ", ".join(abbreviations)
    return context


def _format_date(record: Mapping[str, str], *keys: str) -> str:
    return next((record.get(key, "") for key in keys if record.get(key, "")), "")


def _direct_policy_rows(
    policy_type: str,
    code_rows: pd.DataFrame,
    metadata: Dict[str, dict],
    context: Dict[str, dict],
) -> List[dict]:
    rows = []
    id_column = "lcd_id" if policy_type == "LCD" else "article_id"
    version_column = "lcd_version" if policy_type == "LCD" else "article_version"
    for code_row in code_rows.to_dict("records"):
        policy_id = code_row[id_column]
        version = code_row[version_column]
        key = policy_id + "\x1f" + version
        document = metadata.get(key)
        if document is None:
            continue
        if policy_type == "LCD":
            effective = _format_date(document, "rev_eff_date", "orig_det_eff_date")
            retired = _format_date(document, "ent_det_end_date", "rev_end_date")
            display_id = document.get("display_id", "")
        else:
            effective = document.get("article_eff_date", "")
            retired = _format_date(document, "article_end_date", "date_retired")
            display_id = document.get("display_id", "")
        policy_context = context.get(key, {})
        rows.append(
            {
                "procedure_code": code_row["procedure_code"],
                "code_system": "CPT/HCPCS",
                "policy_type": policy_type,
                "policy_id": policy_id,
                "display_id": display_id,
                "policy_version": version,
                "title": document.get("title", ""),
                "status": document.get("status", ""),
                "effective_date": effective,
                "retirement_date": retired,
                "contractor": policy_context.get("contractor", ""),
                "jurisdiction": policy_context.get("jurisdiction", ""),
                "code_description": (
                    code_row.get("long_description")
                    or code_row.get("short_description")
                    or ""
                ),
                "relationship_type": f"HCPCS/CPT code listed in current {policy_type} code table",
                "source_url": _source_url(policy_type, policy_id, version),
            }
        )
    return rows


def _article_icd_rows(
    coverage_dir: Path,
    archive_path: Path,
    csv_bundle_name: str,
    article_code_rows: pd.DataFrame,
    chunksize: int,
) -> pd.DataFrame:
    if article_code_rows.empty:
        return _empty_frame(ICD_OUTPUT_COLUMNS)
    article_keys = _pair_keys(article_code_rows, "article_id", "article_version")
    relationship_rows = []
    for table_name, relationship in (
        ("article_x_icd10_covered.csv", "supports medical necessity per Article code table"),
        ("article_x_icd10_noncovered.csv", "does not support medical necessity per Article code table"),
    ):
        rows = _filter_pair_chunks(
            archive_path,
            csv_bundle_name,
            table_name,
            "article_id",
            "article_version",
            article_keys,
            chunksize,
        )
        if rows.empty:
            continue
        codes = article_code_rows[
            ["article_id", "article_version", "procedure_code"]
        ].drop_duplicates()
        merged = rows.merge(codes, on=["article_id", "article_version"], how="inner")
        for row in merged.to_dict("records"):
            article_id = row["article_id"]
            version = row["article_version"]
            relationship_rows.append(
                {
                    "procedure_code": row["procedure_code"],
                    "policy_id": article_id,
                    "policy_version": version,
                    "policy_title": "",
                    "icd10_code": row.get("icd10_code_id", ""),
                    "icd10_description": row.get("description", ""),
                    "code_relationship": relationship,
                    "source_url": _source_url("Article", article_id, version),
                }
            )
    result = pd.DataFrame(relationship_rows, columns=ICD_OUTPUT_COLUMNS)
    if result.empty:
        return result
    article_ids = result[["policy_id", "policy_version"]].rename(
        columns={"policy_id": "article_id", "policy_version": "article_version"}
    )
    article_keys = _pair_keys(article_ids, "article_id", "article_version")
    metadata = _metadata_by_pair(
        archive_path,
        csv_bundle_name,
        "article.csv",
        "article_id",
        "article_version",
        article_keys,
        chunksize,
    )
    result["policy_title"] = [
        metadata.get(policy_id + "\x1f" + version, {}).get("title", "")
        for policy_id, version in zip(result.policy_id, result.policy_version)
    ]
    return result


def _related_ncd_rows(
    policy_type: str,
    source_code_rows: pd.DataFrame,
    archive_path: Path,
    csv_bundle_name: str,
    ncd_metadata: Dict[str, dict],
    chunksize: int,
) -> List[dict]:
    if source_code_rows.empty:
        return []
    prefix = "article" if policy_type == "Article" else "lcd"
    id_column = prefix + "_id"
    version_column = prefix + "_version"
    source_keys = _pair_keys(source_code_rows, id_column, version_column)
    related = _filter_pair_chunks(
        archive_path,
        csv_bundle_name,
        prefix + "_related_ncd_documents.csv",
        id_column,
        version_column,
        source_keys,
        chunksize,
    )
    if related.empty:
        return []
    related = related[related["r_ncd_id"].astype(str).str.strip().ne("0")]
    related = related[related["r_ncd_id"].astype(str).str.strip().ne("")]
    if related.empty:
        return []
    code_lookup = source_code_rows[
        [id_column, version_column, "procedure_code"]
    ].drop_duplicates()
    related = related.merge(code_lookup, on=[id_column, version_column], how="inner")
    rows = []
    for link in related.to_dict("records"):
        ncd_id = str(link["r_ncd_id"])
        ncd_version = str(link.get("r_ncd_version", ""))
        ncd = ncd_metadata.get(ncd_id + "\x1f" + ncd_version)
        if ncd is None:
            continue
        rows.append(
            {
                "procedure_code": link["procedure_code"],
                "source_policy_type": policy_type,
                "source_policy_id": link[id_column],
                "source_policy_version": link[version_column],
                "source_policy_title": "",
                "related_policy_type": "NCD",
                "related_policy_id": ncd_id,
                "related_policy_version": ncd_version,
                "related_policy_title": ncd.get("NCD_mnl_sect_title", ""),
                "effective_date": ncd.get("NCD_efctv_dt", ""),
                "relationship_type": f"NCD listed as related to this {policy_type}",
                "source_url": _source_url("NCD", ncd_id, ncd_version),
            }
        )
    return rows


def build_local_coverage_relationships(
    coverage_dir: Path = RAW_DATA_DIR / "coverage",
    processed_dir: Path = Path("data/processed"),
    derived_dir: Path = DERIVED_DATA_DIR,
    procedure_codes: Optional[Sequence[str]] = None,
    chunksize: int = 100_000,
) -> Dict[str, int]:
    """Build observed-code links from CMS current LCD/Article/NCD CSV bundles."""
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")
    for policy_type in ARCHIVES:
        _archive_path(coverage_dir, policy_type)
    if procedure_codes is None:
        stats_path = derived_dir / "procedure_stats.parquet"
        if not stats_path.is_file():
            raise FileNotFoundError(
                f"Missing procedure summary: {stats_path}. Build Milestone 2 first."
            )
        codes = pd.read_parquet(stats_path, columns=["procedure_code"])[
            "procedure_code"
        ].astype(str)
    else:
        codes = pd.Series(procedure_codes, dtype=str)
    code_set = {code.strip().upper() for code in codes if code.strip()}
    derived_dir.mkdir(parents=True, exist_ok=True)

    code_rows_by_type = {}
    metadata_by_type = {}
    context_by_type = {}
    for policy_type, table_name, id_column, version_column in (
        ("LCD", "lcd_x_hcpc_code.csv", "lcd_id", "lcd_version"),
        ("Article", "article_x_hcpc_code.csv", "article_id", "article_version"),
    ):
        archive_name, bundle_name = ARCHIVES[policy_type]
        archive_path = coverage_dir / archive_name
        code_rows = _filter_code_chunks(
            archive_path,
            bundle_name,
            table_name,
            "hcpc_code_id",
            code_set,
            chunksize,
        )
        code_rows_by_type[policy_type] = code_rows
        keys = _pair_keys(code_rows, id_column, version_column)
        metadata_by_type[policy_type] = _metadata_by_pair(
            archive_path,
            bundle_name,
            "lcd.csv" if policy_type == "LCD" else "article.csv",
            id_column,
            version_column,
            keys,
            chunksize,
        )
        context_by_type[policy_type] = _policy_context(
            policy_type, archive_path, bundle_name, keys, chunksize
        )

    policy_rows = []
    for policy_type in ("LCD", "Article"):
        policy_rows.extend(
            _direct_policy_rows(
                policy_type,
                code_rows_by_type[policy_type],
                metadata_by_type[policy_type],
                context_by_type[policy_type],
            )
        )
    policy_matches = pd.DataFrame(policy_rows, columns=POLICY_OUTPUT_COLUMNS)

    article_archive, article_bundle = ARCHIVES["Article"]
    article_path = coverage_dir / article_archive
    icd_relationships = _article_icd_rows(
        coverage_dir,
        article_path,
        article_bundle,
        code_rows_by_type["Article"],
        chunksize,
    )

    ncd_archive, ncd_bundle = ARCHIVES["NCD"]
    ncd_path = coverage_dir / ncd_archive
    ncd_trunk = _read_table(ncd_path, ncd_bundle, "ncd_trkg.csv", chunksize)
    ncd_id_column, ncd_version_column = "NCD_id", "NCD_vrsn_num"
    ncd_metadata = {
        key: row
        for key, row in zip(
            _pair_key(ncd_trunk, ncd_id_column, ncd_version_column),
            ncd_trunk.to_dict("records"),
        )
    }
    ncd_relationship_rows = []
    for policy_type in ("LCD", "Article"):
        archive_name, bundle_name = ARCHIVES[policy_type]
        ncd_relationship_rows.extend(
            _related_ncd_rows(
                policy_type,
                code_rows_by_type[policy_type],
                coverage_dir / archive_name,
                bundle_name,
                ncd_metadata,
                chunksize,
            )
        )
    ncd_relationships = pd.DataFrame(
        ncd_relationship_rows, columns=DOCUMENT_OUTPUT_COLUMNS
    )

    outputs = {
        "coverage_policy_matches.parquet": policy_matches,
        "coverage_icd10_relationships.parquet": icd_relationships,
        "coverage_ncd_relationships.parquet": ncd_relationships,
    }
    counts = {}
    for filename, frame in outputs.items():
        target = derived_dir / filename
        temporary = target.with_name("." + target.name + ".tmp.parquet")
        temporary.unlink(missing_ok=True)
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
        counts[filename] = len(frame)
    return counts


def load_local_coverage_details(
    procedure_code: str, derived_dir: Path = DERIVED_DATA_DIR
) -> Dict[str, pd.DataFrame]:
    """Read only the relationship rows for one procedure from local Parquet."""
    normalized_code = procedure_code.strip().upper()
    details = {}
    for name, filename in LOCAL_COVERAGE_OUTPUTS.items():
        path = derived_dir / filename
        details[name] = (
            pd.read_parquet(path, filters=[("procedure_code", "==", normalized_code)])
            if path.is_file()
            else pd.DataFrame()
        )
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage-dir", type=Path, default=RAW_DATA_DIR / "coverage")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    counts = build_local_coverage_relationships(
        args.coverage_dir, args.processed_dir, args.derived_dir, chunksize=args.chunksize
    )
    for filename, row_count in counts.items():
        print(f"{filename}: {row_count:,} rows")


if __name__ == "__main__":
    main()