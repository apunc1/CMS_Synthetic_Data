from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence
from urllib.parse import urlencode

import pandas as pd
import requests

from src.config import DATA_DIR


CMS_COVERAGE_API = "https://api.coverage.cms.gov"
CMS_NCD_URL = "https://www.cms.gov/medicare-coverage-database/view/ncd.aspx"
POLICY_INDEX_ENDPOINTS = {
    "NCD": "/v1/reports/national-coverage-ncd/",
    "LCD": "/v1/reports/local-coverage-final-lcds/",
    "Article": "/v1/reports/local-coverage-articles/",
}
POLICY_INDEX_COLUMNS = [
    "policy_type",
    "policy_id",
    "display_id",
    "version",
    "title",
    "status",
    "effective_date",
    "retirement_date",
    "last_updated",
    "contractor",
    "jurisdiction",
    "source_url",
    "api_endpoint",
    "fetched_at",
    "relationship_status",
]
POLICY_INDEX_CACHE = DATA_DIR / "coverage_cache" / "policy_index.parquet"


def match_policy_codes(
    policies: Iterable[Mapping[str, Any]], procedure_code: str
) -> list[Mapping[str, Any]]:
    """Match normalized procedure codes in policy records.

    Each policy record is expected to expose a ``procedure_codes`` iterable.
    """
    target = procedure_code.strip().upper()
    matches = []
    for policy in policies:
        codes = policy.get("procedure_codes", ())
        if target in {str(code).strip().upper() for code in codes}:
            matches.append(policy)
    return matches


def normalize_policy_record(
    policy_type: str, record: Mapping[str, Any], endpoint: str, fetched_at: str
) -> dict:
    """Normalize one public CMS MCD summary without inferring code relationships."""
    policy_id = record.get("document_id")
    version = record.get("document_version")
    source_url = record.get("url")
    if policy_type == "NCD" and policy_id is not None:
        query = {"ncdid": policy_id}
        if version is not None:
            query["ncdver"] = version
        source_url = CMS_NCD_URL + "?" + urlencode(query)

    return {
        "policy_type": policy_type,
        "policy_id": str(policy_id) if policy_id is not None else "",
        "display_id": record.get("document_display_id") or "",
        "version": str(version) if version is not None else "",
        "title": record.get("title") or "",
        "status": record.get("document_status") or record.get("status") or "",
        "effective_date": record.get("effective_date") or "",
        "retirement_date": record.get("retirement_date") or "",
        "last_updated": record.get("last_updated") or record.get("updated_on") or "",
        "contractor": record.get("contractor_name_type") or "",
        "jurisdiction": "National" if policy_type == "NCD" else "",
        "source_url": source_url or "",
        "api_endpoint": CMS_COVERAGE_API + endpoint,
        "fetched_at": fetched_at,
        "relationship_status": "Not checked: policy code tables are not loaded",
    }


def fetch_policy_index(
    *,
    session: Any = requests,
    timeout: int = 30,
    fetched_at: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch public NCD, final LCD, and Article summary indexes from CMS."""
    snapshot_time = fetched_at or datetime.now(timezone.utc).isoformat()
    rows: List[dict] = []
    for policy_type, endpoint in POLICY_INDEX_ENDPOINTS.items():
        response = session.get(CMS_COVERAGE_API + endpoint, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        records = payload.get("data")
        if not isinstance(records, list):
            raise ValueError(f"CMS returned an invalid {policy_type} policy index.")
        rows.extend(
            normalize_policy_record(policy_type, record, endpoint, snapshot_time)
            for record in records
        )
    return pd.DataFrame(rows, columns=POLICY_INDEX_COLUMNS)


def save_policy_index(
    policy_index: pd.DataFrame, path: Path = POLICY_INDEX_CACHE
) -> Path:
    """Save a local CMS policy-index snapshot as Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name("." + path.name + ".tmp.parquet")
    temporary_path.unlink(missing_ok=True)
    policy_index.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)
    return path


def load_policy_index(path: Path = POLICY_INDEX_CACHE) -> Optional[pd.DataFrame]:
    """Load a cached CMS policy index, if one exists."""
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def search_policy_index(
    policy_index: pd.DataFrame,
    query: str = "",
    policy_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Filter public policy summaries by document type and title/ID text."""
    result = policy_index
    if policy_types:
        result = result[result["policy_type"].isin(policy_types)]
    query = query.strip()
    if query:
        searchable = (
            result["title"].fillna("").astype(str)
            + " "
            + result["policy_id"].fillna("").astype(str)
            + " "
            + result["display_id"].fillna("").astype(str)
        )
        result = result[searchable.str.contains(query, case=False, regex=False)]
    return result.reset_index(drop=True)