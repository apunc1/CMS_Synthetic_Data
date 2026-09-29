from pathlib import Path

import pandas as pd

from src.coverage import (
    POLICY_INDEX_ENDPOINTS,
    fetch_policy_index,
    load_policy_index,
    match_policy_codes,
    save_policy_index,
    search_policy_index,
)


def test_match_policy_codes_is_case_insensitive_and_exact() -> None:
    policies = [
        {"id": "one", "procedure_codes": ["12345"]},
        {"id": "two", "procedure_codes": ["123456"]},
    ]
    assert match_policy_codes(policies, " 12345 ") == [policies[0]]


class FakeResponse:
    def __init__(self, data: list[dict]) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"data": self._data}


class FakeSession:
    records = {
        "NCD": [
            {
                "document_id": 108,
                "document_version": 1,
                "document_display_id": "100.3",
                "title": "Esophageal monitoring",
                "url": "/data/ncd?ncdid=108&ncdver=1",
            }
        ],
        "LCD": [
            {
                "document_id": 38441,
                "document_version": 17,
                "document_display_id": "L38441",
                "title": "Blood product molecular typing",
                "effective_date": "09/24/2026",
                "retirement_date": "N/A",
                "contractor_name_type": "WPS Insurance Corporation",
                "url": "https://www.cms.gov/medicare-coverage-database/view/lcd.aspx?lcdid=38441&ver=17",
            }
        ],
        "Article": [
            {
                "document_id": 57878,
                "document_version": 28,
                "document_display_id": "A57878",
                "title": "Billing and coding article",
                "url": "https://www.cms.gov/medicare-coverage-database/view/article.aspx?articleid=57878&ver=28",
            }
        ],
    }

    def get(self, url: str, timeout: int) -> FakeResponse:
        endpoint = url.removeprefix("https://api.coverage.cms.gov")
        policy_type = next(
            kind for kind, path in POLICY_INDEX_ENDPOINTS.items() if path == endpoint
        )
        return FakeResponse(self.records[policy_type])


def test_fetch_policy_index_normalizes_public_cms_records() -> None:
    index = fetch_policy_index(
        session=FakeSession(), fetched_at="2026-09-29T00:00:00+00:00"
    )

    assert list(index.policy_type) == ["NCD", "LCD", "Article"]
    assert index.iloc[0].source_url.endswith("ncdid=108&ncdver=1")
    assert index.iloc[1].effective_date == "09/24/2026"
    assert index.iloc[1].contractor == "WPS Insurance Corporation"
    assert index.iloc[1].jurisdiction == ""
    assert index.iloc[0].jurisdiction == "National"
    assert index.relationship_status.str.startswith("Not checked").all()


def test_search_policy_index_filters_title_id_and_type() -> None:
    index = fetch_policy_index(session=FakeSession(), fetched_at="snapshot")

    result = search_policy_index(index, "100.3", ["NCD"])

    assert len(result) == 1
    assert result.iloc[0].policy_id == "108"
    assert search_policy_index(index, "not present").empty


def test_policy_index_cache_round_trip(tmp_path: Path) -> None:
    index = fetch_policy_index(session=FakeSession(), fetched_at="snapshot")
    path = save_policy_index(index, tmp_path / "policy_index.parquet")
    loaded = load_policy_index(path)

    assert loaded is not None
    assert loaded.equals(index)