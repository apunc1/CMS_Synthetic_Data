from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from src.coverage_ingest import (
    build_local_coverage_relationships,
    load_local_coverage_details,
)


def _csv_bytes(rows: list[dict]) -> bytes:
    buffer = BytesIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue()


def _write_mcd_bundle(
    directory: Path,
    archive_name: str,
    nested_name: str,
    tables: dict[str, list[dict]],
) -> None:
    nested_buffer = BytesIO()
    with ZipFile(nested_buffer, "w", ZIP_DEFLATED) as nested:
        for table_name, rows in tables.items():
            nested.writestr(table_name, _csv_bytes(rows))
    with ZipFile(directory / archive_name, "w", ZIP_DEFLATED) as outer:
        outer.writestr(nested_name, nested_buffer.getvalue())


def test_build_local_coverage_relationships_joins_current_versions(
    tmp_path: Path,
) -> None:
    coverage = tmp_path / "coverage"
    coverage.mkdir()
    _write_mcd_bundle(
        coverage,
        "current_article.zip",
        "current_article_csv.zip",
        {
            "article.csv": [
                {
                    "article_id": "10",
                    "article_version": "2",
                    "display_id": "A10",
                    "title": "Article for procedure A123",
                    "article_eff_date": "2026-01-01",
                    "article_end_date": "",
                    "status": "A",
                },
                {
                    "article_id": "10",
                    "article_version": "1",
                    "display_id": "A10",
                    "title": "Old Article Version",
                    "article_eff_date": "2024-01-01",
                    "article_end_date": "2025-01-01",
                    "status": "R",
                },
            ],
            "article_x_hcpc_code.csv": [
                {
                    "article_id": "10",
                    "article_version": "2",
                    "hcpc_code_id": "A123",
                    "long_description": "Procedure A123",
                    "short_description": "A123",
                }
            ],
            "article_x_icd10_covered.csv": [
                {
                    "article_id": "10",
                    "article_version": "2",
                    "icd10_code_id": "I10",
                    "description": "Covered diagnosis",
                }
            ],
            "article_x_icd10_noncovered.csv": [
                {
                    "article_id": "10",
                    "article_version": "2",
                    "icd10_code_id": "F01",
                    "description": "Noncovered diagnosis",
                }
            ],
            "article_related_ncd_documents.csv": [
                {
                    "article_id": "10",
                    "article_version": "2",
                    "r_ncd_id": "77",
                    "r_ncd_version": "1",
                },
                {
                    "article_id": "10",
                    "article_version": "2",
                    "r_ncd_id": "0",
                    "r_ncd_version": "1",
                },
            ],
            "article_x_contractor.csv": [
                {
                    "article_id": "10",
                    "article_version": "2",
                    "contractor_id": "4",
                    "contractor_type_id": "2",
                    "contractor_version": "1",
                }
            ],
            "contractor.csv": [
                {
                    "contractor_id": "4",
                    "contractor_type_id": "2",
                    "contractor_version": "1",
                    "contractor_bus_name": "Test MAC",
                }
            ],
            "contractor_jurisdiction.csv": [
                {
                    "contractor_id": "4",
                    "contractor_type_id": "2",
                    "contractor_version": "1",
                    "state_id": "12",
                    "active_date": "2020-01-01",
                    "term_date": "",
                }
            ],
            "article_x_primary_jurisdiction.csv": [
                {
                    "article_id": "10",
                    "article_version": "2",
                    "state_id": "12",
                }
            ],
            "state_lookup.csv": [
                {"state_id": "12", "state_abbrev": "CA"}
            ],
        },
    )
    _write_mcd_bundle(
        coverage,
        "current_lcd.zip",
        "current_lcd_csv.zip",
        {
            "lcd.csv": [
                {
                    "lcd_id": "20",
                    "lcd_version": "3",
                    "display_id": "L20",
                    "title": "LCD for procedure A123",
                    "status": "A",
                    "rev_eff_date": "2026-02-01",
                    "ent_det_end_date": "",
                }
            ],
            "lcd_x_hcpc_code.csv": [
                {
                    "lcd_id": "20",
                    "lcd_version": "3",
                    "hcpc_code_id": "A123",
                    "long_description": "Procedure A123",
                }
            ],
            "lcd_related_ncd_documents.csv": [
                {
                    "lcd_id": "20",
                    "lcd_version": "3",
                    "r_ncd_id": "77",
                    "r_ncd_version": "1",
                }
            ],
            "lcd_x_contractor.csv": [
                {
                    "lcd_id": "999",
                    "lcd_version": "1",
                    "contractor_id": "8",
                    "contractor_type_id": "2",
                    "contractor_version": "1",
                }
            ],
                "contractor_jurisdiction.csv": [
                    {
                        "contractor_id": "8",
                        "contractor_type_id": "2",
                        "contractor_version": "1",
                        "state_id": "12",
                        "active_date": "2020-01-01",
                        "term_date": "",
                    }
                ],
            "lcd_x_primary_jurisdiction.csv": [
                {"lcd_id": "999", "lcd_version": "1", "state_id": "12"}
            ],
            "state_lookup.csv": [
                {"state_id": "12", "state_abbrev": "CA"}
            ],
        },
    )
    _write_mcd_bundle(
        coverage,
        "ncd.zip",
        "ncd_csv.zip",
        {
            "ncd_trkg.csv": [
                {
                    "NCD_id": "77",
                    "NCD_vrsn_num": "1",
                    "NCD_mnl_sect_title": "National policy for A123",
                    "NCD_efctv_dt": "2025-01-01",
                    "NCD_trmntn_dt": "",
                }
            ]
        },
    )
    derived = tmp_path / "derived"

    counts = build_local_coverage_relationships(
        coverage_dir=coverage,
        derived_dir=derived,
        procedure_codes=["a123"],
    )

    policies = pd.read_parquet(derived / "coverage_policy_matches.parquet")
    diagnoses = pd.read_parquet(derived / "coverage_icd10_relationships.parquet")
    ncds = pd.read_parquet(derived / "coverage_ncd_relationships.parquet")
    filtered = load_local_coverage_details(" a123 ", derived)
    assert counts["coverage_policy_matches.parquet"] == 2
    assert set(policies.policy_type) == {"Article", "LCD"}
    assert "Old Article Version" not in set(policies.title)
    article = policies[policies.policy_type == "Article"].iloc[0]
    assert article.short_description == "A123"
    assert article.medium_description == ""
    assert article.long_description == "Procedure A123"
    assert set(policies.contractor) == {"", "Test MAC"}
    assert set(policies.jurisdiction) == {"", "CA"}
    assert set(diagnoses.icd10_code) == {"I10", "F01"}
    assert set(diagnoses.code_relationship) == {
        "supports medical necessity per Article code table",
        "does not support medical necessity per Article code table",
    }
    assert len(ncds) == 2
    assert set(ncds.related_policy_id) == {"77"}
    assert len(filtered["policy_matches"]) == 2
    assert len(filtered["icd10_relationships"]) == 2