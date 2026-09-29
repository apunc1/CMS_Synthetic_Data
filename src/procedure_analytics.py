"""Build local procedure utilization tables from standardized Parquet sources."""

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import duckdb
import pyarrow.parquet as pq

from src.config import DERIVED_DATA_DIR, PROCESSED_DATA_DIR


SOURCE_FILES = {
    "carrier": "carrier.parquet",
    "outpatient": "outpatient.parquet",
    "inpatient": "inpatient.parquet",
}
PROVIDER_FIELDS = {
    "carrier": (
        ("rendering_provider_npi", "NPI"),
        ("organization_npi", "NPI"),
    ),
    "outpatient": (
        ("rendering_provider_npi", "NPI"),
        ("attending_provider_npi", "NPI"),
        ("organization_npi", "NPI"),
        ("facility_provider_id", "CMS_PROVIDER_NUMBER"),
    ),
    "inpatient": (
        ("attending_provider_npi", "NPI"),
        ("operating_provider_npi", "NPI"),
        ("organization_npi", "NPI"),
        ("facility_provider_id", "CMS_PROVIDER_NUMBER"),
    ),
}
OUTPUT_QUERIES = {
    "procedure_stats.parquet": "SELECT * FROM procedure_stats",
    "diagnosis_procedure_stats.parquet": "SELECT * FROM diagnosis_procedure_stats",
    "provider_procedure_stats.parquet": "SELECT * FROM provider_procedure_stats",
    "pos_procedure_stats.parquet": "SELECT * FROM pos_procedure_stats",
    "procedure_trends.parquet": "SELECT * FROM procedure_trends",
}


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _nullable_text(column: str, columns: Sequence[str]) -> str:
    if column not in columns:
        return "NULL::VARCHAR"
    quoted = _quote_identifier(column)
    return f"NULLIF(TRIM(CAST({quoted} AS VARCHAR)), '')"


def _provider_expressions(
    source_type: str, columns: Sequence[str]
) -> Tuple[str, str]:
    choices = [
        (_nullable_text(column, columns), identifier_type)
        for column, identifier_type in PROVIDER_FIELDS[source_type]
        if column in columns
    ]
    if not choices:
        return "NULL::VARCHAR", "NULL::VARCHAR"
    provider_id = "CASE " + " ".join(
        f"WHEN {expression} IS NOT NULL THEN {_sql_string(identifier_type + ':')} || {expression}"
        for expression, identifier_type in choices
    ) + " ELSE NULL END"
    provider_type = "CASE " + " ".join(
        f"WHEN {expression} IS NOT NULL THEN {_sql_string(identifier_type)}"
        for expression, identifier_type in choices
    ) + " ELSE NULL END"
    return provider_id, provider_type


def _source_query(source_type: str, path: Path) -> str:
    columns = pq.ParquetFile(path).schema_arrow.names
    provider_id, provider_type = _provider_expressions(source_type, columns)
    diagnosis_columns = [
        name
        for name in columns
        if name.startswith("diagnosis_code_") or name == "line_diagnosis_code"
    ]
    if diagnosis_columns:
        diagnosis_values = ", ".join(
            _nullable_text(name, columns) for name in diagnosis_columns
        )
        diagnosis_list = (
            "list_filter(list_value(" + diagnosis_values + "), "
            "item -> item IS NOT NULL)"
        )
    else:
        diagnosis_list = "[]::VARCHAR[]"

    amount = (
        "TRY_CAST(line_allowed_charge AS DOUBLE)"
        if source_type == "carrier" and "line_allowed_charge" in columns
        else "NULL::DOUBLE"
    )
    place_of_service = (
        _nullable_text("place_of_service_code", columns)
        if source_type == "carrier"
        else "NULL::VARCHAR"
    )
    path_literal = _sql_string(path.as_posix())
    return f"""
        SELECT
            {_sql_string(source_type)} AS claim_type,
            'HCPCS/CPT' AS code_system,
            {_nullable_text('hcpcs_code', columns)} AS procedure_code,
            {_nullable_text('claim_id', columns)} AS source_claim_id,
            {_nullable_text('bene_id', columns)} AS bene_id,
            {provider_id} AS provider_id,
            {provider_type} AS provider_id_type,
            {place_of_service} AS place_of_service_code,
            {amount} AS allowed_amount,
            TRY_CAST(claim_from_date AS DATE) AS service_date,
            {diagnosis_list} AS diagnosis_codes,
            CASE
                WHEN {_nullable_text('claim_id', columns)} IS NULL
                THEN {_sql_string(source_type + ':row:')} || CAST(row_number() OVER () AS VARCHAR)
                ELSE {_sql_string(source_type + ':')} || {_nullable_text('claim_id', columns)}
            END AS claim_key
        FROM read_parquet({path_literal})
    """


def _write_query(connection: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    temporary_path = path.with_name("." + path.name + ".tmp.parquet")
    temporary_path.unlink(missing_ok=True)
    connection.execute(
        f"COPY ({query}) TO {_sql_string(temporary_path.as_posix())} (FORMAT PARQUET)"
    )
    temporary_path.replace(path)


def get_procedure_codes(derived_dir: Path = DERIVED_DATA_DIR):
    """Return observed procedure codes and claim volumes for the explorer."""
    path = derived_dir / "procedure_stats.parquet"
    if not path.is_file():
        return None
    connection = duckdb.connect()
    try:
        return connection.execute(
            "SELECT procedure_code, claim_count "
            f"FROM read_parquet({_sql_string(path.as_posix())}) "
            "ORDER BY claim_count DESC, procedure_code"
        ).df()
    finally:
        connection.close()


def get_procedure_details(procedure_code: str, derived_dir: Path = DERIVED_DATA_DIR):
    """Load summary and breakdown tables for one selected procedure code."""
    filenames = {
        "summary": "procedure_stats.parquet",
        "diagnoses": "diagnosis_procedure_stats.parquet",
        "providers": "provider_procedure_stats.parquet",
        "places": "pos_procedure_stats.parquet",
        "trends": "procedure_trends.parquet",
    }
    missing = [name for name in filenames.values() if not (derived_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing derived tables: " + ", ".join(missing))
    connection = duckdb.connect()
    try:
        tables = {}
        for name, filename in filenames.items():
            path = derived_dir / filename
            query = (
                f"SELECT * FROM read_parquet({_sql_string(path.as_posix())}) "
                "WHERE procedure_code = ?"
            )
            if name == "summary":
                query += " LIMIT 1"
            elif name == "diagnoses":
                query += " ORDER BY claim_count DESC LIMIT 20"
            elif name == "providers":
                query += " ORDER BY claim_count DESC LIMIT 20"
            elif name == "places":
                query += " ORDER BY claim_count DESC"
            else:
                query += " ORDER BY service_month"
            frame = connection.execute(query, [procedure_code]).df()
            tables[name] = frame.iloc[0].to_dict() if name == "summary" and not frame.empty else frame
        return tables
    finally:
        connection.close()


def build_procedure_analytics(
    processed_dir: Path = PROCESSED_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
) -> Dict[str, int]:
    """Aggregate observed HCPCS/CPT utilization into local Parquet tables."""
    missing = [filename for filename in SOURCE_FILES.values() if not (processed_dir / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing standardized claims files: " + ", ".join(missing)
        )
    derived_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        source_queries = [
            _source_query(source_type, processed_dir / filename)
            for source_type, filename in SOURCE_FILES.items()
        ]
        connection.execute(
            "CREATE TEMP TABLE procedure_lines AS " + " UNION ALL ".join(source_queries)
        )
        connection.execute(
            """
            CREATE TEMP VIEW hcpcs_lines AS
            SELECT * FROM procedure_lines
            WHERE procedure_code IS NOT NULL AND procedure_code <> ''
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE procedure_stats AS
            WITH core AS (
                SELECT
                    procedure_code,
                    code_system,
                    COUNT(DISTINCT claim_key) AS claim_count,
                    COUNT(DISTINCT bene_id) AS unique_beneficiaries,
                    COUNT(DISTINCT provider_id) AS unique_providers,
                    SUM(allowed_amount) AS total_allowed,
                    AVG(allowed_amount) AS average_allowed,
                    MEDIAN(allowed_amount) AS median_allowed,
                    COUNT(allowed_amount) AS allowed_amount_claims,
                    MIN(service_date) AS first_service_date,
                    MAX(service_date) AS last_service_date
                FROM hcpcs_lines
                GROUP BY procedure_code, code_system
            ), diagnoses AS (
                SELECT procedure_code, COUNT(DISTINCT diagnosis_code) AS unique_diagnoses
                FROM hcpcs_lines, UNNEST(diagnosis_codes) AS codes(diagnosis_code)
                WHERE diagnosis_code IS NOT NULL AND diagnosis_code <> ''
                GROUP BY procedure_code
            ), places AS (
                SELECT procedure_code, COUNT(DISTINCT place_of_service_code) AS unique_places_of_service
                FROM hcpcs_lines
                WHERE place_of_service_code IS NOT NULL
                GROUP BY procedure_code
            )
            SELECT
                core.*,
                COALESCE(diagnoses.unique_diagnoses, 0) AS unique_diagnoses,
                COALESCE(places.unique_places_of_service, 0) AS unique_places_of_service,
                'Carrier line allowed amounts only' AS allowed_amount_scope
            FROM core
            LEFT JOIN diagnoses USING (procedure_code)
            LEFT JOIN places USING (procedure_code)
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE diagnosis_procedure_stats AS
            WITH relationships AS (
                SELECT
                    procedure_code,
                    diagnosis_code,
                    COUNT(DISTINCT claim_key) AS claim_count,
                    COUNT(DISTINCT bene_id) AS beneficiary_count,
                    COUNT(DISTINCT provider_id) AS provider_count
                FROM hcpcs_lines, UNNEST(diagnosis_codes) AS codes(diagnosis_code)
                WHERE diagnosis_code IS NOT NULL AND diagnosis_code <> ''
                GROUP BY procedure_code, diagnosis_code
            )
            SELECT
                relationships.*,
                100.0 * relationships.claim_count / NULLIF(procedures.claim_count, 0)
                    AS percentage_of_procedure_claims
            FROM relationships
            JOIN procedure_stats AS procedures USING (procedure_code)
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE provider_procedure_stats AS
            SELECT
                provider_id,
                provider_id_type,
                procedure_code,
                COUNT(DISTINCT claim_key) AS claim_count,
                COUNT(DISTINCT bene_id) AS beneficiary_count,
                SUM(allowed_amount) AS total_allowed,
                COUNT(DISTINCT claim_key) * 1.0
                    / NULLIF(COUNT(DISTINCT bene_id), 0) AS procedure_rate,
                'Claims per beneficiary observed for this provider and procedure'
                    AS procedure_rate_basis
            FROM hcpcs_lines
            WHERE provider_id IS NOT NULL
            GROUP BY provider_id, provider_id_type, procedure_code
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE pos_procedure_stats AS
            WITH carrier_totals AS (
                SELECT procedure_code, COUNT(DISTINCT claim_key) AS carrier_claim_count
                FROM hcpcs_lines
                WHERE claim_type = 'carrier'
                GROUP BY procedure_code
            ), places AS (
                SELECT
                    procedure_code,
                    place_of_service_code,
                    COUNT(DISTINCT claim_key) AS claim_count,
                    COUNT(DISTINCT bene_id) AS beneficiary_count
                FROM hcpcs_lines
                WHERE claim_type = 'carrier' AND place_of_service_code IS NOT NULL
                GROUP BY procedure_code, place_of_service_code
            )
            SELECT
                places.*,
                100.0 * places.claim_count / NULLIF(carrier_totals.carrier_claim_count, 0)
                    AS percentage_of_carrier_procedure_claims,
                'Carrier claims only' AS data_scope
            FROM places
            JOIN carrier_totals USING (procedure_code)
            """
        )
        connection.execute(
            """
            CREATE TEMP TABLE procedure_trends AS
            SELECT
                procedure_code,
                DATE_TRUNC('month', service_date)::DATE AS service_month,
                COUNT(DISTINCT claim_key) AS claim_count,
                COUNT(DISTINCT bene_id) AS beneficiary_count
            FROM hcpcs_lines
            WHERE service_date IS NOT NULL
            GROUP BY procedure_code, service_month
            ORDER BY procedure_code, service_month
            """
        )

        row_counts = {}
        for filename, query in OUTPUT_QUERIES.items():
            output_path = derived_dir / filename
            _write_query(connection, query, output_path)
            row_counts[filename] = connection.execute(
                f"SELECT COUNT(*) FROM read_parquet({_sql_string(output_path.as_posix())})"
            ).fetchone()[0]
        return row_counts
    finally:
        connection.close()


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    args = parser.parse_args()
    row_counts = build_procedure_analytics(args.processed_dir, args.derived_dir)
    for filename, row_count in row_counts.items():
        print(f"{filename}: {row_count:,} rows")


if __name__ == "__main__":
    _main()