"""Chunked Milestone 1 validation and Parquet standardization pipeline."""

import argparse
import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data_loader import iter_table_chunks
from src.ingest import (
	REQUIRED_COLUMNS,
	SOURCE_COLUMN_MAPS,
	SOURCE_TYPES,
	ZIP_MEMBERS,
	standardize_columns,
)


SOURCE_URL = (
	"https://data.cms.gov/collection/"
	"synthetic-medicare-enrollment-fee-for-service-claims-and-prescription-drug-event"
)
SOURCE_URLS = {
	"beneficiary_2019.csv": "https://data.cms.gov/sites/default/files/2023-04/7f65254b-174e-4460-aa49-6759ea8397af/beneficiary_2019.csv",
	"beneficiary_2020.csv": "https://data.cms.gov/sites/default/files/2023-04/74b86018-5148-45bb-87f7-c551eda070d8/beneficiary_2020.csv",
	"beneficiary_2021.csv": "https://data.cms.gov/sites/default/files/2023-04/98832739-88bd-4243-87fe-ee12b8542baf/beneficiary_2021.csv",
	"beneficiary_2022.csv": "https://data.cms.gov/sites/default/files/2023-04/831cfb41-3a7a-4ce0-b02f-04d27df56b31/beneficiary_2022.csv",
	"beneficiary_2023.csv": "https://data.cms.gov/sites/default/files/2023-04/18fe0f5d-4126-432f-b12b-2569db9bdefa/beneficiary_2023.csv",
	"inpatient.csv": "https://data.cms.gov/sites/default/files/2023-04/67157de9-d962-4af0-bf0e-3578b3afec58/inpatient.csv",
	"carrier.zip": "https://data.cms.gov/sites/default/files/2023-04/b6fe03a8-82e2-49b7-a9fe-9eb68c8ac4d0/Carrier.zip",
	"outpatient.zip": "https://data.cms.gov/sites/default/files/2023-04/c3d8a962-c6b8-4a59-adb5-f0495cc81fda/Outpatient.zip",
}
CODEBOOK_URLS = {
	"beneficiary": "https://www2.ccwdata.org/documents/10280/19022436/codebook-mbsf-abcd.pdf",
	"carrier": "https://www2.ccwdata.org/documents/10280/19022436/codebook-ffs-claims.pdf",
	"inpatient": "https://www2.ccwdata.org/documents/10280/19022436/codebook-ffs-claims.pdf",
	"outpatient": "https://www2.ccwdata.org/documents/10280/19022436/codebook-ffs-claims.pdf",
}
DATE_COLUMNS = {"birth_date", "claim_from_date", "claim_thru_date"}
NUMERIC_COLUMNS = {
	"age_at_end_of_year",
	"line_submitted_charge",
	"line_allowed_charge",
	"line_provider_payment",
	"revenue_center_rate",
	"revenue_center_provider_payment",
	"revenue_center_beneficiary_payment",
	"revenue_center_total_charge",
}
CANONICAL_DESCRIPTIONS = {
	"bene_id": "CMS synthetic beneficiary identifier; preserved as text.",
	"claim_id": "CMS claim identifier; preserved as text.",
	"hcpcs_code": "HCPCS code reported on a claim line; preserved as text.",
	"claim_from_date": "CMS claim start date parsed from the source date field.",
	"claim_thru_date": "CMS claim end date parsed from the source date field.",
	"birth_date": "Beneficiary birth date parsed from the source date field.",
	"age_at_end_of_year": "Age at the end of the beneficiary reference year.",
	"icd10_pcs_procedure_code": "Numbered ICD-10-PCS procedure code from the source claim.",
	"diagnosis_code": "Numbered diagnosis code from the source claim.",
}


def _source_columns(path: Path, source_type: str) -> List[str]:
	if path.suffix.lower() == ".zip":
		member_name = ZIP_MEMBERS[path.name.lower()]
		with ZipFile(path) as archive:
			with archive.open(member_name) as member:
				return list(pd.read_csv(member, sep="|", nrows=0).columns)
	return list(pd.read_csv(path, sep="|", nrows=0).columns)


def _canonical_column(source_type: str, source_column: str) -> str:
	known = SOURCE_COLUMN_MAPS[source_type].get(source_column)
	if known is not None:
		return known
	for prefix, canonical_prefix in (
		("ICD_DGNS_CD", "diagnosis_code_"),
		("ICD_PRCDR_CD", "icd10_pcs_procedure_code_"),
	):
		match = re.fullmatch(prefix + r"(\d+)", source_column)
		if match:
			return canonical_prefix + match.group(1)
	return source_column.lower()


def _semantic_type(source_column: str, canonical_column: str) -> str:
	if source_column.endswith("_DT") or canonical_column in DATE_COLUMNS:
		return "date"
	if (
		source_column.endswith(("_AMT", "_CNT", "_QTY"))
		or canonical_column in NUMERIC_COLUMNS
	):
		return "numeric"
	if (
		"CODE" in canonical_column
		or "code_" in canonical_column
		or canonical_column.endswith(("_id", "_npi", "_cd"))
		or source_column.endswith("_CD")
	):
		return "identifier_or_code"
	return "text_or_category"


def _data_dictionary_rows(path: Path, source_type: str, columns: List[str]) -> List[dict]:
	rows = []
	for source_column in columns:
		canonical_column = _canonical_column(source_type, source_column)
		base_name = re.sub(r"_\d+$", "", canonical_column)
		description = CANONICAL_DESCRIPTIONS.get(
			base_name, "Source field retained; consult the CMS data dictionary for its definition."
		)
		rows.append(
			{
				"source_file": path.name,
				"source_type": source_type,
				"source_column": source_column,
				"standardized_column": canonical_column,
				"semantic_type": _semantic_type(source_column, canonical_column),
				"description": description,
				"definition_source_url": CODEBOOK_URLS[source_type],
			}
		)
	return rows


def _is_code_column(column: str) -> bool:
	upper = column.upper()
	return (
		upper.endswith("_CD")
		or "HCPCS" in upper
		or "ICD_" in upper
		or upper == "REV_CNTR"
	)


def _normalize_chunk(
	chunk: pd.DataFrame, source_type: str
) -> Tuple[pd.DataFrame, Dict[str, int]]:
	quality_counts: Dict[str, int] = {}
	for source_column in chunk.columns:
		if _is_code_column(source_column):
			values = chunk[source_column].astype(str)
			stripped_values = values.str.strip()
			metric = "code_whitespace_rows:" + source_column
			quality_counts[metric] = quality_counts.get(metric, 0) + int(
				(values.ne(stripped_values) & stripped_values.ne("")).sum()
			)

	frame = standardize_columns(chunk, source_type)
	frame = frame.rename(columns={column: column.lower() for column in frame.columns})
	for column in frame.columns:
		values = frame[column].astype(str).str.strip()
		if column.endswith("_dt") or column in DATE_COLUMNS:
			nonempty = values.ne("")
			parsed = pd.to_datetime(values, format="%d-%b-%Y", errors="coerce")
			fallback_rows = nonempty & parsed.isna()
			if fallback_rows.any():
				parsed.loc[fallback_rows] = pd.to_datetime(
					values.loc[fallback_rows], errors="coerce", dayfirst=True
				)
			quality_counts["date_parse_failures"] = quality_counts.get(
				"date_parse_failures", 0
			) + int((nonempty & parsed.isna()).sum())
			frame[column] = parsed
		elif (
			column.endswith(("_amt", "_cnt", "_qty"))
			or column in NUMERIC_COLUMNS
		):
			nonempty = values.ne("")
			numeric = pd.to_numeric(
				values.str.replace(",", "", regex=False).replace("", pd.NA),
				errors="coerce",
			)
			quality_counts["numeric_parse_failures"] = quality_counts.get(
				"numeric_parse_failures", 0
			) + int((nonempty & numeric.isna()).sum())
			frame[column] = numeric.astype("float64")
		else:
			frame[column] = values
	return frame, quality_counts


def _source_hash(path: Path) -> str:
	digest = hashlib.sha256()
	with path.open("rb") as source:
		for block in iter(lambda: source.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _append_metric(
	rows: List[dict], source_file: str, source_type: str, metric: str, value: object, field: str = ""
) -> None:
	rows.append(
		{
			"source_file": source_file,
			"source_type": source_type,
			"metric": metric,
			"field": field,
			"value": value,
		}
	)


def run_pipeline(
	raw_dir: Path = RAW_DATA_DIR,
	data_dir: Path = DATA_DIR,
	chunksize: int = 50_000,
	download_date: str = "",
	release_date: str = "",
) -> Dict[str, int]:
	"""Validate source headers and write chunked standardized Parquet outputs."""
	if chunksize <= 0:
		raise ValueError("chunksize must be greater than zero")
	for label, value in (("download_date", download_date), ("release_date", release_date)):
		if value:
			try:
				date.fromisoformat(value)
			except ValueError as error:
				raise ValueError(f"{label} must use YYYY-MM-DD format") from error

	missing_files = [name for name in SOURCE_TYPES if not (raw_dir / name).is_file()]
	if missing_files:
		raise FileNotFoundError(
			"Missing expected CMS sources: " + ", ".join(missing_files)
		)

	processed_dir = data_dir / "processed"
	reports_dir = data_dir / "reports"
	processed_dir.mkdir(parents=True, exist_ok=True)
	reports_dir.mkdir(parents=True, exist_ok=True)

	manifest_rows: List[dict] = []
	dictionary_rows: List[dict] = []	
	quality_rows: List[dict] = []
	writers: Dict[Path, pq.ParquetWriter] = {}
	temporary_outputs: Dict[Path, Path] = {}
	row_counts: Dict[str, int] = {}

	try:
		for filename, source_type in SOURCE_TYPES.items():
			path = raw_dir / filename
			columns = _source_columns(path, source_type)
			required_missing = sorted(set(REQUIRED_COLUMNS[source_type]) - set(columns))
			if required_missing:
				raise ValueError(
					f"{filename} is missing required columns: {', '.join(required_missing)}"
				)
			dictionary_rows.extend(_data_dictionary_rows(path, source_type, columns))
			output_name = (
				"beneficiaries.parquet"
				if source_type == "beneficiary"
				else source_type + ".parquet"
			)
			output_path = processed_dir / output_name
			temporary_path = processed_dir / ("." + output_name + ".tmp")
			if output_path not in temporary_outputs:
				temporary_outputs[output_path] = temporary_path
				temporary_path.unlink(missing_ok=True)
			seen_hashes: Set[int] = set()
			duplicate_rows = 0
			null_counts: Dict[str, int] = {}
			date_bounds: Dict[str, List[pd.Timestamp]] = {}
			quality_counts: Dict[str, int] = {}
			row_count = 0
			for chunk in iter_table_chunks(
				path,
				chunksize=chunksize,
				delimiter="|",
				zip_member=ZIP_MEMBERS.get(filename.lower()),
			):
				standardized, chunk_quality = _normalize_chunk(chunk, source_type)
				if source_type == "beneficiary":
					year = pd.Series(
						int(path.stem.rsplit("_", 1)[1]),
						index=standardized.index,
						name="year",
						dtype="int64",
					)
					standardized = pd.concat((year, standardized), axis=1)
				for metric, value in chunk_quality.items():
					quality_counts[metric] = quality_counts.get(metric, 0) + value
				for column in standardized.columns:
					missing = standardized[column].isna()
					if standardized[column].dtype == object:
						missing = missing | standardized[column].eq("")
					null_counts[column] = null_counts.get(column, 0) + int(missing.sum())
					if pd.api.types.is_datetime64_any_dtype(standardized[column]):
						valid_dates = standardized[column].dropna()
						if not valid_dates.empty:
							bounds = date_bounds.setdefault(
								column,
								[pd.Timestamp.max, pd.Timestamp.min],
							)
							bounds[0] = min(bounds[0], valid_dates.min())
							bounds[1] = max(bounds[1], valid_dates.max())
				for value in pd.util.hash_pandas_object(chunk, index=False):
					hash_value = int(value)
					if hash_value in seen_hashes:
						duplicate_rows += 1
					else:
						seen_hashes.add(hash_value)
				table = pa.Table.from_pandas(standardized, preserve_index=False)
				writer = writers.get(output_path)
				if writer is None:
					writer = pq.ParquetWriter(temporary_path, table.schema)
					writers[output_path] = writer
				elif not writer.schema.equals(table.schema, check_metadata=False):
					raise ValueError(
						f"Inconsistent standardized schema while appending {filename}."
					)
				writer.write_table(table)
				row_count += len(standardized)
			if row_count == 0:
				raise ValueError(f"No data rows found in {filename}.")
			row_counts[filename] = row_count
			file_size = path.stat().st_size
			manifest_rows.append(
				{
					"source": "CMS Synthetic Medicare claims",
					"file": filename,
					"source_url": SOURCE_URLS[filename],
					"collection_url": SOURCE_URL,
					"download_date": download_date,
					"release_date": release_date,
					"row_count": row_count,
					"file_size_bytes": file_size,
					"sha256": _source_hash(path),
					"processed_file": output_name,
				}
			)
			_append_metric(quality_rows, filename, source_type, "file_exists", True)
			_append_metric(quality_rows, filename, source_type, "file_size_bytes", file_size)
			_append_metric(quality_rows, filename, source_type, "row_count", row_count)
			_append_metric(quality_rows, filename, source_type, "column_count", len(columns))
			_append_metric(quality_rows, filename, source_type, "duplicate_rows", duplicate_rows)
			_append_metric(
				quality_rows,
				filename,
				source_type,
				"required_columns_present",
				not required_missing,
			)
			for column, null_count in sorted(null_counts.items()):
				_append_metric(quality_rows, filename, source_type, "null_count", null_count, column)
				_append_metric(
					quality_rows,
					filename,
					source_type,
					"null_percentage",
					round(100 * null_count / row_count, 4),
					column,
				)
			for column, bounds in sorted(date_bounds.items()):
				_append_metric(
					quality_rows, filename, source_type, "date_min", bounds[0].date().isoformat(), column
				)
				_append_metric(
					quality_rows, filename, source_type, "date_max", bounds[1].date().isoformat(), column
				)
			for metric, value in sorted(quality_counts.items()):
				field = ""
				if metric.startswith("code_whitespace_rows:"):
					metric, field = metric.split(":", 1)
				_append_metric(quality_rows, filename, source_type, metric, value, field)
	except Exception:
		for writer in writers.values():
			writer.close()
		for temporary_path in temporary_outputs.values():
			temporary_path.unlink(missing_ok=True)
		raise
	else:
		for writer in writers.values():
			writer.close()
		for output_path, temporary_path in temporary_outputs.items():
			temporary_path.replace(output_path)

	pd.DataFrame(manifest_rows).to_csv(data_dir / "source_manifest.csv", index=False)
	pd.DataFrame(dictionary_rows).to_csv(data_dir / "data_dictionary.csv", index=False)
	pd.DataFrame(quality_rows).to_csv(
		reports_dir / "data_quality_report.csv", index=False
	)
	return row_counts


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--chunksize", type=int, default=50_000)
	parser.add_argument("--download-date", default="", help="Verified download date (YYYY-MM-DD).")
	parser.add_argument("--release-date", default="", help="Verified CMS release date (YYYY-MM-DD).")
	args = parser.parse_args()
	row_counts = run_pipeline(
		chunksize=args.chunksize,
		download_date=args.download_date,
		release_date=args.release_date,
	)
	for filename, row_count in row_counts.items():
		print(f"{filename}: {row_count:,} rows")
	print("Wrote source manifest, data dictionary, quality report, and Parquet files.")


if __name__ == "__main__":
	main()