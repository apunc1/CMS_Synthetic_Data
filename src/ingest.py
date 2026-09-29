"""Source-specific mappings for the local CMS synthetic files."""

from pathlib import Path

import pandas as pd

from src.data_loader import load_table


SOURCE_TYPES = {
	"beneficiary_2019.csv": "beneficiary",
	"beneficiary_2020.csv": "beneficiary",
	"beneficiary_2021.csv": "beneficiary",
	"beneficiary_2022.csv": "beneficiary",
	"beneficiary_2023.csv": "beneficiary",
	"inpatient.csv": "inpatient",
	"carrier.zip": "carrier",
	"outpatient.zip": "outpatient",
}

ZIP_MEMBERS = {
	"carrier.zip": "carrier.csv",
	"outpatient.zip": "outpatient.csv",
}

REQUIRED_COLUMNS = {
	"beneficiary": ("BENE_ID",),
	"carrier": ("BENE_ID", "CLM_ID", "HCPCS_CD"),
	"outpatient": ("BENE_ID", "CLM_ID", "HCPCS_CD"),
	"inpatient": ("BENE_ID", "CLM_ID"),
}

SOURCE_COLUMN_MAPS = {
	"beneficiary": {
		"bene_id": "BENE_ID",
		"birth_date": "BENE_BIRTH_DT",
		"age_at_end_of_year": "AGE_AT_END_REF_YR",
		"sex_code": "SEX_IDENT_CD",
		"race_code": "BENE_RACE_CD",
	},
	"carrier": {
		"claim_id": "CLM_ID",
		"bene_id": "BENE_ID",
		"claim_from_date": "CLM_FROM_DT",
		"claim_thru_date": "CLM_THRU_DT",
		"referring_provider_npi": "RFR_PHYSN_NPI",
		"rendering_provider_npi": "PRF_PHYSN_NPI",
		"organization_npi": "ORG_NPI_NUM",
		"place_of_service_code": "LINE_PLACE_OF_SRVC_CD",
		"hcpcs_code": "HCPCS_CD",
		"line_diagnosis_code": "LINE_ICD_DGNS_CD",
		"line_submitted_charge": "LINE_SBMTD_CHRG_AMT",
		"line_allowed_charge": "LINE_ALOWD_CHRG_AMT",
		"line_provider_payment": "LINE_PRVDR_PMT_AMT",
	},
	"outpatient": {
		"claim_id": "CLM_ID",
		"bene_id": "BENE_ID",
		"claim_from_date": "CLM_FROM_DT",
		"claim_thru_date": "CLM_THRU_DT",
		"facility_provider_id": "PRVDR_NUM",
		"organization_npi": "ORG_NPI_NUM",
		"attending_provider_npi": "AT_PHYSN_NPI",
		"operating_provider_npi": "OP_PHYSN_NPI",
		"other_provider_npi": "OT_PHYSN_NPI",
		"rendering_provider_npi": "RNDRNG_PHYSN_NPI",
		"revenue_center_code": "REV_CNTR",
		"hcpcs_code": "HCPCS_CD",
		"revenue_center_rate": "REV_CNTR_RATE_AMT",
		"revenue_center_provider_payment": "REV_CNTR_PRVDR_PMT_AMT",
		"revenue_center_beneficiary_payment": "REV_CNTR_BENE_PMT_AMT",
		"revenue_center_total_charge": "REV_CNTR_TOT_CHRG_AMT",
	},
	"inpatient": {
		"claim_id": "CLM_ID",
		"bene_id": "BENE_ID",
		"claim_from_date": "CLM_FROM_DT",
		"claim_thru_date": "CLM_THRU_DT",
		"facility_provider_id": "PRVDR_NUM",
		"organization_npi": "ORG_NPI_NUM",
		"attending_provider_npi": "AT_PHYSN_NPI",
		"operating_provider_npi": "OP_PHYSN_NPI",
		"other_provider_npi": "OT_PHYSN_NPI",
		"revenue_center_code": "REV_CNTR",
		"hcpcs_code": "HCPCS_CD",
	},
}


def source_type_for(path: Path) -> str:
	"""Return the known source type for a file in the CMS data bundle."""
	source_type = SOURCE_TYPES.get(path.name.lower())
	if source_type is None:
		raise ValueError(f"Unsupported CMS source file: {path.name}")
	return source_type


def standardize_columns(frame: pd.DataFrame, source_type: str) -> pd.DataFrame:
	"""Rename known CMS fields, leaving all unmapped source fields intact."""
	if source_type not in SOURCE_COLUMN_MAPS:
		raise ValueError(f"Unsupported CMS source type: {source_type}")
	missing = sorted(set(REQUIRED_COLUMNS[source_type]) - set(frame.columns))
	if missing:
		raise ValueError(
			f"Missing required {source_type} columns: {', '.join(missing)}"
		)
	rename_map = {
		source_column: canonical_name
		for canonical_name, source_column in SOURCE_COLUMN_MAPS[source_type].items()
		if source_column in frame.columns
	}
	for index in range(1, 26):
		numbered_fields = (
			(f"ICD_DGNS_CD{index}", f"diagnosis_code_{index}"),
			(f"ICD_PRCDR_CD{index}", f"icd10_pcs_procedure_code_{index}"),
		)
		for source_column, canonical_name in numbered_fields:
			if source_column in frame.columns:
				rename_map[source_column] = canonical_name
	return frame.rename(columns=rename_map)


def load_source(path: Path) -> pd.DataFrame:
	"""Load one known CMS file and standardize its verified columns."""
	filename = path.name.lower()
	source_type = source_type_for(path)
	frame = load_table(path, delimiter="|", zip_member=ZIP_MEMBERS.get(filename))
	return standardize_columns(frame, source_type)