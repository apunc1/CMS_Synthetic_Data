def normalize_icd10_code(code: str) -> str:
    """Normalize case and surrounding whitespace without changing punctuation."""
    return code.strip().upper()