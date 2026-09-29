def build_explanation_prompt(procedure_summary: str, coverage_evidence: str) -> str:
    """Build an evidence-bounded prompt; provider/API integration is intentionally separate."""
    return (
        "Explain the procedure summary using only the evidence below. "
        "Separate observed facts from uncertainty and do not infer a coverage decision.\n\n"
        f"Procedure summary:\n{procedure_summary}\n\n"
        f"Coverage evidence:\n{coverage_evidence}"
    )