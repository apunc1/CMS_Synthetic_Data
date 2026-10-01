from src.llm import build_explanation_prompt, generate_explanation


def test_build_explanation_prompt_binds_to_evidence_only() -> None:
    prompt = build_explanation_prompt(
        "Observed claim count: 96002",
        "LCD 12345: covered for the selected code.",
        provider_summary="Provider rate 0.60 vs peer median 0.20",
        anomaly_summary="Peer percentile 97th.",
    )

    assert "Use only the evidence supplied below" in prompt
    assert "Observed claim count: 96002" in prompt
    assert "Provider rate 0.60 vs peer median 0.20" in prompt
    assert "Do not infer medical necessity" in prompt


def test_generate_explanation_returns_evidence_based_summary() -> None:
    result = generate_explanation(
        {
            "procedure_code": "99241",
            "description": "Office consultation",
            "summary": {"claim_count": 96002, "unique_beneficiaries": 42000},
            "provider_comparison": {
                "provider_rate": 0.60,
                "peer_median": 0.20,
                "rate_ratio": 3.0,
                "percentile": 97.0,
            },
            "coverage_evidence": [
                {"policy_type": "LCD", "display_id": "L12345", "title": "Office visits"}
            ],
            "source_links": ["https://example.com/policy/L12345"],
        }
    )

    assert result["explanation"]
    assert "99241" in result["explanation"]
    assert "observed" in result["explanation"].lower()
    assert "coverage" in result["explanation"].lower()
    assert result["sources"]
    assert "not a coverage determination" in result["limitations"].lower()
