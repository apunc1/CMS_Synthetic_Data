from __future__ import annotations

import os
from typing import Any, Mapping


def build_explanation_prompt(
    procedure_summary: str,
    coverage_evidence: str,
    provider_summary: str = "",
    anomaly_summary: str = "",
) -> str:
    """Build an evidence-bounded prompt for explanation generation."""
    provider_context = f"Provider/peer summary:\n{provider_summary}\n\n" if provider_summary else ""
    anomaly_context = f"Anomaly summary:\n{anomaly_summary}\n\n" if anomaly_summary else ""
    return (
        "You are explaining healthcare claims analytics.\n"
        "Use only the evidence supplied below.\n"
        "Do not infer medical necessity.\n"
        "Do not make unsupported claims about fraud or improper billing.\n"
        "Distinguish observed data from interpretation.\n"
        "Separate observed facts from uncertainty.\n"
        "Do not infer a coverage decision.\n\n"
        f"Procedure summary:\n{procedure_summary}\n\n"
        f"{provider_context}"
        f"{anomaly_context}"
        f"Coverage evidence:\n{coverage_evidence}"
    )


def _format_summary_value(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _fallback_explanation(context: Mapping[str, Any]) -> str:
    code = context.get("procedure_code", "selected code")
    description = context.get("description") or "No description available in the local references."
    summary = context.get("summary") or {}
    claim_count = summary.get("claim_count")
    beneficiary_count = summary.get("unique_beneficiaries")
    diagnosis_count = summary.get("unique_diagnoses")
    provider_comp = context.get("provider_comparison") or {}
    provider_rate = provider_comp.get("provider_rate")
    peer_median = provider_comp.get("peer_median")
    rate_ratio = provider_comp.get("rate_ratio")
    percentile = provider_comp.get("percentile")
    coverage_rows = context.get("coverage_evidence") or []

    observed = (
        f"Observed data for {code} show {claim_count:,} claims and "
        f"{beneficiary_count:,} unique beneficiaries in the local synthetic dataset."
        if claim_count is not None and beneficiary_count is not None
        else f"Observed data for {code} are available in the local synthetic dataset."
    )
    context_text = (
        f"The local records describe this service as: {description}. "
        if description
        else ""
    )
    peer_text = ""
    if provider_rate is not None and peer_median is not None:
        peer_text = (
            f"For the selected provider, the observed rate is {_format_summary_value(provider_rate)} "
            f"versus a peer median of {_format_summary_value(peer_median)}; the rate ratio is "
            f"{_format_summary_value(rate_ratio)} and the peer percentile is "
            f"{_format_summary_value(percentile)}th."
        )
    elif provider_rate is not None:
        peer_text = f"The selected provider's rate is {_format_summary_value(provider_rate)} in the local peer comparison."

    coverage_text = ""
    if coverage_rows:
        first_row = coverage_rows[0]
        policy_id = first_row.get("display_id") or first_row.get("policy_id") or "policy record"
        policy_type = first_row.get("policy_type") or "coverage document"
        coverage_text = (
            f"Coverage evidence indicates a relevant {policy_type} entry ({policy_id}) in the local CMS files; "
            "this is evidence for a policy relationship, not a final coverage determination."
        )
    else:
        coverage_text = "No direct local coverage relationship was identified for this code in the current evidence set."

    diagnosis_text = ""
    if diagnosis_count is not None:
        diagnosis_text = f"The local code summary includes {diagnosis_count:,} diagnosis associations."

    return (
        f"Observed facts: {observed} {context_text}{diagnosis_text} {peer_text} {coverage_text} "
        "This explanation is an evidence-based summary only. It does not state that the service is medically necessary, "
        "inappropriate, or improperly billed, and it is not a coverage determination."
    )


def generate_explanation(
    context: Mapping[str, Any],
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Return a structured explanation based on local analytical evidence.

    If an OpenAI-compatible API key is present, the function will attempt to use it;
    otherwise it falls back to a deterministic, evidence-based explanation so the app
    remains usable without external credentials.
    """
    procedure_summary = context.get("procedure_summary") or ""
    if not procedure_summary:
        summary = context.get("summary") or {}
        claim_count = summary.get("claim_count")
        claim_text = f"Claim count: {claim_count:,}" if claim_count is not None else "Claim count: not available"
        procedure_summary = (
            f"Procedure code: {context.get('procedure_code', 'selected code')}\n"
            f"Description: {context.get('description', 'not available')}\n"
            f"{claim_text}"
        )

    provider_summary = context.get("provider_summary") or ""
    if not provider_summary and context.get("provider_comparison"):
        provider = context["provider_comparison"]
        provider_summary = (
            f"Provider rate: {provider.get('provider_rate')}; peer median: {provider.get('peer_median')}; "
            f"rate ratio: {provider.get('rate_ratio')}; peer percentile: {provider.get('percentile')}"
        )

    anomaly_summary = context.get("anomaly_summary") or ""
    coverage_rows = context.get("coverage_evidence") or []
    coverage_evidence = "\n".join(
        (
            f"- {row.get('policy_type', 'Coverage')} {row.get('display_id', row.get('policy_id', 'document'))}: "
            f"{row.get('title', 'No title available')}"
        )
        for row in coverage_rows
    ) or "No local CMS coverage relationship rows were available."

    prompt = build_explanation_prompt(
        procedure_summary=procedure_summary,
        coverage_evidence=coverage_evidence,
        provider_summary=provider_summary,
        anomaly_summary=anomaly_summary,
    )

    source_links = context.get("source_links") or []
    if not source_links and coverage_rows:
        source_links = [
            row.get("source_url") for row in coverage_rows if row.get("source_url")
        ]

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if resolved_api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=resolved_api_key)
            response = client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}],
            )
            content = getattr(response, "output_text", None)
            if content:
                return {
                    "prompt": prompt,
                    "explanation": content.strip(),
                    "model": model,
                    "sources": source_links,
                    "limitations": (
                        "This explanation is evidence-based and intentionally is not a coverage determination. "
                        "It does not infer medical necessity, fraud, or improper billing. It is a decision-support "
                        "summary, not a final adjudication."
                    ),
                }
        except Exception:
            pass

    fallback = _fallback_explanation(context)
    return {
        "prompt": prompt,
        "explanation": fallback,
        "model": "evidence-fallback",
        "sources": source_links,
        "limitations": (
            "This explanation is evidence-based and intentionally is not a coverage determination. "
            "It does not infer medical necessity, fraud, or improper billing. It is a decision-support summary, "
            "not a final adjudication."
        ),
    }