from collections.abc import Iterable, Mapping
from typing import Any


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