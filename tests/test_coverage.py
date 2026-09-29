from src.coverage import match_policy_codes


def test_match_policy_codes_is_case_insensitive_and_exact() -> None:
    policies = [
        {"id": "one", "procedure_codes": ["12345"]},
        {"id": "two", "procedure_codes": ["123456"]},
    ]
    assert match_policy_codes(policies, " 12345 ") == [policies[0]]