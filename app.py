import math
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.config import DERIVED_DATA_DIR, RAW_DATA_DIR
from src.coverage import fetch_policy_index, load_policy_index, save_policy_index, search_policy_index
from src.coverage_ingest import (
    build_local_coverage_relationships,
    load_local_coverage_details,
)
from src.anomaly import build_provider_anomaly_table
from src.data_loader import discover_data_files
from src.hcpcs_reference import get_long_description
from src.procedure_analytics import (
    get_procedure_codes,
    get_procedure_details,
    get_provider_peer_comparison,
)


st.set_page_config(page_title="CPT/HCPCS Procedure Intelligence", layout="wide")
st.title("CPT/HCPCS Procedure Intelligence")
st.caption("Local utilization explorer for observed CMS synthetic HCPCS/CPT claims.")

with st.expander("Local source files"):
    files = discover_data_files()
    if files:
        st.dataframe(
            [
                {
                    "File": path.name,
                    "Format": path.suffix.removeprefix("."),
                    "Size (MB)": round(path.stat().st_size / (1024 * 1024), 2),
                }
                for path in files
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Place local source files in data/raw. Data files are excluded from Git.")

st.header("Procedure Explorer")
procedure_codes = get_procedure_codes()
if procedure_codes is None:
    st.info("Procedure summaries have not been built yet.")
    st.code("python -m src.procedure_analytics", language="bash")
    st.stop()
if procedure_codes.empty:
    st.warning("No observed HCPCS/CPT codes were found in the processed data.")
    st.stop()

code_claim_counts = dict(
    zip(procedure_codes["procedure_code"], procedure_codes["claim_count"])
)
selected_code = st.selectbox(
    "Observed HCPCS/CPT code",
    procedure_codes["procedure_code"].tolist(),
    format_func=lambda code: f"{code} ({int(code_claim_counts[code]):,} claims)",
)
coverage_outputs_ready = (
    DERIVED_DATA_DIR / "coverage_policy_matches.parquet"
).is_file()
local_coverage = (
    load_local_coverage_details(selected_code) if coverage_outputs_ready else None
)
selected_description = get_long_description(selected_code)
if selected_description:
    st.caption("HCPCS/CPT long description")
    st.write(selected_description)
else:
    st.caption("Long description is not available for this code in the local reference files.")

details = get_procedure_details(selected_code)
summary = details["summary"]
if summary is None:
    st.warning("No procedure summary is available for this code.")
    st.stop()

st.subheader(selected_code)
claim_column, beneficiary_column, provider_column, diagnosis_column = st.columns(4)
claim_column.metric("Claims", f"{int(summary['claim_count']):,}")
beneficiary_column.metric(
    "Beneficiaries", f"{int(summary['unique_beneficiaries']):,}"
)
provider_column.metric("Providers", f"{int(summary['unique_providers']):,}")
diagnosis_column.metric("Diagnoses", f"{int(summary['unique_diagnoses']):,}")

allowed = summary["total_allowed"]
if allowed is None or math.isnan(float(allowed)):
    allowed_display = "Not available"
else:
    allowed_display = f"${float(allowed):,.2f}"
first_service = summary["first_service_date"]
last_service = summary["last_service_date"]
first_service_display = (
    first_service.strftime("%Y-%m-%d")
    if hasattr(first_service, "strftime")
    else str(first_service or "Not available")
)
last_service_display = (
    last_service.strftime("%Y-%m-%d")
    if hasattr(last_service, "strftime")
    else str(last_service or "Not available")
)
allowed_column, place_column, start_column, end_column = st.columns(4)
allowed_column.metric("Allowed amount", allowed_display)
allowed_column.caption("Carrier claim lines only")
place_column.metric(
    "Carrier places of service",
    f"{int(summary['unique_places_of_service']):,}",
)
start_column.metric("First service", first_service_display)
end_column.metric("Last service", last_service_display)

diagnosis_tab, provider_tab, place_tab, trend_tab = st.tabs(
    ["Diagnoses", "Providers", "Place of service", "Trend"]
)
with diagnosis_tab:
    diagnoses = details["diagnoses"]
    st.caption(
        "A claim can contain multiple diagnoses, so relationship percentages may sum above 100%."
    )
    if diagnoses.empty:
        st.info("No diagnosis codes are available for this procedure.")
    else:
        st.dataframe(diagnoses, hide_index=True, use_container_width=True)

with provider_tab:
    providers = details["providers"]
    st.caption(
        "Procedure rate is claims per beneficiary observed for that provider and code, not an eligible-population utilization rate."
    )
    if providers.empty:
        st.info("No provider identifiers are available for this procedure.")
    else:
        st.dataframe(providers, hide_index=True, use_container_width=True)

with place_tab:
    places = details["places"]
    st.caption("Place-of-service data is available from Carrier claim lines only.")
    if places.empty:
        st.info("No Carrier place-of-service rows are available for this code.")
    else:
        st.bar_chart(places, x="place_of_service_code", y="claim_count")
        st.dataframe(places, hide_index=True, use_container_width=True)

with trend_tab:
    trends = details["trends"]
    if trends.empty:
        st.info("No service dates are available for this procedure.")
    else:
        st.line_chart(
            trends.set_index("service_month")[["claim_count", "beneficiary_count"]]
        )
        st.dataframe(trends, hide_index=True, use_container_width=True)

st.divider()
st.header("Provider Investigation")
provider_stats_path = DERIVED_DATA_DIR / "provider_procedure_stats.parquet"
if provider_stats_path.is_file():
    provider_stats = pd.read_parquet(provider_stats_path)
    provider_options = sorted(provider_stats["provider_id"].dropna().unique().tolist())
    if not provider_options:
        st.info("No provider IDs are available in the local aggregate tables.")
    else:
        selected_provider = st.selectbox("Provider ID", provider_options)
        procedure_options = sorted(
            provider_stats[provider_stats["provider_id"] == selected_provider]["procedure_code"]
            .dropna()
            .unique()
            .tolist()
        )
        if not procedure_options:
            st.info("No procedures were observed for the selected provider.")
        else:
            selected_provider_procedure = st.selectbox(
                "Procedure code for peer comparison",
                procedure_options,
            )
            comparison = get_provider_peer_comparison(selected_provider, selected_provider_procedure)
            provider_matches = provider_stats[
                (provider_stats["provider_id"] == selected_provider)
                & (provider_stats["procedure_code"] == selected_provider_procedure)
            ]
            provider_row = provider_matches.iloc[0].to_dict() if not provider_matches.empty else {}
            peer_rows = provider_stats[
                provider_stats["procedure_code"] == selected_provider_procedure
            ].copy()
            peer_rows = peer_rows.sort_values("procedure_rate", ascending=False)

            provider_rate_display = (
                f"{comparison['provider_rate']:.2f}"
                if comparison["provider_rate"] is not None
                else "N/A"
            )
            peer_median_display = (
                f"{comparison['peer_median']:.2f}"
                if comparison["peer_median"] is not None
                else "N/A"
            )
            rate_ratio_display = (
                f"{comparison['rate_ratio']:.2f}x"
                if comparison["rate_ratio"] is not None
                else "N/A"
            )
            percentile_display = (
                f"{comparison['percentile']:.2f}th"
                if comparison["percentile"] is not None
                else "N/A"
            )

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Provider rate", provider_rate_display)
            col_b.metric("Peer median", peer_median_display)
            col_c.metric("Rate ratio", rate_ratio_display)
            col_d.metric("Peer percentile", percentile_display)

            st.caption(
                "Peer comparison uses the observed claim rate for the selected procedure across all providers in the local aggregate tables."
            )
            st.dataframe(
                peer_rows[
                    ["provider_id", "procedure_code", "claim_count", "beneficiary_count", "procedure_rate"]
                ].head(25),
                hide_index=True,
                use_container_width=True,
            )
            st.caption("Provider detail")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Metric": [
                            "Provider ID",
                            "Procedure code",
                            "Claim count",
                            "Beneficiary count",
                            "Procedure rate",
                            "Peer median",
                            "Peer 90th percentile",
                            "Peer 95th percentile",
                            "Peer percentile",
                        ],
                        "Value": [
                            selected_provider,
                            selected_provider_procedure,
                            provider_row.get("claim_count"),
                            provider_row.get("beneficiary_count"),
                            provider_row.get("procedure_rate"),
                            comparison["peer_median"],
                            comparison["peer_p90"],
                            comparison["peer_p95"],
                            comparison["percentile"],
                        ],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
else:
    st.info("Provider-level procedure statistics have not been built yet. Run python -m src.procedure_analytics first.")

st.divider()
st.header("Anomaly Intelligence")
try:
    anomaly_table = build_provider_anomaly_table(DERIVED_DATA_DIR)
    if anomaly_table.empty:
        st.info("No provider-level anomaly scores are available yet.")
    else:
        flagged = anomaly_table[anomaly_table["anomaly_flag"] | (anomaly_table["peer_percentile"] >= 90)].copy()
        st.caption(
            "Anomaly flags are a prioritization signal only; they do not establish inappropriate care or billing."
        )
        if flagged.empty:
            st.info("No strong utilization outliers were detected in the current local peer comparison.")
        else:
            st.dataframe(
                flagged[
                    [
                        "provider_id",
                        "procedure_code",
                        "provider_rate",
                        "peer_median",
                        "provider_to_peer_ratio",
                        "peer_percentile",
                        "anomaly_score",
                        "anomaly_flag",
                    ]
                ].head(25),
                hide_index=True,
                use_container_width=True,
            )
except FileNotFoundError:
    st.info("Provider-level anomaly scores are not available until the analytical tables are built.")

st.divider()
st.header("Coverage Intelligence")
st.caption("Browse CMS MCD policy summaries and their official source pages.")
coverage_archive_names = ["current_lcd.zip", "current_article.zip", "ncd.zip"]
coverage_archives_ready = all(
    (RAW_DATA_DIR / "coverage" / filename).is_file()
    for filename in coverage_archive_names
)
if coverage_outputs_ready:
    st.info(
        f"Showing code relationships from the local current MCD downloads for {selected_code}. "
        "An absent link in these files is not a coverage determination."
    )
else:
    local_coverage = None
    st.warning(
        f"Code-level coverage relationship for {selected_code} has not been checked. "
        "The public summary index below is not a procedure match; absence from it "
        "must not be interpreted as noncoverage."
    )

if coverage_archives_ready:
    st.caption("Local current LCD, Article, and NCD archives are available.")
    if st.button("Rebuild code relationships from local MCD files"):
        try:
            with st.spinner("Scanning local code tables for observed procedures..."):
                import_counts = build_local_coverage_relationships()
                st.session_state.coverage_import_counts = import_counts
            st.rerun()
        except Exception as error:
            st.error(f"Local coverage import failed: {error}")
elif not coverage_outputs_ready:
    st.info(
        "Place current_lcd.zip, current_article.zip, and ncd.zip in "
        "data/raw/coverage to enable local code-table matching."
    )

if local_coverage is not None:
    policy_matches = local_coverage["policy_matches"]
    st.subheader("Direct policy code links")
    if policy_matches.empty:
        st.info(
            "No direct HCPCS/CPT link for this code was found in the current local "
            "LCD/Article code tables. This does not establish noncoverage."
        )
    else:
        st.dataframe(
            policy_matches[
                [
                    "policy_type",
                    "display_id",
                    "title",
                    "status",
                    "effective_date",
                    "retirement_date",
                    "contractor",
                    "jurisdiction",
                    "relationship_type",
                    "source_url",
                ]
            ],
            column_config={
                "source_url": st.column_config.LinkColumn(
                    "CMS source", display_text="Open policy"
                )
            },
            hide_index=True,
            use_container_width=True,
        )

    icd_relationships = local_coverage["icd10_relationships"]
    st.subheader("Article ICD-10 relationships")
    st.caption(
        "These are diagnosis-code entries in matched Article tables. A noncovered "
        "entry is specific to that Article and is not a global service determination."
    )
    if icd_relationships.empty:
        st.info("No Article ICD-10 relationship rows were found for this procedure.")
    else:
        st.dataframe(
            icd_relationships[
                [
                    "policy_id",
                    "policy_title",
                    "icd10_code",
                    "icd10_description",
                    "code_relationship",
                    "source_url",
                ]
            ],
            column_config={
                "source_url": st.column_config.LinkColumn(
                    "CMS source", display_text="Open Article"
                )
            },
            hide_index=True,
            use_container_width=True,
            height=360,
        )

    ncd_relationships = local_coverage["ncd_relationships"]
    st.subheader("Related NCD references")
    if ncd_relationships.empty:
        st.info(
            "No positive NCD cross-reference was listed by the matched Articles/LCDs "
            "in these current exports. This is not proof that no NCD applies."
        )
    else:
        st.dataframe(
            ncd_relationships[
                [
                    "source_policy_type",
                    "source_policy_id",
                    "source_policy_version",
                    "related_policy_id",
                    "related_policy_title",
                    "effective_date",
                    "relationship_type",
                    "source_url",
                ]
            ],
            column_config={
                "source_url": st.column_config.LinkColumn(
                    "CMS source", display_text="Open NCD"
                )
            },
            hide_index=True,
            use_container_width=True,
        )

if "coverage_policy_index" not in st.session_state:
    st.session_state.coverage_policy_index = load_policy_index()

refresh_column, snapshot_column = st.columns([1, 3])
refresh_requested = refresh_column.button("Refresh CMS policy index")
if refresh_requested:
    try:
        with st.spinner("Retrieving current CMS policy summaries..."):
            index = fetch_policy_index()
            save_policy_index(index)
            st.session_state.coverage_policy_index = index
            st.session_state.coverage_policy_index_fetched_at = datetime.now(
                timezone.utc
            ).isoformat()
    except Exception as error:
        st.error(f"CMS policy index refresh failed: {error}")

policy_index = st.session_state.coverage_policy_index
if policy_index is None:
    st.info("No local CMS policy snapshot yet. Refresh to retrieve public policy summaries.")
else:
    fetched_at = (
        st.session_state.get("coverage_policy_index_fetched_at")
        or policy_index["fetched_at"].iloc[0]
    )
    snapshot_column.caption(f"Local CMS snapshot fetched: {fetched_at}")
    selected_policy_types = st.multiselect(
        "Policy types",
        ["NCD", "LCD", "Article"],
        default=["NCD", "LCD", "Article"],
    )
    policy_query = st.text_input(
        "Search policy title or ID",
        placeholder="Enter policy text or a document ID",
    )
    filtered_policies = search_policy_index(
        policy_index, policy_query, selected_policy_types
    )
    st.caption(f"{len(filtered_policies):,} policy summaries")
    if filtered_policies.empty:
        st.info("No policy summaries match those title/ID filters.")
    else:
        display_columns = [
            "policy_type",
            "display_id",
            "title",
            "status",
            "effective_date",
            "retirement_date",
            "last_updated",
            "contractor",
            "jurisdiction",
            "source_url",
        ]
        st.dataframe(
            filtered_policies[display_columns],
            column_config={
                "source_url": st.column_config.LinkColumn(
                    "CMS source", display_text="Open policy"
                )
            },
            hide_index=True,
            use_container_width=True,
            height=420,
        )