import math

import streamlit as st

from src.config import DERIVED_DATA_DIR
from src.data_loader import discover_data_files
from src.procedure_analytics import get_procedure_codes, get_procedure_details


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
st.caption(
    "Codes are observed in the current claims files; a CMS HCPCS description reference has not been added yet."
)

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