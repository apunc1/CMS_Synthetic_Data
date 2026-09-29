import streamlit as st

from src.data_loader import discover_data_files


st.set_page_config(page_title="CPT/HCPCS Procedure Intelligence", layout="wide")
st.title("CPT/HCPCS Procedure Intelligence")
st.caption("Local prototype for exploring procedure utilization and coverage evidence.")

files = discover_data_files()
st.subheader("Local source files")
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

st.subheader("Investigate a procedure")
st.text_input("CPT/HCPCS code", placeholder="Enter a procedure code")
st.info("Procedure analysis becomes available after source columns are mapped in the ingestion workflow.")