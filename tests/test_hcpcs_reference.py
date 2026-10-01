from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from src.hcpcs_reference import load_anweb_descriptions, load_cpt_descriptions


def test_load_anweb_descriptions_reads_long_description_from_archive(tmp_path) -> None:
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "HCPC": ["A1001", 96156, "B2002"],
                "LONG DESCRIPTION": [
                    "Dressing for one wound",
                    "Cognitive assessment",
                    "Dressing for two wounds",
                ],
            }
        ).to_excel(writer, index=False, sheet_name="HCPC2026_OCT_contr")

    archive_path = tmp_path / "hcpc2026_oct_anweb_09232026.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "HCPC2026_OCT_ANWEB_09232026.xlsx",
            excel_buffer.getvalue(),
        )

    descriptions = load_anweb_descriptions(tmp_path)

    assert descriptions["A1001"] == "Dressing for one wound"
    assert descriptions["96156"] == "Cognitive assessment"
    assert descriptions["B2002"] == "Dressing for two wounds"


def test_load_cpt_descriptions_reads_cpt_reference_file(tmp_path) -> None:
    cpt_path = tmp_path / "cpt_descriptions.csv"
    cpt_path.write_text(
        "CPT,SHORT DESCRIPTION,LONG DESCRIPTION\n99241,Office consult,Office consultation for a new or established patient\n",
        encoding="utf-8",
    )

    descriptions = load_cpt_descriptions(tmp_path)

    assert descriptions["99241"] == "Office consultation for a new or established patient"
