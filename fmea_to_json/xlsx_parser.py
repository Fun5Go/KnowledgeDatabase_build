import os
import json
import math
import pandas as pd
import numpy as np
from fmea_to_json.common_utils import (
    to_scalar,
    extract_metadata_from_file,
    is_numeric_like
)

###############################################################################
# Helpers
###############################################################################
def norm_col(x):
    return " ".join(str(x).lower().strip().split())


def build_col_map(header_row):
    return {
        norm_col(v): i
        for i, v in enumerate(header_row.tolist())
        if str(v).strip() != ""
    }


def get_cell(row, col_map, *header_names, default_idx=None):
    """
    Get value using header map first, fallback to fixed index.
    """
    for name in header_names:
        idx = col_map.get(norm_col(name))
        if idx is not None:
            return to_scalar(row.iloc[idx])

    if default_idx is not None:
        return to_scalar(row.iloc[default_idx])

    return ""


def get_int_cell(row, col_map, *header_names, default_idx=None):
    """
    Same as get_cell, but ensures numeric-like value.
    """
    val = get_cell(row, col_map, *header_names, default_idx=default_idx)
    return val if is_numeric_like(val) else ""


###############################################################################
# Core extraction
###############################################################################
def extract_old_fmea_failures(df, metadata, file_name):
    records = []

    # ------------------------------------------------------------
    # 1. Locate header row
    # ------------------------------------------------------------
    header_idx = -1
    for i in range(min(20, len(df))):
        row_text = " ".join(df.iloc[i].astype(str).str.lower())
        if "process step" in row_text:
            header_idx = i
            break

    if header_idx == -1:
        return records

    header_row = df.iloc[header_idx]
    col_map = build_col_map(header_row)

    df_data = df.iloc[header_idx + 1:].dropna(how="all")

    # ------------------------------------------------------------
    # 2. Iterate rows
    # ------------------------------------------------------------
    for _, row in df_data.iterrows():

        failure_cause = get_cell(
            row, col_map,
            "potential cause(s) of failure",
            default_idx=5
        )

        if not failure_cause:
            continue

        failure_type = get_cell(
            row, col_map,
            "process step",
            default_idx=1
        )

        failure_mode = get_cell(
            row, col_map,
            "potential failure mode",
            default_idx=2
        )

        failure_effect = get_cell(
            row, col_map,
            "potential effect(s) of failure",
            default_idx=3
        )

        severity = get_int_cell(
            row, col_map,
            "severity",
            default_idx=4
        )

        occurrence = get_int_cell(
            row, col_map,
            "occurrence",
            default_idx=6
        )

        detection = get_int_cell(
            row, col_map,
            "detection",
            default_idx=9
        )

        rpn = get_int_cell(
            row, col_map,
            "rpn", "so",
            default_idx=10
        )

        current_detection = get_cell(
            row, col_map,
            "current controls",
            default_idx=8
        )

        recommended_action = get_cell(
            row, col_map,
            "recommended actions",
            "recommended action",
            default_idx=11
        )

        # ------------------------------------------------------------
        # 3. Build record (JSON schema UNCHANGED)
        # ------------------------------------------------------------
        record = {
            "source_type": "old_fmea",
            "file_name": file_name,

            "project_description": metadata["project_description"],
            "fmea_date": metadata["fmea_date"],

            "failure_type": failure_type,
            "failure_mode": failure_mode,
            "failure_effect": failure_effect,

            "severity": severity,
            "occurrence": occurrence,
            "detection": detection,
            "rpn": rpn,

            "failure_cause": failure_cause,
            "current_detection": current_detection,
            "recommended_action": recommended_action,

            "text": (
                f"Failure mode {failure_mode}. "
                f"Cause {failure_cause} leads to effect {failure_effect}. "
                f"Severity {severity}, "
                f"Occurrence {occurrence}, "
                f"Detection {detection}, "
                f"RPN {rpn}. "
                f"Action {recommended_action}."
            )
        }

        records.append(record)

    return records


###############################################################################
# Main entry
###############################################################################
def process_old_fmea_xlsx(path, output_json):
    file_name = os.path.splitext(os.path.basename(path))[0]

    df = pd.read_excel(
        path,
        sheet_name="FMEA",
        header=None,
        engine="openpyxl"
    )

    meta = extract_metadata_from_file(
        path,
        sheet_name="FMEA",
        project_cell="E2",
        date_cell="J4",
        date_fallback_cell="J3"
    )

    records = extract_old_fmea_failures(df, meta, file_name)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("Old FMEA JSON saved to:", output_json)
