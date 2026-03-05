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

def has_failure_content(row, col_map):
    """
    Return True if this row contains any failure-related information
    (mode / effect / cause).
    """
    keys = [
        "potential failure mode",
        "potential effect(s) of failure",
        "potential cause(s) of failure",
    ]
    for k in keys:
        val = get_cell(row, col_map, k)
        if isinstance(val, str) and val.strip():
            return True
    return False



###############################################################################
# Core extraction
###############################################################################
def extract_old_fmea_failures(df, metadata, file_name):
    records = []

    # 1. Locate header row
    header_idx = -1
    for i in range(min(20, len(df))):
        row_text = " ".join(df.iloc[i].astype(str).str.lower())
        if ("process step" in row_text) or ("function" in row_text):
            header_idx = i
            break
    if header_idx == -1:
        raise ValueError("Header row not found (process step / function)")
    header_row = df.iloc[header_idx]
    col_map = build_col_map(header_row)
    df_data = df.iloc[header_idx + 1:].dropna(how="all")

    last_process_step = ""

    # 2. Iterate rows
    for row_idx, row in df_data.iterrows():

        process_step_raw = get_cell(
            row, col_map,
            "process step",
            default_idx=1
        )
        if not process_step_raw:
            process_step_raw = get_cell(
                row, col_map,
                "function"
            )

        failure_cause = get_cell(
            row, col_map,
            "potential cause(s) of failure",
        )
        failure_mode = get_cell(
            row, col_map,
            "potential failure mode",
        )
        failure_effect = get_cell(
            row, col_map,
            "potential effect(s) of failure",
        )
        
        is_failure_row = bool(failure_mode or failure_effect or failure_cause)
        if process_step_raw:
            process_step = process_step_raw
            last_process_step = process_step_raw
        else:
            process_step = last_process_step if is_failure_row else ""

        severity = get_int_cell(row, col_map, "severity", default_idx=4)
        occurrence = get_int_cell(row, col_map, "occurrence", default_idx=6)
        detection = get_int_cell(row, col_map, "detection", default_idx=9)

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
        def safe_str(x):
            return x.strip() if isinstance(x, str) and x.strip() else ""

        def detect_cause_discipline(cause_text, element_text):

            def norm(x):
                return x.lower().strip() if isinstance(x, str) else ""

            element = norm(element_text)
            cause = norm(cause_text)

            hardware_keys = [
                "hardware", "hw",
                "electronic", "electronics",
                "electrical",
            ]

            software_keys = [
                "software", "sw", "esw"
            ]

            mech_keys = [
                "mechanic", "mechanical", "mch", "mech"
            ]

            for k in hardware_keys:
                if k in element:
                    return "hardware"

            for k in software_keys:
                if k in element:
                    return "software"

            for k in mech_keys:
                if k in element:
                    return "mechanics"

            # fallback to cause
            for k in hardware_keys:
                if k in cause:
                    return "hardware"

            for k in software_keys:
                if k in cause:
                    return "software"

            for k in mech_keys:
                if k in cause:
                    return "mechanics"

            return "unknown"

        cause_discipline = detect_cause_discipline(failure_cause,process_step)
        text = (
            f"Product: {safe_str(metadata.get('productName'))}. "
            f"Process step: {safe_str(process_step)}. "
            f"Failure mode: {safe_str(failure_mode)}. "
            f"Failure cause: {safe_str(failure_cause)}. "
            f"Failure effect: {safe_str(failure_effect)}. "
            f"Current controls: {safe_str(current_detection)}. "
            f"Recommended action: {safe_str(recommended_action)}."
        )

        record = {
            "source_type": "old_fmea",
            "file_name": file_name,

            # ===== metadata 扩展 =====
            "project_description": metadata.get("project_description"),
            "released": metadata.get("released"),
            "productId": metadata.get("productId"),
            "productPnId": metadata.get("productPnId"),
            "productName": metadata.get("productName"),

            "process_step": process_step,
            "failure_mode": failure_mode,
            "failure_effect": failure_effect,

            "severity": severity,
            "occurrence": occurrence,
            "detection": detection,
            "rpn": rpn,

            "failure_cause": failure_cause,
            "current_detection": current_detection,
            "recommended_action": recommended_action,
            "cause_discipline": cause_discipline, 

            "text" : text,
        }

        records.append(record)

    return records

###############################################################################
# Main entry
###############################################################################
def process_old_fmea_xlsx(path, output_json, fmea_index):
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
    fmea_date = meta.get("fmea_date", "")


    # ===== Add metadata from json list=====
    idx = {}
    if fmea_index and file_name in fmea_index:
        if isinstance(fmea_index[file_name], dict):
            idx = fmea_index.get(file_name, {})
        else:
            print(f"⚠ fmea_index[{file_name}] is not dict:", fmea_index[file_name])
    meta.update({
        "released": idx.get("released") or fmea_date or None,
        "productId": idx.get("productId"),
        "productPnId": idx.get("productPnId"),
        "productName": idx.get("productName"),
    })

    records = extract_old_fmea_failures(df, meta, file_name)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("Old FMEA JSON saved to:", output_json)
