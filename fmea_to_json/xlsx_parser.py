import os
import json
import pandas as pd
from fmea_to_json.common_utils import to_scalar
from fmea_to_json.common_utils import extract_metadata_from_file, is_numeric_like


def extract_metadata_old(df):
    metadata = {"project_description": "Unknown", "fmea_date": "Unknown"}

    try:
        metadata["project_description"] = to_scalar(df.iloc[1, 4])
    except:
        pass

    try:
        metadata["fmea_date"] = to_scalar(df.iloc[3, 9]) or to_scalar(df.iloc[2, 9])
    except:
        pass

    return metadata


# --------- minimal helpers (NEW) ----------
def norm_col(x):
    return " ".join(str(x).lower().strip().split())


def build_col_map(header_row):
    return {
        norm_col(v): i
        for i, v in enumerate(header_row.tolist())
        if str(v).strip() != ""
    }
# ------------------------------------------


def extract_old_fmea_failures(df, metadata, file_name):
    records = []

    header_idx = -1
    for i in range(min(15, len(df))):
        if "process step" in " ".join(df.iloc[i].astype(str).str.lower()):
            header_idx = i
            break

    if header_idx == -1:
        return records

    # NEW: header-based column map
    header_row = df.iloc[header_idx]
    col_map = build_col_map(header_row)

    df_data = df.iloc[header_idx + 1:]

    for _, row in df_data.iterrows():
        if len(row) < 12:
            continue

        # default (fixed index)
        failure_cause = to_scalar(row.iloc[5])

        # NEW: numeric cause -> use header name map
        if is_numeric_like(failure_cause):
            failure_cause = to_scalar(
                row.iloc[col_map.get("potential cause(s) of failure")]
            )
            current_detection = to_scalar(
                row.iloc[col_map.get("current controls")]
            )
            recommended_action = to_scalar(
                row.iloc[col_map.get("recommended actions")]
            )
        else:
            current_detection = to_scalar(row.iloc[8])
            recommended_action = to_scalar(row.iloc[11])

        if not failure_cause:
            continue

        record = {
            "source_type": "old_fmea",
            "file_name": file_name,

            "project_description": metadata["project_description"],
            "fmea_date": metadata["fmea_date"],

            # NEW: Process Step -> failure_type
            "failure_type": to_scalar(
                row.iloc[col_map.get("process step", 1)]
            ),
            "failure_mode": to_scalar(row.iloc[2]),
            "failure_effect": to_scalar(row.iloc[3]),

            "severity": to_scalar(row.iloc[4]),
            "occurrence": to_scalar(row.iloc[6]),
            "detection": to_scalar(row.iloc[9]),
            "rpn": to_scalar(row.iloc[10]),

            "failure_cause": failure_cause,
            "current_detection": current_detection,
            "recommended_action": recommended_action,

            "text": (
                f"Failure mode {to_scalar(row.iloc[2])}. "
                f"Cause {failure_cause} leads to effect {to_scalar(row.iloc[3])}. "
                f"Severity {to_scalar(row.iloc[4])}, "
                f"Occurrence {to_scalar(row.iloc[6])}, "
                f"Detection {to_scalar(row.iloc[9])}, "
                f"RPN {to_scalar(row.iloc[10])}. "
                f"Action {recommended_action}."
            )
        }

        records.append(record)

    return records


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
