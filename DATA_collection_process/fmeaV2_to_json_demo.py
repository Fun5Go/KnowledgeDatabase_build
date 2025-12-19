import pandas as pd
import json
import numpy as np
import os
import math
from datetime import datetime,date
###############################################################################
# Helper: Clean and convert types
###############################################################################
def to_scalar(x):
    """Convert Pandas/NumPy objects into clean scalars (date -> YYYY-MM-DD)."""
    if isinstance(x, pd.Series):
        return to_scalar(x.iloc[0])

    # Handle pandas Timestamp (common from read_excel)
    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()

    # Handle python datetime/date
    if isinstance(x, datetime):
        return x.date().isoformat()
    if isinstance(x, date):
        return x.isoformat()

    if isinstance(x, np.generic):
        x = x.item()

    if isinstance(x, float) and math.isnan(x):
        return ""
    if x is None:
        return ""

    return str(x).strip()


###############################################################################
# 1. Extract Header Metadata (Fixed Coordinates)
###############################################################################
def extract_metadata(df):
    """
    Extracts Project Description (E2) and Date (J4) based on Excel coordinates.
    Note: Pandas is 0-indexed, so Excel Row 2 is Index 2, Column E is Index 4.
    """
    metadata = {
        "project_description": "Unknown",
        "fmea_date": "Unknown"
    }
    
    # Extract Project Description from E2 (Row index 1, Col index 4 in the provided CSV)
    try:
        # Based on CSV provided: Row 2 (Index 1) contains the description in Column E (Index 4)
        val = df.iloc[1, 4] 
        if to_scalar(val):
            metadata["project_description"] = to_scalar(val)
    except IndexError:
        pass

    # Extract Date from J4 (Row index 3, Col index 9 in the provided CSV)
    try:
        val = df.iloc[3, 9]
        if to_scalar(val):
            metadata["fmea_date"] = to_scalar(val)
        else:
            # Fallback: Try J3 (Index 2, 9) if J4 is empty (common in merged cells)
            val_fallback = df.iloc[2, 9]
            if to_scalar(val_fallback):
                metadata["fmea_date"] = to_scalar(val_fallback)
    except IndexError:
        pass

    return metadata

###############################################################################
# 2. Extract Failure Rows (Specific Column Mapping)
###############################################################################
def extract_failures(df, metadata, file_name):
    records = []
    
    # Find the header row usually starting with "Item" or "Process Step"
    # In the provided file, this is Row 7 (Index 6)
    header_idx = -1
    for i in range(min(15, len(df))):
        row_str = " ".join([str(x).lower() for x in df.iloc[i]])
        if "process step" in row_str and "potential effect" in row_str:
            header_idx = i
            break
            
    if header_idx == -1:
        print(f"Could not find header row in {file_name}")
        return []

    # Slice the dataframe to get only the data table
    df_data = df.iloc[header_idx + 1:].copy()
    
    # -------------------------------------------------------
    # Column Mapping (0-based Indexing)
    # A=0, B=1, C=2, D=3, E=4, F=5, G=6, H=7, I=8, J=9, K=10, L=11
    # -------------------------------------------------------
    # User Requests:
    # Failure Type (B) -> Index 1
    # Failure Mode (C) -> Index 2 (Implicitly needed for context, though not explicitly requested, extracting it as 'failure_mode_detail')
    # Failure Effect (D) -> Index 3
    # Severity (E) -> Index 4
    # Failure Cause (F) -> Index 5
    # Occurrence (G) -> Index 6
    # Current Control (I) -> Index 8 (Skipping H)
    # Detection (J) -> Index 9
    # RPN (K) -> Index 10
    # Recommended Action (L) -> Index 11
    # -------------------------------------------------------

    for _, row in df_data.iterrows():
        # Ensure row has enough columns
        if len(row) < 12: 
            continue
            
        # Extract columns by integer position
        failure_type = to_scalar(row.iloc[1]) # Column B
        # Column C is usually the actual "Failure Mode" in standard FMEA. 
        # Even though you mapped B to type, I'll grab C to ensure the text description is complete.
        failure_mode_detail = to_scalar(row.iloc[2]) # Column C
        
        failure_effect = to_scalar(row.iloc[3]) # Column D
        severity = to_scalar(row.iloc[4])       # Column E
        failure_cause = to_scalar(row.iloc[5])  # Column F
        occurrence = to_scalar(row.iloc[6])     # Column G
        # Column H is usually S*O, skipping
        current_control = to_scalar(row.iloc[8]) # Column I
        detection = to_scalar(row.iloc[9])      # Column J
        rpn = to_scalar(row.iloc[10])           # Column K
        rec_action = to_scalar(row.iloc[11])    # Column L
        
        # Skip empty rows (must have at least a failure type or cause)
        if not failure_type and not failure_cause:
            continue

        # Create Searchable Text Summary
        text_summary = (
            f"Project: {metadata['project_description']}. "
            f"Type/Step: {failure_type}. "
            f"Failure Mode: {failure_mode_detail}. "
            f"Cause: {failure_cause} leading to Effect: {failure_effect}. "
            f"Severity: {severity}, Occurrence: {occurrence}, Detection: {detection}, RPN: {rpn}. "
            f"Controls: {current_control}. Actions: {rec_action}."
        )

        record = {
            "file_name": file_name,
            "source_type": "old_fmea",
            "project_description": metadata['project_description'],
            "fmea_date": metadata['fmea_date'],
            
            "failure_type": failure_type,       # From Col B
            "failure_mode": failure_mode_detail,# From Col C (Added for clarity)
            "failure_effect": failure_effect,   # From Col D
            "severity": severity,               # From Col E
            "failure_cause": failure_cause,     # From Col F
            "occurrence": occurrence,           # From Col G
            "current_control": current_control, # From Col I
            "detection": detection,             # From Col J
            "rpn": rpn,                         # From Col K
            "recommended_action": rec_action,   # From Col L
            
            "text": text_summary
        }
        records.append(record)

    return records

###############################################################################
# 3. Main Processor
###############################################################################
def process_csv_fmea(file_path, output_json_path):
    file_name = os.path.basename(file_path)
    print(f"Processing {file_name}...")
    
    try:
        # Load CSV without header first to access by index
        df = pd.read_excel(file_path, sheet_name="FMEA", header=None, engine="openpyxl")
        
        # 1. Extract Metadata
        meta = extract_metadata(df)
        print(f"   Found Project: {meta['project_description']}")
        
        # 2. Extract Failures
        json_data = extract_failures(df, meta, file_name)
        
        # 3. Save to JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
            
        print(f"   Saved {len(json_data)} records to {output_json_path}")

    except Exception as e:
        print(f"Error processing file: {e}")

# Example Usage
if __name__ == "__main__":
    file_path = r"C:\Users\FW\Desktop\FMEA_AI\Project_Phase\DATA\FMEA\FMEA6337210047R02.xlsx"
    process_csv_fmea(file_path, 'output_old_format_example.json')