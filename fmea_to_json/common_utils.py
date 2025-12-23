import pandas as pd
import numpy as np
import math
import re
from datetime import datetime, date

###############################################################################
# Type Conversion
###############################################################################

def to_scalar(x):
    if isinstance(x, pd.Series):
        return to_scalar(x.iloc[0])

    if isinstance(x, pd.Timestamp):
        return x.date().isoformat()

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
# Text Cleaning
###############################################################################

def strip_prefix(text):
    return re.sub(r"^\[[^\]]+\]\s*", "", str(text)).strip()


def extract_discipline(cause_raw):
    m = re.match(r"^\[[^\]-]*-\s*([A-Za-z]+)\]\s*(.*)", str(cause_raw))
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, strip_prefix(cause_raw)




def excel_a1_to_rc(a1: str):
    a1 = a1.strip().upper()
    m = re.match(r"^([A-Z]+)(\d+)$", a1)
    if not m:
        raise ValueError(f"Invalid A1 cell address: {a1}")

    col_letters, row_num = m.group(1), int(m.group(2))
    col_num = 0
    for ch in col_letters:
        col_num = col_num * 26 + (ord(ch) - ord("A") + 1)

    return row_num - 1, col_num - 1  # 0-based


def extract_cell_from_df(df, cell: str):
    """Read an A1 cell like 'J4' from an already-loaded df (header=None)."""
    r, c = excel_a1_to_rc(cell)
    try:
        return to_scalar(df.iloc[r, c])
    except Exception:
        return ""


def extract_metadata_df(df, project_cell="E2", date_cell="J4", date_fallback_cell=None):
    """
    Generic metadata extractor from a DataFrame (header=None).

    Defaults:
      - project_cell: E2
      - date_cell: J4
      - date_fallback_cell: optional, e.g. "J3"
    """
    meta = {"project_description": "Unknown", "fmea_date": "Unknown"}

    proj = extract_cell_from_df(df, project_cell)
    if proj:
        meta["project_description"] = proj

    dt = extract_cell_from_df(df, date_cell)
    if dt:
        meta["fmea_date"] = dt
    elif date_fallback_cell:
        dt2 = extract_cell_from_df(df, date_fallback_cell)
        if dt2:
            meta["fmea_date"] = dt2

    return meta


def extract_metadata_from_file(
    path: str,
    sheet_name=None,
    sheet_index: int = 0,
    project_cell="E2",
    date_cell="J4",
    date_fallback_cell=None,
    engine="openpyxl",
):
    """
    Generic metadata extractor directly from an Excel file.
    Works for both .xlsx and .xlsm.

    - Use sheet_name if you know it (e.g. old .xlsx uses 'FMEA')
    - Otherwise use sheet_index (for your .xlsm parser you used sheet_index=1)

    Example:
      xlsx: extract_metadata_from_file(path, sheet_name="FMEA", date_cell="J4", date_fallback_cell="J3")
      xlsm: extract_metadata_from_file(path, sheet_index=1, date_cell="T4")
    """
    # only need to read up to max row of cells used (E2 -> row2, J4/T4 -> row4)
    df = pd.read_excel(
        path,
        sheet_name=sheet_name if sheet_name is not None else sheet_index,
        header=None,
        nrows=10,
        engine=engine
    )
    return extract_metadata_df(
        df,
        project_cell=project_cell,
        date_cell=date_cell,
        date_fallback_cell=date_fallback_cell
    )

def is_numeric_like(value):
    """Return True if value is numeric or numeric-like string."""
    try:
        if value is None:
            return False
        float(str(value).strip())
        return True
    except:
        return False