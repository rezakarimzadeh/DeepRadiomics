class config:
    DATA_PATH = "../../dataset"
    clinical_data = f"{DATA_PATH}/ClinicalData.xlsx"
    SAVE_DIR = "./"


import os, re, unicodedata, difflib
import pandas as pd

SPACE_LIKE = {
    "\u00A0": " ",  # NBSP
    "\u2007": " ",  # figure space
    "\u202F": " ",  # narrow NBSP
    "\u200B": "",   # zero-width space
    "\uFEFF": "",   # BOM
}

def normalize_id(s: str) -> str:
    if s is None:
      return ""
    # unify unicode, replace odd spaces
    s = unicodedata.normalize("NFKC", str(s))
    for k, v in SPACE_LIKE.items():
        s = s.replace(k, v)
    # uppercase
    s = s.upper()
    # replace separators/punctuation with space (keeps letters/digits)
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    # collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def strip_trailing_code(s: str) -> str:
    """
    If the LAST token looks like a code (pure digits or mix like 1234, 24-08-2023, etc.), drop it.
    Otherwise keep the full name (don't blindly drop the last token).
    """
    toks = s.split()
    if not toks:
        return s
    last = toks[-1]
    if re.fullmatch(r"[0-9\-_.]+", last):
        return " ".join(toks[:-1])
    return s

def clean_table():
    # 1) Excel IDs
    raw_df = pd.read_excel(config.clinical_data, sheet_name=0)
    excel_patient_ids = [normalize_id(x) for x in raw_df['patient_id'].astype(str).tolist()]
    excel_set = set(filter(None, excel_patient_ids))

    # 2) Folder IDs
    patients_ids_raw = os.listdir(config.DATA_PATH)
    # keep folder name stem (in case of extensions)
    patients_ids_raw = [os.path.splitext(pid)[0] for pid in patients_ids_raw]

    # optional: if your folder names end with a numeric code, strip it; else keep full
    folder_norm = []
    for pid in patients_ids_raw:
        pid = normalize_id(pid)
        pid = strip_trailing_code(pid)
        folder_norm.append(pid)
    folder_set = set(filter(None, folder_norm))

    # 3) Compare
    matches = folder_set & excel_set
    only_in_folders = sorted(folder_set - excel_set)
    only_in_excel = sorted(excel_set - folder_set)

    print(f"Total folders (normalized): {len(folder_set)}")
    print(f"Total excel (normalized):   {len(excel_set)}")
    print(f"Total matching patient IDs: {len(matches)}")

    # Show a few non-matches with close suggestions
    print("\nNon-matching from folders (up to 50):")
    for pid in only_in_folders[:50]:
        sugg = difflib.get_close_matches(pid, excel_set, n=1, cutoff=0.85)
        print(f"  {pid}" + (f"  -> maybe: {sugg[0]}" if sugg else ""))

    print("\nNon-matching from excel (up to 50):")
    for pid in only_in_excel[:50]:
        sugg = difflib.get_close_matches(pid, folder_set, n=1, cutoff=0.85)
        print(f"  {pid}" + (f"  -> maybe: {sugg[0]}" if sugg else ""))

    # select the rows of matching IDs in the df
    matched_rows = raw_df[raw_df['patient_id'].astype(str).apply(lambda x: normalize_id(x) in matches)]
    print(f"\nMatched rows in excel: {len(matched_rows)}")
    print(matched_rows.head())
    matched_rows.to_csv(os.path.join(config.SAVE_DIR, "matched_clinical_data.csv"), index=False)
if __name__ == "__main__":
    clean_table()
