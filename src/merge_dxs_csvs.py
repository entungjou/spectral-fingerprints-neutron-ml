from pathlib import Path
import re
import pandas as pd

"""
Merge all parsed DXS CSVs (created by parse_dxs_to_csv.py) into one master table.

Expected input CSV naming:
  outputs/<DXS_STEM>_E_vs_XS.csv
where <DXS_STEM> is like:
  14Si0c, 26Fe0c, 31Ga0c, 25Mn55c, 21Sc45c, ...

Output:
  outputs/dxs_master_dataset.csv
"""

# ----------------------------
# CONFIG
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"
MASTER_CSV = OUT_DIR / "dxs_master_dataset.csv"


def extract_element_symbol(stem: str) -> str:
    """
    Extract element symbol from a DXS stem.

    Examples:
      14Si0c   -> Si
      03Li0c   -> Li
      31Ga0c   -> Ga
      25Mn55c  -> Mn
      21Sc45c  -> Sc
      30Zn0c   -> Zn

    Rule: digits + (Uppercase + optional lowercase) at the start
    """
    m = re.match(r"^\d+([A-Z][a-z]?)", stem)
    return m.group(1) if m else "Unknown"


def extract_za(stem: str) -> int | None:
    """
    Optional: extract the leading integer (often Z*1000 + A style in some nuclear datasets,
    but for these filenames it's just the number prefix like 14, 26, 31...)
    We'll store it as 'prefix_id' for traceability.
    """
    m = re.match(r"^(\d+)", stem)
    return int(m.group(1)) if m else None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(OUT_DIR.glob("*_E_vs_XS.csv"))
    if not csv_files:
        raise SystemExit(f"No '*_E_vs_XS.csv' found in {OUT_DIR}. Run parse_dxs_to_csv.py first.")

    all_rows = []
    unknown_count = 0

    for csv_path in csv_files:
        # e.g. "14Si0c_E_vs_XS.csv" -> "14Si0c"
        dxs_stem = csv_path.stem.replace("_E_vs_XS", "")

        element = extract_element_symbol(dxs_stem)
        prefix_id = extract_za(dxs_stem)
        if element == "Unknown":
            unknown_count += 1

        df = pd.read_csv(csv_path)

        # normalize column names if needed
        # expected: energy_MeV, displacement_xs_barn
        if "energy_MeV" not in df.columns or "displacement_xs_barn" not in df.columns:
            raise SystemExit(
                f"{csv_path.name} does not have expected columns. "
                f"Found columns: {list(df.columns)}"
            )

        df.insert(0, "dxs_file", csv_path.name)
        df.insert(1, "dxs_stem", dxs_stem)
        df.insert(2, "element", element)
        df.insert(3, "prefix_id", prefix_id)

        all_rows.append(df)

    master = pd.concat(all_rows, ignore_index=True)
    master.to_csv(MASTER_CSV, index=False)

    print(f"Found parsed CSVs: {len(csv_files)}")
    print(f"Total rows merged: {len(master)}")
    print(f"Unknown element stems: {unknown_count}")
    print(f"Saved master CSV -> {MASTER_CSV}")

    # quick sanity preview
    print("\nElement counts (top 20):")
    print(master["element"].value_counts().head(20))


if __name__ == "__main__":
    main()
