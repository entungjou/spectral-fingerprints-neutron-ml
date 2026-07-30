from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd


# =========================
# CONFIG (edit if needed)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DXS_DIR = PROJECT_ROOT / "data" / "raw" / "iasa_dxs"
OUT_DIR = PROJECT_ROOT / "outputs"
PER_MAT_DIR = OUT_DIR / "per_material_csv_v2"

MASTER_CSV = OUT_DIR / "dxs_master_dataset_all_v2.csv"
FAILED_TXT = OUT_DIR / "dxs_failed_v2.txt"

PER_MAT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ENDF identifiers we want
TARGET_MF = 3
TARGET_MTS = {900, 901}  # 900: arc-dpa, 901: NRT-based (often)

# Optional: if you only want MT=900, set TARGET_MTS={900}


# =========================
# Helpers
# =========================
def element_from_stem(stem: str) -> str:
    """
    DXS filenames look like '14Si0c.dxs', '31Ga0c.dxs', '26Fe0c.dxs', etc.
    Extract element symbol from the stem: digits + Element + ...
    """
    m = re.match(r"^\d+([A-Za-z]{1,2})", stem)
    return m.group(1) if m else "Unknown"


def parse_endf_tail_fixed(line: str) -> Optional[Tuple[int, int, int]]:
    """
    ENDF-6 fixed-width tail parsing.
    Columns (1-indexed):
      67-70 MAT, 71-72 MF, 73-75 MT, 76-80 line number
    (0-indexed python slices):
      [66:70], [70:72], [72:75], [75:80]

    Returns (MAT, MF, MT) or None.
    """
    if len(line) < 75:
        return None

    try:
        mat = int(line[66:70])
        mf = int(line[70:72])
        mt = int(line[72:75])
        return mat, mf, mt
    except ValueError:
        # fallback: regex from end-of-line integers
        m = re.search(r"\s(\d{1,4})\s+(\d{1,2})\s+(\d{1,3})\s+\d+\s*$", line)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2)), int(m.group(3))


def parse_six_fields(line: str) -> List[Optional[float]]:
    """
    Parse the left ENDF numeric area into up to 6 float fields.
    ENDF uses 11-char fields, typically 6 per line (66 chars).
    We only use the first 6*11 chars (first 66 chars).
    """
    left = line[:66]
    fields = [left[i:i + 11] for i in range(0, 66, 11)]
    out: List[Optional[float]] = []

    for f in fields:
        s = f.strip()
        if not s:
            out.append(None)
            continue

        # ENDF may write like 1.23456+3 (no 'E')
        # Convert to python float by inserting 'E' before last sign in exponent.
        # Examples: " 1.00000-5" -> "1.00000E-5"
        #           " 3.68130+7" -> "3.68130E+7"
        # Also handle already-normal "1.23E+4"
        if ("E" not in s) and re.search(r"[+\-]\d+$", s):
            s = re.sub(r"([+\-]\d+)$", r"E\1", s)

        try:
            out.append(float(s))
        except ValueError:
            out.append(None)

    return out


def extract_tab1_xy(lines: List[str], mf: int, mt: int) -> List[Tuple[float, float]]:
    """
    Extract TAB1-style XY pairs from an ENDF MF=3 MT=(900/901) section.

    The exact TAB1 bookkeeping is a bit complex, but for these .dxs files
    the numeric table typically appears as repeated pairs (E, XS) across lines
    after the MF=3 MT block begins.

    Strategy (robust and practical):
    - Once inside MF=3 MT target, parse each line into 6 fields
    - Collect float values in order, skipping None
    - Then interpret as consecutive pairs: (x0,y0),(x1,y1),...

    We also drop clearly-nonphysical x (<=0) and y (<0).
    """
    values: List[float] = []
    for line in lines:
        tail = parse_endf_tail_fixed(line)
        if not tail:
            continue
        _mat, _mf, _mt = tail
        if _mf != mf or _mt != mt:
            continue

        nums = parse_six_fields(line)
        for v in nums:
            if v is None:
                continue
            values.append(v)

    xy: List[Tuple[float, float]] = []
    # Pair them up
    for i in range(0, len(values) - 1, 2):
        x = values[i]
        y = values[i + 1]
        if x is None or y is None:
            continue
        # Basic sanity
        if x <= 0:
            continue
        if y < 0:
            continue
        xy.append((float(x), float(y)))

    # De-duplicate exact duplicates
    if xy:
        seen = set()
        uniq = []
        for x, y in xy:
            key = (x, y)
            if key in seen:
                continue
            seen.add(key)
            uniq.append((x, y))
        xy = uniq

    return xy


def find_target_sections(lines: List[str]) -> List[Tuple[int, int]]:
    """
    Return list of (MF, MT) sections present in the file (unique).
    """
    secs = set()
    for line in lines:
        tail = parse_endf_tail_fixed(line)
        if not tail:
            continue
        _mat, mf, mt = tail
        secs.add((mf, mt))
    return sorted(secs)


def parse_one_file(dxs_path: Path) -> pd.DataFrame:
    """
    Parse one .dxs into a DataFrame with columns:
    dxs_file, dxs_stem, element, MT, energy_MeV, displacement_xs_barn
    """
    text = dxs_path.read_text(errors="ignore")
    lines = text.splitlines()

    stem = dxs_path.stem
    element = element_from_stem(stem)

    # Check sections exist
    secs = find_target_sections(lines)
    mts_present = {mt for (mf, mt) in secs if mf == TARGET_MF}

    rows: List[Dict] = []

    for mt in sorted(TARGET_MTS):
        if mt not in mts_present:
            continue

        xy = extract_tab1_xy(lines, TARGET_MF, mt)
        for x, y in xy:
            rows.append(
                dict(
                    dxs_file=dxs_path.name,
                    dxs_stem=stem,
                    element=element,
                    MT=int(mt),
                    energy_MeV=float(x),
                    displacement_xs_barn=float(y),
                )
            )

    return pd.DataFrame(rows)


def main():
    dxs_files = sorted(DXS_DIR.glob("*.dxs"))
    print(f"Found .dxs files: {len(dxs_files)} in {DXS_DIR}")

    failed: List[str] = []
    all_frames: List[pd.DataFrame] = []

    for f in dxs_files:
        try:
            df = parse_one_file(f)
            if df.empty:
                failed.append(f.name)
                continue

            # Save per-material CSV (per file)
            out_csv = PER_MAT_DIR / f"{f.stem}_E_vs_XS.csv"
            df.to_csv(out_csv, index=False)

            all_frames.append(df)
        except Exception as e:
            failed.append(f"{f.name}  |  {repr(e)}")

    # Write failures
    if failed:
        FAILED_TXT.write_text("\n".join(failed), encoding="utf-8")
        print(f"Failed/empty files: {len(failed)} (saved to {FAILED_TXT})")
    else:
        print("Failed/empty files: 0")

    if not all_frames:
        raise RuntimeError("No data parsed. All files were empty. Check FAILED_TXT and parser logic.")

    master = pd.concat(all_frames, ignore_index=True)

    # Sort for nicer output
    master = master.sort_values(["element", "MT", "energy_MeV"]).reset_index(drop=True)

    master.to_csv(MASTER_CSV, index=False)

    print(f"Parsed OK: {len(all_frames)}")
    print(f"Saved per-material CSVs -> {PER_MAT_DIR}")
    print(f"Saved master CSV -> {MASTER_CSV}")
    print("\nMaster preview:")
    print(master.head(10))
    print("\nCounts by element (top 15):")
    print(master["element"].value_counts().head(15))


if __name__ == "__main__":
    main()