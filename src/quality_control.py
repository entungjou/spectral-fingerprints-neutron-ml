# src/quality_control.py
# Quality Control Agent for IAEA DXS (2018) neutron displacement cross-section data
#
# Input:  outputs/dxs_master_dataset_all_v2.csv
# Columns expected (from your parser v2):
#   ['dxs_file','dxs_stem','element','MT','energy_MeV','displacement_xs_barn']
#
# Output (in outputs/):
#   - dxs_master_dataset_all_v2_clean.csv        (cleaned dataset)
#   - dxs_coverage_summary.csv                   (coverage per material/MT)
#   - dxs_outliers_top.csv                       (top suspicious points)
#   - dxs_qc_report.txt                          (human-readable summary)
#   - plots/                                     (per-material log-log plots)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# CONFIG (edit if needed)
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_CSV = PROJECT_ROOT / "outputs" / "dxs_master_dataset_all_v2.csv"
OUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLEAN = OUT_DIR / "dxs_master_dataset_all_v2_clean.csv"
OUT_COVERAGE = OUT_DIR / "dxs_coverage_summary.csv"
OUT_OUTLIERS = OUT_DIR / "dxs_outliers_top.csv"
OUT_REPORT = OUT_DIR / "dxs_qc_report.txt"

# Required columns
REQ_COLS = ["dxs_file", "dxs_stem", "element", "MT", "energy_MeV", "displacement_xs_barn"]

# QC knobs
MAX_PLOTS = 25  # avoid making 78 plots every run; set to 999 to plot all
PLOT_MT = 900   # displacement XS curve you care about (MT=900). Keep; can change to 901 if needed.
MIN_POSITIVE_ENERGY = 0.0  # keep 0 rows? we will drop E<=0 anyway for log-log plots
DROP_NONPOSITIVE_FOR_CLEAN = True  # for "clean" dataset, drop E<=0 or XS<=0
OUTLIER_Z_THRESHOLD = 8.0  # robust z-score threshold on log10(XS) per material
LOG_ENERGY_STEP_JUMP = 1.0  # "big jump" in log10(E) between consecutive points flagged


@dataclass
class QCStats:
    n_rows_raw: int
    n_rows_clean: int
    n_removed: int
    n_materials_raw: int
    n_materials_clean: int


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce energy/xs/MT to numeric; drop rows where they can't be parsed."""
    df = df.copy()
    df["MT"] = pd.to_numeric(df["MT"], errors="coerce").astype("Int64")
    df["energy_MeV"] = pd.to_numeric(df["energy_MeV"], errors="coerce")
    df["displacement_xs_barn"] = pd.to_numeric(df["displacement_xs_barn"], errors="coerce")
    return df


def _standardize_element(el: str) -> str:
    """Make element look like 'Si', 'Fe'. Keeps 'Unknown' if not valid."""
    if el is None:
        return "Unknown"
    el = str(el).strip()
    if not el:
        return "Unknown"
    if el.lower() == "unknown":
        return "Unknown"
    # Normalize: first letter uppercase, rest lowercase
    el2 = el[0].upper() + el[1:].lower() if len(el) > 1 else el.upper()
    # Basic element symbol validation (1-2 letters)
    if re.fullmatch(r"[A-Z][a-z]?", el2):
        return el2
    return "Unknown"


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score using median absolute deviation (MAD)."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad):
        return np.full_like(x, 0.0)
    return 0.6745 * (x - med) / mad


def _write_report(lines: list[str]) -> None:
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def make_plots(clean_df: pd.DataFrame, max_plots: int = MAX_PLOTS, mt: int = PLOT_MT) -> list[str]:
    """
    Save per-material log-log scatter plots for a selected MT.
    Returns list of saved filenames.
    """
    saved = []
    subset = clean_df[clean_df["MT"].astype("Int64") == mt].copy()
    if subset.empty:
        return saved

    # For log-log plots, ensure positive
    subset = subset[(subset["energy_MeV"] > 0) & (subset["displacement_xs_barn"] > 0)].copy()

    materials = sorted(subset["element"].unique().tolist())
    for i, el in enumerate(materials[:max_plots]):
        sub = subset[subset["element"] == el].sort_values("energy_MeV")
        if sub.empty:
            continue

        fig = plt.figure()
        plt.scatter(sub["energy_MeV"].values, sub["displacement_xs_barn"].values, s=6)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Incident neutron energy (MeV)")
        plt.ylabel("Displacement cross section (barn)")
        plt.title(f"Neutron displacement XS – {el} (MT={mt})")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        out_path = PLOTS_DIR / f"dxs_{el}_MT{mt}_loglog.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        saved.append(str(out_path))
    return saved


def coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage per element and MT: points and energy range."""
    g = df.groupby(["element", "MT"], dropna=False)
    cov = g.agg(
        n_points=("energy_MeV", "size"),
        E_min_MeV=("energy_MeV", "min"),
        E_max_MeV=("energy_MeV", "max"),
        XS_min_barn=("displacement_xs_barn", "min"),
        XS_max_barn=("displacement_xs_barn", "max"),
    ).reset_index()
    cov = cov.sort_values(["element", "MT"])
    return cov


def find_outliers(clean_df: pd.DataFrame, mt: int = PLOT_MT) -> pd.DataFrame:
    """
    Identify suspicious points:
      - robust z-score on log10(XS) per element (within MT)
      - big gaps in log10(E) sequence
    Returns a dataframe of flagged points (top suspicious).
    """
    df = clean_df.copy()
    df = df[df["MT"].astype("Int64") == mt].copy()
    if df.empty:
        return pd.DataFrame()

    # Only positive for log metrics
    df = df[(df["energy_MeV"] > 0) & (df["displacement_xs_barn"] > 0)].copy()
    if df.empty:
        return pd.DataFrame()

    df["logE"] = np.log10(df["energy_MeV"].values)
    df["logXS"] = np.log10(df["displacement_xs_barn"].values)

    flagged_rows = []

    for el, sub in df.groupby("element"):
        sub = sub.sort_values("energy_MeV").copy()
        z = _robust_z(sub["logXS"].values)
        sub["robust_z_logXS"] = z

        # Energy gap checks
        logE = sub["logE"].values
        dlogE = np.diff(logE)
        # mark index positions where big jump occurs (gap after row i)
        gap_idx = np.where(dlogE > LOG_ENERGY_STEP_JUMP)[0]
        sub["big_logE_gap_after"] = False
        if len(gap_idx) > 0:
            sub.iloc[gap_idx, sub.columns.get_loc("big_logE_gap_after")] = True

        out = sub[(np.abs(sub["robust_z_logXS"]) >= OUTLIER_Z_THRESHOLD) | (sub["big_logE_gap_after"])]
        if not out.empty:
            flagged_rows.append(out)

    if not flagged_rows:
        return pd.DataFrame()

    flagged = pd.concat(flagged_rows, ignore_index=True)

    # Rank: high |z| first, then higher XS
    flagged["abs_z"] = flagged["robust_z_logXS"].abs()
    flagged = flagged.sort_values(["abs_z", "displacement_xs_barn"], ascending=[False, False])

    # Keep useful columns
    keep = [
        "element", "MT", "dxs_file", "dxs_stem",
        "energy_MeV", "displacement_xs_barn",
        "robust_z_logXS", "big_logE_gap_after"
    ]
    keep = [c for c in keep if c in flagged.columns]
    return flagged[keep].head(500).reset_index(drop=True)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean:
      - standardize element
      - drop NaNs in critical numeric fields
      - drop duplicate (element,MT,energy_MeV) keeping the last
      - optionally drop non-positive energy/xs (for log-based ML/plots)
    Returns (clean_df, removal_stats).
    """
    df = df.copy()

    # Standardize element
    df["element"] = df["element"].apply(_standardize_element)

    # Coerce numeric
    df = _coerce_numeric(df)

    n0 = len(df)

    # Drop NaNs in required numeric columns
    df = df.dropna(subset=["MT", "energy_MeV", "displacement_xs_barn"])
    n1 = len(df)

    # Drop duplicates
    # (for same material+MT+energy, keep last occurrence)
    df = df.sort_values(["element", "MT", "energy_MeV"])
    df = df.drop_duplicates(subset=["element", "MT", "energy_MeV"], keep="last")
    n2 = len(df)

    # Optional: drop non-positive
    if DROP_NONPOSITIVE_FOR_CLEAN:
        df = df[(df["energy_MeV"] > 0) & (df["displacement_xs_barn"] > 0)]
    n3 = len(df)

    removal = {
        "raw_rows": n0,
        "after_dropna_rows": n1,
        "after_dedup_rows": n2,
        "after_positive_filter_rows": n3,
        "dropped_na": n0 - n1,
        "dropped_dupes": n1 - n2,
        "dropped_nonpositive": n2 - n3,
        "final_rows": n3,
    }
    return df.reset_index(drop=True), removal


def main() -> None:
    # Validate input exists
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {IN_CSV}")

    # Load
    df = pd.read_csv(IN_CSV)
    missing = set(REQ_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {IN_CSV}: {sorted(missing)}")

    n_rows_raw = len(df)
    n_materials_raw = df["element"].nunique(dropna=False)

    # Clean
    clean_df, removal = clean_dataset(df)
    n_rows_clean = len(clean_df)
    n_materials_clean = clean_df["element"].nunique(dropna=False)

    stats = QCStats(
        n_rows_raw=n_rows_raw,
        n_rows_clean=n_rows_clean,
        n_removed=n_rows_raw - n_rows_clean,
        n_materials_raw=int(n_materials_raw),
        n_materials_clean=int(n_materials_clean),
    )

    # Coverage summary
    cov = coverage_summary(clean_df)
    cov.to_csv(OUT_COVERAGE, index=False)

    # Outliers
    outliers = find_outliers(clean_df, mt=PLOT_MT)
    if not outliers.empty:
        outliers.to_csv(OUT_OUTLIERS, index=False)
    else:
        # write empty file with headers so you know it ran
        pd.DataFrame(columns=["element", "MT", "dxs_file", "dxs_stem", "energy_MeV", "displacement_xs_barn"]).to_csv(
            OUT_OUTLIERS, index=False
        )

    # Save clean
    clean_df.to_csv(OUT_CLEAN, index=False)

    # Plots
    saved_plots = make_plots(clean_df, max_plots=MAX_PLOTS, mt=PLOT_MT)

    # Human-readable report
    lines = []
    lines.append(f"Input file: {IN_CSV}")
    lines.append(f"Rows raw: {stats.n_rows_raw}")
    lines.append(f"Rows clean: {stats.n_rows_clean}")
    lines.append(f"Removed: {stats.n_removed}")
    lines.append("")
    lines.append(f"Materials (raw): {stats.n_materials_raw}")
    lines.append(f"Materials (clean): {stats.n_materials_clean}")
    lines.append("")
    lines.append("=== Removal breakdown ===")
    for k, v in removal.items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("=== Column checks ===")
    lines.append(f"Energy min/max (MeV): {clean_df['energy_MeV'].min():.6g} / {clean_df['energy_MeV'].max():.6g}")
    lines.append(
        f"XS min/max (barn): {clean_df['displacement_xs_barn'].min():.6g} / {clean_df['displacement_xs_barn'].max():.6g}"
    )
    lines.append("")
    lines.append("=== MT counts (top 10) ===")
    mt_counts = clean_df["MT"].value_counts().head(10)
    for mt, c in mt_counts.items():
        lines.append(f"MT={mt}: {c}")
    lines.append("")
    lines.append(f"Coverage summary saved: {OUT_COVERAGE}")
    lines.append(f"Clean dataset saved: {OUT_CLEAN}")
    lines.append(f"Outliers saved: {OUT_OUTLIERS}")
    lines.append(f"Plots saved in: {PLOTS_DIR}")
    if saved_plots:
        lines.append(f"Example plot: {saved_plots[0]}")
    _write_report(lines)

    # Console output (what you show in screenshots)
    print(f"Loaded: {IN_CSV}")
    print(f"Rows raw: {stats.n_rows_raw} | Rows clean: {stats.n_rows_clean} | Removed: {stats.n_removed}")
    print(f"Materials (raw): {stats.n_materials_raw} | Materials (clean): {stats.n_materials_clean}")
    print("")
    print(f"Saved coverage: {OUT_COVERAGE.name}")
    print(f"Saved clean:    {OUT_CLEAN.name}")
    print(f"Saved outliers: {OUT_OUTLIERS.name}")
    print(f"Saved report:   {OUT_REPORT.name}")
    print(f"Saved plots in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()