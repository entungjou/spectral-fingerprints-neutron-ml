from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_CSV = PROJECT_ROOT / "outputs" / "material_features.csv"
CURVE_CSV = PROJECT_ROOT / "outputs" / "dxs_master_dataset_all_v2_clean.csv"
OUT_CSV = PROJECT_ROOT / "outputs" / "ml_dataset_with_peak_features.csv"


def main():
    features_df = pd.read_csv(IN_CSV)
    curve_df = pd.read_csv(CURVE_CSV)

    # Rename old column names if needed
    curve_df = curve_df.rename(columns={"energy_MeV": "energy_eV"})

    required_cols = {"element", "MT", "energy_eV", "displacement_xs_barn"}
    missing = required_cols - set(curve_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in curve CSV: {missing}")

    curve_df["MT"] = pd.to_numeric(curve_df["MT"], errors="coerce")
    curve_df["energy_eV"] = pd.to_numeric(curve_df["energy_eV"], errors="coerce")
    curve_df["displacement_xs_barn"] = pd.to_numeric(
        curve_df["displacement_xs_barn"], errors="coerce"
    )

    curve_df = curve_df.dropna(
        subset=["element", "MT", "energy_eV", "displacement_xs_barn"]
    ).copy()

    curve_df = curve_df[
        (curve_df["MT"] == 900)
        & (curve_df["energy_eV"] > 0)
        & (curve_df["displacement_xs_barn"] > 0)
    ].copy()

    peak_rows = []

    for element, sub in curve_df.groupby("element", sort=True):
        sub = sub.sort_values("energy_eV")
        idx = sub["displacement_xs_barn"].idxmax()

        peak_rows.append({
            "element": element,
            "energy_at_peak_eV": sub.loc[idx, "energy_eV"],
            "peak_xs_barn": sub.loc[idx, "displacement_xs_barn"],
        })

    peak_df = pd.DataFrame(peak_rows)

    # Remove old MeV column if it exists
    features_df = features_df.drop(columns=["energy_at_peak_MeV"], errors="ignore")

    merged = pd.merge(features_df, peak_df, on="element", how="left")
    merged.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    print("Rows:", len(merged))
    print("Columns:", list(merged.columns))


if __name__ == "__main__":
    main()