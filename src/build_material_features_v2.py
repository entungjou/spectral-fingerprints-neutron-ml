from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CURVE_FILE = PROJECT_ROOT / "outputs" / "dxs_master_dataset_all_v2_clean.csv"
BASE_ML_FILE = PROJECT_ROOT / "outputs" / "ml_dataset_with_peak_features.csv"
OUT_FILE = PROJECT_ROOT / "outputs" / "ml_dataset_v8.csv"


def compute_peak_width_log10E(x, y, frac=0.5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan

    peak_y = np.max(y)
    threshold = frac * peak_y
    idx = np.where(y >= threshold)[0]

    if len(idx) < 2:
        return np.nan

    return np.log10(x[idx[-1]]) - np.log10(x[idx[0]])


def compute_group_features(sub):
    sub = sub.sort_values("energy_eV").copy()

    x = sub["energy_eV"].to_numpy(dtype=float)
    y = sub["displacement_xs_barn"].to_numpy(dtype=float)

    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return {
            "log10_energy_at_peak": np.nan,
            "peak_to_mean_ratio": np.nan,
            "peak_to_median_ratio": np.nan,
            "xs_dynamic_range_log10": np.nan,
            "peak_width_log10E_halfmax": np.nan,
            "energy_span_log10": np.nan,
            "curve_center_log10E": np.nan,
            "weighted_center_log10E": np.nan,
        }

    peak_idx = np.argmax(y)
    peak_x = x[peak_idx]
    peak_y = y[peak_idx]

    xs_mean = np.mean(y)
    xs_median = np.median(y)
    xs_min = np.min(y)
    xs_max = np.max(y)

    logx = np.log10(x)

    return {
        "log10_energy_at_peak": np.log10(peak_x),
        "peak_to_mean_ratio": peak_y / xs_mean if xs_mean > 0 else np.nan,
        "peak_to_median_ratio": peak_y / xs_median if xs_median > 0 else np.nan,
        "xs_dynamic_range_log10": np.log10(xs_max / xs_min) if xs_min > 0 else np.nan,
        "peak_width_log10E_halfmax": compute_peak_width_log10E(x, y, frac=0.5),
        "energy_span_log10": np.log10(np.max(x)) - np.log10(np.min(x)),
        "curve_center_log10E": np.mean(logx),
        "weighted_center_log10E": np.sum(logx * y) / np.sum(y),
    }


def main():
    curve_df = pd.read_csv(CURVE_FILE)
    ml_df = pd.read_csv(BASE_ML_FILE)

    # Rename old column names if needed
    curve_df = curve_df.rename(columns={"energy_MeV": "energy_eV"})
    ml_df = ml_df.rename(columns={"energy_at_peak_MeV": "energy_at_peak_eV"})

    required_curve = {"element", "MT", "energy_eV", "displacement_xs_barn"}
    missing_curve = required_curve - set(curve_df.columns)
    if missing_curve:
        raise ValueError(f"Missing columns in curve file: {sorted(missing_curve)}")

    curve_df["MT"] = pd.to_numeric(curve_df["MT"], errors="coerce")
    curve_df["energy_eV"] = pd.to_numeric(curve_df["energy_eV"], errors="coerce")
    curve_df["displacement_xs_barn"] = pd.to_numeric(
        curve_df["displacement_xs_barn"], errors="coerce"
    )

    curve_df = curve_df.dropna(
        subset=["element", "MT", "energy_eV", "displacement_xs_barn"]
    ).copy()

    curve_df = curve_df[
        (curve_df["energy_eV"] > 0)
        & (curve_df["displacement_xs_barn"] > 0)
        & (curve_df["MT"] == 900)
    ].copy()

    feature_rows = []
    for element, sub in curve_df.groupby("element", sort=True):
        feats = compute_group_features(sub)
        feats["element"] = element
        feature_rows.append(feats)

    v8_df = pd.DataFrame(feature_rows)

    merged = pd.merge(ml_df, v8_df, on="element", how="left")
    merged.to_csv(OUT_FILE, index=False)

    print("Saved:", OUT_FILE)
    print("Rows:", len(merged))
    print("Columns:", list(merged.columns))


if __name__ == "__main__":
    main()