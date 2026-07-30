from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CURVE_FILE = PROJECT_ROOT / "outputs" / "dxs_master_dataset_all_v2_clean.csv"
BASE_ML_FILE = PROJECT_ROOT / "outputs" / "ml_dataset_with_peak_features.csv"
OUT_FILE = PROJECT_ROOT / "outputs" / "ml_dataset_v8.csv"


def safe_log10(x):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    mask = x > 0
    out[mask] = np.log10(x[mask])
    return out


def compute_peak_width_log10E(x, y, frac=0.5):
    """
    Width in log10(E) where y >= frac * peak_y.
    Returns np.nan if not enough valid points.
    """
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

    x_left = x[idx[0]]
    x_right = x[idx[-1]]

    if x_left <= 0 or x_right <= 0:
        return np.nan

    return np.log10(x_right) - np.log10(x_left)


def compute_group_features(sub):
    """
    sub: one element's MT=900 curve
    """
    sub = sub.sort_values("energy_MeV").copy()

    x = sub["energy_MeV"].to_numpy(dtype=float)
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

    # simple center of the sampled curve in logE space
    curve_center_log10E = np.mean(logx)

    # weighted center of the curve in logE space
    weighted_center_log10E = np.sum(logx * y) / np.sum(y)

    features = {
        "log10_energy_at_peak": np.log10(peak_x) if peak_x > 0 else np.nan,
        "peak_to_mean_ratio": peak_y / xs_mean if xs_mean > 0 else np.nan,
        "peak_to_median_ratio": peak_y / xs_median if xs_median > 0 else np.nan,
        "xs_dynamic_range_log10": np.log10(xs_max / xs_min) if xs_min > 0 else np.nan,
        "peak_width_log10E_halfmax": compute_peak_width_log10E(x, y, frac=0.5),
        "energy_span_log10": np.log10(np.max(x)) - np.log10(np.min(x)),
        "curve_center_log10E": curve_center_log10E,
        "weighted_center_log10E": weighted_center_log10E,
    }

    return features


def main():
    if not CURVE_FILE.exists():
        raise FileNotFoundError(f"Cannot find curve file: {CURVE_FILE}")
    if not BASE_ML_FILE.exists():
        raise FileNotFoundError(f"Cannot find ML file: {BASE_ML_FILE}")

    curve_df = pd.read_csv(CURVE_FILE)
    ml_df = pd.read_csv(BASE_ML_FILE)

    required_curve = {"element", "MT", "energy_MeV", "displacement_xs_barn"}
    missing_curve = required_curve - set(curve_df.columns)
    if missing_curve:
        raise ValueError(f"Missing columns in curve file: {sorted(missing_curve)}")

    if "element" not in ml_df.columns:
        raise ValueError("Base ML file must contain column 'element'")

    # numeric cleanup
    curve_df["MT"] = pd.to_numeric(curve_df["MT"], errors="coerce")
    curve_df["energy_MeV"] = pd.to_numeric(curve_df["energy_MeV"], errors="coerce")
    curve_df["displacement_xs_barn"] = pd.to_numeric(curve_df["displacement_xs_barn"], errors="coerce")

    curve_df = curve_df.dropna(subset=["element", "MT", "energy_MeV", "displacement_xs_barn"]).copy()
    curve_df = curve_df[(curve_df["energy_MeV"] > 0) & (curve_df["displacement_xs_barn"] > 0)].copy()

    # IMPORTANT: match your ML dataset, which is already MT=900 only
    curve_df = curve_df[curve_df["MT"] == 900].copy()

    feature_rows = []
    for element, sub in curve_df.groupby("element", sort=True):
        feats = compute_group_features(sub)
        feats["element"] = element
        feature_rows.append(feats)

    v8_df = pd.DataFrame(feature_rows)

    merged = pd.merge(
        ml_df,
        v8_df,
        on="element",
        how="left"
    )

    merged.to_csv(OUT_FILE, index=False)

    print("Loaded curve file:", CURVE_FILE)
    print("Loaded base ML file:", BASE_ML_FILE)
    print("Rows in output:", len(merged))
    print("Saved ->", OUT_FILE)

    print("\nNew columns added:")
    new_cols = [c for c in v8_df.columns if c != "element"]
    for c in new_cols:
        print("-", c)

    print("\nPreview:")
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()