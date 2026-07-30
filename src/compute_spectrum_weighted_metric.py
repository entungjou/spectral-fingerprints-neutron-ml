from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr


# =========================
# File paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DXS_FILE = PROJECT_ROOT / "outputs" / "dxs_master_dataset_all_v2_clean.csv"
ML_FILE = PROJECT_ROOT / "outputs" / "ml_dataset_v8.csv"

OUT_CSV = PROJECT_ROOT / "outputs" / "auc_vs_spectrum_weighted_metric.csv"
OUT_FIG = PROJECT_ROOT / "outputs" / "auc_vs_spectrum_weighted_metric.png"
OUT_TXT = PROJECT_ROOT / "outputs" / "auc_dphi_correlation_results.txt"


# =========================
# Column names
# =========================
TARGET_AUC = "auc_log10xs_over_log10E"
XS_COL = "displacement_xs_barn"


def find_energy_column(df):
    """
    Detect energy column and standardize to energy values.

    Note:
    Some older files may still use the column name 'energy_MeV',
    even though the values appear to be in eV.
    """
    if "energy_eV" in df.columns:
        return "energy_eV"

    if "energy_MeV" in df.columns:
        print("Warning: using column 'energy_MeV' as energy values.")
        print("If your values range up to ~2e8, they are likely eV despite the old column name.")
        return "energy_MeV"

    raise ValueError(
        "Could not find an energy column. Expected 'energy_eV' or 'energy_MeV'. "
        f"Available columns: {df.columns.tolist()}"
    )


def clean_curve(sub_df, energy_col):
    """
    Clean one material's DXS curve.
    """
    sub_df = sub_df[[energy_col, XS_COL]].dropna().copy()

    sub_df[energy_col] = pd.to_numeric(sub_df[energy_col], errors="coerce")
    sub_df[XS_COL] = pd.to_numeric(sub_df[XS_COL], errors="coerce")

    sub_df = sub_df.dropna()
    sub_df = sub_df[(sub_df[energy_col] > 0) & (sub_df[XS_COL] > 0)].copy()

    if len(sub_df) < 2:
        return None

    sub_df = sub_df.sort_values(energy_col)

    return sub_df


def compute_flat_log_spectrum_metric(sub_df, energy_col):
    """
    Compute a reference spectrum-weighted displacement-response metric.

    Here we use a flat weighting in log-energy space:

        D_phi = integral sigma_d(E) d(log10 E)

    This is not an experimental dpa value. It is a reference
    spectrum-weighted displacement-response metric used to physically
    anchor the AUC proxy.
    """
    sub_df = clean_curve(sub_df, energy_col)

    if sub_df is None:
        return np.nan

    energy = sub_df[energy_col].to_numpy()
    xs = sub_df[XS_COL].to_numpy()

    logE = np.log10(energy)

    # NumPy 2.x uses np.trapezoid instead of np.trapz
    dphi = np.trapezoid(xs, logE)

    return dphi


def compute_linear_energy_metric(sub_df, energy_col):
    """
    Optional comparison metric:

        D_phi_linear = integral sigma_d(E) dE

    This can be very sensitive to the high-energy range and units.
    It is included only as a secondary comparison.
    """
    sub_df = clean_curve(sub_df, energy_col)

    if sub_df is None:
        return np.nan

    energy = sub_df[energy_col].to_numpy()
    xs = sub_df[XS_COL].to_numpy()

    dphi_linear = np.trapezoid(xs, energy)

    return dphi_linear


def main():
    if not DXS_FILE.exists():
        raise FileNotFoundError(f"Cannot find DXS file: {DXS_FILE}")

    if not ML_FILE.exists():
        raise FileNotFoundError(f"Cannot find ML dataset: {ML_FILE}")

    dxs = pd.read_csv(DXS_FILE)
    ml = pd.read_csv(ML_FILE)

    energy_col = find_energy_column(dxs)

    required_dxs = ["element", energy_col, XS_COL]
    missing_dxs = [c for c in required_dxs if c not in dxs.columns]

    if missing_dxs:
        raise ValueError(f"Missing DXS columns: {missing_dxs}")

    required_ml = ["element", TARGET_AUC]
    missing_ml = [c for c in required_ml if c not in ml.columns]

    if missing_ml:
        raise ValueError(f"Missing ML columns: {missing_ml}")

    print("Loaded DXS file:", DXS_FILE)
    print("Loaded ML file :", ML_FILE)
    print("Energy column  :", energy_col)

    records = []

    for element, sub in dxs.groupby("element"):
        dphi_log = compute_flat_log_spectrum_metric(sub, energy_col)
        dphi_linear = compute_linear_energy_metric(sub, energy_col)

        records.append({
            "element": element,
            "Dphi_flat_logE": dphi_log,
            "Dphi_linear_E": dphi_linear,
        })

    dphi_df = pd.DataFrame(records)

    merged = ml[["element", TARGET_AUC]].merge(
        dphi_df,
        on="element",
        how="inner"
    )

    merged = merged.dropna(subset=[TARGET_AUC, "Dphi_flat_logE"]).copy()

    # Log-transform Dphi because its magnitude can span several orders of magnitude
    merged["log10_Dphi_flat_logE"] = np.log10(merged["Dphi_flat_logE"])

    # =========================
    # Correlations
    # =========================

    pearson_r, pearson_p = pearsonr(
        merged[TARGET_AUC],
        merged["log10_Dphi_flat_logE"]
    )

    spearman_rho, spearman_p = spearmanr(
        merged[TARGET_AUC],
        merged["log10_Dphi_flat_logE"]
    )

    # Optional raw-Dphi correlation
    pearson_raw_r, pearson_raw_p = pearsonr(
        merged[TARGET_AUC],
        merged["Dphi_flat_logE"]
    )

    spearman_raw_rho, spearman_raw_p = spearmanr(
        merged[TARGET_AUC],
        merged["Dphi_flat_logE"]
    )

    merged.to_csv(OUT_CSV, index=False)

    # =========================
    # Scatter plot
    # =========================

    plt.figure(figsize=(6, 5))

    plt.scatter(
        merged[TARGET_AUC],
        merged["log10_Dphi_flat_logE"],
        s=45,
        alpha=0.8
    )

    plt.xlabel("AUC of Log-Transformed DXS Spectrum")
    plt.ylabel("log10 Spectrum-Weighted Metric")
    plt.title("AUC vs. Reference Spectrum-Weighted Displacement Metric")

    text = (
        f"Pearson r = {pearson_r:.3f}\n"
        f"Spearman rho = {spearman_rho:.3f}\n"
        f"N = {len(merged)}"
    )

    plt.text(
        0.05,
        0.95,
        text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=600, bbox_inches="tight")
    plt.show()

    # =========================
    # Save text results
    # =========================

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("AUC vs Spectrum-Weighted Displacement Metric\n")
        f.write("===========================================\n\n")
        f.write(f"Number of matched materials: {len(merged)}\n")
        f.write(f"Energy column used: {energy_col}\n\n")

        f.write("Metric definition:\n")
        f.write("Dphi_flat_logE = integral sigma_d(E) d(log10 E)\n")
        f.write("This is a reference spectrum-weighted displacement-response metric,\n")
        f.write("not an experimental dpa measurement.\n\n")

        f.write("Correlation using log10(Dphi_flat_logE):\n")
        f.write(f"Pearson r     = {pearson_r:.6f}\n")
        f.write(f"Pearson p     = {pearson_p:.6e}\n")
        f.write(f"Spearman rho  = {spearman_rho:.6f}\n")
        f.write(f"Spearman p    = {spearman_p:.6e}\n\n")

        f.write("Correlation using raw Dphi_flat_logE:\n")
        f.write(f"Pearson r     = {pearson_raw_r:.6f}\n")
        f.write(f"Pearson p     = {pearson_raw_p:.6e}\n")
        f.write(f"Spearman rho  = {spearman_raw_rho:.6f}\n")
        f.write(f"Spearman p    = {spearman_raw_p:.6e}\n\n")

        f.write("Output files:\n")
        f.write(str(OUT_CSV) + "\n")
        f.write(str(OUT_FIG) + "\n")

    # =========================
    # Console output
    # =========================

    print("\n===========================================")
    print("AUC vs Spectrum-Weighted Metric Correlation")
    print("===========================================")
    print("Matched materials:", len(merged))

    print("\nUsing log10(Dphi_flat_logE):")
    print(f"Pearson r     : {pearson_r:.4f}")
    print(f"Pearson p     : {pearson_p:.4e}")
    print(f"Spearman rho  : {spearman_rho:.4f}")
    print(f"Spearman p    : {spearman_p:.4e}")

    print("\nUsing raw Dphi_flat_logE:")
    print(f"Pearson r     : {pearson_raw_r:.4f}")
    print(f"Pearson p     : {pearson_raw_p:.4e}")
    print(f"Spearman rho  : {spearman_raw_rho:.4f}")
    print(f"Spearman p    : {spearman_raw_p:.4e}")

    print("\nSaved:")
    print(OUT_CSV)
    print(OUT_FIG)
    print(OUT_TXT)


if __name__ == "__main__":
    main()