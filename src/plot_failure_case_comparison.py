from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# =========================
# Plot style
# =========================
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12
mpl.rcParams["legend.fontsize"] = 12

# =========================
# File paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "outputs" / "dxs_master_dataset_all_v2_clean.csv"
OUT_FIG = PROJECT_ROOT / "outputs" / "failure_case_curve_comparison.png"

# =========================
# Materials to compare
# =========================
FAILURE_CASES = ["Bi", "Nb", "Ne"]
GOOD_CASES = ["Fe", "Si", "Al"]

ALL_ELEMENTS = FAILURE_CASES + GOOD_CASES


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Cannot find data file: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    # Rename old energy column if needed
    df = df.rename(columns={"energy_MeV": "energy_eV"})

    required_cols = ["element", "energy_eV", "displacement_xs_barn"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert numeric columns safely
    df["energy_eV"] = pd.to_numeric(df["energy_eV"], errors="coerce")
    df["displacement_xs_barn"] = pd.to_numeric(
        df["displacement_xs_barn"],
        errors="coerce"
    )

    # Clean invalid rows
    df = df.dropna(subset=required_cols).copy()
    df = df[
        (df["energy_eV"] > 0)
        & (df["displacement_xs_barn"] > 0)
    ].copy()

    # Keep only selected materials
    df = df[df["element"].isin(ALL_ELEMENTS)].copy()

    if df.empty:
        raise ValueError("No selected elements found in the dataset.")

    # =========================
    # Plot
    # =========================
    plt.figure(figsize=(9, 6))

    # Failure cases: thicker solid lines
    for el in FAILURE_CASES:
        sub = df[df["element"] == el].sort_values("energy_eV")

        if sub.empty:
            print(f"Warning: no data found for {el}")
            continue

        plt.plot(
            sub["energy_eV"],
            sub["displacement_xs_barn"],
            linewidth=2.5,
            label=f"{el} (failure case)"
        )

    # Well-predicted references: thinner dashed lines
    for el in GOOD_CASES:
        sub = df[df["element"] == el].sort_values("energy_eV")

        if sub.empty:
            print(f"Warning: no data found for {el}")
            continue

        plt.plot(
            sub["energy_eV"],
            sub["displacement_xs_barn"],
            linewidth=1.8,
            linestyle="--",
            alpha=0.9,
            label=f"{el} (well predicted)"
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Neutron Energy (eV)")
    plt.ylabel("Displacement Cross Section (barn)")
    plt.title("Comparison of DXS Curves Between Failure-Case and Well-Predicted Materials")

    plt.legend()
    plt.tight_layout()

    plt.savefig(OUT_FIG, dpi=600, bbox_inches="tight")
    plt.show()

    print("\nSaved:")
    print(OUT_FIG)


if __name__ == "__main__":
    main()