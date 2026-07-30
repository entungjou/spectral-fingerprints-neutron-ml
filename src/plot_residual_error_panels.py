from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_ROOT / "outputs" / "loo_predictions_v8.csv"
OUT_FIG = PROJECT_ROOT / "outputs" / "figure5_residual_error_panels.png"

def main():
    df = pd.read_csv(IN_FILE)

    # Expected columns: element, true_auc, pred_auc, error, abs_error
    required = ["true_auc", "pred_auc", "error"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available columns: {df.columns.tolist()}")

    # Residual = predicted - true
    df["residual"] = df["pred_auc"] - df["true_auc"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))

    # -------------------------
    # (a) Residual plot
    # -------------------------
    ax = axes[0]
    ax.scatter(df["true_auc"], df["residual"], s=35, alpha=0.8)
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_title("Residual Plot")
    ax.set_xlabel("True Radiation Damage AUC")
    ax.set_ylabel("Residual (Predicted - True)")

    ax.text(
        0.02, 0.95, "(a)",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left"
    )

    # -------------------------
    # (b) Error distribution
    # -------------------------
    ax = axes[1]
    ax.hist(df["residual"], bins=15)

    ax.set_title("Error Distribution")
    ax.set_xlabel("Prediction Error (pred - true)")
    ax.set_ylabel("Count")

    ax.text(
        0.02, 0.95, "(b)",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left"
    )

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=600, bbox_inches="tight")
    plt.show()

    print("Saved figure:", OUT_FIG)

if __name__ == "__main__":
    main()