from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"

# best model files
PRED_FILE = OUT_DIR / "loo_predictions_v8.csv"
IMP_FILE = OUT_DIR / "feature_importance.csv"

# optional comparison files
COMPARE_FILES = {
    "v4": OUT_DIR / "loo_predictions_v4.csv",
    "v7": OUT_DIR / "loo_predictions_v7.csv",
    "v8": OUT_DIR / "loo_predictions_v8.csv",
    "v9": OUT_DIR / "loo_predictions_v9.csv",
    "nn_v5": OUT_DIR / "loo_predictions_nn_v5.csv",
}

# outputs
OUT_TRUE_PRED = OUT_DIR / "true_vs_prediction.png"
OUT_ERR_HIST = OUT_DIR / "error_histogram.png"
OUT_ABSERR_TRUE = OUT_DIR / "abs_error_vs_true.png"
OUT_IMPORTANCE = OUT_DIR / "feature_importance.png"

OUT_WORST = OUT_DIR / "final_top10_worst_v8.csv"
OUT_BEST = OUT_DIR / "final_top10_best_v8.csv"
OUT_COMPARE = OUT_DIR / "final_model_comparison.csv"
OUT_SUMMARY = OUT_DIR / "final_summary_v8.txt"


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find prediction file: {path}")

    df = pd.read_csv(path)
    required = {"element", "true_auc", "pred_auc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")

    df["error"] = df["pred_auc"] - df["true_auc"]
    df["abs_error"] = df["error"].abs()
    return df


def plot_true_vs_pred(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # ===== Font settings =====
    mpl.rcParams['font.family'] = 'Times New Roman'

    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.labelsize'] = 12

    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    mpl.rcParams['legend.fontsize'] = 12

    mpl.rcParams['figure.titlesize'] = 12
    plt.figure(figsize=(7, 6))
    plt.scatter(df["true_auc"], df["pred_auc"], s=45)

    min_val = min(df["true_auc"].min(), df["pred_auc"].min())
    max_val = max(df["true_auc"].max(), df["pred_auc"].max())

    plt.plot([min_val, max_val], [min_val, max_val], linewidth=2)

    plt.xlabel("True AUC")
    plt.ylabel("Predicted AUC")
    plt.title("True vs Predicted AUC")
    plt.tight_layout()
    plt.savefig(OUT_TRUE_PRED, dpi=300)
    plt.close()


def plot_error_hist(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # ===== Font settings =====
    mpl.rcParams['font.family'] = 'Times New Roman'

    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.labelsize'] = 12

    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    mpl.rcParams['legend.fontsize'] = 12

    mpl.rcParams['figure.titlesize'] = 12
    plt.figure(figsize=(7, 5.5))
    plt.hist(df["error"], bins=15)
    plt.xlabel("Prediction Error (pred - true)")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.tight_layout()
    plt.savefig(OUT_ERR_HIST, dpi=300)
    plt.close()


def plot_abs_error_vs_true(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # ===== Font settings =====
    mpl.rcParams['font.family'] = 'Times New Roman'

    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.labelsize'] = 12

    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    mpl.rcParams['legend.fontsize'] = 12

    mpl.rcParams['figure.titlesize'] = 12
    plt.figure(figsize=(7, 5.5))
    plt.scatter(df["true_auc"], df["abs_error"], s=45)
    plt.xlabel("True AUC")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs True AUC")
    plt.tight_layout()
    plt.savefig(OUT_ABSERR_TRUE, dpi=300)
    plt.close()


def plot_feature_importance(imp_df: pd.DataFrame):
    required = {"feature", "importance"}
    missing = required - set(imp_df.columns)
    if missing:
        raise ValueError(f"Missing columns in importance file: {sorted(missing)}")

    imp_df = imp_df.sort_values("importance", ascending=True).copy()

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # ===== Font settings =====
    mpl.rcParams['font.family'] = 'Times New Roman'

    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.labelsize'] = 12

    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    mpl.rcParams['legend.fontsize'] = 12

    mpl.rcParams['figure.titlesize'] = 12
    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["feature"], imp_df["importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(OUT_IMPORTANCE, dpi=300)
    plt.close()


def summarize_predictions(df: pd.DataFrame):
    best = df.sort_values("abs_error", ascending=True).head(10).copy()
    worst = df.sort_values("abs_error", ascending=False).head(10).copy()

    best.to_csv(OUT_BEST, index=False)
    worst.to_csv(OUT_WORST, index=False)

    return best, worst


def compare_models():
    rows = []

    for name, path in COMPARE_FILES.items():
        if not path.exists():
            continue

        df = load_predictions(path)

        mae = np.mean(np.abs(df["pred_auc"] - df["true_auc"]))
        ss_res = np.sum((df["true_auc"] - df["pred_auc"]) ** 2)
        ss_tot = np.sum((df["true_auc"] - df["true_auc"].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        rows.append({
            "model": name,
            "n_samples": len(df),
            "MAE": mae,
            "R2": r2,
        })

    if not rows:
        return pd.DataFrame()

    comp = pd.DataFrame(rows).sort_values(["R2", "MAE"], ascending=[False, True]).reset_index(drop=True)
    comp.to_csv(OUT_COMPARE, index=False)
    return comp


def write_summary(df: pd.DataFrame, imp_df: pd.DataFrame, best: pd.DataFrame, worst: pd.DataFrame, comp: pd.DataFrame):
    mae = np.mean(np.abs(df["pred_auc"] - df["true_auc"]))
    ss_res = np.sum((df["true_auc"] - df["pred_auc"]) ** 2)
    ss_tot = np.sum((df["true_auc"] - df["true_auc"].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    top_features = imp_df.sort_values("importance", ascending=False).head(5)

    lines = []
    lines.append("Final Model Summary (v8)")
    lines.append("=" * 40)
    lines.append(f"Samples: {len(df)}")
    lines.append(f"MAE: {mae:.4f}")
    lines.append(f"R2: {r2:.4f}")
    lines.append("")

    lines.append("Top 5 important features:")
    for _, row in top_features.iterrows():
        lines.append(f"- {row['feature']}: {row['importance']:.6f}")
    lines.append("")

    lines.append("Top 5 best-predicted materials:")
    for _, row in best.head(5).iterrows():
        lines.append(
            f"- {row['element']}: true={row['true_auc']:.4f}, pred={row['pred_auc']:.4f}, abs_error={row['abs_error']:.4f}"
        )
    lines.append("")

    lines.append("Top 5 worst-predicted materials:")
    for _, row in worst.head(5).iterrows():
        lines.append(
            f"- {row['element']}: true={row['true_auc']:.4f}, pred={row['pred_auc']:.4f}, abs_error={row['abs_error']:.4f}"
        )
    lines.append("")

    if not comp.empty:
        lines.append("Model comparison:")
        for _, row in comp.iterrows():
            lines.append(
                f"- {row['model']}: MAE={row['MAE']:.4f}, R2={row['R2']:.4f}"
            )
        lines.append("")

    lines.append("Main interpretation:")
    lines.append("- v8 is the best-performing model among the tested versions.")
    lines.append("- Physics-inspired curve-shape descriptors improved performance substantially.")
    lines.append("- Peak-location and curve-distribution features dominate prediction.")
    lines.append("- Some unusual materials still show large prediction errors and remain important future cases for analysis.")

    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def main():
    pred_df = load_predictions(PRED_FILE)
    imp_df = pd.read_csv(IMP_FILE)

    best, worst = summarize_predictions(pred_df)
    comp = compare_models()

    plot_true_vs_pred(pred_df)
    plot_error_hist(pred_df)
    plot_abs_error_vs_true(pred_df)
    plot_feature_importance(imp_df)
    write_summary(pred_df, imp_df, best, worst, comp)

    print("Saved final analysis files:")
    print("-", OUT_TRUE_PRED)
    print("-", OUT_ERR_HIST)
    print("-", OUT_ABSERR_TRUE)
    print("-", OUT_IMPORTANCE)
    print("-", OUT_BEST)
    print("-", OUT_WORST)
    if not comp.empty:
        print("-", OUT_COMPARE)
    print("-", OUT_SUMMARY)

    if not comp.empty:
        print("\nModel comparison:")
        print(comp.to_string(index=False))

    print("\nTop 10 worst materials:")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()