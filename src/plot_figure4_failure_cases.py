from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "outputs"

IN_FILE = OUT_DIR / "loo_predictions_v8.csv"
OUT_FIG = OUT_DIR / "failure_cases.png"
OUT_CSV = OUT_DIR / "top10_failure_cases.csv"

df = pd.read_csv(IN_FILE)

df["abs_error"] = (df["pred_auc"] - df["true_auc"]).abs()

top10 = df.sort_values("abs_error", ascending=False).head(10)
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===== Font settings =====
mpl.rcParams['font.family'] = 'Times New Roman'

mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 11

mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

mpl.rcParams['legend.fontsize'] = 10

mpl.rcParams['figure.titlesize'] = 12
plt.figure(figsize=(8,6))

plt.barh(top10["element"], top10["abs_error"])

plt.xlabel("Absolute Error in Radiation Damage AUC", fontsize=12)
plt.ylabel("Material", fontsize=12)

plt.title(
    "Failure Cases of the Optimized Random Forest Model",
    fontsize=14
)

plt.gca().invert_yaxis()

plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.tight_layout()

plt.savefig(OUT_FIG, dpi=600)

top10.to_csv(OUT_CSV, index=False)

print("Saved figure ->", OUT_FIG)
print("Saved table  ->", OUT_CSV)
print("\nTop 10 failure cases:")
print(top10[["element", "true_auc", "pred_auc", "abs_error"]].to_string(index=False))