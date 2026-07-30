from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 12

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "outputs" / "ml_dataset_v8.csv"
OUT_FIG = PROJECT_ROOT / "outputs" / "permutation_importance.png"
OUT_CSV = PROJECT_ROOT / "outputs" / "permutation_importance.csv"

TARGET = "auc_log10xs_over_log10E"

FEATURES = [
    "energy_at_peak_eV",
    "std_log10_xs",
    "peak_xs_barn",
    "slope_high",
    "Z",
    "atomic_mass",
    "log10_energy_at_peak",
    "peak_to_mean_ratio",
    "peak_to_median_ratio",
    "xs_dynamic_range_log10",
    "peak_width_log10E_halfmax",
    "energy_span_log10",
    "curve_center_log10E",
    "weighted_center_log10E",
]

df = pd.read_csv(DATA_FILE)

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET]

model = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

perm = permutation_importance(
    model,
    X,
    y,
    n_repeats=50,
    random_state=42,
    scoring="r2"
)

importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
})

importance_df = importance_df.sort_values(
    "importance_mean",
    ascending=True
)

importance_df.to_csv(OUT_CSV, index=False)

plt.figure(figsize=(7, 5))

plt.barh(
    importance_df["feature"],
    importance_df["importance_mean"],
    xerr=importance_df["importance_std"]
)

plt.xlabel("Permutation Importance (Decrease in R²)")
plt.title("Permutation Importance of Physics-Informed Features")

plt.tight_layout()

plt.savefig(OUT_FIG, dpi=600, bbox_inches="tight")

plt.show()

print("\nSaved:")
print(OUT_FIG)
print(OUT_CSV)