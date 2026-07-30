from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs"

IN_FILE = OUT_DIR / "ml_dataset_v8.csv"
OUT_FIG = OUT_DIR / "figure_feature_correlation_heatmap.png"

df = pd.read_csv(IN_FILE)

df = df.rename(columns={
    "energy_at_peak_MeV": "energy_at_peak_eV"
})

drop_cols = [
    "element",
    "material",
    "symbol",
    "MT",
    "target",
    "true_auc",
    "pred_auc",
    "auc",
    "auc_log10xs_over_log10E",
]

feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
feature_df = feature_df.select_dtypes(include=[np.number])

feature_df = feature_df.dropna(axis=1, how="all")
feature_df = feature_df.loc[:, feature_df.nunique(dropna=True) > 1]

corr = feature_df.corr(method="pearson")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(corr.columns)))
ax.set_yticks(np.arange(len(corr.columns)))

ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Pearson Correlation", fontsize=12)
cbar.ax.tick_params(labelsize=12)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        value = corr.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

ax.set_title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig(OUT_FIG, dpi=600, bbox_inches="tight")
plt.show()

print("Saved:", OUT_FIG)