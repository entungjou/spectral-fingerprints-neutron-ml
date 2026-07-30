from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_ROOT / "outputs" / "material_features.csv"
OUT_FILE = PROJECT_ROOT / "outputs" / "ml_dataset.csv"

df = pd.read_csv(IN_FILE)

print("Original rows:", len(df))

# keep only arc-dpa model
df = df[df["MT"] == 900].copy()

print("Rows after MT=900 filter:", len(df))

# choose features
features = [
    "log10_xs_mean",
    "log10_xs_median",
    "auc_log10xs_over_log10E",
    "slope_low",
    "slope_mid",
    "slope_high"
]

cols = ["element"] + features

df = df[cols]

df.to_csv(OUT_FILE, index=False)

print("Saved ML dataset ->", OUT_FILE)
print(df.head())