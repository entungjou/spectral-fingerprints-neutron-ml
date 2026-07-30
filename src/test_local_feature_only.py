from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =========================
# File paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "outputs" / "ml_dataset_v8.csv"
OUT_FILE = PROJECT_ROOT / "outputs" / "local_features_only_results.csv"


# =========================
# Target
# =========================
TARGET = "auc_log10xs_over_log10E"


# =========================
# Local / non-integral-like features only
# =========================
LOCAL_FEATURES = [
    "energy_at_peak_eV",
    "log10_energy_at_peak",
    "peak_xs_barn",
    "peak_width_log10E_halfmax",
    "slope_high",
    "peak_to_mean_ratio",
    "peak_to_median_ratio",
    "Z",
    "atomic_mass",
]

# These are intentionally removed because they are more global/integral-like
REMOVED_FEATURES = [
    "weighted_center_log10E",
    "curve_center_log10E",
    "xs_dynamic_range_log10",
    "std_log10_xs",
    "energy_span_log10",
]


def evaluate_rf_loocv(df, features):
    X = df[features].values
    y = df[TARGET].values
    elements = df["element"].values

    loo = LeaveOneOut()

    y_true = []
    y_pred = []
    names = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features=1.0,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]

        y_true.append(y_test[0])
        y_pred.append(pred)
        names.append(elements[test_idx][0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    results = pd.DataFrame({
        "element": names,
        "true_auc": y_true,
        "pred_auc": y_pred,
        "error": y_pred - y_true,
        "abs_error": np.abs(y_pred - y_true),
    })

    return r2, mae, rmse, results


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Cannot find dataset: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required_cols = ["element", TARGET] + LOCAL_FEATURES
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print("\nAvailable columns:")
        print(df.columns.tolist())
        raise ValueError(f"\nMissing required columns: {missing}")

    df = df.dropna(subset=required_cols).copy()

    print("\nLoaded:", DATA_FILE)
    print("Rows after cleaning:", len(df))
    print("Target:", TARGET)

    print("\nLocal / non-integral-like features used:")
    for f in LOCAL_FEATURES:
        print(" -", f)

    print("\nIntegral-like/global descriptors intentionally removed:")
    for f in REMOVED_FEATURES:
        print(" -", f)

    r2, mae, rmse, results = evaluate_rf_loocv(df, LOCAL_FEATURES)

    results.to_csv(OUT_FILE, index=False)

    print("\n======================================")
    print("LOCAL-FEATURES-ONLY RANDOM FOREST")
    print("======================================")
    print(f"R2   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    print("\nWorst predictions:")
    print(results.sort_values("abs_error", ascending=False).head(10).to_string(index=False))

    print("\nSaved:")
    print(OUT_FILE)


if __name__ == "__main__":
    main()