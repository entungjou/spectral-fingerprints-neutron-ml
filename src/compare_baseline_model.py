from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_ROOT / "outputs" / "ml_dataset_v8.csv"
OUT_FILE = PROJECT_ROOT / "outputs" / "baseline_model_comparison.csv"

TARGET = "auc_log10xs_over_log10E"

BASELINE_FEATURES = [
    "Z",
    "atomic_mass",
]

FULL_FEATURES = [
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


def evaluate_model(model, X, y):
    loo = LeaveOneOut()

    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]

        y_true.append(y_test.values[0])
        y_pred.append(pred)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mae, r2


def main():
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Cannot find input file: {IN_FILE}")

    df = pd.read_csv(IN_FILE)

    # Rename old energy column if it still exists
    df = df.rename(columns={
        "energy_at_peak_MeV": "energy_at_peak_eV"
    })

    required_cols = list(set(BASELINE_FEATURES + FULL_FEATURES + [TARGET]))

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    # Convert to numeric and remove invalid values
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_cols).copy()

    print("Loaded:", IN_FILE)
    print("Rows after cleaning:", len(df))

    y = df[TARGET]

    results = []

    # =========================
    # Baseline 1: Linear Regression with atomic descriptors only
    # =========================
    X_atomic = df[BASELINE_FEATURES]

    mae, r2 = evaluate_model(
        make_pipeline(StandardScaler(), LinearRegression()),
        X_atomic,
        y
    )

    results.append({
        "Model": "Linear Regression",
        "Input Features": "Atomic descriptors only (Z, atomic mass)",
        "R2": r2,
        "MAE": mae,
    })

    # =========================
    # Baseline 2: Ridge Regression with atomic descriptors only
    # =========================
    mae, r2 = evaluate_model(
        make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        X_atomic,
        y
    )

    results.append({
        "Model": "Ridge Regression",
        "Input Features": "Atomic descriptors only (Z, atomic mass)",
        "R2": r2,
        "MAE": mae,
    })

    # =========================
    # Baseline 3: Random Forest with atomic descriptors only
    # =========================
    mae, r2 = evaluate_model(
        RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
        X_atomic,
        y
    )

    results.append({
        "Model": "Random Forest",
        "Input Features": "Atomic descriptors only (Z, atomic mass)",
        "R2": r2,
        "MAE": mae,
    })

    # =========================
    # Proposed Model: Random Forest with physics-informed features
    # =========================
    X_full = df[FULL_FEATURES]

    mae, r2 = evaluate_model(
        RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
        X_full,
        y
    )

    results.append({
        "Model": "Random Forest (Proposed)",
        "Input Features": "Physics-informed cross-section descriptors",
        "R2": r2,
        "MAE": mae,
    })

    results_df = pd.DataFrame(results)

    print("\nBaseline model comparison:")
    print(results_df.to_string(index=False))

    results_df.to_csv(OUT_FILE, index=False)

    print("\nSaved:", OUT_FILE)


if __name__ == "__main__":
    main()