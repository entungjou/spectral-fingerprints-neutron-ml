from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =========================
# File paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "outputs" / "ml_dataset_v8.csv"
OUT_FILE = PROJECT_ROOT / "outputs" / "mlp_baseline_results.csv"


# =========================
# Target and features
# =========================
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


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Cannot find dataset: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required_cols = ["element", TARGET] + FEATURES
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        print("\nAvailable columns:")
        print(df.columns.tolist())
        raise ValueError(f"\nMissing required columns: {missing}")

    df = df.dropna(subset=required_cols).copy()

    X = df[FEATURES].values
    y = df[TARGET].values
    elements = df["element"].values

    print("Loaded:", DATA_FILE)
    print("Rows:", len(df))
    print("Target:", TARGET)
    print("Features:", len(FEATURES))
    print("MLP hidden layers: (64, 32)")

    loo = LeaveOneOut()

    y_true = []
    y_pred = []
    names = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=5000,
                random_state=42,
                early_stopping=False
            ))
        ])

        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]

        y_true.append(y_test[0])
        y_pred.append(pred)
        names.append(elements[test_idx][0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    residual_std = np.std(errors)

    results = pd.DataFrame({
        "element": names,
        "true_auc": y_true,
        "pred_auc": y_pred,
        "error": errors,
        "abs_error": abs_errors
    })

    results.to_csv(OUT_FILE, index=False)

    print("\n==============================")
    print("MLP BASELINE RESULTS")
    print("==============================")
    print("Hidden layers: (64, 32)")
    print("Activation: ReLU")
    print("Optimizer: Adam")
    print(f"LOO R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Residual STD: {residual_std:.4f}")

    print("\nSaved:")
    print(OUT_FILE)


if __name__ == "__main__":
    main()