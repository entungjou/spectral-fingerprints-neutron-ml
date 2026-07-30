from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import resample

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_ROOT / "outputs" / "ml_dataset_v8.csv"
OUT_FILE = PROJECT_ROOT / "outputs" / "statistical_validation_results.txt"

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


def loo_predict(X, y):
    loo = LeaveOneOut()

    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=1,
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)[0]

        y_true.append(y_test.values[0])
        y_pred.append(pred)

    return np.array(y_true), np.array(y_pred)


def bootstrap_ci(y_true, y_pred, n_bootstrap=1000):
    r2_values = []
    mae_values = []

    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = np.random.choice(np.arange(n), size=n, replace=True)

        yt = y_true[idx]
        yp = y_pred[idx]

        r2_values.append(r2_score(yt, yp))
        mae_values.append(mean_absolute_error(yt, yp))

    return {
        "r2_mean": np.mean(r2_values),
        "r2_low": np.percentile(r2_values, 2.5),
        "r2_high": np.percentile(r2_values, 97.5),

        "mae_mean": np.mean(mae_values),
        "mae_low": np.percentile(mae_values, 2.5),
        "mae_high": np.percentile(mae_values, 97.5),
    }


def permutation_test(X, y, n_perm=50):
    perm_r2 = []

    for i in range(n_perm):
        y_shuffled = pd.Series(np.random.permutation(y.values))

        yt, yp = loo_predict(X, y_shuffled)

        r2 = r2_score(yt, yp)

        perm_r2.append(r2)

        print(f"Permutation {i+1}/{n_perm}: R2 = {r2:.4f}")

    return np.array(perm_r2)


def main():
    df = pd.read_csv(IN_FILE)

    df = df.rename(columns={
        "energy_at_peak_MeV": "energy_at_peak_eV"
    })

    required = FEATURES + [TARGET]

    df = df.dropna(subset=required).copy()

    X = df[FEATURES]
    y = df[TARGET]

    print("Running LOOCV on real model...")

    y_true, y_pred = loo_predict(X, y)

    real_r2 = r2_score(y_true, y_pred)
    real_mae = mean_absolute_error(y_true, y_pred)

    print("\nReal model:")
    print(f"R2  = {real_r2:.4f}")
    print(f"MAE = {real_mae:.4f}")

    print("\nRunning bootstrap confidence intervals...")

    ci = bootstrap_ci(y_true, y_pred)

    print("\nBootstrap Results:")
    print(f"R2 95% CI : [{ci['r2_low']:.4f}, {ci['r2_high']:.4f}]")
    print(f"MAE 95% CI: [{ci['mae_low']:.4f}, {ci['mae_high']:.4f}]")

    print("\nRunning permutation test...")

    perm_r2 = permutation_test(X, y, n_perm=200)

    perm_mean = np.mean(perm_r2)
    perm_max = np.max(perm_r2)

    print("\nPermutation Test Results:")
    print(f"Mean shuffled R2: {perm_mean:.4f}")
    print(f"Best shuffled R2: {perm_max:.4f}")

    with open(OUT_FILE, "w") as f:
        f.write("STATISTICAL VALIDATION RESULTS\n")
        f.write("===============================\n\n")

        f.write(f"Real R2: {real_r2:.4f}\n")
        f.write(f"Real MAE: {real_mae:.4f}\n\n")

        f.write("Bootstrap Confidence Intervals:\n")
        f.write(f"R2 95% CI : [{ci['r2_low']:.4f}, {ci['r2_high']:.4f}]\n")
        f.write(f"MAE 95% CI: [{ci['mae_low']:.4f}, {ci['mae_high']:.4f}]\n\n")

        f.write("Permutation Test:\n")
        f.write(f"Mean shuffled R2: {perm_mean:.4f}\n")
        f.write(f"Best shuffled R2: {perm_max:.4f}\n")

    print("\nSaved:", OUT_FILE)


if __name__ == "__main__":
    main()