from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_FILE = PROJECT_ROOT / "outputs" / "ml_dataset_v8.csv"
OUT_MODEL = PROJECT_ROOT / "outputs" / "rf_damage_model_v8.joblib"
OUT_PRED = PROJECT_ROOT / "outputs" / "loo_predictions_v8.csv"
OUT_IMPORTANCE = PROJECT_ROOT / "outputs" / "feature_importance_v8.csv"

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
    df = pd.read_csv(IN_FILE)

    # Rename old column name if still present
    df = df.rename(columns={"energy_at_peak_MeV": "energy_at_peak_eV"})

    required = ["element", TARGET] + FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df.dropna(subset=required).copy()

    X = df[FEATURES]
    y = df[TARGET]
    elements = df["element"]

    print("Loaded:", IN_FILE)
    print("Rows:", len(df))
    print("Target:", TARGET)
    print("AUC is used ONLY as target, not input feature.")
    print("Number of features:", len(FEATURES))

    loo = LeaveOneOut()
    preds, trues, names = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)[0]

        preds.append(y_pred)
        trues.append(y_test.values[0])
        names.append(elements.iloc[test_idx].values[0])

    results = pd.DataFrame({
        "element": names,
        "true_auc": trues,
        "pred_auc": preds,
    })

    results["error"] = results["pred_auc"] - results["true_auc"]
    results["abs_error"] = results["error"].abs()
    results["pct_error"] = results["abs_error"] / results["true_auc"].abs() * 100

    mae = mean_absolute_error(results["true_auc"], results["pred_auc"])
    r2 = r2_score(results["true_auc"], results["pred_auc"])

    print("\nRandom Forest Results, LOOCV")
    print(f"MAE: {mae:.4f}")
    print(f"R2 : {r2:.4f}")

    final_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(X, y)

    importances = pd.DataFrame({
        "feature": FEATURES,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    results.to_csv(OUT_PRED, index=False)
    importances.to_csv(OUT_IMPORTANCE, index=False)
    joblib.dump(final_model, OUT_MODEL)

    print("\nSaved:")
    print("Predictions ->", OUT_PRED)
    print("Importance  ->", OUT_IMPORTANCE)
    print("Model       ->", OUT_MODEL)


if __name__ == "__main__":
    main()