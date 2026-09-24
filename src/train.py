"""Train baseline + candidate on the same split; write raw metrics."""
import json
import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from validate_data import load_data


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    df = load_data()
    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SPLIT_SEED)

    # Optional weakening of the training data (Failure A demo only)
    if config.TRAIN_FRACTION < 1.0:
        X_train = X_train.sample(frac=config.TRAIN_FRACTION, random_state=config.SPLIT_SEED)
        y_train = y_train.loc[X_train.index]

    # Baseline: predicts the training mean
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_rmse = rmse(y_val, baseline.predict(X_val))

    # Candidate: scaler is fitted on training data only (inside the Pipeline)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=config.N_ESTIMATORS,
                                     max_depth=config.MAX_DEPTH,
                                     random_state=config.MODEL_SEED, n_jobs=-1)),
    ]).fit(X_train, y_train)
    model_rmse = rmse(y_val, model.predict(X_val))

    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "features": config.FEATURES}, config.MODEL_PATH)

    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.METRICS_PATH.write_text(json.dumps({
        "metric": config.METRIC_NAME,
        "baseline_score": baseline_rmse,
        "model_score": model_rmse,
        "n_train_rows": int(len(X_train)),
        "n_val_rows": int(len(X_val)),
    }, indent=2))
    print(f"Baseline RMSE={baseline_rmse:.4f}  Model RMSE={model_rmse:.4f}  "
          f"(train rows={len(X_train)}, val rows={len(X_val)})")


if __name__ == "__main__":
    main()
