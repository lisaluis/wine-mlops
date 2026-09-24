"""Prediction application. Self-contained so it can ship in the model package.

Usage: python predict.py '{"fixed acidity": 7.4, ... "alcohol": 9.4}'
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

_HERE = Path(__file__).resolve().parent


def _default_model_path() -> Path:
    for p in (_HERE / "model.joblib", _HERE.parent / "models" / "model.joblib"):
        if p.exists():
            return p
    return _HERE.parent / "models" / "model.joblib"


def load_model(path=None):
    return joblib.load(path or _default_model_path())


def predict(record: dict, bundle=None) -> float:
    bundle = bundle or load_model()
    features = bundle["features"]
    missing = [f for f in features if f not in record]
    if missing:
        raise ValueError(f"Missing required feature(s): {missing}")
    X = pd.DataFrame([{f: record[f] for f in features}])
    return float(bundle["pipeline"].predict(X)[0])


if __name__ == "__main__":
    try:
        print(predict(json.loads(sys.argv[1])))
    except (ValueError, IndexError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
