"""Single source of truth for the pipeline. Metric, split and margin must stay
fixed during the failure/recovery demonstration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "winequality-red.csv"
MODEL_PATH = ROOT / "models" / "model.joblib"
METRICS_PATH = ROOT / "metrics" / "metrics.json"
REPORT_PATH = ROOT / "metrics" / "metrics_report.json"

TARGET = "quality"
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]

# Evaluation setup (do NOT change during the demo)
METRIC_NAME = "RMSE"          # lower is better
TEST_SIZE = 0.2
SPLIT_SEED = 42
MARGIN = 0.15                 # model_rmse must be <= baseline_rmse - MARGIN

# Model settings
N_ESTIMATORS = 200
MAX_DEPTH = None
MODEL_SEED = 42

# Fraction of the TRAINING split used to fit the model (1.0 = all of it).
# Failure A demo: lower this (e.g. 0.02) and push. Restore to 1.0 afterwards.
TRAIN_FRACTION = 0.02