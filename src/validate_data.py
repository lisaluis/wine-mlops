"""Load the dataset and fail (exit 1) if it does not look as expected."""
import sys
import pandas as pd
from config import DATA_PATH, FEATURES, TARGET


def load_data(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def validate(df: pd.DataFrame) -> list[str]:
    errors = []
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
        return errors
    if len(df) < 500:
        errors.append(f"Too few rows: {len(df)} (expected >= 500)")
    cols = FEATURES + [TARGET]
    if df[cols].isnull().any().any():
        errors.append("Null values found in features/target")
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        errors.append(f"Non-numeric columns: {non_numeric}")
    return errors


if __name__ == "__main__":
    data = load_data()
    problems = validate(data)
    if problems:
        for p in problems:
            print(f"DATA VALIDATION FAILED: {p}")
        sys.exit(1)
    print(f"Data validation passed: {len(data)} rows, {len(FEATURES)} features.")
