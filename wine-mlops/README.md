# Lab 4 – Automated ML Pipeline with GitHub Actions

## Dataset and task
- **Source:** UCI Wine Quality (Cortez et al., 2009) – red Vinho Verde wine, `data/winequality-red.csv` (84 KB, committed to the repo; semicolon-separated).
- **Task:** regression – predict the sensory `quality` score (median expert rating, 0–10).
- **Target:** `quality`
- **Features (11):** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.
- **Model:** `StandardScaler` + `RandomForestRegressor(n_estimators=200)` in one sklearn `Pipeline` (scaler fitted on training data only). **Baseline:** `DummyRegressor(strategy="mean")`.
- **Split:** 80/20, `random_state=42`. Same validation set for baseline and model.
- **Metric:** RMSE (lower is better). **Margin:** 0.15 RMSE units – see answers below.

## Setup and run locally
```bash
pip install -r requirements.txt
python src/validate_data.py
python src/train.py
python src/evaluate.py      # quality gate, exit 1 on failure
pytest -v                   # application tests
python src/predict.py '{"fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0, "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0, "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51, "sulphates": 0.56, "alcohol": 9.4}'
```

## Pipeline (`.github/workflows/pipeline.yml`, runs on push to `main`, one job)
checkout → setup Python → install deps → validate data → train baseline + model → quality gate → pytest → assemble package → upload artifact `wine-model-package-run-<run_number>-<sha>`.
Every check exits non-zero on failure; there is no `continue-on-error` and no `if: always()`, so the upload step runs only if everything before it passed.

Package contents: `model.joblib` (model + preprocessing), `metrics_report.json` (baseline score, model score, margin, gate result), `predict.py`, `requirements.txt`.

## Evidence
| Run | Link | Result |
|---|---|---|
| Failure A (quality gate) | TODO | TODO |
| Failure B (application test) | TODO | TODO |
| Final success | TODO | TODO – artifact: `wine-model-package-run-…` (local copy downloaded: yes/no) |

**What I changed**
- Failure A: TODO (e.g. `TRAIN_FRACTION = 0.02` in `src/config.py`; gate, metric, split, margin unchanged).
- Failure B: TODO (e.g. removed the missing-feature check in `src/predict.py`).
- Recovery: TODO.

## Answers
1. **Why RMSE?** TODO
2. **Why margin 0.15?** TODO (what if too low / too high?)
3. **What caused each failed run? Which check prevented publication?** TODO
4. **CI and artifact delivery in the workflow:** TODO
5. **MLOps maturity level:** TODO
