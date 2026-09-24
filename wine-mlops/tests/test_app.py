import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config  # noqa: E402
import predict as app  # noqa: E402

SAMPLE = {"fixed acidity": 7.4, "volatile acidity": 0.7, "citric acid": 0.0,
          "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
          "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
          "sulphates": 0.56, "alcohol": 9.4}


@pytest.fixture(scope="module")
def bundle():
    return app.load_model(config.MODEL_PATH)


def test_model_loads(bundle):
    assert "pipeline" in bundle and bundle["features"] == config.FEATURES


def test_valid_sample_output_type_and_range(bundle):
    out = app.predict(SAMPLE, bundle)
    assert isinstance(out, float)
    assert 0 <= out <= 10          # wine quality scale


def test_missing_feature_rejected(bundle):
    bad = {k: v for k, v in SAMPLE.items() if k != "alcohol"}
    with pytest.raises(ValueError, match="alcohol"):
        app.predict(bad, bundle)
