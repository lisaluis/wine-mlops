"""Quality gate: exit 1 unless model_rmse <= baseline_rmse - MARGIN."""
import json
import sys
import config


def main():
    m = json.loads(config.METRICS_PATH.read_text())
    threshold = m["baseline_score"] - config.MARGIN
    passed = m["model_score"] <= threshold
    report = {**m, "margin": config.MARGIN, "required_max_model_score": threshold,
              "improvement": m["baseline_score"] - m["model_score"],
              "gate_passed": bool(passed)}
    config.REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not passed:
        print(f"QUALITY GATE FAILED: model RMSE {m['model_score']:.4f} > "
              f"allowed {threshold:.4f}")
        sys.exit(1)
    print("Quality gate passed.")


if __name__ == "__main__":
    main()
