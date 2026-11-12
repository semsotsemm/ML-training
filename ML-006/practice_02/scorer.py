"""Score both submission files against the local answer key."""

import argparse
from datetime import date
import json
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COLUMNS = ["query_id", "prediction", "lower_90", "upper_90"]


def concentration_metrics(scored):
    truth = scored["true_concentration"].to_numpy()
    prediction = scored["prediction"].to_numpy()
    low, high = scored[["lower_90", "upper_90"]].to_numpy().T
    width = high - low
    outside = np.maximum(low - truth, 0) + np.maximum(truth - high, 0)
    return {
        "rows": len(scored),
        "rmse_ug_l": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "mae_ug_l": float(np.mean(np.abs(prediction - truth))),
        "coverage_90": float(np.mean(outside == 0)),
        "mean_width_ug_l": float(width.mean()),
        "interval_score_90": float((width + 20 * outside).mean()),
    }


def score_predictions(file, key, queries, domain):
    submission = pd.read_csv(file)
    if set(submission.columns) != set(COLUMNS) or submission.columns.duplicated().any():
        raise ValueError(f"predictions.csv must have columns {', '.join(COLUMNS)}")
    if len(submission) != len(key) or set(submission["query_id"]) != set(
        key["query_id"]
    ):
        raise ValueError("Submit exactly one row per query_id")
    submission[COLUMNS[1:]] = submission[COLUMNS[1:]].apply(
        pd.to_numeric, errors="raise"
    )
    values = submission[COLUMNS[1:]].to_numpy(dtype=float)
    pred, low, high = values.T
    if (
        not np.isfinite(values).all()
        or not ((0 <= low) & (low <= pred) & (pred <= high)).all()
    ):
        raise ValueError(
            "Values must be finite and satisfy 0 <= lower_90 <= prediction <= upper_90"
        )
    scored = key.merge(submission, on="query_id", validate="one_to_one")
    scored = scored.merge(
        queries[["query_id", "t"]], on="query_id", validate="one_to_one"
    )
    future = pd.to_datetime(scored["t"]).ge(domain["mass_date"])
    return {
        "all": concentration_metrics(scored),
        "historical": concentration_metrics(scored[~future]),
        "future": concentration_metrics(scored[future]),
    }


def score_summary(file, key, domain):
    result = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or set(result) != {"mass_kg", "arrival_date"}:
        raise ValueError("summary.json must have mass_kg and arrival_date")
    mass = result["mass_kg"]
    if type(mass) not in (int, float) or not isfinite(mass) or mass < 0:
        raise ValueError("mass_kg must be a finite nonnegative number")
    arrival = result["arrival_date"]
    submitted_date = None
    if arrival is not None:
        if not isinstance(arrival, str):
            raise ValueError("arrival_date must be YYYY-MM-DD or null")
        submitted_date = date.fromisoformat(arrival)
        if (
            submitted_date.isoformat() != arrival
            or not domain["mass_date"] <= arrival <= domain["forecast_end"]
        ):
            raise ValueError(
                "arrival_date must use YYYY-MM-DD and be within the forecast"
            )
    expected = key["arrival_date"]
    both_dates = expected is not None and submitted_date is not None
    return {
        "mass_absolute_error_kg": abs(mass - key["mass_kg"]),
        "mass_relative_error": abs(mass - key["mass_kg"]) / key["mass_kg"],
        "arrival_status_correct": (expected is None) == (submitted_date is None),
        "arrival_absolute_error_days": (
            abs((submitted_date - date.fromisoformat(expected)).days)
            if both_dates
            else None
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path, nargs="?", default=Path("submission"))
    args = parser.parse_args()
    try:
        domain = json.loads((ROOT / "domain.json").read_text(encoding="utf-8"))
        key = pd.read_csv(ROOT / "_scoring" / "queries_key.csv")
        summary_key = json.loads(
            (ROOT / "_scoring" / "summary_key.json").read_text(encoding="utf-8")
        )
        queries = pd.read_csv(ROOT / "queries_test.csv")
        points = score_predictions(
            args.submission / "predictions.csv", key, queries, domain
        )
        summary = score_summary(args.submission / "summary.json", summary_key, domain)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {"concentrations": points, "summary": summary}, indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
