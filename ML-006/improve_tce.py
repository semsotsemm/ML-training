"""Fit a smooth plume model to the first ML-006 dataset and evaluate it."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit

DATA = Path(__file__).resolve().parent / "data" / "raw"
START = pd.Timestamp("2020-01-01")
FEATURES = ["x", "y", "z", "days"]
# The old generator adds a fixed offset and rounds the clean targets to cents.
CLEAN_BACKGROUND = 0.10


def add_days(table):
    iso = pd.to_datetime(table["t"], format="%Y-%m-%d", errors="coerce")
    european = pd.to_datetime(table["t"], format="%d.%m.%Y", errors="coerce")
    dates = iso.fillna(european)
    return table.assign(days=(dates - START).dt.days)


def load_observations(data_dir):
    wells = pd.read_csv(data_dir / "wells.csv").drop_duplicates()
    # Negative x can occur near the site edge. -999 is the missing-value code.
    wells = wells[wells["x"].ne(-999)].dropna(subset=["x", "y", "z"])
    wells["z"] = -wells["z"].abs()

    samples = add_days(pd.read_csv(data_dir / "concentration.csv"))
    raw = samples["concentration"].astype("string").str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    limit = pd.to_numeric(raw.str.extract(r"^<\s*([\d.]+)$")[0], errors="coerce")
    # Use half the detection limit as a simple approximation for censored values.
    samples["concentration"] = numeric.fillna(limit / 2).mask(raw.eq("ND"), 0.025)
    # This exact repeated value was injected by the first dataset's generator.
    samples = samples.query(
        "concentration >= 0 and concentration != 999.99 and days >= 0"
    )
    samples = samples.drop_duplicates(["well_id", "days", "concentration"])
    samples = samples.groupby(["well_id", "days"], as_index=False)[
        "concentration"
    ].median()
    samples = samples.merge(wells, on="well_id", validate="many_to_one")
    # Near-duplicate well IDs from the generator stay on the same side of a split.
    samples["block"] = (
        (samples["x"] // 300).astype(int).astype(str)
        + "/"
        + (samples["y"] // 300).astype(int).astype(str)
    )
    return samples


def predict(parameters, points):
    (
        x0,
        y0,
        depth_halo,
        depth_plume,
        log_halo,
        log_plume,
        halo_xy,
        halo_z,
        length,
        width,
        plume_z,
        speed,
        angle,
        length_growth,
        width_growth,
        background,
    ) = parameters
    x, y, z, days = points.T
    dx = x - x0
    dy = y - y0
    along = dx * np.cos(angle) + dy * np.sin(angle) - speed * days
    across = -dx * np.sin(angle) + dy * np.cos(angle)
    halo_radius_sq = (dx**2 + dy**2) / halo_xy**2 + ((z - depth_halo) / halo_z) ** 2
    plume_radius_sq = (
        (along / (length + length_growth * days)) ** 2
        + (across / (width + width_growth * days)) ** 2
        + ((z - depth_plume) / plume_z) ** 2
    )
    halo = np.exp(log_halo - 0.5 * halo_radius_sq)
    plume = np.exp(log_plume - 0.5 * plume_radius_sq)
    return background + halo + plume


def predict_clean(parameters, points):
    lab_background = parameters[-1]
    return (predict(parameters, points) - lab_background + CLEAN_BACKGROUND).round(2)


def fit(samples, source, intake):
    x0, y0 = source
    angle = np.arctan2(intake[1] - y0, intake[0] - x0)
    start = [
        x0,
        y0,
        -30,
        -55,
        np.log(700),
        np.log(300),
        400,
        30,
        500,
        260,
        30,
        0.9,
        angle,
        0.1,
        0.04,
        0.15,
    ]
    lower = [
        x0 - 600,
        y0 - 600,
        -125,
        -125,
        0,
        0,
        60,
        5,
        70,
        40,
        5,
        0.15,
        angle - 1,
        0,
        0,
        0,
    ]
    upper = [
        x0 + 600,
        y0 + 600,
        -5,
        -5,
        10,
        10,
        1000,
        90,
        1400,
        950,
        90,
        3,
        angle + 1,
        0.7,
        0.5,
        1,
    ]

    points = samples[FEATURES].to_numpy()
    measured = samples["concentration"].to_numpy()
    target = np.log1p(measured)

    def residuals(parameters):
        # Multiplicative measurement errors become closer to additive on this scale.
        return np.log1p(predict(parameters, points)) - target

    result = least_squares(
        residuals,
        start,
        bounds=(lower, upper),
        x_scale="jac",
        loss="soft_l1",
        f_scale=0.3,
        max_nfev=500,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def rmse(actual, predicted):
    return np.sqrt(np.mean((np.asarray(actual) - predicted) ** 2))


def report(label, actual, predicted):
    actual = np.asarray(actual)
    squared = (actual - predicted) ** 2
    top = actual >= np.quantile(actual, 0.9)
    print(
        f"{label}: RMSE={np.sqrt(squared.mean()):.2f}, "
        f"MAE={np.abs(actual - predicted).mean():.2f}, "
        f"top 10% of concentrations = {squared[top].sum() / squared.sum():.1%} of SSE"
    )


def validate(samples, source, intake):
    block_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_rows, val_rows = next(block_split.split(samples, groups=samples["block"]))
    last_year = (pd.Timestamp("2025-01-01") - START).days
    splits = [
        ("spatial holdout", samples.iloc[train_rows], samples.iloc[val_rows]),
        (
            "time holdout",
            samples[samples["days"] < last_year],
            samples[samples["days"] >= last_year],
        ),
    ]
    for label, train, val in splits:
        fitted = fit(train, source, intake)
        values = predict(fitted, val[FEATURES].to_numpy())
        baseline = rmse(val["concentration"], train["concentration"].median())
        report(f"Noisy {label}", val["concentration"], values)
        print(f"  Training median baseline RMSE={baseline:.2f} ({len(val)} rows)")


def bootstrap_predictions(samples, points, source, intake, repeats):
    """Resample spatial blocks and refit to measure uncertainty of the fitted field."""
    rng = np.random.default_rng(42)
    blocks = list(samples.groupby("block", sort=True).indices.values())
    predictions = []
    for _ in range(repeats):
        choices = rng.integers(len(blocks), size=len(blocks))
        rows = np.concatenate([blocks[index] for index in choices])
        parameters = fit(samples.iloc[rows], source, intake)
        predictions.append(predict_clean(parameters, points))
    return np.quantile(predictions, [0.05, 0.95], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=Path("predictions_tce.csv"))
    parser.add_argument("--bootstrap", type=int, default=60)
    parser.add_argument(
        "--score",
        action="store_true",
        help="Evaluate the saved predictions on the old labels.",
    )
    args = parser.parse_args()
    if args.bootstrap < 20:
        parser.error("--bootstrap must be at least 20")

    samples = load_observations(args.data)
    sources = pd.read_csv(args.data / "sources.csv")
    source = sources.iloc[0][["x", "y"]].to_numpy(dtype=float)
    intake = (
        pd.read_csv(args.data / "abstraction.csv")
        .iloc[0][["x_center", "y_center"]]
        .to_numpy(dtype=float)
    )
    print(
        f"Cleaned measurements: {len(samples)} from {samples['well_id'].nunique()} wells"
    )
    validate(samples, source, intake)

    queries = add_days(pd.read_csv(args.data / "queries_test.csv"))
    points = queries[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("Invalid coordinates or dates in queries_test.csv")

    fitted = fit(samples, source, intake)
    center = predict_clean(fitted, points)
    low, high = bootstrap_predictions(samples, points, source, intake, args.bootstrap)
    submission = pd.DataFrame(
        {
            "query_id": queries["query_id"],
            "prediction": center,
            "lower_90": np.minimum(low, center),
            "upper_90": np.maximum(high, center),
        }
    )
    submission.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

    if args.score:
        truth = pd.read_csv(args.data / "_ground_truth_answers.csv")
        scored = submission.merge(truth, on="query_id", validate="one_to_one")
        if len(scored) != len(queries):
            raise ValueError("Some query IDs do not have ground truth")
        actual = scored["true_concentration"]
        report("Clean query labels", actual, scored["prediction"])
        print(
            f"  Clean label median={actual.median():.2f}, p95={actual.quantile(.95):.2f}, max={actual.max():.2f}"
        )
        print(
            f"  Training median baseline RMSE={rmse(actual, samples['concentration'].median()):.2f}"
        )
        coverage = actual.between(scored["lower_90"], scored["upper_90"]).mean()
        print(
            f"  Bootstrap coverage={coverage:.1%}, mean width={(scored['upper_90'] - scored['lower_90']).mean():.2f}"
        )


if __name__ == "__main__":
    main()
