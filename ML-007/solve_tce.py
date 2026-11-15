"""Train XGBoost and produce the TCE exercise submission from cleaned data."""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
START = pd.Timestamp("2020-01-01")
MASS_DATE = pd.Timestamp("2026-01-01")
CELL_SIZE = 50  # metres, from the raster description
THRESHOLD = 5.0  # µg/L
LEVEL = 0.90


def month_start(dates):
    return dates.dt.to_period("M").dt.to_timestamp()


def make_features(points, raster, zones, sources, recharge, abstraction):
    """Use the same x, y, z, t inputs for training and every kind of prediction."""
    features = points[["x", "y", "z"]].astype(float).reset_index(drop=True)
    date = points["t"].reset_index(drop=True)
    if date.isna().any() or not np.isfinite(features).all().all():
        raise ValueError("Every point needs finite x, y, z and a valid date")

    angle = 2 * np.pi * (date.dt.month - 1) / 12
    features["month_sin"] = np.sin(angle)
    features["month_cos"] = np.cos(angle)
    features["days_since_start"] = (date - START).dt.days

    row = np.floor(features["y"] / CELL_SIZE).to_numpy(dtype=int)
    col = np.floor(features["x"] / CELL_SIZE).to_numpy(dtype=int)
    height, width = raster.shape
    if not ((0 <= row) & (row < height) & (0 <= col) & (col < width)).all():
        raise ValueError("Point outside the permeability raster")
    values = raster[row, col]
    features["permeability"] = np.where(values == -9999, np.nan, values)

    zone_id = pd.Series(pd.NA, index=features.index, dtype="string")
    for zone in zones.itertuples(index=False):
        inside = (
            features["x"].between(zone.x_min, zone.x_max, inclusive="left")
            & features["y"].between(zone.y_min, zone.y_max, inclusive="left")
        )
        zone_id.loc[inside] = zone.zone
    month = month_start(date)
    schedule_keys = pd.MultiIndex.from_arrays([zone_id, month], names=["zone", "t"])
    features["recharge_mm"] = recharge.reindex(schedule_keys).to_numpy(dtype=float)
    features["abstraction_m3_day"] = abstraction.reindex(month).to_numpy(dtype=float)
    if features[["recharge_mm", "abstraction_m3_day"]].isna().any().any():
        raise ValueError("Missing recharge or abstraction schedule for a point")

    for source in sources.itertuples(index=False):
        dx = features["x"] - source.x
        dy = features["y"] - source.y
        dz = features["z"] - (source.z_top + source.z_bottom) / 2
        features[f"distance_{source.source_id}"] = np.sqrt(dx**2 + dy**2 + dz**2)
        active = date.ge(source.start_date)
        if pd.notna(source.end_date):
            active &= date.le(source.end_date)
        features[f"release_{source.source_id}"] = np.where(
            active, source.estimated_release_rate_g_day, 0.0
        )

    return features


def fit_model(features, target):
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=4,
        subsample=0.85,
        tree_method="hist",
        random_state=42,
        n_jobs=2,
    )
    # Keep dense clusters from dominating the fit. Counts use training rows only.
    blocks = (features[["x", "y"]] // 400).astype(int)
    count = blocks.groupby(["x", "y"])["x"].transform("size")
    weight = 1 / np.sqrt(count)
    model.fit(features, target, sample_weight=weight / weight.mean())
    return model


def predict(model, features):
    return np.maximum(model.predict(features).astype(float), 0.0)


def calibrate_interval(model, features, target):
    prediction = predict(model, features)
    # Larger predicted concentrations can have larger absolute errors.
    scores = np.abs(target.to_numpy(dtype=float) - prediction) / np.sqrt(1 + prediction)
    rank = min(len(scores), int(np.ceil((len(scores) + 1) * LEVEL)))
    factor = np.partition(scores, rank - 1)[rank - 1]
    # The model file carries the calibration factor as well as the trees.
    model.get_booster().set_attr(interval_factor=str(float(factor)))


def predict_interval(model, features):
    center = predict(model, features)
    factor = float(model.get_booster().attr("interval_factor"))
    radius = factor * np.sqrt(1 + center)
    return center, np.maximum(center - radius, 0.0), center + radius


def train_and_check(features, target, date):
    before_2024 = date.lt("2024-01-01")
    year_2024 = date.dt.year.eq(2024)
    year_2025 = date.dt.year.eq(2025)
    if not (before_2024.any() and year_2024.any() and year_2025.any()):
        raise ValueError("Training requires measurements before 2024 and in 2024 and 2025")

    check_model = fit_model(features.loc[before_2024], target.loc[before_2024])
    calibrate_interval(check_model, features.loc[year_2024], target.loc[year_2024])
    pred, low, high = predict_interval(check_model, features.loc[year_2025])
    actual = target.loc[year_2025].to_numpy(dtype=float)
    print(f"2025 holdout RMSE: {np.sqrt(np.mean((actual - pred)**2)):.3f} µg/L")
    print(f"2025 holdout interval coverage: {np.mean((low <= actual) & (actual <= high)):.1%}")

    model = fit_model(features.loc[~year_2025], target.loc[~year_2025])
    calibrate_interval(model, features.loc[year_2025], target.loc[year_2025])
    return model


def estimate_mass(model, build_features, raster, layers):
    rows, cols = np.indices(raster.shape)
    x = (cols.ravel() + 0.5) * CELL_SIZE
    y = (rows.ravel() + 0.5) * CELL_SIZE

    mass_mg = 0.0
    for layer in layers.itertuples(index=False):
        depth = (layer.z_top + layer.z_bottom) / 2
        points = pd.DataFrame({"x": x, "y": y, "z": depth, "t": MASS_DATE})
        concentration = predict(model, build_features(points))
        water_m3 = CELL_SIZE**2 * (layer.z_top - layer.z_bottom) * layer.porosity
        # 1 µg/L equals 1 mg/m³.
        mass_mg += concentration.sum() * water_m3
    return float(mass_mg / 1_000_000)


def forecast_intake(model, build_features, wells, pumping, end_date):
    intake = wells.loc[wells["role"].eq("intake"), ["well_id", "x", "y", "z"]]
    days = pd.DataFrame({"t": pd.date_range(MASS_DATE, end_date, freq="D")})
    requests = days.merge(intake, how="cross")
    requests["month"] = month_start(requests["t"])
    rates = pumping[["well_id", "t", "pumping_rate"]].rename(columns={"t": "month"})
    requests = requests.merge(rates, on=["well_id", "month"], how="left", validate="many_to_one")
    if intake.empty or requests["pumping_rate"].isna().any():
        raise ValueError("Missing intake wells or monthly pumping schedule")

    requests["mass_mg_day"] = predict(model, build_features(requests)) * requests["pumping_rate"]
    daily = requests.groupby("t", as_index=False)[["mass_mg_day", "pumping_rate"]].sum()
    daily["prediction"] = daily["mass_mg_day"] / daily["pumping_rate"].where(daily["pumping_rate"] > 0)
    return daily[["t", "pumping_rate", "prediction"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission")
    args = parser.parse_args()
    data = ROOT / "data" / "iterim"

    def read(name):
        return pd.read_parquet(data / f"{name}.parquet")

    wells = read("wells")
    train = read("concentration").merge(
        wells[["well_id", "x", "y", "z"]], on="well_id", how="left", validate="many_to_one"
    )
    queries = read("queries_test")
    layers = read("cores")[["z_top", "z_bottom", "porosity"]].drop_duplicates()
    raster = tifffile.imread(ROOT / "data" / "raw" / "permeability.tif")
    build_features = partial(
        make_features,
        raster=raster,
        zones=read("zones"),
        sources=read("sources").sort_values("source_id"),
        recharge=read("recharge").set_index(["zone", "t"])["recharge"],
        abstraction=read("abstraction").set_index("t")["total_abstraction_m3_day"],
    )
    model = train_and_check(build_features(train), train["concentration"], train["t"])
    prediction, low, high = predict_interval(model, build_features(queries))
    predictions = pd.DataFrame({
        "query_id": queries["query_id"],
        "prediction": prediction,
        "lower_90": low,
        "upper_90": high,
    })

    mass_kg = estimate_mass(model, build_features, raster, layers)
    intake = forecast_intake(model, build_features, wells, read("pumping"), queries["t"].max())
    crossings = intake.loc[intake["prediction"] > THRESHOLD, "t"]
    arrival = crossings.min()
    summary = {
        "mass_kg": mass_kg,
        "arrival_date": arrival.date().isoformat() if pd.notna(arrival) else None,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_model(args.output / "model.json")
    predictions.to_csv(args.output / "predictions.csv", index=False)
    intake.to_csv(args.output / "intake_forecast.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Mass on {MASS_DATE.date()}: {mass_kg:.2f} kg")
    if pd.notna(arrival):
        print(f"Threshold exceeded on {arrival.date()} ({(arrival - MASS_DATE).days} days)")
    else:
        print(f"Threshold not exceeded through {intake['t'].max().date()}")
    print(f"Saved {len(predictions)} forecasts and model to {args.output.resolve()}")


if __name__ == "__main__":
    main()
