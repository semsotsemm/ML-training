"""Basic space-time interpolation with interval and summary submissions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
ORIGIN = pd.Timestamp("2020-01-01")
FEATURES = ["x", "y", "z", "days"]
SCALE = np.array([280, 200, 17, 240])


def dates(values):
    iso = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    european = pd.to_datetime(values, format="%d.%m.%Y", errors="coerce")
    parsed = iso.fillna(european)
    if parsed.isna().any():
        raise ValueError("Expected dates in YYYY-MM-DD or DD.MM.YYYY format")
    return parsed


def load_observations(root, wells):
    table = pd.read_csv(root / "concentration.csv").drop_duplicates()
    table = table[table["quality"].isin(["ok", "replicate"])].copy()
    table["t"] = dates(table["t"])
    value = pd.to_numeric(table["concentration"], errors="coerce")
    censored = table["concentration"].str.startswith("<") | table["concentration"].eq(
        "ND"
    )
    table["value"] = value.mask(censored, table["lod_ug_l"] / 2)
    table = table.dropna(subset=["value"])
    table = table.groupby(["well_id", "t"], as_index=False)["value"].median()
    table = table.merge(wells, on="well_id", validate="many_to_one")
    return table.assign(days=(table["t"] - ORIGIN).dt.days)


def make_predictor(observations):
    tree = cKDTree(observations[FEATURES].to_numpy() / SCALE)
    labels = observations["value"].to_numpy()
    k = min(12, len(labels))

    def predict(points):
        distance, index = tree.query(points / SCALE, k=list(range(1, k + 1)))
        weight = 1 / (0.25 + distance) ** 2
        return np.average(labels[index], weights=weight, axis=1)

    return predict


def interval_scale(predictions):
    return (1 + predictions) ** 0.65


def calibrate(table):
    # Hold out full spatial cells and the latest year for a calibration check.
    block = (table["x"] // 250 + 3 * (table["y"] // 250)).mod(5).eq(0)
    holdout = block | table["t"].ge("2025-01-01")
    predict = make_predictor(table[~holdout])
    cal = table[holdout]
    pred = predict(cal[FEATURES].to_numpy())
    normalized_error = np.abs(cal["value"].to_numpy() - pred) / interval_scale(pred)
    return np.quantile(normalized_error, 0.90, method="higher")


def estimate_mass(predict, domain):
    dx = domain["cell_size_m"]
    x = domain["x_min"] + (np.arange(domain["nx"]) + 0.5) * dx
    y = domain["y_min"] + (np.arange(domain["ny"]) + 0.5) * dx
    xx, yy = np.meshgrid(x, y)
    mass_day = (pd.Timestamp(domain["mass_date"]) - ORIGIN).days
    mass_mg = 0.0
    for layer in domain["layers"]:
        center = (layer["z_top"] + layer["z_bottom"]) / 2
        points = np.column_stack(
            (
                xx.ravel(),
                yy.ravel(),
                np.full(xx.size, center),
                np.full(xx.size, mass_day),
            )
        )
        pore_m3 = (
            dx**2 * (layer["z_top"] - layer["z_bottom"]) * layer["effective_porosity"]
        )
        # 1 µg/L = 1 mg/m³.
        mass_mg += predict(points).sum() * pore_m3
    return float(mass_mg / 1e6)


def estimate_arrival(predict, domain, wells, root):
    intake = wells[wells["role"].eq("intake")]
    daily = pd.date_range(domain["mass_date"], domain["forecast_end"], freq="D")
    pumping = pd.read_csv(root / "pumping.csv")
    pumping["month"] = dates(pumping["t"])
    pumping["rate_m3_day"] = pumping["pumping_rate"] * pumping["units"].map(
        {"m3/day": 1, "l/s": 86.4}
    )
    monthly = pumping.pivot(index="month", columns="well_id", values="rate_m3_day")[
        intake["well_id"]
    ]
    rates = monthly.reindex(daily.to_period("M").to_timestamp()).to_numpy()
    if (
        not np.isfinite(rates).all()
        or (rates < 0).any()
        or (rates.sum(axis=1) <= 0).any()
    ):
        raise ValueError("Missing or invalid forecast pumping schedule")

    xyz = intake[["x", "y", "z"]].to_numpy()
    points = np.column_stack(
        (np.tile(xyz, (len(daily), 1)), np.repeat((daily - ORIGIN).days, len(intake)))
    )
    concentrations = predict(points).reshape(rates.shape)
    mixed = np.average(concentrations, axis=1, weights=rates)
    crossings = daily[mixed > domain["intake_threshold_ug_l"]]
    return crossings[0].date().isoformat() if len(crossings) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("submission"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    domain = json.loads((ROOT / "domain.json").read_text(encoding="utf-8"))
    wells = pd.read_csv(ROOT / "wells.csv")
    table = load_observations(ROOT, wells)
    quantile = calibrate(table)
    predict = make_predictor(table)

    queries = pd.read_csv(ROOT / "queries_test.csv")
    queries["days"] = (dates(queries["t"]) - ORIGIN).dt.days
    values = predict(queries[FEATURES].to_numpy())
    radius = quantile * interval_scale(values)
    results = pd.DataFrame(
        {
            "query_id": queries["query_id"],
            "prediction": values,
            "lower_90": np.maximum(0, values - radius),
            "upper_90": values + radius,
        }
    )
    results.to_csv(args.output / "predictions.csv", index=False)
    summary = {
        "mass_kg": estimate_mass(predict, domain),
        "arrival_date": estimate_arrival(predict, domain, wells, ROOT),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
