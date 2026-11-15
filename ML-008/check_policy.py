"""Run a TorchScript policy on the selected scenario seeds and write metrics."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import torch

from tram_env import ACTION_COUNT, TramEnv, read_rows


def run_episode(env, model, seed):
    observations = env.reset(seed)
    known_trams = set()
    times_ms = []
    with torch.inference_mode():
        while True:
            actions = {}
            for tram_id, obs in observations.items():
                if tram_id not in known_trams:
                    model.reset(tram_id)
                    known_trams.add(tram_id)
                tensor = torch.tensor(obs, dtype=torch.float32)
                start = perf_counter()
                action = model.act(tram_id, tensor)
                times_ms.append((perf_counter() - start) * 1000)
                if type(action) is not int or not 0 <= action < ACTION_COUNT:
                    raise ValueError(f"act() must return an integer from 0 to {ACTION_COUNT - 1}")
                actions[tram_id] = action

            observations, _, _, done, metrics = env.step(actions)
            if done:
                metrics["mean_act_ms"] = sum(times_ms) / max(1, len(times_ms))
                return metrics


def score(metrics, rules):
    red_per_tram = metrics["red_interventions"] / metrics["scheduled"]
    gap_per_tram = metrics["gap_interventions"] / metrics["scheduled"]
    passed = (
        metrics["completion_rate"] >= float(rules["min_completion_rate"])
        and metrics["mean_trip_s"] <= float(rules["max_mean_trip_s"])
        and metrics["mean_car_queue"] <= float(rules["max_mean_car_queue"])
        and red_per_tram <= float(rules["max_red_per_tram"])
        and gap_per_tram <= float(rules["max_gap_per_tram"])
    )
    return {"red_per_tram": red_per_tram, "gap_per_tram": gap_per_tram, "passed": passed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("--output", type=Path, default=Path("metrics.csv"))
    parser.add_argument("--split", choices=["train", "public"], default="public")
    parser.add_argument("--scenario", help="Run just one scenario")
    args = parser.parse_args()

    torch.set_num_threads(1)
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    if not all(hasattr(policy, name) for name in ["reset", "act"]):
        parser.error("TorchScript policy must expose reset(tram_id) and act(tram_id, obs)")

    limits = {row["scenario_id"]: row for row in read_rows("thresholds")}
    seeds = [row for row in read_rows("seeds") if row["split"] == args.split
             and (args.scenario is None or args.scenario == row["scenario_id"])]
    if not seeds:
        parser.error("No matching scenario and split")

    rows = []
    for row in seeds:
        scenario = row["scenario_id"]
        seed = int(row["seed"])
        metrics = run_episode(TramEnv(scenario), policy, seed)
        result = score(metrics, limits[scenario])
        rows.append({"scenario_id": scenario, "seed": seed, "split": args.split, **metrics, **result})
        print(f"{scenario} seed={seed}: {'PASS' if result['passed'] else 'FAIL'} "
              f"| trips {metrics['completed']}/{metrics['scheduled']} "
              f"| mean trip {metrics['mean_trip_s']:.1f} s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
