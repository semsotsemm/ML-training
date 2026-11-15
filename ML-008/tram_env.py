"""Deterministic multi-tram simulator driven by the scenario CSV files."""

import csv
import random
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path


DATA = Path(__file__).resolve().parent / "data"
OBS_SIZE = 14
ACTION_COUNT = 5
ACCELERATION = (-2.2, -0.12, 1.35, -0.12, 1.35)
SAFE_GAP = 18.0
PRIORITY_RANGE = 70.0
PRIORITY_SECONDS = 9
PRIORITY_COOLDOWN = 60


def read_rows(name, data=DATA):
    with (Path(data) / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


@dataclass(frozen=True)
class Segment:
    track: str
    start: float
    end: float
    speed_limit: float


@dataclass(frozen=True)
class Route:
    segments: tuple[Segment, ...]
    stops: tuple[tuple[float, int], ...]
    signals: tuple[str, ...]

    @property
    def end(self):
        return self.segments[-1].end

    def segment_at(self, position):
        return next((part for part in self.segments if position < part.end), self.segments[-1])


@dataclass
class Signal:
    position: float
    cycle: int
    green_time: int
    offset: int
    queue: int = 0
    priority_until: int = 0
    next_priority: int = 0

    def phase(self, time):
        phase = (time + self.offset) % self.cycle
        normal_green = phase < self.green_time
        normal_change = self.green_time - phase if normal_green else self.cycle - phase
        forced = self.priority_until - time
        if forced <= 0:
            return normal_green, normal_change
        # Priority adds green time. An overlapping ordinary green can extend it further.
        if normal_green:
            return True, max(forced, normal_change)
        return True, forced if forced < normal_change else normal_change + self.green_time

    def accepts_priority(self, time, distance):
        return (
            distance <= PRIORITY_RANGE
            and not self.phase(time)[0]
            and time >= self.next_priority
            and self.queue <= 14
        )


@dataclass(frozen=True)
class Departure:
    tram_id: int
    route_id: str
    spawn_s: int
    planned_trip_s: int


@dataclass
class Tram:
    departure: Departure
    route: Route
    position: float = 0.0
    speed: float = 0.0
    dwell: int = 0
    next_stop: int = 0
    next_signal: int = 0


class TramEnv:
    """Use reset(seed), then step({tram_id: action}) once per simulated second."""

    def __init__(self, scenario_id, data=DATA):
        self.scenario = scenario_id
        scenarios = {row["scenario_id"]: row for row in read_rows("scenarios", data)}
        spec = scenarios[scenario_id]
        self.horizon = int(spec["horizon_s"])
        network = spec["network_id"]

        signal_rows = [row for row in read_rows("signals", data) if row["network_id"] == network]
        self._signal_specs = {
            row["signal_id"]: (
                float(row["position_m"]), int(row["cycle_s"]),
                int(row["tram_green_s"]), int(row["offset_s"]),
            )
            for row in sorted(signal_rows, key=lambda row: row["signal_id"])
        }
        stops = [row for row in read_rows("stops", data) if row["network_id"] == network]
        segments = [row for row in read_rows("segments", data) if row["network_id"] == network]
        self.routes = {}
        for route_id in sorted({row["route_id"] for row in segments}):
            ordered = sorted(
                (row for row in segments if row["route_id"] == route_id),
                key=lambda row: int(row["segment_order"]),
            )
            parts = tuple(
                Segment(row["track_id"], float(row["start_m"]), float(row["end_m"]),
                        float(row["speed_limit_mps"]))
                for row in ordered
            )
            track_ids = {part.track for part in parts}
            route_stops = tuple(sorted(
                (float(row["position_m"]), int(row["dwell_s"]))
                for row in stops if row["track_id"] in track_ids
            ))
            route_signals = tuple(
                row["signal_id"] for row in sorted(signal_rows, key=lambda row: float(row["position_m"]))
                if row["track_id"] in track_ids
            )
            self.routes[route_id] = Route(parts, route_stops, route_signals)

        self.departures = sorted(
            (
                Departure(int(row["tram_id"]), row["route_id"], int(row["spawn_s"]),
                          int(row["planned_trip_s"]))
                for row in read_rows("tram_spawns", data) if row["scenario_id"] == scenario_id
            ),
            key=lambda row: (row.spawn_s, row.tram_id),
        )
        self._traffic = {signal: [None] * self.horizon for signal in self._signal_specs}
        for row in read_rows("traffic", data):
            if row["scenario_id"] != scenario_id:
                continue
            first, last = int(row["start_s"]), int(row["end_s"])
            probability = float(row["cars_per_hour"]) / 3600
            if not (0 <= first < last <= self.horizon and 0 <= probability <= 1):
                raise ValueError(f"Invalid traffic window: {row}")
            schedule = self._traffic[row["signal_id"]]
            if any(value is not None for value in schedule[first:last]):
                raise ValueError(f"Overlapping traffic windows: {row}")
            schedule[first:last] = [probability] * (last - first)
        if any(None in schedule for schedule in self._traffic.values()):
            raise ValueError(f"Incomplete traffic schedule for {scenario_id}")
        self.done = True

    def reset(self, seed):
        self.time = 0
        self.done = False
        self._rng = random.Random(seed)
        self._pending = self.departures.copy()
        self.active: dict[int, Tram] = {}
        self.signals = {name: Signal(*spec) for name, spec in self._signal_specs.items()}
        self.trip_times = []
        self.priority_grants = 0
        self.red_interventions = 0
        self.gap_interventions = 0
        self.car_queue_total = 0
        self.episode_return = 0.0
        self._spawn()
        return self.observations()

    def _spawn(self):
        pending = []
        for departure in self._pending:
            if departure.spawn_s > self.time:
                pending.append(departure)
                continue
            route = self.routes[departure.route_id]
            track = route.segments[0].track
            blocked = any(
                tram.position < SAFE_GAP and tram.route.segment_at(tram.position).track == track
                for tram in self.active.values()
            )
            if blocked:
                pending.append(departure)
            else:
                self.active[departure.tram_id] = Tram(departure, route)
        self._pending = pending

    def _leader(self, tram):
        # A leader on a shared segment ahead can be seen before the merge.
        ahead = (
            other for other in self.active.values()
            if other is not tram
            and (other.position, -other.departure.tram_id) > (tram.position, -tram.departure.tram_id)
            and other.position < tram.route.end
            and other.route.segment_at(other.position).track == tram.route.segment_at(other.position).track
        )
        return min(ahead, key=lambda item: item.position, default=None)

    def _next_signal(self, tram):
        if tram.next_signal == len(tram.route.signals):
            return None
        return self.signals[tram.route.signals[tram.next_signal]]

    def _observe(self, tram):
        position = tram.position
        route = tram.route
        if tram.next_stop < len(route.stops):
            stop_distance = min(300.0, route.stops[tram.next_stop][0] - position)
        else:
            stop_distance = 300.0
        leader = self._leader(tram)
        if leader is not None and leader.position - position < 250:
            gap, leader_speed = leader.position - position, leader.speed
        else:
            gap, leader_speed = 250.0, 0.0

        signal = self._next_signal(tram)
        if signal is not None and signal.position - position < 300:
            signal_distance = signal.position - position
            green, phase_time = signal.phase(self.time)
            queue = signal.queue
            priority = signal.accepts_priority(self.time, signal_distance)
        else:
            signal_distance, green, phase_time, queue, priority = 300.0, True, 0, 0, False

        progress = position / route.end
        delay = self.time - tram.departure.spawn_s - progress * tram.departure.planned_trip_s
        return (
            tram.speed, route.segment_at(position).speed_limit, stop_distance, float(tram.dwell),
            gap, leader_speed, signal_distance, float(green), float(phase_time),
            float(queue), float(priority), delay, progress, route.end - position,
        )

    def observations(self):
        return {tram_id: self._observe(self.active[tram_id]) for tram_id in sorted(self.active)}

    def _update_cars(self):
        for name, signal in self.signals.items():
            # Draw once for every signal and second so the seed fixes arrivals across policies.
            signal.queue += int(self._rng.random() < self._traffic[name][self.time])
            if self.time % 2 == 0 and not signal.phase(self.time)[0] and signal.queue > 0:
                signal.queue -= 1
            self.car_queue_total += signal.queue

    def _advance(self, tram, action):
        if tram.dwell:
            tram.dwell -= 1
            return -0.06

        route = tram.route
        position = tram.position
        limit = route.segment_at(position).speed_limit
        speed = min(limit, max(0.0, tram.speed + ACCELERATION[action]))
        target = min(position + speed, route.end)
        reward = -0.06 - (0.015 if action in (2, 4) else 0.0)

        leader = self._leader(tram)
        if leader is not None:
            safe_target = max(position, leader.position - SAFE_GAP)
            if target > safe_target:
                target = safe_target
                self.gap_interventions += 1
                reward -= 0.6

        next_stop = route.stops[tram.next_stop] if tram.next_stop < len(route.stops) else None
        if next_stop is not None:
            target = min(target, next_stop[0])

        signal = self._next_signal(tram)
        if signal is not None and target >= signal.position:
            if signal.phase(self.time)[0]:
                tram.next_signal += 1
            else:
                target = max(position, signal.position - 1.0)
                self.red_interventions += 1
                reward -= 0.8

        progress = target - position
        tram.position = target
        tram.speed = progress
        if next_stop is not None and target == next_stop[0]:
            tram.next_stop += 1
            tram.dwell = next_stop[1]
            tram.speed = 0.0
            reward += 1.0 - 0.1 * max(0.0, speed - 3.0)
        return reward + 0.025 * progress

    def step(self, actions):
        if self.done:
            raise RuntimeError("Call reset(seed) before stepping a finished episode")
        if set(actions) != set(self.active):
            raise ValueError("Send exactly one action for each current tram ID")
        if any(isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < ACTION_COUNT
               for action in actions.values()):
            raise ValueError("Actions must be integers between 0 and 4")

        rewards = dict.fromkeys(actions, 0.0)
        for tram_id in sorted(actions):
            tram = self.active[tram_id]
            signal = self._next_signal(tram)
            if actions[tram_id] in (3, 4) and not tram.dwell and signal is not None:
                rewards[tram_id] -= 0.02
                if signal.accepts_priority(self.time, signal.position - tram.position):
                    signal.priority_until = self.time + PRIORITY_SECONDS
                    signal.next_priority = self.time + PRIORITY_COOLDOWN
                    self.priority_grants += 1
                    rewards[tram_id] -= 0.25 + 0.045 * signal.queue
        self._update_cars()

        terminated = set()
        front_first = sorted(self.active, key=lambda key: (-self.active[key].position, key))
        for tram_id in front_first:
            tram = self.active[tram_id]
            rewards[tram_id] += self._advance(tram, actions[tram_id])
            if tram.position >= tram.route.end:
                self.trip_times.append(self.time + 1 - tram.departure.spawn_s)
                rewards[tram_id] += 12.0
                terminated.add(tram_id)
                del self.active[tram_id]

        self.time += 1
        self.done = self.time >= self.horizon or not (self.active or self._pending)
        if self.done:
            for tram_id in self.active:
                terminated.add(tram_id)
                rewards[tram_id] -= 12.0
            self.active.clear()
        else:
            self._spawn()
        self.episode_return += sum(rewards.values())
        return self.observations(), rewards, terminated, self.done, self.metrics() if self.done else {}

    def metrics(self):
        completed = len(self.trip_times)
        return {
            "scheduled": len(self.departures),
            "completed": completed,
            "completion_rate": completed / len(self.departures),
            "mean_trip_s": sum(self.trip_times) / completed if completed else float(self.horizon),
            "mean_car_queue": self.car_queue_total / (max(1, self.time) * len(self.signals)),
            "priority_grants": self.priority_grants,
            "red_interventions": self.red_interventions,
            "gap_interventions": self.gap_interventions,
            "episode_return": self.episode_return,
        }
