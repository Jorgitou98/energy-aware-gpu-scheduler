"""Portable planning interface for new experiments and deployment validation."""

from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from threading import RLock
from time import perf_counter


ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC = ROOT / "scientific" / "planning"
DEFAULT_GRID = ROOT / "data" / "inputs" / "admission_grid.csv"
_IMPORT_LOCK = RLock()


@contextmanager
def _alias(name, module):
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCIENTIFIC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _runtime(admission, grid):
    # A notebook can replace a custom grid in place between experiments. The
    # path alone must not reuse quantiles cached from its previous contents.
    digest = hashlib.sha256(Path(grid).read_bytes()).hexdigest() if admission != "nominal" else None
    return _cached_runtime(admission, grid, digest)


@lru_cache(maxsize=8)
def _cached_runtime(admission, grid, grid_digest):
    fingerprint = hashlib.sha256(f"{admission}|{grid}|{grid_digest}".encode()).hexdigest()[:16]
    prefix = f"efficient_mig._frozen_{fingerprint}"
    with _IMPORT_LOCK:
        rates = _load(prefix + "_rates", "rate_intervals.py")
        with _alias("rate_intervals", rates):
            model = _load(prefix + "_model", "model_adapter.py")
            if admission != "nominal":
                field = f"{admission.replace('p', 'q')}_1024"
                previous = os.environ.get("ADMISSION_QUANTILE_FIELD")
                os.environ["ADMISSION_QUANTILE_FIELD"] = field
                try:
                    guard = _load(prefix + "_guard", "guard_intervals.py")
                finally:
                    if previous is None:
                        os.environ.pop("ADMISSION_QUANTILE_FIELD", None)
                    else:
                        os.environ["ADMISSION_QUANTILE_FIELD"] = previous
                guard.GRID = Path(grid)
                model.direct_intervals = guard.guarded_intervals
        heuristics = _load(prefix + "_heuristics", "heuristics.py")
    return model, heuristics


@lru_cache(maxsize=8)
def _bounded_runtime(algorithms):
    with _IMPORT_LOCK, _alias("heuristics", algorithms):
        return _load(algorithms.__name__ + "_bounded", "heuristics_bounded.py")


def build_model(case, catalog, *, admission="p95", grid=None, frequency=None):
    """Compile one unchanged workload against the measured catalog."""
    if admission not in ("nominal", "p95", "p99"):
        raise ValueError("Admission must be nominal, p95, or p99")
    if catalog.get("status") == "pending" or catalog.get("missing_configurations"):
        raise ValueError("An incomplete catalog cannot define a planning model")
    if catalog.get("expected_profiles") is not None:
        profiles = {p["catalog_id"] for p in catalog["profiles"]}
        exclusions = {p["id"] for p in catalog.get("exclusions", [])}
        if (profiles & exclusions or len(profiles) != len(catalog["profiles"])
                or len(profiles | exclusions) != catalog["expected_profiles"]):
            raise ValueError("Profiles and exclusions must partition the declared catalog")
    grid_path = str(Path(grid or DEFAULT_GRID).resolve())
    if admission != "nominal" and not Path(grid_path).is_file():
        raise FileNotFoundError(grid_path)
    module, _ = _runtime(admission, grid_path)
    calibration = deepcopy(catalog)
    if frequency is not None:
        calibration["profiles"] = [p for p in calibration["profiles"]
                                   if p["freq_MHz"] == frequency]
    return module.ModelAdapter(case["services"], case["gpus"], calibration,
                               cap_watts=case["cap_W"])


def validate_plan(case, catalog, plan, *, admission="p95", grid=None):
    """Validate a saved deployment without invoking an optimizer."""
    model = build_model(case, catalog, admission=admission, grid=grid)
    model.validate(plan)
    return {key: plan[key] for key in ("power", "utility", "ratio", "replicas", "active")}


def _greedy_stages(model, algorithms):
    refined = algorithms.greedy(model, include_trace=True)
    if not refined.get("assignment_json"):
        return dict(refined), refined
    last = refined["insertion_trace"][-1]
    groups = [dict(model.lookup[(sid, gpu, rid)], n=1, rate=model.ss[sid]["lam"])
              for sid, gpu, rid in last["configurations"]]
    construction = dict(status="feasible", **model.from_groups(groups),
                        algorithm="greedy_construction_without_refinement",
                        seconds=last["elapsed_seconds"])
    model.validate(construction)
    return construction, refined


def _heuristic_stages(model, algorithms, beam_seconds, *, bounded=False):
    construction, greedy = _greedy_stages(model, algorithms)
    beam_algorithm = _bounded_runtime(algorithms) if bounded else algorithms
    beam = beam_algorithm.beam(model, greedy, width=512, per_signature=2,
                               time_limit=beam_seconds)
    candidates = [(name, result) for name, result in (("greedy", greedy), ("beam", beam))
                  if result.get("assignment_json")]
    if candidates:
        selected, result = max(candidates, key=lambda entry: entry[1]["ratio"])
        full = dict(result, algorithm="greedy_and_beam_bounded" if bounded else "greedy_and_beam",
                    selected_stage=selected,
                    seconds=greedy["seconds"] + beam["seconds"])
    else:
        full = dict(status="construction_failed", selected_stage=None,
                    seconds=greedy["seconds"] + beam["seconds"])
        if bounded:
            full["algorithm"] = "greedy_and_beam_bounded"
    return dict(greedy_construction=construction, greedy=greedy, beam=beam, full=full)


def _common_plan(common_model, result):
    if not result.get("assignment_json"):
        return result
    lookup = {(o["s"], o["g"], o["physical_config_id"], o["alternative_id"]): o
              for o in common_model.options}
    groups = []
    for group in json.loads(result["assignment_json"]):
        key = (group["s"], group["g"], group["physical_config_id"], group["alternative_id"])
        groups.append(dict(lookup[key], n=group["n"], rate=group["rate"]))
    result = dict(result, **common_model.from_groups(groups))
    common_model.validate(result)
    return result


def plan_case(case, catalog, *, method="greedy-beam", admission="p95", grid=None,
              beam_seconds=None, exact_seconds=None, warm_start=False, random_seed=None):
    """Run one explicitly selected method; returned timings are new measurements."""
    if method not in ("prepare", "greedy-c", "greedy", "greedy-beam", "greedy-beam-bounded", "exact", "fixed-frequency", "sequential", "least-allocated", "most-allocated"):
        raise ValueError("Unknown planning method")
    if method in ("least-allocated", "most-allocated"):
        if isinstance(random_seed, bool) or not isinstance(random_seed, int):
            raise ValueError("Occupancy policies require an explicit integer random_seed")
    elif random_seed is not None:
        raise ValueError("random_seed applies only to occupancy policies")
    if beam_seconds is not None and (not math.isfinite(beam_seconds) or beam_seconds <= 0):
        raise ValueError("Beam seconds must be positive and finite, or None")
    if exact_seconds is not None and (not math.isfinite(exact_seconds) or exact_seconds <= 0):
        raise ValueError("Exact seconds must be positive and finite, or None")
    started = perf_counter()
    model = build_model(case, catalog, admission=admission, grid=grid)
    prepared = perf_counter() - started
    _, algorithms = _runtime(admission, str(Path(grid or DEFAULT_GRID).resolve()))
    record = dict(case=case["case"], method=method, admission=admission,
                  option_count=len(model.options), preparation_seconds=prepared,
                  beam_seconds=beam_seconds, beam_width=512, per_signature=2,
                  exact_seconds=exact_seconds, solver_threads=1,
                  timing_scope="new local invocation; not an archived paper benchmark")
    results = {}
    if method in ("least-allocated", "most-allocated"):
        occupancy = _load("efficient_mig._occupancy", "occupancy.py")
        policy = "LeastAllocated" if method == "least-allocated" else "MostAllocated"
        record["random_seed"] = random_seed
        results[method] = occupancy.plan(model, policy, seed=random_seed, include_trace=True)
    if method == "greedy-c":
        occupancy = _load("efficient_mig._occupancy", "occupancy.py")
        results["greedy_construction"] = occupancy.plan(model, "GreedyScore", include_trace=True)
    if method == "greedy":
        construction, greedy = _greedy_stages(model, algorithms)
        results = dict(greedy_construction=construction, greedy=greedy)
    elif method in ("greedy-beam", "greedy-beam-bounded") or (method == "exact" and warm_start):
        results = _heuristic_stages(model, algorithms, beam_seconds,
                                    bounded=method == "greedy-beam-bounded")
    if method == "exact":
        results["exact"] = model.solve(time_limit=exact_seconds, threads=1,
                                        seed_plan=results.get("full"))
        record["warm_start"] = warm_start
    if method in ("fixed-frequency", "sequential"):
        trials = []
        # A valid custom catalog can measure baselines at more frequencies
        # than its inference profiles. An empty profile subset cannot define
        # a restricted model and must not prevent trials of measured choices.
        frequencies = sorted({profile["freq_MHz"] for profile in model.profiles.values()})
        for frequency in frequencies:
            restricted = build_model(case, catalog, admission=admission, grid=grid,
                                     frequency=frequency)
            seed = _heuristic_stages(restricted, algorithms, beam_seconds)["full"]
            exact = restricted.solve(time_limit=exact_seconds, threads=1, seed_plan=seed)
            trials.append(dict(_common_plan(model, exact), frequency_MHz=frequency,
                               initializer_seconds=seed["seconds"]))
        feasible = [r for r in trials if r.get("assignment_json")]
        results["fixed_frequency_trials"] = trials
        results["fixed_frequency"] = (max(feasible, key=lambda r: r["ratio"])
                                        if feasible else {"status": "no_feasible_fixed_frequency"})
        if method == "sequential":
            refinements = []
            for seed in feasible:
                exact = model.solve(time_limit=exact_seconds, threads=1, seed_plan=seed,
                                    frozen_groups=json.loads(seed["assignment_json"]))
                refinements.append(dict(exact, seed_frequency_MHz=seed["frequency_MHz"]))
            results["sequential_trials"] = refinements
            feasible_refinements = [r for r in refinements if r.get("assignment_json")]
            results["sequential"] = (max(feasible_refinements, key=lambda r: r["ratio"])
                                        if feasible_refinements else {"status": "no_feasible_sequential_seed"})
    record.update(results=results, total_seconds=perf_counter() - started)
    return record
