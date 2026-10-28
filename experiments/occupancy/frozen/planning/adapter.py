"""Measured-catalog adapter for the physical deployment comparison.

The optimization and search implementations are frozen from the existing
current-model study. This file only supplies benchmark orchestration, static
provisioning baselines, and an explicit per-GPU replay plan.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
import model_adapter as _model_adapter
from guard_intervals import guarded_intervals

_model_adapter.direct_intervals = guarded_intervals
ModelAdapter = _model_adapter.ModelAdapter
import heuristics

LAYOUTS = ((4,), (2, 2), (2, 1, 1), (1, 1, 1, 1))


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(case, calibration, *, frequency=None):
    if calibration.get("status") == "pending" or calibration.get("missing_configurations"):
        raise ValueError("An incomplete calibration cannot define the comparative planning catalog")
    if calibration.get("expected_profiles") is not None:
        profiles = {p["catalog_id"] for p in calibration["profiles"]}
        exclusions = {p["id"] for p in calibration.get("exclusions", [])}
        if (profiles & exclusions or len(profiles) != len(calibration["profiles"])
                or len(profiles | exclusions) != calibration["expected_profiles"]):
            raise ValueError("Measured and explicitly excluded profiles must partition the declared catalog")
    document = deepcopy(calibration)
    if frequency is not None:
        document["profiles"] = [p for p in document["profiles"]
                                if p["freq_MHz"] == frequency]
    return ModelAdapter(case["services"], case["gpus"], document,
                        cap_watts=case["cap_W"])


def _groups_to_common(source_model, common_model, result):
    """Alternative integer IDs can change after a frequency restriction."""
    if not result.get("assignment_json"):
        return result
    lookup = {(o["s"], o["g"], o["physical_config_id"], o["alternative_id"]): o
              for o in common_model.options}
    groups = []
    for group in json.loads(result["assignment_json"]):
        key = (group["s"], group["g"], group["physical_config_id"], group["alternative_id"])
        groups.append(dict(lookup[key], n=group["n"], rate=group["rate"]))
    rebuilt = common_model.from_groups(groups)
    common_model.validate(rebuilt)
    return dict(result, **rebuilt)


def greedy_stages(model):
    """One execution supplies the construction-only and refined greedy plans."""
    refined = heuristics.greedy(model, include_trace=True)
    if not refined.get("assignment_json"):
        return dict(refined), refined
    final = refined["insertion_trace"][-1]
    groups = [dict(model.lookup[(sid, gpu, rid)], n=1, rate=model.ss[sid]["lam"])
              for sid, gpu, rid in final["configurations"]]
    construction = dict(status="feasible", **model.from_groups(groups),
                        algorithm="greedy_construction_without_refinement",
                        seconds=final["elapsed_seconds"])
    model.validate(construction)
    return construction, refined


def placement_score(policy, used, capacity):
    """Kubernetes v1.35 score for the one requested MIG extended resource.

    These are the same integer formulas as the archived MostAllocated and
    LeastAllocated experiments. Provisioning and power admission are external
    additions, not claims about the default Kubernetes scheduler.
    """
    requested = used + 1
    if capacity <= 0 or requested > capacity:
        return 0
    if policy == "MostAllocated":
        return 100 * requested // capacity
    if policy == "LeastAllocated":
        return 100 * (capacity - requested) // capacity
    raise ValueError(policy)


def static_trial(model, layout, frequency, policy, seed):
    """Existing static provision-then-pack pipeline, using calibrated options.

    Each service is provisioned independently with a uniform replica family.
    Candidate variants maximize utility / incremental replica power. Ties
    prefer fewer slices and lower power, as in the archived comparison.
    """
    inventory = Counter(layout)
    choices = []
    for sid, service in enumerate(model.ss):
        candidates = []
        for option in model.options:
            if option["s"] != sid or option["g"] != 0 or option["f"] != frequency:
                continue
            if option["k"] not in inventory:
                continue
            n = heuristics.minimum_replicas(service["lam"], option["upper"])
            rate = service["lam"] / n
            if n > min(model.R, model.G * inventory[option["k"]]):
                continue
            if not option["lower"] - 1e-9 <= rate <= option["upper"] + 1e-9:
                continue
            power = n * option["intercept"] + option["slope"] * service["lam"]
            if power <= 0:
                raise ValueError("Positive incremental power is required by the static provisioner")
            score = option["u"] * service["lam"] / power
            candidates.append(dict(option, n=n, rate=rate, provision_score=score,
                                   service_power=power))
        if not candidates:
            return dict(status="no_service_configuration", failed_service=sid)
        candidates.sort(key=lambda x: (-x["provision_score"], x["n"] * x["k"],
                                       x["service_power"], x["rid"]))
        choices.append((len(candidates), -service["lam"], sid, candidates[0]))
    choices.sort(key=lambda entry: entry[:3])
    rng = random.Random(seed)
    used = [Counter() for _ in range(model.G)]
    reserved = [model.active_base(frequency) for _ in range(model.G)]
    placed = []
    for _, _, _, choice in choices:
        for replica in range(choice["n"]):
            eligible = []
            for gpu in range(model.G):
                capacity = inventory[choice["k"]]
                if used[gpu][choice["k"]] >= capacity:
                    continue
                if reserved[gpu] + choice["peak"] > model.cap + 1e-8:
                    continue
                eligible.append((placement_score(policy, used[gpu][choice["k"]], capacity), gpu))
            if not eligible:
                return dict(status="placement_failed", failed_service=choice["s"],
                            placed_replicas=len(placed))
            score = max(x[0] for x in eligible)
            gpu = rng.choice([g for value, g in eligible if value == score])
            used[gpu][choice["k"]] += 1
            reserved[gpu] += choice["peak"]
            placed.append(dict(choice, g=gpu, n=1))
    merged = defaultdict(lambda: [0, 0.0])
    for group in placed:
        totals = merged[(group["s"], group["g"], group["rid"])]
        totals[0] += 1
        totals[1] += group["rate"]
    groups = [dict(model.lookup[key], n=n, rate=rate)
              for key, (n, rate) in sorted(merged.items())]
    result = model.from_groups(groups)
    model.validate(result)
    return dict(status="feasible", **result,
                static_layout=list(layout), static_frequency_MHz=frequency,
                placement_policy=policy, placement_seed=seed,
                external_power_cap_filter=True)


def static_baseline(model, policy, *, seed=0):
    """Best feasible static template, with a declared fixed placement seed."""
    start = perf_counter()
    trials = []
    best = None
    for frequency in sorted(model.base_by_frequency):
        for layout in LAYOUTS:
            result = static_trial(model, layout, frequency, policy, seed)
            trials.append(dict(layout=list(layout), frequency=frequency,
                               status=result["status"], ratio=result.get("ratio")))
            if result.get("assignment_json") and (best is None or result["ratio"] > best["ratio"]):
                best = result
    return dict(best or {"status": "construction_failed"}, seconds=perf_counter() - start,
                trials=trials, template_selection="best feasible utility/J among all static templates")


def deployment_replay(model, plan):
    """Serialize every logical GPU, with actual per-replica request rates.

    Physical GPU identity is not a planning variable for this homogeneous
    calibration. Canonicalization removes only whole-GPU permutations; it never
    merges replicas, removes empty-GPU energy, or changes their traffic rates.
    """
    if not plan.get("assignment_json"):
        return None
    model.validate(plan)
    original = json.loads(plan["assignment_json"])
    # Solver symmetry only needs frequency/slices. Replay deduplication also
    # orders equally occupied GPUs by their full replica contents.
    def gpu_key(gpu):
        local = [group for group in original if group["g"] == gpu]
        if not local:
            return (1, 0, ())
        contents = tuple(sorted((group["s"], group["physical_config_id"], group["n"],
                                 float(group["rate"]).hex()) for group in local))
        return (0, local[0]["f"], contents)
    mapping = {old:new for new,old in enumerate(sorted(range(model.G), key=gpu_key))}
    groups = [dict(group, g=mapping[group["g"]]) for group in original]
    entries = []
    for gpu in range(model.G):
        local = sorted((g for g in groups if g["g"] == gpu),
                       key=lambda x: (x["s"], x["physical_config_id"], x["rate"]))
        if not local:
            entries.append(dict(logical_gpu=gpu, mode="idle_unlocked", replicas=[],
                                predicted_power_W=model.idle))
            continue
        frequency = local[0]["f"]
        replicas = []
        for group in local:
            profile = model.profiles[group["physical_config_id"]]
            for replica in range(group["n"]):
                replicas.append(dict(service=group["s"], variant_id=group["model"],
                    physical_config_id=group["physical_config_id"], catalog_id=group["catalog_id"],
                    arch=group["arch"], batch_size=group["B"], slices=group["k"],
                    frequency_MHz=group["f"], rate_requests_s=group["rate"] / group["n"],
                    feasible_interval=[group["lower"], group["upper"]],
                    input_spec=profile.get("input_spec"), quality=group["quality"],
                    utility_per_request=group["u"]))
        entries.append(dict(logical_gpu=gpu, mode="active", frequency_MHz=frequency,
            power_cap_W=model.cap, replicas=replicas,
            provisioned_mig_layout=plan.get("static_layout"),
            predicted_power_W=model.active_base(frequency) + math.fsum(
                group["n"] * group["intercept"] + group["slope"] * group["rate"] for group in local),
            predicted_reserved_power_W=model.active_base(frequency) + math.fsum(
                group["n"] * group["peak"] for group in local)))
    payload = dict(gpus=entries, services=model.ss, utility_rate=plan["utility"],
                   predicted_power_W=plan["power"], predicted_utility_per_J=plan["ratio"],
                   static_layout=plan.get("static_layout"),
                   placement_policy=plan.get("placement_policy"))
    assert math.isclose(sum(g["predicted_power_W"] for g in entries), plan["power"], abs_tol=1e-6)
    physical_identity = {key:payload[key] for key in ("gpus", "services", "utility_rate")}
    payload["deployment_sha256"] = hashlib.sha256(json.dumps(physical_identity, sort_keys=True).encode()).hexdigest()
    return payload


def compare_case(case, calibration, *, exact_seconds=30.0, beam=True, beam_seconds=10., progress=None,
                 checkpoint=None):
    """Execute the common-options comparison without selecting cases by outcome."""
    start = perf_counter()
    model = load_model(case, calibration)
    result = dict(case=case, option_count=len(model.options), results={},
                  option_build_seconds=perf_counter()-start)
    construction, greedy = greedy_stages(model)
    result["results"].update(greedy_construction=construction, greedy=greedy)
    def save():
        if checkpoint is not None:
            checkpoint(dict(result, total_seconds=perf_counter()-start))
    save()
    result["results"]["exact"] = model.solve(time_limit=exact_seconds)
    save()
    fixed = []
    for frequency in sorted(model.base_by_frequency):
        restricted = load_model(case, calibration, frequency=frequency)
        plan = restricted.solve(time_limit=exact_seconds)
        plan = _groups_to_common(restricted, model, plan)
        plan["fixed_frequency_MHz"] = frequency
        fixed.append(plan)
    candidates = [x for x in fixed if x.get("assignment_json")]
    result["fixed_frequency_trials"] = fixed
    result["results"]["fixed_frequency"] = (max(candidates, key=lambda x: x["ratio"])
                                                if candidates else {"status": "no_feasible_fixed_frequency"})
    result["results"]["fixed_frequency"] = dict(result["results"]["fixed_frequency"],
        selected_solve_seconds=result["results"]["fixed_frequency"].get("seconds"),
        seconds=sum(x.get("seconds",0.) for x in fixed),
        all_frequencies_resolved=all(x["status"] in ("optimal", "infeasible_in_calibrated_domain",
            "individual_infeasible_in_calibrated_domain") for x in fixed))
    save()
    # Frequency is optimized after fixing all discrete deployment choices from
    # each fixed-frequency seed. Taking the best seed matches the old protocol.
    sequential = []
    for seed in candidates:
        groups = json.loads(seed["assignment_json"])
        plan = model.solve(time_limit=exact_seconds, seed_plan=seed, frozen_groups=groups)
        plan["seed_frequency_MHz"] = seed["fixed_frequency_MHz"]
        sequential.append(plan)
    result["sequential_trials"] = sequential
    result["results"]["sequential"] = (max(sequential, key=lambda x: x.get("ratio", -1))
                                           if sequential else {"status": "no_feasible_sequential_seed"})
    result["results"]["sequential"] = dict(result["results"]["sequential"],
        selected_refinement_seconds=result["results"]["sequential"].get("seconds"),
        seconds=sum(x.get("seconds",0.) for x in fixed+sequential),
        all_refinements_resolved=all(x["status"] in ("optimal", "infeasible_in_calibrated_domain",
            "individual_infeasible_in_calibrated_domain") for x in sequential))
    save()
    for policy in ("MostAllocated", "LeastAllocated"):
        result["results"][policy] = static_baseline(model, policy, seed=case.get("placement_seed", 0))
    save()
    if beam:
        beam_plan = heuristics.beam(model, None, progress=progress, time_limit=beam_seconds)
        candidates = [x for x in (greedy, beam_plan) if x.get("assignment_json")]
        full = max(candidates, key=lambda x: x["ratio"]) if candidates else {"status": "construction_failed"}
        full = dict(full, seconds=greedy["seconds"] + beam_plan["seconds"],
                    selected_stage="greedy" if full is greedy else "beam",
                    algorithm="greedy_and_beam")
        result["results"].update(beam=beam_plan, full=full)
    result["replay_plans"] = {key: deployment_replay(model, plan)
                              for key, plan in result["results"].items()}
    result["total_seconds"] = perf_counter() - start
    return result
