"""Shared calibrated alternatives for exact and heuristic deployment planning.

Only measured ``a30_current_model_v1`` data are accepted. The homogeneous fleet
may use one explicitly selected calibrated GPU or an explicitly declared pooled
calibration with cross-device baseline/anchor provenance. It does not silently
pool observations from different devices or extrapolate load-response curves.

The power model is P_idle for unused GPUs; active GPU baseline P_base(f), plus
sum(n*e + alpha*rate). The power-cap reservation uses n*(e+alpha*mu), never the
average traffic-dependent power. Each disconnected feasible-rate interval is a
separate alternative of the same physical configuration and variant.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

from rate_intervals import direct_intervals, response_values


class CalibrationError(ValueError):
    """Missing, unmeasured, inconsistent, or unsupported calibration data."""


def safe_ceil_ratio(numerator, denominator):
    """Ceil of the exact ratio of stored floats, without a positive epsilon.

    Integer arithmetic avoids rounding a quotient just above an integer down to
    that integer. A conservative extra replica at a floating boundary is safe;
    rounding the bound down could discard a necessary replica.
    """
    if not math.isfinite(numerator) or not math.isfinite(denominator) or numerator <= 0 or denominator <= 0:
        raise ValueError("Positive finite demand and capacity are required")
    a, b = float(numerator).as_integer_ratio()
    c, d = float(denominator).as_integer_ratio()
    return (a * d + b * c - 1) // (b * c)


def finite(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CalibrationError(f"{label} must be a measured finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise CalibrationError(f"{label} has an invalid sign")
    return float(value)


def normalized_profile(raw, utility_reference):
    """Validate a measured physical profile and prepare the root compiler view."""
    if "calibration" in raw:
        raw = dict(raw, **raw["calibration"])
    if raw.get("measurement_status") != "measured":
        raise CalibrationError(f"Profile {raw.get('physical_config_id')} is not measured")
    required = ("physical_config_id", "catalog_id", "variant_id", "gpu_uuid", "task", "arch", "B", "slices", "freq_MHz")
    if any(raw.get(key) is None for key in required):
        raise CalibrationError("A measured profile lacks physical configuration metadata")
    for key in ("B", "slices", "freq_MHz"):
        if isinstance(raw[key], bool) or not isinstance(raw[key], int) or raw[key] <= 0:
            raise CalibrationError(f"{key} must be a positive integer")
    if raw["slices"] not in (1, 2, 4):
        raise CalibrationError("This A30 adapter supports one-, two-, and four-slice MIG instances")
    if raw["task"] not in utility_reference:
        raise CalibrationError("Missing task utility reference")
    profile = dict(raw)
    for key in ("e_W", "alpha_J_per_request", "saturation_contribution_W", "quality"):
        profile[key] = finite(raw.get(key), key, nonnegative=True)
    for key in ("busy_L_s", "mu_requests_s"):
        profile[key] = finite(raw.get(key), key, positive=True)
    profile["busy_cs2"] = finite(raw.get("busy_cs2"), "busy_cs2", nonnegative=True)
    if not math.isclose(profile["mu_requests_s"], raw["B"] / profile["busy_L_s"], rel_tol=1e-7):
        raise CalibrationError("Busy service rate must equal B / busy_L_s")
    reservation = profile["e_W"] + profile["alpha_J_per_request"] * profile["mu_requests_s"]
    if not math.isclose(reservation, profile["saturation_contribution_W"], rel_tol=1e-7, abs_tol=1e-7):
        raise CalibrationError("Saturation contribution differs from e + alpha*mu")
    points = []
    for point in raw.get("points", []):
        points.append([
            finite(point.get("rate_requests_s"), "rate_requests_s", positive=True),
            finite(point.get("L_s"), "L_s", positive=True),
            finite(point.get("cs2"), "cs2", nonnegative=True),
        ])
    if len(points) < 2 or any(a[0] >= b[0] for a, b in zip(points, points[1:])):
        raise CalibrationError("At least two strictly ordered measured load points are required")
    domain = raw.get("domain")
    if domain != [points[0][0], points[-1][0]]:
        raise CalibrationError("Profile domain must equal the measured point range")
    reference = finite(utility_reference[raw["task"]], "utility reference", positive=True)
    profile.update(points=points, u=profile["quality"] / reference)
    return profile


MODEL_VARIANTS = ("full", "simple_same_campaign", "constant_latency_only",
                  "zero_residency_only", "baseline_31_only")


def calibration_variant(calibration, variant):
    """Create a labeled modeling ablation without changing the measured input.

    All response variants retain the same measured rate domain. In zero-residency
    variants, alpha absorbs e/mu so that the isolated busy reservation is exactly
    preserved. Paired model ablations use measurements from the same campaign.
    The separate 31-W baseline ablation changes the base.
    """
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model variant: {variant}")
    derived = deepcopy(calibration)
    source_hash = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    bases = {"gpus": calibration.get("gpus"), "pooled_baseline": calibration.get("pooled_baseline")}
    base_hash = hashlib.sha256(json.dumps(bases, sort_keys=True).encode()).hexdigest()
    constant_latency = variant in ("simple_same_campaign", "constant_latency_only")
    zero_residency = variant in ("simple_same_campaign", "zero_residency_only")
    for profile in derived.get("profiles", []):
        if "calibration" in profile:
            profile.update(profile.pop("calibration"))
        if profile.get("measurement_status") != "measured":
            raise CalibrationError("Cannot derive a model ablation from pending measurements")
        if constant_latency:
            latency = finite(profile.get("busy_L_s"), "busy_L_s", positive=True)
            variance = finite(profile.get("busy_cs2"), "busy_cs2", nonnegative=True)
            for point in profile["points"]:
                point["L_s"], point["cs2"] = latency, variance
        if zero_residency:
            capacity = finite(profile.get("mu_requests_s"), "mu_requests_s", positive=True)
            reservation = finite(profile.get("saturation_contribution_W"), "saturation_contribution_W", nonnegative=True)
            profile["e_W"] = 0.0
            profile["alpha_J_per_request"] = reservation / capacity
    if variant == "baseline_31_only":
        devices = list(derived.get("gpus", {}).values())
        if derived.get("pooled_baseline"):
            devices.append(derived["pooled_baseline"])
        for device in devices:
            device["idle_W"] = 31.0
            device["base_by_freq_W"] = {key: 31.0 for key in device["base_by_freq_W"]}
    derived["model_variant"] = dict(label=variant, source_calibration_sha256=source_hash,
        measured_baselines_sha256=base_hash, rate_domain_policy="common_measured_domain_no_extrapolation",
        baseline_scope="fixed_31W_ablation" if variant == "baseline_31_only" else "same_new_campaign_frequency_baselines",
        busy_reservation_preserved=True,
        measurement_values_retained_in="source calibration; this object contains derived model coefficients")
    return derived


def finalize_measured_document(completed_plan, *, allow_explicit_subset=False):
    """Explicitly convert the measurement plan layout to the model input schema.

    This performs no fitting, imputation, baseline pooling, or anchor decision.
    Those results must already be present. Pending rows are rejected by the
    adapter. The original plan/coefficient files remain the measurement record.
    """
    document = deepcopy(completed_plan)
    allowed = {"measured"}
    if allow_explicit_subset:
        allowed.add("completed_with_exclusions")
    if document.get("status") not in allowed:
        raise CalibrationError("Only an explicitly finalized measured campaign can be converted")
    document["schema_version"] = "a30_current_model_v1"
    references = dict(document.get("utility_reference", {}))
    for profile in document.get("profiles", []):
        reference = profile.get("quality_reference", references.get(profile["task"]))
        if reference is None:
            raise CalibrationError("Profile lacks a task quality reference")
        if profile["task"] in references and references[profile["task"]] != reference:
            raise CalibrationError("Inconsistent utility references within a task")
        references[profile["task"]] = reference
        if "calibration" in profile:
            profile.update(profile.pop("calibration"))
    document["utility_reference"] = references
    return document


class ModelAdapter:
    """One compiled alternative list consumed by MIP and both search stages."""

    def __init__(self, services, gpu_count, calibration, *, calibration_gpu_uuid=None,
                 cap_watts=165.0, require_catalog_ids=None):
        if calibration.get("schema_version") != "a30_current_model_v1":
            raise CalibrationError("Unsupported calibration schema")
        self.fleet_model = calibration.get("fleet_model", "single_reference_gpu")
        declared_gpu = calibration.get("calibration_gpu_uuid")
        if calibration_gpu_uuid is None:
            calibration_gpu_uuid = declared_gpu
        elif declared_gpu is not None and calibration_gpu_uuid != declared_gpu:
            raise CalibrationError("Requested reference GPU differs from the calibrated document")
        if self.fleet_model == "pooled_homogeneous":
            if calibration_gpu_uuid is not None:
                raise CalibrationError("A pooled fleet must not also select an individual GPU")
            device = calibration.get("pooled_baseline", {})
            if not device.get("source_gpu_uuids") or not device.get("pooling_method"):
                raise CalibrationError("Pooled baselines require source GPUs and a declared pooling method")
            if not calibration.get("cross_device_validation"):
                raise CalibrationError("A pooled calibration must retain cross-device anchor evidence")
        elif self.fleet_model == "single_reference_gpu":
            if not calibration_gpu_uuid:
                raise CalibrationError("An explicit calibrated reference GPU UUID is required")
            devices = calibration.get("gpus", {})
            if calibration_gpu_uuid not in devices:
                raise CalibrationError("Reference GPU has no baseline measurements")
            device = devices[calibration_gpu_uuid]
        else:
            raise CalibrationError("Unsupported fleet calibration policy")
        if device.get("measurement_status") != "measured":
            raise CalibrationError("Reference GPU baseline is not measured")
        self.idle = finite(device.get("idle_W"), "idle_W", positive=True)
        self.base_by_frequency = {int(key): finite(value, f"base[{key}]", positive=True)
                                  for key, value in device.get("base_by_freq_W", {}).items()}
        if not self.base_by_frequency:
            raise CalibrationError("Empty per-frequency GPU baseline table")
        if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count <= 0:
            raise ValueError("gpu_count must be a positive integer")
        if not isinstance(services, (list, tuple)) or not services:
            raise ValueError("At least one service is required")
        self.ss = deepcopy(services)
        self.G = gpu_count
        self.R = 4 * gpu_count
        self.cap = finite(cap_watts, "cap_watts", positive=True)
        if self.idle > self.cap:
            raise CalibrationError("The unused-device baseline exceeds the power cap")
        self.calibration_gpu_uuid = calibration_gpu_uuid
        self.calibration_sha256 = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
        reference = dict(calibration.get("utility_reference", {}))
        for profile in calibration.get("profiles", []):
            if profile.get("quality_reference") is not None:
                if profile["task"] in reference and reference[profile["task"]] != profile["quality_reference"]:
                    raise CalibrationError("Inconsistent utility references within a task")
                reference[profile["task"]] = profile["quality_reference"]
        raw_profiles = [profile for profile in calibration.get("profiles", [])
                        if self.fleet_model == "pooled_homogeneous"
                        or profile.get("gpu_uuid") == calibration_gpu_uuid]
        if not raw_profiles:
            raise CalibrationError("The reference GPU has no profile measurements")
        self.profiles = {}
        catalog_ids = set()
        for raw in raw_profiles:
            profile = normalized_profile(raw, reference)
            key = profile["physical_config_id"]
            if key in self.profiles or profile["catalog_id"] in catalog_ids:
                raise CalibrationError("Duplicate physical configuration or catalog row")
            if profile["freq_MHz"] not in self.base_by_frequency:
                raise CalibrationError("Profile frequency has no measured empty-GPU baseline")
            self.profiles[key] = profile
            catalog_ids.add(profile["catalog_id"])
        if require_catalog_ids is not None and set(require_catalog_ids) != catalog_ids:
            raise CalibrationError("Measured catalog coverage differs from the required catalog IDs")
        self.options = []
        self.interval_diagnostics = {}
        alternative_index = 0
        for sid, service in enumerate(self.ss):
            finite(service["lam"], "service demand", positive=True)
            finite(service["slo"], "service deadline", positive=True)
            finite(service["qmin"], "minimum quality", nonnegative=True)
            if service.get("ca2", 1.0) != 1.0:
                raise CalibrationError("The current response model assumes Poisson arrivals")
            for profile in self.profiles.values():
                if profile["task"] != service["task"] or profile["quality"] < service["qmin"]:
                    continue
                frequency = profile["freq_MHz"]
                peak = profile["saturation_contribution_W"]
                if self.active_base(frequency) + peak > self.cap + 1e-8:
                    continue
                cache_key = (profile["physical_config_id"], float(service["slo"]))
                if cache_key not in self.interval_diagnostics:
                    self.interval_diagnostics[cache_key] = direct_intervals(profile, service["slo"])
                intervals, _ = self.interval_diagnostics[cache_key]
                for ordinal, (lower, upper) in enumerate(intervals):
                    if upper <= 0 or lower > service["lam"]:
                        continue
                    alternative_id = (f"{profile['physical_config_id']}__slo{service['slo']:.17g}"
                                      f"__interval{ordinal}")
                    for gpu in range(self.G):
                        self.options.append(dict(s=sid, g=gpu, rid=alternative_index,
                            alternative_id=alternative_id, physical_config_id=profile["physical_config_id"],
                            catalog_id=profile["catalog_id"], model=profile["variant_id"], arch=profile["arch"],
                            k=profile["slices"], f=frequency, B=profile["B"],
                            quality=profile["quality"], u=profile["u"], lower=lower, upper=upper,
                            peak=peak, slope=profile["alpha_J_per_request"], intercept=profile["e_W"]))
                    alternative_index += 1
        self.lookup = {(option["s"], option["g"], option["rid"]): option for option in self.options}

    def active_base(self, frequency):
        return self.base_by_frequency[frequency]

    def totals(self, assignments):
        frequencies = {}
        for group in assignments:
            previous = frequencies.setdefault(group["g"], group["f"])
            if previous != group["f"]:
                raise ValueError("A GPU cannot run two frequencies")
        power = self.G * self.idle
        power += sum(self.active_base(frequency) - self.idle for frequency in frequencies.values())
        power += math.fsum(group["intercept"] * group["n"] + group["slope"] * group["rate"] for group in assignments)
        utility = math.fsum(group["u"] * group["rate"] for group in assignments)
        return power, utility

    def from_groups(self, groups):
        rebuilt = []
        for group in groups:
            count, rate = group["n"], group["rate"]
            if (isinstance(count, bool) or not isinstance(count, (int, float))
                    or not math.isfinite(count) or int(count) != count or count <= 0):
                raise ValueError("Active groups require positive integer counts and positive rates")
            finite(rate, "group rate", positive=True)
            if any(isinstance(group[key], bool) or not isinstance(group[key], int)
                   for key in ("s", "g", "rid")):
                raise ValueError("Service, GPU and alternative indices must be integers")
            option = self.lookup[(group["s"], group["g"], group["rid"])]
            profile = self.profiles[option["physical_config_id"]]
            per_replica = rate / count
            response, latency, utilization = response_values(profile, [per_replica])
            rebuilt.append(dict(option, n=int(count), rate=float(rate),
                                rate_per_replica=per_replica, response=float(response[0]),
                                L=float(latency[0]), utilization=float(utilization[0])))
        power, utility = self.totals(rebuilt)
        return dict(power=power, utility=utility, ratio=utility / power,
                    replicas=sum(group["n"] for group in rebuilt),
                    active=len({group["g"] for group in rebuilt}), assignment_json=json.dumps(rebuilt))

    def validate(self, result):
        if not result.get("assignment_json"):
            raise ValueError("No deployment to validate")
        groups = json.loads(result["assignment_json"])
        checked = self.from_groups(groups)
        for key in ("power", "utility", "ratio", "replicas", "active"):
            if not math.isclose(result[key], checked[key], rel_tol=1e-8, abs_tol=1e-7):
                raise ValueError(f"Incorrect {key}")
        rebuilt = json.loads(checked["assignment_json"])
        # Rebuilding by (s, g, rid) recomputes trusted metrics, but must not hide
        # a corrupted saved deployment naming another frequency or variant.
        for supplied, canonical in zip(groups, rebuilt):
            for key, value in canonical.items():
                if key not in supplied:
                    continue
                observed = supplied[key]
                if isinstance(value, float):
                    equal = (isinstance(observed, (int, float)) and not isinstance(observed, bool)
                             and math.isfinite(observed)
                             and math.isclose(observed, value, rel_tol=1e-8, abs_tol=1e-7))
                else:
                    equal = type(observed) is type(value) and observed == value
                if not equal:
                    raise ValueError(f"Incorrect assignment field: {key}")
        for sid, service in enumerate(self.ss):
            selected = [group for group in rebuilt if group["s"] == sid]
            if abs(sum(group["rate"] for group in selected) - service["lam"]) > 1e-5:
                raise ValueError("Service demand is not fully assigned")
            if len({group["model"] for group in selected}) != 1:
                raise ValueError("Exactly one variant is required per service")
            if not 1 <= sum(group["n"] for group in selected) <= self.R:
                raise ValueError("Service replica count exceeds its bound")
            for group in selected:
                rate = group["rate_per_replica"]
                if not group["lower"] - 1e-7 <= rate <= group["upper"] + 1e-7:
                    raise ValueError("Rate is outside the selected feasible interval")
                if group["utilization"] >= 1 or group["response"] > service["slo"] + 1e-7:
                    raise ValueError("Unstable or noncompliant response prediction")
        for gpu in range(self.G):
            selected = [group for group in rebuilt if group["g"] == gpu]
            if not selected:
                continue
            frequencies = {group["f"] for group in selected}
            if len(frequencies) != 1 or sum(group["k"] * group["n"] for group in selected) > 4:
                raise ValueError("Invalid GPU frequency or slice allocation")
            reservation = self.active_base(next(iter(frequencies))) + sum(group["peak"] * group["n"] for group in selected)
            if reservation > self.cap + 1e-6:
                raise ValueError("Busy power-cap reservation exceeded")
        return True

    def canonical_groups(self, groups):
        """Order identical GPUs by frequency rank, then occupied slices."""
        frequency_rank = {frequency: rank + 1 for rank, frequency in enumerate(sorted(self.base_by_frequency))}
        keys = {}
        for gpu in range(self.G):
            selected = [group for group in groups if group["g"] == gpu]
            keys[gpu] = ((frequency_rank[selected[0]["f"]], sum(group["k"] * group["n"] for group in selected))
                         if selected else (0, 0))
        ordering = sorted(range(self.G), key=lambda gpu: (-keys[gpu][0], -keys[gpu][1], gpu))
        mapping = {old: new for new, old in enumerate(ordering)}
        return [dict(group, g=mapping[group["g"]]) for group in groups]

    def remove_boundary_surplus(self, groups):
        """Remove the closed-relaxation boundary replica described in Section 4.1.

        Do not change the aggregate traffic or variant. The 1e-7 equality
        tolerance is accompanied by the same interval and response checks used
        by deployment validation, so numerical proximity alone is insufficient.
        Fixed-membership Sequential solves must retain their prescribed counts.
        """
        reduced = []
        for group in groups:
            item = dict(group)
            count, rate = item["n"], item["rate"]
            if count > 1 and abs(rate - item["upper"] * (count - 1)) <= 1e-7:
                per_replica = rate / (count - 1)
                profile = self.profiles[item["physical_config_id"]]
                if not profile["points"][0][0] <= per_replica <= profile["points"][-1][0]:
                    reduced.append(item)
                    continue
                response, _, utilization = response_values(profile, [per_replica])
                if (item["lower"] - 1e-7 <= per_replica <= item["upper"] + 1e-7
                        and utilization[0] < 1
                        and response[0] <= self.ss[item["s"]]["slo"] + 1e-7):
                    item["n"] = count - 1
            reduced.append(item)
        return reduced

    def solve(self, *, time_limit=30.0, seed_plan=None, threads=1, env=None, progress=None,
              frozen_groups=None):
        """Fractional MILP: q0=0 cold; q0=eta(validated seed) warm.

        ``time_limit`` is an explicit budget for the entire exact method,
        including model construction. Gurobi is imported only when solving.
        A returned time-limit incumbent is not labeled optimal without a final
        optimal Dinkelbach subproblem and a vanishing residual.
        """
        import gurobipy as gp
        started = perf_counter()
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
            raise ValueError("time_limit must be positive and finite, or None for no time limit")
        if any(not any(option["s"] == sid for option in self.options) for sid in range(len(self.ss))):
            return dict(status="individual_infeasible_in_calibrated_domain", seconds=perf_counter() - started)
        model = gp.Model(env=env)
        model.Params.OutputFlag = 0
        model.Params.Threads = threads
        model.Params.MIPGap = 1e-8
        model.Params.FeasibilityTol = 1e-8
        model.Params.IntFeasTol = 1e-8
        counts, rates = {}, {}
        count_bounds = {}
        by_service, by_gpu, by_variant, by_frequency = (defaultdict(list) for _ in range(4))
        for index, option in enumerate(self.options):
            count_bounds[index] = min(self.R, 4 // option["k"])
            if frozen_groups is None:
                count_bounds[index] = min(count_bounds[index],
                    safe_ceil_ratio(self.ss[option["s"]]["lam"], option["upper"]))
            counts[index] = model.addVar(vtype=gp.GRB.INTEGER, lb=0,
                                        ub=count_bounds[index], name=f"n{index}")
            rates[index] = model.addVar(lb=0, ub=self.ss[option["s"]]["lam"], name=f"r{index}")
            model.addConstr(rates[index] >= option["lower"] * counts[index])
            model.addConstr(rates[index] <= option["upper"] * counts[index])
            # Closed relaxation of the paper's strict minimum-count inequality.
            # Never replace strictness by a positive feasibility epsilon.
            if frozen_groups is None:
                model.addConstr(rates[index] >= option["upper"] * (counts[index] - 1))
            by_service[option["s"]].append(index)
            by_gpu[option["g"]].append(index)
            by_variant[option["s"], option["model"]].append(index)
            by_frequency[option["g"], option["f"]].append(index)
        frequencies = sorted(self.base_by_frequency)
        modes = {(gpu, frequency): model.addVar(vtype=gp.GRB.BINARY, name=f"z{gpu}_{frequency}")
                 for gpu in range(self.G) for frequency in frequencies}
        variants = {key: model.addVar(vtype=gp.GRB.BINARY) for key in by_variant}
        if frozen_groups is not None:
            fixed = defaultdict(int)
            for group in frozen_groups:
                fixed[(group["s"], group["g"], group["model"], group["k"])] += group["n"]
            available = defaultdict(list)
            for index, option in enumerate(self.options):
                available[(option["s"], option["g"], option["model"], option["k"])].append(index)
            for key in set(fixed) | set(available):
                model.addConstr(gp.quicksum(counts[index] for index in available[key]) == fixed[key])
        mode_order = []
        for sid, service in enumerate(self.ss):
            model.addConstr(gp.quicksum(rates[index] for index in by_service[sid]) == service["lam"])
            model.addConstr(gp.quicksum(counts[index] for index in by_service[sid]) >= 1)
            model.addConstr(gp.quicksum(counts[index] for index in by_service[sid]) <= self.R)
            model.addConstr(gp.quicksum(variable for key, variable in variants.items() if key[0] == sid) == 1)
        for key, indices in by_variant.items():
            model.addConstr(gp.quicksum(counts[index] for index in indices) <= self.R * variants[key])
        for gpu in range(self.G):
            active = gp.quicksum(modes[gpu, frequency] for frequency in frequencies)
            number = gp.quicksum(counts[index] for index in by_gpu[gpu])
            model.addConstr(active <= 1)
            model.addConstr(active <= number)
            model.addConstr(gp.quicksum(self.options[index]["k"] * counts[index] for index in by_gpu[gpu]) <= 4 * active)
            reservation = gp.quicksum(self.options[index]["peak"] * counts[index] for index in by_gpu[gpu])
            model.addConstr(reservation + gp.quicksum((self.active_base(frequency) - self.cap) * modes[gpu, frequency]
                                                     for frequency in frequencies) <= 0)
            for frequency in frequencies:
                model.addConstr(gp.quicksum(self.options[index]["k"] * counts[index]
                    for index in by_frequency[gpu, frequency]) <= 4 * modes[gpu, frequency])
            frequency_rank = gp.quicksum((rank + 1) * modes[gpu, frequency]
                                         for rank, frequency in enumerate(frequencies))
            occupied_slices = gp.quicksum(self.options[index]["k"] * counts[index] for index in by_gpu[gpu])
            mode_order.append(5 * frequency_rank + occupied_slices)
            if gpu and frozen_groups is None:
                model.addConstr(mode_order[gpu - 1] >= mode_order[gpu])
        power = self.G * self.idle + gp.quicksum((self.active_base(frequency) - self.idle) * variable
                                                for (gpu, frequency), variable in modes.items())
        power += gp.quicksum(option["intercept"] * counts[index] + option["slope"] * rates[index]
                             for index, option in enumerate(self.options))
        utility = gp.quicksum(option["u"] * rates[index] for index, option in enumerate(self.options))
        best = None
        warm_start_canonicalized = False
        if seed_plan and seed_plan.get("assignment_json"):
            self.validate(seed_plan)
            original_groups = json.loads(seed_plan["assignment_json"])
            start_groups = (self.canonical_groups(original_groups) if frozen_groups is None
                            else original_groups)
            warm_start_canonicalized = any(old["g"] != new["g"] for old, new in zip(original_groups, start_groups))
            best = self.from_groups(start_groups)
            index_by_key = {(option["s"], option["g"], option["rid"]): index for index, option in enumerate(self.options)}
            model.update()
            initial_counts = defaultdict(int)
            initial_rates = defaultdict(float)
            for variable in (*counts.values(), *rates.values(), *modes.values(), *variants.values()):
                variable.Start = 0
            for group in start_groups:
                index = index_by_key[group["s"], group["g"], group["rid"]]
                initial_counts[index] += group["n"]
                initial_rates[index] += group["rate"]
                modes[group["g"], group["f"]].Start = 1
                variants[group["s"], group["model"]].Start = 1
            for index, value in initial_counts.items():
                if value > count_bounds[index]:
                    raise ValueError("Warm seed contains dominated replica counts above the exact-method bound")
                counts[index].Start = value
                rates[index].Start = initial_rates[index]
        q_initial = best["ratio"] if best is not None else 0.0
        q, certified, last_status = q_initial, False, None
        first_solver_incumbent_seconds = None
        construction_seconds = perf_counter() - started
        def incumbent_callback(solver, where):
            nonlocal first_solver_incumbent_seconds
            if where == gp.GRB.Callback.MIPSOL and first_solver_incumbent_seconds is None:
                first_solver_incumbent_seconds = perf_counter() - started
                if progress is not None:
                    progress(dict(event="first_solver_incumbent", solve_elapsed_seconds=first_solver_incumbent_seconds,
                        mip_construction_seconds=construction_seconds,
                        incumbent_scope="complete feasible MILP incumbent; final returned plan is independently validated"))
        history = []
        while time_limit is None or perf_counter() - started < time_limit:
            model.setObjective(utility - q * power, gp.GRB.MAXIMIZE)
            model.Params.TimeLimit = (gp.GRB.INFINITY if time_limit is None else
                                      max(0.001, time_limit - (perf_counter() - started)))
            model.optimize(incumbent_callback)
            last_status = model.Status
            if not model.SolCount:
                break
            groups = [dict(option, n=int(round(counts[index].X)), rate=float(rates[index].X))
                      for index, option in enumerate(self.options)
                      if counts[index].X > 0.5 and rates[index].X > 1e-9]
            if frozen_groups is None:
                groups = self.remove_boundary_surplus(groups)
            candidate = self.from_groups(groups)
            self.validate(candidate)
            residual = candidate["utility"] - q * candidate["power"]
            history.append(dict(q=q, ratio=candidate["ratio"], residual=residual,
                                status=int(last_status), seconds=perf_counter() - started))
            if best is None or candidate["ratio"] > best["ratio"]:
                best = candidate
            if last_status == gp.GRB.OPTIMAL and abs(residual) <= 1e-7 * max(1, candidate["utility"]):
                certified = True
                break
            if last_status != gp.GRB.OPTIMAL:
                break
            q = candidate["ratio"]
        variables = model.NumVars
        model.dispose()
        status = ("optimal" if certified else "feasible_limit" if best is not None
                  else "infeasible_in_calibrated_domain" if last_status == gp.GRB.INFEASIBLE
                  else "no_solution_limit")
        result = dict(best or {})
        result.update(status=status, seconds=perf_counter() - started,
                      variables=variables, history=history, time_limit_seconds=time_limit,
                      q_initial=q_initial, mip_start=bool(seed_plan and seed_plan.get("assignment_json")),
                      warm_start_canonicalized=warm_start_canonicalized,
                      first_solver_incumbent_seconds=first_solver_incumbent_seconds,
                      time_to_first=0.0 if seed_plan and seed_plan.get("assignment_json") else first_solver_incumbent_seconds,
                      mip_construction_seconds=construction_seconds,
                      exact_strengthening=dict(
                          count_upper="min(fleet limit, per-GPU slice limit, exact rational ceil(lambda/h))",
                          rate_count_dominance="r >= h*(n-1), closed relaxation of strict paper inequality; no positive epsilon",
                          strict_paper_inequality_implemented_literally=False,
                          boundary_cleanup="remove one surplus replica at equality within 1e-7 after interval/response checks",
                          optimum_preserved="nonnegative residency permits removal of surplus replicas without changing traffic/utility",
                          frequency_capacity="sum(k*n) <= 4*z[g,f]",
                          symmetry="all homogeneous GPUs ordered by 5*frequency_rank+occupied_slices, descending"),
                      sequential_frozen_membership=frozen_groups is not None,
                      timing_scope="solve entry includes MIP construction/validation; excludes option compilation and seed generation")
        return result


def load_model(case, calibration_path, *, calibration_gpu_uuid=None, require_catalog_ids=None):
    calibration = json.loads(Path(calibration_path).read_text())
    return ModelAdapter(case["services"], case["gpus"], calibration,
                        cap_watts=case["cap_W"],
                        calibration_gpu_uuid=calibration_gpu_uuid,
                        require_catalog_ids=require_catalog_ids)
