from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Iterable

from .data import ProjectInstance
from .graph import (
    Package,
    Partition,
    induced_weakly_connected,
    normalize_partition,
    partition_is_valid,
)
from .schedule import package_duration
from .valuation import PolicyParams, ValuationResult, evaluate_partition


@dataclass(frozen=True)
class CandidatePackage:
    package: Package
    workload: float
    duration: float


@dataclass(frozen=True)
class GurobiBaselineConfig:


    max_connected_package_size: int = 4
    max_generated_candidates: int = 6000
    time_limit_seconds: float = 45.0
    mip_gap: float = 0.0
    pwl_points: int = 48
    output_flag: int = 0
    threads: int = 0
    improvement_tolerance: float = 1e-9


@dataclass(frozen=True)
class GurobiBaselineResult:
    partition: Partition
    valuation: ValuationResult
    mip_objective: float
    selected_exact_value: float
    reported_exact_value: float
    runtime_seconds: float
    solver_runtime_seconds: float
    candidate_generation_seconds: float
    status: str
    status_code: int
    mip_gap: float | None
    best_bound: float | None
    candidate_count: int
    selected_candidate_count: int
    precedence_constraint_count: int
    finish_bucket_count: int
    reported_partition_source: str


def _undirected_neighbors(instance: ProjectInstance) -> tuple[tuple[int, ...], ...]:
    neighbors: list[set[int]] = [set() for _ in range(instance.n_tasks)]
    for task, successors in enumerate(instance.successors):
        for successor in successors:
            neighbors[task].add(successor)
            neighbors[successor].add(task)
    return tuple(tuple(sorted(items)) for items in neighbors)


def _package_allowed(instance: ProjectInstance, package: Iterable[int]) -> bool:
    normalized = tuple(sorted(set(int(task) for task in package)))
    if not normalized:
        return False
    if len(normalized) > 1 and any(instance.tasks[task].inactive for task in normalized):
        return False
    return induced_weakly_connected(instance, normalized)


def _add_partition_packages(
    instance: ProjectInstance,
    candidates: set[Package],
    partitions: Iterable[Partition],
) -> None:
    for partition in partitions:
        for package in partition:
            normalized = tuple(sorted(package))
            if _package_allowed(instance, normalized):
                candidates.add(normalized)


def generate_candidate_packages(
    instance: ProjectInstance,
    *,
    extra_partitions: Iterable[Partition] = (),
    max_connected_package_size: int = 4,
    max_generated_candidates: int = 6000,
) -> tuple[CandidatePackage, ...]:


    if max_connected_package_size < 1:
        raise ValueError("max_connected_package_size must be at least 1")

    neighbors = _undirected_neighbors(instance)
    candidates: set[Package] = {(task,) for task in range(instance.n_tasks)}

    for start in range(instance.n_tasks):
        stack: list[Package] = [(start,)]
        seen_from_start: set[Package] = {(start,)}
        while stack:
            current = stack.pop()
            if len(current) >= max_connected_package_size:
                continue
            expansion_frontier: set[int] = set()
            for task in current:
                expansion_frontier.update(neighbors[task])
            for next_task in sorted(expansion_frontier.difference(current)):
                expanded = tuple(sorted((*current, next_task)))
                if expanded in seen_from_start:
                    continue
                seen_from_start.add(expanded)
                if not _package_allowed(instance, expanded):
                    continue
                candidates.add(expanded)
                if len(candidates) >= max_generated_candidates:
                    break
                stack.append(expanded)
            if len(candidates) >= max_generated_candidates:
                break
        if len(candidates) >= max_generated_candidates:
            break

    _add_partition_packages(instance, candidates, extra_partitions)

    ordered = sorted(candidates, key=lambda pkg: (len(pkg), pkg))
    return tuple(
        CandidatePackage(
            package=package,
            workload=sum(instance.tasks[task].workload for task in package),
            duration=package_duration(instance, package),
        )
        for package in ordered
    )


def _precedence_pairs(
    instance: ProjectInstance,
    candidates: tuple[CandidatePackage, ...],
) -> set[tuple[int, int]]:
    task_to_candidates: list[list[int]] = [[] for _ in range(instance.n_tasks)]
    package_sets = [set(candidate.package) for candidate in candidates]
    for idx, candidate in enumerate(candidates):
        for task in candidate.package:
            task_to_candidates[task].append(idx)

    pairs: set[tuple[int, int]] = set()
    for source, successors in enumerate(instance.successors):
        for target in successors:
            for package_i in task_to_candidates[source]:
                for package_j in task_to_candidates[target]:
                    if package_i == package_j:
                        continue
                    if package_sets[package_i].intersection(package_sets[package_j]):
                        continue
                    pairs.add((package_i, package_j))
    return pairs


def _finish_bucket_count(max_time: float, tax_period: float) -> int:
    return max(1, int(math.ceil(max_time / tax_period)) + 1)


def _discount_breakpoints(max_time: float, discount_rate: float, points: int) -> tuple[list[float], list[float]]:
    n_points = max(2, points)
    if max_time <= 0:
        return [0.0, 1.0], [1.0, math.exp(-discount_rate)]
    step = max_time / (n_points - 1)
    xs = [idx * step for idx in range(n_points)]
    ys = [math.exp(-discount_rate * x) for x in xs]
    return xs, ys


def _best_exact_incumbent(
    instance: ProjectInstance,
    params: PolicyParams,
    partitions: Iterable[tuple[str, Partition]],
) -> tuple[str, Partition, ValuationResult] | None:
    best: tuple[str, Partition, ValuationResult] | None = None
    for name, partition in partitions:
        if not partition_is_valid(instance, partition):
            continue
        valuation = evaluate_partition(instance, partition, params)
        if best is None or valuation.total_value > best[2].total_value:
            best = (name, partition, valuation)
    return best


def _status_name(gp_module, status_code: int) -> str:
    for name in (
        "OPTIMAL",
        "TIME_LIMIT",
        "SUBOPTIMAL",
        "INFEASIBLE",
        "INF_OR_UNBD",
        "UNBOUNDED",
        "INTERRUPTED",
        "NUMERIC",
    ):
        if getattr(gp_module.GRB, name, None) == status_code:
            return name
    return str(status_code)


def solve_gurobi_candidate_pool(
    instance: ProjectInstance,
    params: PolicyParams,
    *,
    extra_partitions: Iterable[tuple[str, Partition]] = (),
    warm_start_partition: Partition | None = None,
    config: GurobiBaselineConfig | None = None,
) -> GurobiBaselineResult:


    config = config or GurobiBaselineConfig()
    candidate_start = time.perf_counter()
    named_extra = tuple(extra_partitions)
    candidates = generate_candidate_packages(
        instance,
        extra_partitions=(partition for _, partition in named_extra),
        max_connected_package_size=config.max_connected_package_size,
        max_generated_candidates=config.max_generated_candidates,
    )
    candidate_generation_seconds = time.perf_counter() - candidate_start
    if not candidates:
        raise ValueError("No candidate packages were generated")

    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("carbon_work_package_candidate_pool_mip")
    model.Params.OutputFlag = config.output_flag
    model.Params.TimeLimit = config.time_limit_seconds
    model.Params.MIPGap = config.mip_gap
    if config.threads > 0:
        model.Params.Threads = config.threads

    k_count = len(candidates)
    max_time = sum(instance.durations)
    big_m = max(1.0, max_time + max(candidate.duration for candidate in candidates))
    bucket_count = _finish_bucket_count(max_time, params.tax_period)
    xpts, ypts = _discount_breakpoints(max_time, params.discount_rate, config.pwl_points)

    y = model.addVars(k_count, vtype=GRB.BINARY, name="select")
    start = model.addVars(k_count, lb=0.0, ub=big_m, name="start")
    finish = model.addVars(k_count, lb=0.0, ub=big_m, name="finish")
    start_discount = model.addVars(k_count, lb=min(ypts), ub=1.0, name="start_discount")
    active_start_discount = model.addVars(k_count, lb=0.0, ub=1.0, name="active_start_discount")
    finish_bucket = model.addVars(k_count, bucket_count, vtype=GRB.BINARY, name="finish_bucket")

    for task in range(instance.n_tasks):
        covering = [idx for idx, candidate in enumerate(candidates) if task in candidate.package]
        model.addConstr(gp.quicksum(y[idx] for idx in covering) == 1, name=f"cover_{task}")

    for idx, candidate in enumerate(candidates):
        model.addConstr(start[idx] <= big_m * y[idx], name=f"active_start_{idx}")
        model.addConstr(finish[idx] <= big_m * y[idx], name=f"active_finish_{idx}")
        model.addConstr(
            finish[idx] >= start[idx] + candidate.duration - big_m * (1 - y[idx]),
            name=f"duration_lb_{idx}",
        )
        model.addConstr(
            finish[idx] <= start[idx] + candidate.duration + big_m * (1 - y[idx]),
            name=f"duration_ub_{idx}",
        )
        model.addGenConstrPWL(start[idx], start_discount[idx], xpts, ypts, name=f"discount_pwl_{idx}")
        model.addConstr(active_start_discount[idx] <= y[idx], name=f"active_disc_y_{idx}")
        model.addConstr(active_start_discount[idx] <= start_discount[idx], name=f"active_disc_z_{idx}")
        model.addConstr(
            active_start_discount[idx] >= start_discount[idx] + y[idx] - 1,
            name=f"active_disc_lb_{idx}",
        )
        model.addConstr(
            gp.quicksum(finish_bucket[idx, bucket] for bucket in range(bucket_count)) == y[idx],
            name=f"one_finish_bucket_{idx}",
        )
        for bucket in range(bucket_count):
            upper = (bucket + 1) * params.tax_period
            lower = bucket * params.tax_period
            model.addConstr(
                finish[idx] <= upper + big_m * (1 - finish_bucket[idx, bucket]),
                name=f"finish_bucket_ub_{idx}_{bucket}",
            )
            if bucket > 0:
                model.addConstr(
                    finish[idx] >= lower + 1e-6 - big_m * (1 - finish_bucket[idx, bucket]),
                    name=f"finish_bucket_lb_{idx}_{bucket}",
                )

    precedence_pairs = _precedence_pairs(instance, candidates)
    for left, right in precedence_pairs:
        model.addConstr(
            start[right] >= finish[left] - big_m * (2 - y[left] - y[right]),
            name=f"precedence_{left}_{right}",
        )

    objective_terms = []
    for idx, candidate in enumerate(candidates):
        workload = candidate.workload
        start_value = (
            params.subsidy_rate * (workload ** params.subsidy_concavity)
            - params.setup_cost
        )
        objective_terms.append(start_value * active_start_discount[idx])
        tax_base = params.tax_rate * (workload ** params.tax_convexity)
        for bucket in range(bucket_count):
            filing_time = (bucket + 1) * params.tax_period
            objective_terms.append(
                -tax_base
                * math.exp(-params.discount_rate * filing_time)
                * finish_bucket[idx, bucket]
            )

    if warm_start_partition is not None:
        warm_start_packages = {tuple(sorted(package)) for package in warm_start_partition}
        for idx, candidate in enumerate(candidates):
            y[idx].Start = 1.0 if candidate.package in warm_start_packages else 0.0

    model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)

    solve_start = time.perf_counter()
    model.optimize()
    solver_runtime = time.perf_counter() - solve_start

    status = _status_name(gp, model.Status)
    has_solution = model.SolCount > 0
    if not has_solution:
        incumbent = _best_exact_incumbent(instance, params, named_extra)
        if incumbent is None:
            raise RuntimeError(f"Gurobi did not return a feasible solution; status={status}")
        source, partition, valuation = incumbent
        runtime_seconds = candidate_generation_seconds + solver_runtime
        return GurobiBaselineResult(
            partition=partition,
            valuation=valuation,
            mip_objective=float("nan"),
            selected_exact_value=valuation.total_value,
            reported_exact_value=valuation.total_value,
            runtime_seconds=runtime_seconds,
            solver_runtime_seconds=solver_runtime,
            candidate_generation_seconds=candidate_generation_seconds,
            status=status,
            status_code=model.Status,
            mip_gap=None,
            best_bound=None,
            candidate_count=k_count,
            selected_candidate_count=len(partition),
            precedence_constraint_count=len(precedence_pairs),
            finish_bucket_count=bucket_count,
            reported_partition_source=f"fallback_{source}",
        )

    selected_packages = tuple(
        candidates[idx].package for idx in range(k_count) if y[idx].X >= 0.5
    )
    selected_partition = normalize_partition(selected_packages)
    if not partition_is_valid(instance, selected_partition):
        raise RuntimeError("Gurobi selected partition is not graph-feasible")

    selected_valuation = evaluate_partition(instance, selected_partition, params)
    exact_candidates: list[tuple[str, Partition]] = [("gurobi_selected", selected_partition)]
    exact_candidates.extend(named_extra)
    best_exact = _best_exact_incumbent(instance, params, exact_candidates)
    if best_exact is None:
        raise RuntimeError("No graph-feasible partition available for exact reporting")

    source, reported_partition, reported_valuation = best_exact
    runtime_seconds = candidate_generation_seconds + solver_runtime
    mip_gap = model.MIPGap if model.SolCount > 0 else None
    best_bound = model.ObjBound if model.SolCount > 0 else None

    return GurobiBaselineResult(
        partition=reported_partition,
        valuation=reported_valuation,
        mip_objective=model.ObjVal,
        selected_exact_value=selected_valuation.total_value,
        reported_exact_value=reported_valuation.total_value,
        runtime_seconds=runtime_seconds,
        solver_runtime_seconds=solver_runtime,
        candidate_generation_seconds=candidate_generation_seconds,
        status=status,
        status_code=model.Status,
        mip_gap=mip_gap,
        best_bound=best_bound,
        candidate_count=k_count,
        selected_candidate_count=len(selected_partition),
        precedence_constraint_count=len(precedence_pairs),
        finish_bucket_count=bucket_count,
        reported_partition_source=source,
    )
