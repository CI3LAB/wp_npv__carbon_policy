from __future__ import annotations

import itertools
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .algorithms import MICFResult, run_micf
from .data import ProjectInstance, Task
from .graph import (
    Partition,
    atomistic_partition,
    feasible_adjacent_merge_indices,
    merge_packages,
    partition_is_valid,
)
from .initialization import serial_dp_initial_partition
from .valuation import PolicyParams, ValuationResult, evaluate_partition


@dataclass(frozen=True)
class SyntheticInstanceSpec:
    instance_id: str
    network_type: str
    n_tasks: int
    replicate: int


@dataclass(frozen=True)
class ExactBenchmarkResult:
    optimum_partition: Partition
    optimum_value: float
    optimum_packages: int
    feasible_partitions: int
    enumerated_partitions: int
    runtime_seconds: float


@dataclass(frozen=True)
class ComparatorResult:
    method: str
    partition: Partition
    valuation: ValuationResult
    iterations: int
    runtime_seconds: float
    stopped_by: str
    initial_value: float
    initial_packages: int


def synthetic_instance_specs() -> tuple[SyntheticInstanceSpec, ...]:
    specs: list[SyntheticInstanceSpec] = []
    for n_tasks in (6, 8, 9):
        for network_type in ("serial", "parallel_chains", "diamond", "layered"):
            for replicate in (1, 2):
                specs.append(
                    SyntheticInstanceSpec(
                        instance_id=f"{network_type}_n{n_tasks}_r{replicate}",
                        network_type=network_type,
                        n_tasks=n_tasks,
                        replicate=replicate,
                    )
                )
    return tuple(specs)


def build_synthetic_instance(spec: SyntheticInstanceSpec) -> ProjectInstance:
    successors = _synthetic_successors(spec.network_type, spec.n_tasks, spec.replicate)
    tasks: list[Task] = []
    for task_id in range(spec.n_tasks):
        workload = 4 + ((3 * task_id + 2 * spec.replicate + spec.n_tasks) % 9)
        duration = 2 + ((5 * task_id + spec.replicate) % 7)
        inactive = spec.n_tasks >= 8 and (task_id + spec.replicate) % 11 == 0
        tasks.append(
            Task(
                task_id=task_id,
                inactive=inactive,
                workload=float(workload),
                duration=float(duration),
                successors=tuple(sorted(successors[task_id])),
            )
        )

    return ProjectInstance(
        path=Path(f"synthetic/{spec.instance_id}.rcp"),
        metadata_value=spec.n_tasks,
        tasks=tuple(tasks),
    )


def write_synthetic_instance(path: Path, instance: ProjectInstance) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(instance.metadata_value)]
    for task in instance.tasks:
        successors = [str(successor + 1) for successor in task.successors]
        row = [
            "1" if task.inactive else "0",
            str(int(task.workload)),
            str(int(task.duration)),
            str(len(successors)),
            *successors,
        ]
        lines.append(" ".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def exact_optimum(instance: ProjectInstance, params: PolicyParams) -> ExactBenchmarkResult:
    start = time.perf_counter()
    best_partition: Partition | None = None
    best_value = -math.inf
    feasible_partitions = 0
    enumerated_partitions = 0

    for partition in enumerate_task_partitions(instance):
        enumerated_partitions += 1
        if not partition_is_valid(instance, partition):
            continue
        feasible_partitions += 1
        valuation = evaluate_partition(instance, partition, params)
        if valuation.total_value > best_value:
            best_value = valuation.total_value
            best_partition = partition

    if best_partition is None:
        raise ValueError(f"No feasible partition found for {instance.path}")

    runtime_seconds = time.perf_counter() - start
    return ExactBenchmarkResult(
        optimum_partition=best_partition,
        optimum_value=best_value,
        optimum_packages=len(best_partition),
        feasible_partitions=feasible_partitions,
        enumerated_partitions=enumerated_partitions,
        runtime_seconds=runtime_seconds,
    )


def enumerate_task_partitions(instance: ProjectInstance) -> Iterable[Partition]:
    blocks: list[list[int]] = []
    inactive = instance.inactive_flags

    def search(task_id: int) -> Iterable[Partition]:
        if task_id == instance.n_tasks:
            yield tuple(tuple(block) for block in blocks)
            return

        blocks.append([task_id])
        yield from search(task_id + 1)
        blocks.pop()

        if inactive[task_id]:
            return

        for block in blocks:
            if any(inactive[member] for member in block):
                continue
            block.append(task_id)
            yield from search(task_id + 1)
            block.pop()

    yield from search(0)


def run_comparator(
    instance: ProjectInstance,
    params: PolicyParams,
    method: str,
    rng_seed: int = 1009,
    improvement_tolerance: float = 1e-9,
) -> ComparatorResult:
    if method == "atomistic":
        return _fixed_partition_result(instance, params, method, atomistic_partition(instance))
    if method == "serial_dp_initial":
        return _fixed_partition_result(instance, params, method, serial_dp_initial_partition(instance))
    if method == "micf_atomistic":
        return _micf_result(instance, params, method, atomistic_partition(instance))
    if method == "micf_serial_dp":
        return _micf_result(instance, params, method, serial_dp_initial_partition(instance))

    if method == "first_positive":
        selector = _select_first_positive
        initial_partition = atomistic_partition(instance)
        accept_by = "full_delta"
    elif method == "setup_ranked_positive":
        selector = _select_setup_ranked_positive
        initial_partition = atomistic_partition(instance)
        accept_by = "full_delta"
    elif method == "setup_saving_only":
        selector = _select_setup_saving_only
        initial_partition = atomistic_partition(instance)
        accept_by = "score"
    elif method == "random_positive":
        selector = _select_random_positive(rng_seed)
        initial_partition = atomistic_partition(instance)
        accept_by = "full_delta"
    elif method == "direct_no_externality":
        selector = _select_direct_no_externality
        initial_partition = atomistic_partition(instance)
        accept_by = "score"
    else:
        raise ValueError(f"Unknown comparator method: {method}")

    return _ranked_greedy_result(
        instance=instance,
        params=params,
        method=method,
        initial_partition=initial_partition,
        selector=selector,
        accept_by=accept_by,
        improvement_tolerance=improvement_tolerance,
    )


def package_size_std(partition: Partition) -> float:
    sizes = [len(package) for package in partition]
    if len(sizes) <= 1:
        return 0.0
    return statistics.pstdev(sizes)


def partition_to_string(partition: Partition) -> str:
    return ";".join("-".join(str(task + 1) for task in package) for package in partition)


def component_deltas(
    initial: ValuationResult, final: ValuationResult
) -> tuple[float, float, float, float]:
    subsidy_gain = final.subsidy_pv - initial.subsidy_pv
    tax_saving = initial.tax_pv - final.tax_pv
    setup_saving = initial.setup_pv - final.setup_pv
    total_gain = final.total_value - initial.total_value
    residual = total_gain - subsidy_gain - tax_saving - setup_saving
    return subsidy_gain, tax_saving, setup_saving, residual


def policy_scenarios() -> tuple[tuple[str, PolicyParams], ...]:
    return (
        ("balanced", PolicyParams(tax_rate=0.5, subsidy_rate=0.5, discount_rate=0.0001)),
        ("high_tax", PolicyParams(tax_rate=0.8, subsidy_rate=0.4, discount_rate=0.0001)),
        ("high_subsidy", PolicyParams(tax_rate=0.4, subsidy_rate=0.8, discount_rate=0.0001)),
    )


def _fixed_partition_result(
    instance: ProjectInstance, params: PolicyParams, method: str, partition: Partition
) -> ComparatorResult:
    start = time.perf_counter()
    valuation = evaluate_partition(instance, partition, params)
    runtime_seconds = time.perf_counter() - start
    return ComparatorResult(
        method=method,
        partition=partition,
        valuation=valuation,
        iterations=0,
        runtime_seconds=runtime_seconds,
        stopped_by="fixed_partition",
        initial_value=valuation.total_value,
        initial_packages=len(partition),
    )


def _micf_result(
    instance: ProjectInstance, params: PolicyParams, method: str, initial_partition: Partition
) -> ComparatorResult:
    initial_value = evaluate_partition(instance, initial_partition, params)
    start = time.perf_counter()
    result: MICFResult = run_micf(instance, params, initial_partition=initial_partition)
    runtime_seconds = time.perf_counter() - start
    return ComparatorResult(
        method=method,
        partition=result.partition,
        valuation=result.valuation,
        iterations=len(result.history) - 1,
        runtime_seconds=runtime_seconds,
        stopped_by=result.stopped_by,
        initial_value=initial_value.total_value,
        initial_packages=len(initial_partition),
    )


CandidateSelector = Callable[[list[dict[str, object]]], dict[str, object] | None]


def _ranked_greedy_result(
    instance: ProjectInstance,
    params: PolicyParams,
    method: str,
    initial_partition: Partition,
    selector: CandidateSelector,
    accept_by: str,
    improvement_tolerance: float,
) -> ComparatorResult:
    start = time.perf_counter()
    partition = initial_partition
    valuation = evaluate_partition(instance, partition, params)
    initial_value = valuation.total_value
    initial_packages = len(partition)
    iterations = 0

    while True:
        candidates = _candidate_records(instance, params, partition, valuation)
        selected = selector(candidates)
        if selected is None:
            stopped_by = "converged"
            break

        threshold_value = float(selected["actual_delta"] if accept_by == "full_delta" else selected["score"])
        if threshold_value <= improvement_tolerance:
            stopped_by = "converged"
            break

        partition = selected["partition"]  # type: ignore[assignment]
        valuation = selected["valuation"]  # type: ignore[assignment]
        iterations += 1

    runtime_seconds = time.perf_counter() - start
    return ComparatorResult(
        method=method,
        partition=partition,
        valuation=valuation,
        iterations=iterations,
        runtime_seconds=runtime_seconds,
        stopped_by=stopped_by,
        initial_value=initial_value,
        initial_packages=initial_packages,
    )


def _candidate_records(
    instance: ProjectInstance,
    params: PolicyParams,
    partition: Partition,
    valuation: ValuationResult,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for idx_a, idx_b in feasible_adjacent_merge_indices(instance, partition):
        candidate_partition = merge_packages(partition, idx_a, idx_b)
        candidate_valuation = evaluate_partition(instance, candidate_partition, params)
        records.append(
            {
                "idx_a": idx_a,
                "idx_b": idx_b,
                "partition": candidate_partition,
                "valuation": candidate_valuation,
                "actual_delta": candidate_valuation.total_value - valuation.total_value,
                "setup_saving": valuation.setup_pv - candidate_valuation.setup_pv,
                "direct_delta": _direct_no_externality_delta(
                    instance, params, partition, valuation, idx_a, idx_b
                ),
            }
        )
    return records


def _select_first_positive(candidates: list[dict[str, object]]) -> dict[str, object] | None:
    for candidate in candidates:
        if float(candidate["actual_delta"]) > 0:
            candidate["score"] = candidate["actual_delta"]
            return candidate
    return None


def _select_setup_ranked_positive(candidates: list[dict[str, object]]) -> dict[str, object] | None:
    positive = [candidate for candidate in candidates if float(candidate["actual_delta"]) > 0]
    if not positive:
        return None
    selected = max(positive, key=lambda row: (float(row["setup_saving"]), float(row["actual_delta"])))
    selected["score"] = selected["setup_saving"]
    return selected


def _select_setup_saving_only(candidates: list[dict[str, object]]) -> dict[str, object] | None:
    if not candidates:
        return None
    selected = max(candidates, key=lambda row: float(row["setup_saving"]))
    selected["score"] = selected["setup_saving"]
    return selected


def _select_random_positive(rng_seed: int) -> CandidateSelector:
    rng = random.Random(rng_seed)

    def select(candidates: list[dict[str, object]]) -> dict[str, object] | None:
        positive = [candidate for candidate in candidates if float(candidate["actual_delta"]) > 0]
        if not positive:
            return None
        selected = rng.choice(positive)
        selected["score"] = selected["actual_delta"]
        return selected

    return select


def _select_direct_no_externality(candidates: list[dict[str, object]]) -> dict[str, object] | None:
    if not candidates:
        return None
    selected = max(candidates, key=lambda row: float(row["direct_delta"]))
    selected["score"] = selected["direct_delta"]
    return selected


def _direct_no_externality_delta(
    instance: ProjectInstance,
    params: PolicyParams,
    partition: Partition,
    valuation: ValuationResult,
    idx_a: int,
    idx_b: int,
) -> float:
    package_a = partition[idx_a]
    package_b = partition[idx_b]
    merged = tuple(sorted(package_a + package_b))
    start_time = min(
        valuation.package_start_times[idx_a],
        valuation.package_start_times[idx_b],
    )
    finish_time = max(
        valuation.package_finish_times[idx_a],
        valuation.package_finish_times[idx_b],
    )
    return (
        _package_value_at_time(instance, params, merged, start_time, finish_time)
        - _package_value_at_time(
            instance,
            params,
            package_a,
            valuation.package_start_times[idx_a],
            valuation.package_finish_times[idx_a],
        )
        - _package_value_at_time(
            instance,
            params,
            package_b,
            valuation.package_start_times[idx_b],
            valuation.package_finish_times[idx_b],
        )
    )


def _package_value_at_time(
    instance: ProjectInstance,
    params: PolicyParams,
    package: tuple[int, ...],
    start_time: float,
    finish_time: float,
) -> float:
    content = sum(instance.tasks[task].workload for task in package)
    subsidy = params.subsidy_rate * (content**params.subsidy_concavity)
    subsidy *= math.exp(-params.discount_rate * start_time)
    tax = params.tax_rate * (content**params.tax_convexity)
    filing_time = math.ceil(finish_time / params.tax_period) * params.tax_period
    tax *= math.exp(-params.discount_rate * filing_time)
    setup = params.setup_cost * math.exp(-params.discount_rate * start_time)
    return subsidy - tax - setup


def _synthetic_successors(network_type: str, n_tasks: int, replicate: int) -> tuple[set[int], ...]:
    if network_type == "serial":
        return _serial_successors(n_tasks)
    if network_type == "parallel_chains":
        return _parallel_chain_successors(n_tasks, replicate)
    if network_type == "diamond":
        return _diamond_successors(n_tasks)
    if network_type == "layered":
        return _layered_successors(n_tasks, replicate)
    raise ValueError(f"Unknown synthetic network type: {network_type}")


def _serial_successors(n_tasks: int) -> tuple[set[int], ...]:
    successors = tuple(set() for _ in range(n_tasks))
    for task in range(n_tasks - 1):
        successors[task].add(task + 1)
    return successors


def _parallel_chain_successors(n_tasks: int, replicate: int) -> tuple[set[int], ...]:
    successors = tuple(set() for _ in range(n_tasks))
    n_chains = 2 + (replicate % 2)
    for chain_start in range(n_chains):
        chain_nodes = list(range(chain_start, n_tasks, n_chains))
        for source, target in itertools.pairwise(chain_nodes):
            successors[source].add(target)
    return successors


def _diamond_successors(n_tasks: int) -> tuple[set[int], ...]:
    successors = tuple(set() for _ in range(n_tasks))
    if n_tasks <= 2:
        return _serial_successors(n_tasks)
    for task in range(1, n_tasks - 1):
        successors[0].add(task)
        successors[task].add(n_tasks - 1)
    return successors


def _layered_successors(n_tasks: int, replicate: int) -> tuple[set[int], ...]:
    successors = tuple(set() for _ in range(n_tasks))
    layers: list[list[int]] = [[], [], []]
    for task in range(n_tasks):
        layers[min(2, (3 * task) // max(1, n_tasks))].append(task)

    for layer_idx in range(len(layers) - 1):
        for source in layers[layer_idx]:
            for offset, target in enumerate(layers[layer_idx + 1]):
                if (source + target + offset + replicate) % 2 == 0:
                    successors[source].add(target)
            if not successors[source] and layers[layer_idx + 1]:
                successors[source].add(layers[layer_idx + 1][source % len(layers[layer_idx + 1])])

    for task in range(n_tasks - 1):
        if task % 3 == replicate % 3:
            successors[task].add(task + 1)

    return successors
