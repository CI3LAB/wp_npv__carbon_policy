from __future__ import annotations

import time
from dataclasses import dataclass

from .data import ProjectInstance
from .graph import Partition, feasible_adjacent_merge_indices, merge_packages, partition_is_valid
from .valuation import PolicyParams, ValuationResult, evaluate_partition


@dataclass(frozen=True)
class LookaheadPath:
    partition: Partition
    valuation: ValuationResult
    path: tuple[tuple[int, int], ...]
    delta: float


@dataclass(frozen=True)
class CertifiedLookaheadResult:
    partition: Partition
    valuation: ValuationResult
    initial_value: float
    depth: int
    accepted_sequences: int
    accepted_merges: int
    evaluated_paths: int
    runtime_seconds: float
    final_best_delta: float
    final_path_count: int
    certified_depth_stable: bool
    stopped_by: str


def canonical_partition(partition: Partition) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted(tuple(sorted(package)) for package in partition))


def best_merge_lookahead_path(
    instance: ProjectInstance,
    partition: Partition,
    params: PolicyParams,
    depth: int,
    current_value: float | None = None,
) -> tuple[LookaheadPath | None, int]:

    if depth < 1:
        raise ValueError("depth must be at least 1")

    start_value = (
        evaluate_partition(instance, partition, params).total_value
        if current_value is None
        else current_value
    )
    best: LookaheadPath | None = None
    evaluated_paths = 0
    seen: set[tuple[tuple[int, ...], ...]] = {canonical_partition(partition)}

    def search(current_partition: Partition, path: tuple[tuple[int, int], ...]) -> None:
        nonlocal best, evaluated_paths
        if len(path) >= depth:
            return

        for idx_a, idx_b in feasible_adjacent_merge_indices(instance, current_partition):
            candidate_partition = merge_packages(current_partition, idx_a, idx_b)
            key = canonical_partition(candidate_partition)
            if key in seen:
                continue
            seen.add(key)
            candidate_valuation = evaluate_partition(instance, candidate_partition, params)
            candidate_path = (*path, (idx_a, idx_b))
            delta = candidate_valuation.total_value - start_value
            evaluated_paths += 1
            if best is None or delta > best.delta:
                best = LookaheadPath(
                    partition=candidate_partition,
                    valuation=candidate_valuation,
                    path=candidate_path,
                    delta=delta,
                )
            search(candidate_partition, candidate_path)

    search(partition, tuple())
    return best, evaluated_paths


def run_certified_merge_lookahead(
    instance: ProjectInstance,
    params: PolicyParams,
    initial_partition: Partition,
    depth: int = 2,
    improvement_tolerance: float = 1e-9,
) -> CertifiedLookaheadResult:

    if not partition_is_valid(instance, initial_partition):
        raise ValueError("Initial partition is not graph-feasible")

    start = time.perf_counter()
    partition = initial_partition
    valuation = evaluate_partition(instance, partition, params)
    initial_value = valuation.total_value
    accepted_sequences = 0
    accepted_merges = 0
    evaluated_paths = 0

    while True:
        best, path_count = best_merge_lookahead_path(
            instance=instance,
            partition=partition,
            params=params,
            depth=depth,
            current_value=valuation.total_value,
        )
        evaluated_paths += path_count
        if best is None or best.delta <= improvement_tolerance:
            final_best_delta = 0.0 if best is None else best.delta
            final_path_count = path_count
            stopped_by = "converged"
            break

        partition = best.partition
        valuation = best.valuation
        accepted_sequences += 1
        accepted_merges += len(best.path)

    runtime_seconds = time.perf_counter() - start
    return CertifiedLookaheadResult(
        partition=partition,
        valuation=valuation,
        initial_value=initial_value,
        depth=depth,
        accepted_sequences=accepted_sequences,
        accepted_merges=accepted_merges,
        evaluated_paths=evaluated_paths,
        runtime_seconds=runtime_seconds,
        final_best_delta=final_best_delta,
        final_path_count=final_path_count,
        certified_depth_stable=final_best_delta <= improvement_tolerance,
        stopped_by=stopped_by,
    )
