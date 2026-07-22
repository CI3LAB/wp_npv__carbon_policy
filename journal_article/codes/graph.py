from __future__ import annotations

from collections import deque
from typing import Sequence

from .data import ProjectInstance

Package = tuple[int, ...]
Partition = tuple[Package, ...]


def normalize_partition(partition: Sequence[Sequence[int]]) -> Partition:
    return tuple(tuple(sorted(int(task) for task in package)) for package in partition)


def atomistic_partition(instance: ProjectInstance) -> Partition:
    return tuple((task_id,) for task_id in range(instance.n_tasks))


def predecessors(successors: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    pred: list[list[int]] = [[] for _ in successors]
    for i, succs in enumerate(successors):
        for j in succs:
            pred[j].append(i)
    return tuple(tuple(items) for items in pred)


def package_successors(instance: ProjectInstance, partition: Partition) -> tuple[tuple[int, ...], ...]:
    task_to_package = {}
    for package_idx, package in enumerate(partition):
        for task in package:
            task_to_package[task] = package_idx

    successors_by_package: list[set[int]] = [set() for _ in partition]
    for task, task_successors in enumerate(instance.successors):
        source_package = task_to_package[task]
        for successor in task_successors:
            target_package = task_to_package[successor]
            if source_package != target_package:
                successors_by_package[source_package].add(target_package)

    return tuple(tuple(sorted(items)) for items in successors_by_package)


def is_acyclic(successors: Sequence[Sequence[int]]) -> bool:
    n = len(successors)
    indegree = [0] * n
    for succs in successors:
        for j in succs:
            indegree[j] += 1

    queue = deque(i for i, degree in enumerate(indegree) if degree == 0)
    seen = 0
    while queue:
        i = queue.popleft()
        seen += 1
        for j in successors[i]:
            indegree[j] -= 1
            if indegree[j] == 0:
                queue.append(j)

    return seen == n


def induced_weakly_connected(instance: ProjectInstance, package: Sequence[int]) -> bool:
    tasks = set(package)
    if len(tasks) <= 1:
        return True

    neighbors: dict[int, set[int]] = {task: set() for task in tasks}
    for task in tasks:
        for successor in instance.successors[task]:
            if successor in tasks:
                neighbors[task].add(successor)
                neighbors[successor].add(task)

    start = next(iter(tasks))
    visited = {start}
    stack = [start]
    while stack:
        task = stack.pop()
        for neighbor in neighbors[task]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    return visited == tasks


def packages_adjacent(instance: ProjectInstance, package_a: Sequence[int], package_b: Sequence[int]) -> bool:
    a = set(package_a)
    b = set(package_b)
    for task in a:
        if any(successor in b for successor in instance.successors[task]):
            return True
    for task in b:
        if any(successor in a for successor in instance.successors[task]):
            return True
    return False


def package_has_inactive_task(instance: ProjectInstance, package: Sequence[int]) -> bool:
    inactive = instance.inactive_flags
    return any(inactive[task] for task in package)


def partition_is_valid(instance: ProjectInstance, partition: Partition) -> bool:
    all_tasks = [task for package in partition for task in package]
    if sorted(all_tasks) != list(range(instance.n_tasks)):
        return False

    for package in partition:
        if not induced_weakly_connected(instance, package):
            return False
        if len(package) > 1 and package_has_inactive_task(instance, package):
            return False

    return is_acyclic(package_successors(instance, partition))


def merge_packages(partition: Partition, idx_a: int, idx_b: int) -> Partition:
    if idx_a == idx_b:
        raise ValueError("Cannot merge a package with itself")
    keep = [package for idx, package in enumerate(partition) if idx not in {idx_a, idx_b}]
    merged = tuple(sorted(partition[idx_a] + partition[idx_b]))
    keep.append(merged)
    return tuple(keep)


def feasible_adjacent_merge_indices(instance: ProjectInstance, partition: Partition) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    candidate_pairs = set()
    for i, successors in enumerate(package_successors(instance, partition)):
        for j in successors:
            candidate_pairs.add(tuple(sorted((i, j))))

    for i, j in sorted(candidate_pairs):
        if package_has_inactive_task(instance, partition[i]):
            continue
        if package_has_inactive_task(instance, partition[j]):
            continue
        if not packages_adjacent(instance, partition[i], partition[j]):
            continue
        candidate = merge_packages(partition, i, j)
        if partition_is_valid(instance, candidate):
            pairs.append((i, j))
    return pairs
