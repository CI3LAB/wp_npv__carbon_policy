from __future__ import annotations

import math
from dataclasses import dataclass

from .data import ProjectInstance
from .graph import Partition, atomistic_partition, partition_is_valid, predecessors


@dataclass(frozen=True)
class SerialDPParams:
    fixed_cost: float = 50.0
    discount_penalty: float = 50.0
    discount_rate: float = 0.00025
    workload_concavity: float = 0.8
    workload_convexity: float = 1.2
    variable_scale: float = 6.0


def active_serial_edges(instance: ProjectInstance) -> tuple[tuple[int, int], ...]:
    pred = predecessors(instance.successors)
    edges: list[tuple[int, int]] = []
    inactive = instance.inactive_flags

    for task, successors in enumerate(instance.successors):
        if inactive[task] or len(successors) != 1:
            continue
        successor = successors[0]
        if inactive[successor] or len(pred[successor]) != 1:
            continue
        edges.append((task, successor))

    return tuple(edges)


def serial_components(instance: ProjectInstance) -> tuple[tuple[int, ...], ...]:
    edges = active_serial_edges(instance)
    if not edges:
        return tuple()

    neighbors: dict[int, set[int]] = {}
    for u, v in edges:
        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)

    components: list[tuple[int, ...]] = []
    visited: set[int] = set()
    for node in sorted(neighbors):
        if node in visited:
            continue
        stack = [node]
        comp: list[int] = []
        visited.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in neighbors.get(cur, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        components.append(tuple(sorted(comp)))

    return tuple(components)


def _serial_package_cost(workload: float, completion_time: float, params: SerialDPParams) -> float:
    return (
        params.fixed_cost
        + params.variable_scale * (workload**params.workload_concavity)
        + (workload**params.workload_convexity)
        + params.discount_penalty
        * workload
        * (1.0 - math.exp(-params.discount_rate * completion_time))
    )


def serial_dp_partition(
    instance: ProjectInstance, component: tuple[int, ...], params: SerialDPParams
) -> list[tuple[int, ...]]:


    n = len(component)
    if n <= 1:
        return [component]

    prefix_workload = [0.0]
    prefix_duration = [0.0]
    for task in component:
        prefix_workload.append(prefix_workload[-1] + instance.tasks[task].workload)
        prefix_duration.append(prefix_duration[-1] + instance.tasks[task].duration)

    dp = [float("inf")] * (n + 1)
    prev = [-1] * (n + 1)
    dp[0] = 0.0

    for end in range(1, n + 1):
        for start in range(end):
            workload = prefix_workload[end] - prefix_workload[start]
            duration = prefix_duration[end] - prefix_duration[start]
            candidate = dp[start] + _serial_package_cost(workload, duration, params)
            if candidate < dp[end]:
                dp[end] = candidate
                prev[end] = start

    packages: list[tuple[int, ...]] = []
    end = n
    while end > 0:
        start = prev[end]
        if start < 0:
            raise ValueError("Failed to recover serial DP partition")
        packages.append(tuple(component[start:end]))
        end = start
    packages.reverse()
    return packages


def serial_dp_initial_partition(
    instance: ProjectInstance, params: SerialDPParams | None = None
) -> Partition:
    params = params or SerialDPParams()
    covered: set[int] = set()
    packages: list[tuple[int, ...]] = []

    for component in serial_components(instance):
        component_packages = serial_dp_partition(instance, component, params)
        packages.extend(component_packages)
        covered.update(component)

    for task in range(instance.n_tasks):
        if task not in covered:
            packages.append((task,))

    partition = tuple(packages)
    if not partition_is_valid(instance, partition):
        return atomistic_partition(instance)
    return partition
