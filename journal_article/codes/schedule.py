from __future__ import annotations

from collections import deque
from typing import Sequence

from .data import ProjectInstance
from .graph import Partition, package_successors, predecessors


def earliest_start_finish(
    durations: Sequence[float], successors: Sequence[Sequence[int]]
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    n = len(durations)
    pred = predecessors(successors)
    indegree = [len(items) for items in pred]
    queue = deque(i for i, degree in enumerate(indegree) if degree == 0)
    order: list[int] = []

    while queue:
        i = queue.popleft()
        order.append(i)
        for j in successors[i]:
            indegree[j] -= 1
            if indegree[j] == 0:
                queue.append(j)

    if len(order) != n:
        raise ValueError("Cannot schedule a cyclic graph")

    est = [0.0] * n
    eft = [0.0] * n
    for i in order:
        est[i] = max((eft[p] for p in pred[i]), default=0.0)
        eft[i] = est[i] + float(durations[i])

    return tuple(est), tuple(eft), max(eft, default=0.0)


def package_duration(instance: ProjectInstance, package: Sequence[int]) -> float:
    tasks = tuple(package)
    if len(tasks) == 1:
        return instance.tasks[tasks[0]].duration

    local_index = {task: idx for idx, task in enumerate(tasks)}
    local_successors: list[list[int]] = [[] for _ in tasks]
    for task in tasks:
        i = local_index[task]
        for successor in instance.successors[task]:
            if successor in local_index:
                local_successors[i].append(local_index[successor])

    durations = [instance.tasks[task].duration for task in tasks]
    _, _, makespan = earliest_start_finish(durations, local_successors)
    return makespan


def package_schedule(
    instance: ProjectInstance, partition: Partition
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], float]:
    durations = tuple(package_duration(instance, package) for package in partition)
    successors = package_successors(instance, partition)
    est, eft, makespan = earliest_start_finish(durations, successors)
    return durations, est, eft, makespan
