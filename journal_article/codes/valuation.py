from __future__ import annotations

import math
from dataclasses import dataclass

from .data import ProjectInstance
from .graph import Partition
from .schedule import package_schedule


@dataclass(frozen=True)
class PolicyParams:
    tax_rate: float = 0.5
    subsidy_rate: float = 0.5
    discount_rate: float = 0.0001
    tax_period: float = 30.0
    setup_cost: float = 20.0
    tax_convexity: float = 1.2
    subsidy_concavity: float = 0.8


@dataclass(frozen=True)
class ValuationResult:
    total_value: float
    makespan: float
    subsidy_pv: float
    tax_pv: float
    setup_pv: float
    package_durations: tuple[float, ...]
    package_start_times: tuple[float, ...]
    package_finish_times: tuple[float, ...]


def _filing_time(finish_time: float, tax_period: float) -> float:
    if tax_period <= 0:
        raise ValueError("tax_period must be positive")
    return math.ceil(finish_time / tax_period) * tax_period


def evaluate_partition(
    instance: ProjectInstance, partition: Partition, params: PolicyParams
) -> ValuationResult:
    durations, starts, finishes, makespan = package_schedule(instance, partition)
    subsidy_pv = 0.0
    tax_pv = 0.0
    setup_pv = 0.0

    for package, start, finish in zip(partition, starts, finishes):
        content = sum(instance.tasks[task].workload for task in package)

        subsidy_base = params.subsidy_rate * (content ** params.subsidy_concavity)
        subsidy_pv += subsidy_base * math.exp(-params.discount_rate * start)

        tax_base = params.tax_rate * (content ** params.tax_convexity)
        filing_time = _filing_time(finish, params.tax_period)
        tax_pv += tax_base * math.exp(-params.discount_rate * filing_time)

        setup_pv += params.setup_cost * math.exp(-params.discount_rate * start)

    total_value = subsidy_pv - tax_pv - setup_pv
    return ValuationResult(
        total_value=total_value,
        makespan=makespan,
        subsidy_pv=subsidy_pv,
        tax_pv=tax_pv,
        setup_pv=setup_pv,
        package_durations=durations,
        package_start_times=starts,
        package_finish_times=finishes,
    )
