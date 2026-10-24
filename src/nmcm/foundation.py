"""Construction of the frozen deterministic FOUNDATION_0 CPU experiment bundle."""

from __future__ import annotations

import resource
import time
from dataclasses import dataclass

import torch

from .synthetic import (
    GatedRecurrentA2,
    IncompatibleA3,
    RecurrentA1,
    RecurrentB,
    RecurrentC,
    SyntheticSplit,
    Teacher,
    TrainingReport,
    build_splits,
    build_teacher,
    configure_determinism,
    module_digest,
    run_a,
    train_a,
    train_b,
)


@dataclass(frozen=True)
class FoundationBundle:
    teacher: Teacher
    splits: dict[str, SyntheticSplit]
    a1: RecurrentA1
    a2: GatedRecurrentA2
    a3: IncompatibleA3
    b: RecurrentB
    c: RecurrentC
    training_reports: dict[str, TrainingReport]
    initial_parameter_digests: dict[str, str]
    frozen_b_digest: str
    replacement_b_optimizer_steps: int
    total_wall_time_seconds: float
    total_cpu_time_seconds: float
    process_peak_memory_bytes: int


def build_trained_bundle() -> FoundationBundle:
    configure_determinism(401)
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    teacher = build_teacher()
    splits = build_splits(teacher)
    a1 = RecurrentA1(teacher)
    a2 = GatedRecurrentA2(teacher)
    b = RecurrentB(teacher)
    c = RecurrentC()
    initial_digests = {
        "a1": module_digest(a1),
        "a2": module_digest(a2),
        "b": module_digest(b),
        "c": module_digest(c),
    }
    reports = {
        "a1": train_a(a1, splits["train"], seed=4401),
        "a2": train_a(a2, splits["train"], seed=4402),
    }
    with torch.no_grad():
        train_latent, _ = run_a(a1, splits["train"])
    reports["b"] = train_b(
        b,
        train_latent.detach(),
        splits["train"].labels,
        splits["train"].task_scores,
    )
    frozen_b_digest = module_digest(b)
    a3 = IncompatibleA3(a1)
    for parameter in a1.parameters():
        parameter.requires_grad_(False)
    for parameter in a2.parameters():
        parameter.requires_grad_(False)
    for parameter in a3.parameters():
        parameter.requires_grad_(False)
    for parameter in c.parameters():
        parameter.requires_grad_(False)
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return FoundationBundle(
        teacher=teacher,
        splits=splits,
        a1=a1,
        a2=a2,
        a3=a3,
        b=b,
        c=c,
        training_reports=reports,
        initial_parameter_digests=initial_digests,
        frozen_b_digest=frozen_b_digest,
        replacement_b_optimizer_steps=0,
        total_wall_time_seconds=time.perf_counter() - wall_started,
        total_cpu_time_seconds=time.process_time() - cpu_started,
        process_peak_memory_bytes=int(peak_kib * 1024),
    )
