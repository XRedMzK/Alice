from __future__ import annotations

import copy
from dataclasses import dataclass

import pytest
import torch

from nmcm.foundation import FoundationBundle, build_trained_bundle
from nmcm.synthetic import (
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
class LocalAResult:
    teacher: Teacher
    splits: dict[str, SyntheticSplit]
    module: RecurrentA1 | GatedRecurrentA2
    report: TrainingReport
    initial_digest: str
    loaded_module_ids: tuple[str, ...]


@dataclass(frozen=True)
class LocalA1BResult:
    teacher: Teacher
    splits: dict[str, SyntheticSplit]
    a1: RecurrentA1
    b: RecurrentB
    a3: IncompatibleA3
    c: RecurrentC
    training_reports: dict[str, TrainingReport]
    frozen_b_digest: str
    replacement_b_optimizer_steps: int
    loaded_module_ids: tuple[str, ...]


@pytest.fixture(scope="session")
def a1_local_result() -> LocalAResult:
    configure_determinism(401)
    teacher = build_teacher()
    splits = build_splits(teacher)
    module = RecurrentA1(teacher)
    initial_digest = module_digest(module)
    report = train_a(module, splits["train"], seed=4401)
    return LocalAResult(
        teacher, splits, module, report, initial_digest, ("synthetic.a1",)
    )


@pytest.fixture(scope="session")
def a2_local_result() -> LocalAResult:
    configure_determinism(401)
    teacher = build_teacher()
    splits = build_splits(teacher)
    module = GatedRecurrentA2(teacher)
    initial_digest = module_digest(module)
    report = train_a(module, splits["train"], seed=4402)
    return LocalAResult(
        teacher, splits, module, report, initial_digest, ("synthetic.a2",)
    )


@pytest.fixture(scope="session")
def a1b_local_result(a1_local_result: LocalAResult) -> LocalA1BResult:
    """Build only the declared A1 dependency and B/C probes; never load/train A2."""

    a1 = copy.deepcopy(a1_local_result.module)
    b = RecurrentB(a1_local_result.teacher)
    with torch.no_grad():
        train_latent, _ = run_a(a1, a1_local_result.splits["train"])
    b_report = train_b(
        b,
        train_latent,
        a1_local_result.splits["train"].labels,
        a1_local_result.splits["train"].task_scores,
    )
    a3 = IncompatibleA3(a1)
    c = RecurrentC()
    for module in (a1, a3, c):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return LocalA1BResult(
        teacher=a1_local_result.teacher,
        splits=a1_local_result.splits,
        a1=a1,
        b=b,
        a3=a3,
        c=c,
        training_reports={"a1": a1_local_result.report, "b": b_report},
        frozen_b_digest=module_digest(b),
        replacement_b_optimizer_steps=0,
        loaded_module_ids=("synthetic.a1", "synthetic.b", "synthetic.a3", "synthetic.c"),
    )


@pytest.fixture(scope="session")
def foundation_bundle() -> FoundationBundle:
    return build_trained_bundle()
