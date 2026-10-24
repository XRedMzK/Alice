"""Deterministic synthetic neural populations for FOUNDATION_0 CPU evidence."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .neural import ModuleOutput, NeuralModule, TensorMap


INPUT_DIM = 16
LATENT_DIM = 32
TASK_CLASSES = 2
SEQUENCE_LENGTH = 8
TASK_MARGIN = 0.15


@dataclass(frozen=True)
class Teacher:
    projection: Tensor
    recurrence: Tensor
    task_projection: Tensor
    task_recurrence: Tensor
    task_readout: Tensor


@dataclass(frozen=True)
class SyntheticSplit:
    split: str
    seed: int
    stimulus: Tensor
    latent: Tensor
    labels: Tensor
    task_scores: Tensor


@dataclass(frozen=True)
class TrainingReport:
    module_id: str
    seed: int
    optimizer: str
    steps: int
    initial_loss: float
    final_loss: float
    wall_time_seconds: float
    cpu_time_seconds: float


def configure_determinism(seed: int = 401) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def build_teacher(seed: int = 4101) -> Teacher:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    projection = torch.randn(LATENT_DIM, INPUT_DIM, generator=generator) / math.sqrt(INPUT_DIM)
    recurrence = torch.randn(LATENT_DIM, LATENT_DIM, generator=generator) / math.sqrt(LATENT_DIM)
    spectral_radius = torch.linalg.eigvals(recurrence.double()).abs().max().real.float()
    recurrence = recurrence * (0.45 / spectral_radius)
    task_projection = (
        torch.randn(LATENT_DIM, LATENT_DIM, generator=generator) / math.sqrt(LATENT_DIM)
    )
    task_recurrence = (
        torch.randn(LATENT_DIM, LATENT_DIM, generator=generator) / math.sqrt(LATENT_DIM)
    )
    task_radius = torch.linalg.eigvals(task_recurrence.double()).abs().max().real.float()
    task_recurrence = task_recurrence * (0.35 / task_radius)
    task_readout = torch.randn(LATENT_DIM, generator=generator)
    task_readout = task_readout / task_readout.norm()
    return Teacher(
        projection.float(),
        recurrence.float(),
        task_projection.float(),
        task_recurrence.float(),
        task_readout.float(),
    )


def teacher_latent(stimulus: Tensor, teacher: Teacher) -> Tensor:
    batch_size, steps, _ = stimulus.shape
    state = torch.zeros(batch_size, LATENT_DIM, dtype=torch.float32)
    outputs: list[Tensor] = []
    for step in range(steps):
        state = torch.tanh(
            F.linear(stimulus[:, step], teacher.projection)
            + F.linear(state, teacher.recurrence)
        )
        outputs.append(state)
    return torch.stack(outputs, dim=1).contiguous()


def teacher_task_scores(latent: Tensor, teacher: Teacher) -> Tensor:
    state = torch.zeros(latent.shape[0], LATENT_DIM, dtype=torch.float32)
    for step in range(latent.shape[1]):
        state = torch.tanh(
            F.linear(latent[:, step], teacher.task_projection)
            + F.linear(state, teacher.task_recurrence)
        )
    return state @ teacher.task_readout


def build_split(
    split: str,
    *,
    seed: int,
    samples: int,
    teacher: Teacher,
    sequence_length: int = SEQUENCE_LENGTH,
) -> SyntheticSplit:
    if samples % 2:
        raise ValueError("synthetic split size must be even for antipodal balancing")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    required = samples // 2
    accepted: list[Tensor] = []
    accepted_count = 0
    while accepted_count < required:
        candidates = torch.randn(
            required,
            sequence_length,
            INPUT_DIM,
            generator=generator,
            dtype=torch.float32,
        )
        candidate_latent = teacher_latent(candidates, teacher)
        candidate_scores = teacher_task_scores(candidate_latent, teacher)
        selected = candidates[candidate_scores.abs() >= TASK_MARGIN]
        if selected.numel():
            take = min(required - accepted_count, selected.shape[0])
            accepted.append(selected[:take])
            accepted_count += take
    half = torch.cat(accepted, dim=0)
    stimulus = torch.cat((half, -half), dim=0)
    order = torch.randperm(samples, generator=generator)
    stimulus = stimulus[order].contiguous()
    latent = teacher_latent(stimulus, teacher)
    score = teacher_task_scores(latent, teacher)
    labels = (score >= 0.0).long()
    return SyntheticSplit(split, seed, stimulus, latent, labels, score)


def build_splits(teacher: Teacher) -> dict[str, SyntheticSplit]:
    return {
        "train": build_split("train", seed=4201, samples=4096, teacher=teacher),
        "dev": build_split("dev", seed=4202, samples=512, teacher=teacher),
        "heldout": build_split("heldout", seed=4203, samples=1024, teacher=teacher),
    }


class RecurrentA1(NeuralModule):
    module_id = "synthetic.a1"
    compatibility_version = "dense_latent/v1"
    input_ports = ("stimulus",)
    output_ports = ("dense_latent",)
    input_contracts = {"stimulus": "synthetic_stimulus/v1"}
    output_contracts = {"dense_latent": "dense_latent/v1"}
    input_features = {"stimulus": INPUT_DIM}
    output_features = {"dense_latent": LATENT_DIM}
    state_features = {"hidden": LATENT_DIM}

    def __init__(self, teacher: Teacher, *, seed: int = 4301, noise: float = 0.08) -> None:
        super().__init__()
        self.input_projection = nn.Linear(INPUT_DIM, LATENT_DIM)
        self.recurrence = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.input_projection.weight.copy_(
                teacher.projection
                + noise * torch.randn(teacher.projection.shape, generator=generator)
            )
            self.input_projection.bias.copy_(
                noise * torch.randn(LATENT_DIM, generator=generator)
            )
            self.recurrence.weight.copy_(
                teacher.recurrence
                + noise * torch.randn(teacher.recurrence.shape, generator=generator)
            )

    def initial_state(
        self, batch_size: int, *, device: torch.device | str = "cpu"
    ) -> TensorMap:
        return {"hidden": torch.zeros(batch_size, LATENT_DIM, device=device)}

    def forward_sequence(
        self, inputs: Mapping[str, Tensor], state: Mapping[str, Tensor]
    ) -> ModuleOutput:
        stimulus = inputs["stimulus"]
        hidden = state["hidden"]
        outputs: list[Tensor] = []
        for step in range(stimulus.shape[1]):
            hidden = torch.tanh(
                self.input_projection(stimulus[:, step]) + self.recurrence(hidden)
            )
            outputs.append(hidden)
        return ModuleOutput(
            {"dense_latent": torch.stack(outputs, dim=1).contiguous()},
            {"hidden": hidden},
        )


class GatedRecurrentA2(NeuralModule):
    module_id = "synthetic.a2"
    compatibility_version = "dense_latent/v1"
    input_ports = ("stimulus",)
    output_ports = ("dense_latent",)
    input_contracts = {"stimulus": "synthetic_stimulus/v1"}
    output_contracts = {"dense_latent": "dense_latent/v1"}
    input_features = {"stimulus": INPUT_DIM}
    output_features = {"dense_latent": LATENT_DIM}
    state_features = {"hidden": LATENT_DIM}

    def __init__(self, teacher: Teacher, *, seed: int = 4302, noise: float = 0.08) -> None:
        super().__init__()
        self.candidate_input = nn.Linear(INPUT_DIM, LATENT_DIM)
        self.candidate_recurrence = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)
        self.update_input = nn.Linear(INPUT_DIM, LATENT_DIM)
        self.update_recurrence = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)
        self.output_projection = nn.Linear(LATENT_DIM, LATENT_DIM)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            self.candidate_input.weight.copy_(
                teacher.projection
                + noise * torch.randn(teacher.projection.shape, generator=generator)
            )
            self.candidate_input.bias.copy_(
                noise * torch.randn(LATENT_DIM, generator=generator)
            )
            self.candidate_recurrence.weight.copy_(
                teacher.recurrence
                + noise * torch.randn(teacher.recurrence.shape, generator=generator)
            )
            self.update_input.weight.zero_()
            self.update_input.bias.fill_(3.0)
            self.update_recurrence.weight.zero_()
            self.output_projection.weight.copy_(torch.eye(LATENT_DIM))
            self.output_projection.bias.zero_()

    def initial_state(
        self, batch_size: int, *, device: torch.device | str = "cpu"
    ) -> TensorMap:
        return {"hidden": torch.zeros(batch_size, LATENT_DIM, device=device)}

    def forward_sequence(
        self, inputs: Mapping[str, Tensor], state: Mapping[str, Tensor]
    ) -> ModuleOutput:
        stimulus = inputs["stimulus"]
        hidden = state["hidden"]
        outputs: list[Tensor] = []
        for step in range(stimulus.shape[1]):
            current = stimulus[:, step]
            candidate = torch.tanh(
                self.candidate_input(current) + self.candidate_recurrence(hidden)
            )
            update = torch.sigmoid(
                self.update_input(current) + self.update_recurrence(hidden)
            )
            hidden = update * candidate + (1.0 - update) * hidden
            outputs.append(self.output_projection(hidden))
        return ModuleOutput(
            {"dense_latent": torch.stack(outputs, dim=1).contiguous()},
            {"hidden": hidden},
        )


class IncompatibleA3(NeuralModule):
    """A structurally plausible negative provider with a wrong latent basis."""

    module_id = "synthetic.a3"
    compatibility_version = "dense_latent/v1"
    input_ports = ("stimulus",)
    output_ports = ("dense_latent",)
    input_contracts = {"stimulus": "synthetic_stimulus/v1"}
    output_contracts = {"dense_latent": "dense_latent/v1"}
    input_features = {"stimulus": INPUT_DIM}
    output_features = {"dense_latent": LATENT_DIM}
    state_features = {"hidden": LATENT_DIM}

    def __init__(self, source: RecurrentA1) -> None:
        super().__init__()
        self.input_projection = nn.Linear(INPUT_DIM, LATENT_DIM)
        self.recurrence = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)
        with torch.no_grad():
            self.input_projection.load_state_dict(source.input_projection.state_dict())
            self.recurrence.load_state_dict(source.recurrence.state_dict())
        self.register_buffer("orthogonal_sign", torch.tensor(-1.0))

    def initial_state(
        self, batch_size: int, *, device: torch.device | str = "cpu"
    ) -> TensorMap:
        return {"hidden": torch.zeros(batch_size, LATENT_DIM, device=device)}

    def forward_sequence(
        self, inputs: Mapping[str, Tensor], state: Mapping[str, Tensor]
    ) -> ModuleOutput:
        stimulus = inputs["stimulus"]
        hidden = state["hidden"]
        outputs: list[Tensor] = []
        for step in range(stimulus.shape[1]):
            hidden = torch.tanh(
                self.input_projection(stimulus[:, step]) + self.recurrence(hidden)
            )
            outputs.append(hidden * self.orthogonal_sign)
        return ModuleOutput(
            {"dense_latent": torch.stack(outputs, dim=1).contiguous()},
            {"hidden": hidden},
        )


class RecurrentB(NeuralModule):
    module_id = "synthetic.b"
    module_version = "1.1.0"
    compatibility_version = "task_logits/v1"
    input_ports = ("dense_latent",)
    output_ports = ("task_logits",)
    input_contracts = {"dense_latent": "dense_latent/v1"}
    output_contracts = {"task_logits": "task_logits/v1"}
    input_features = {"dense_latent": LATENT_DIM}
    output_features = {"task_logits": 2}
    state_features = {"hidden": LATENT_DIM}

    def __init__(
        self, teacher: Teacher | None = None, *, seed: int = 4303, noise: float = 0.08
    ) -> None:
        super().__init__()
        forked_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.input_projection = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)
        self.recurrence = nn.Linear(LATENT_DIM, LATENT_DIM, bias=False)
        self.state_score = nn.Linear(LATENT_DIM, 1, bias=False)
        torch.random.set_rng_state(forked_state)
        with torch.no_grad():
            if teacher is None:
                self.recurrence.weight.mul_(0.2)
            else:
                generator = torch.Generator(device="cpu").manual_seed(seed)
                self.input_projection.weight.copy_(
                    teacher.task_projection
                    + noise * torch.randn(teacher.task_projection.shape, generator=generator)
                )
                self.recurrence.weight.copy_(
                    teacher.task_recurrence
                    + noise * torch.randn(teacher.task_recurrence.shape, generator=generator)
                )
                self.state_score.weight.copy_(
                    teacher.task_readout.reshape(1, -1)
                    + noise * torch.randn((1, LATENT_DIM), generator=generator)
                )

    def initial_state(
        self, batch_size: int, *, device: torch.device | str = "cpu"
    ) -> TensorMap:
        return {"hidden": torch.zeros(batch_size, LATENT_DIM, device=device)}

    def forward_sequence(
        self, inputs: Mapping[str, Tensor], state: Mapping[str, Tensor]
    ) -> ModuleOutput:
        latent = inputs["dense_latent"]
        hidden = state["hidden"]
        logits: list[Tensor] = []
        for step in range(latent.shape[1]):
            hidden = torch.tanh(
                self.input_projection(latent[:, step]) + self.recurrence(hidden)
            )
            score = self.state_score(hidden)
            logits.append(torch.cat((-score, score), dim=-1))
        return ModuleOutput(
            {"task_logits": torch.stack(logits, dim=1).contiguous()},
            {"hidden": hidden},
        )


class RecurrentC(NeuralModule):
    module_id = "synthetic.c"
    compatibility_version = "side_latent/v1"
    input_ports = ("dense_latent",)
    output_ports = ("side_latent",)
    input_contracts = {"dense_latent": "dense_latent/v1"}
    output_contracts = {"side_latent": "side_latent/v1"}
    input_features = {"dense_latent": LATENT_DIM}
    output_features = {"side_latent": 16}
    state_features = {"hidden": 16}

    def __init__(self, *, seed: int = 4304) -> None:
        super().__init__()
        forked_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        self.input_projection = nn.Linear(LATENT_DIM, 16)
        self.recurrence = nn.Linear(16, 16, bias=False)
        torch.random.set_rng_state(forked_state)
        with torch.no_grad():
            self.recurrence.weight.mul_(0.2)

    def initial_state(
        self, batch_size: int, *, device: torch.device | str = "cpu"
    ) -> TensorMap:
        return {"hidden": torch.zeros(batch_size, 16, device=device)}

    def forward_sequence(
        self, inputs: Mapping[str, Tensor], state: Mapping[str, Tensor]
    ) -> ModuleOutput:
        latent = inputs["dense_latent"]
        hidden = state["hidden"]
        outputs: list[Tensor] = []
        for step in range(latent.shape[1]):
            hidden = torch.tanh(
                self.input_projection(latent[:, step]) + self.recurrence(hidden)
            )
            outputs.append(hidden)
        return ModuleOutput(
            {"side_latent": torch.stack(outputs, dim=1).contiguous()},
            {"hidden": hidden},
        )


def run_a(module: NeuralModule, split: SyntheticSplit) -> tuple[Tensor, TensorMap]:
    output = module.forward_sequence(
        {"stimulus": split.stimulus}, module.initial_state(split.stimulus.shape[0])
    )
    return output.tensors["dense_latent"], output.state


def run_b(module: RecurrentB, latent: Tensor) -> tuple[Tensor, TensorMap]:
    output = module.forward_sequence(
        {"dense_latent": latent}, module.initial_state(latent.shape[0])
    )
    return output.tensors["task_logits"], output.state


def train_a(
    module: NeuralModule,
    train: SyntheticSplit,
    *,
    seed: int,
    steps: int = 180,
    learning_rate: float = 0.012,
) -> TrainingReport:
    import time

    configure_determinism(seed)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial, _ = run_a(module, train)
        initial_loss = F.mse_loss(initial, train.latent).item()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    module.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        predicted, _ = run_a(module, train)
        loss = F.mse_loss(predicted, train.latent)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
        optimizer.step()
    wall_time = time.perf_counter() - wall_started
    cpu_time = time.process_time() - cpu_started
    module.eval()
    with torch.no_grad():
        final, _ = run_a(module, train)
        final_loss = F.mse_loss(final, train.latent).item()
    return TrainingReport(
        module.module_id,
        seed,
        "Adam",
        steps,
        initial_loss,
        final_loss,
        wall_time,
        cpu_time,
    )


def train_b(
    module: RecurrentB,
    train_latent: Tensor,
    labels: Tensor,
    target_scores: Tensor,
    *,
    seed: int = 4403,
    steps: int = 240,
    learning_rate: float = 0.015,
) -> TrainingReport:
    import time

    configure_determinism(seed)
    optimizer = torch.optim.Adam(module.parameters(), lr=learning_rate)
    with torch.no_grad():
        initial_logits, _ = run_b(module, train_latent)
        initial_loss = (
            F.cross_entropy(initial_logits[:, -1], labels)
            + F.mse_loss(initial_logits[:, -1, 1], target_scores)
        ).item()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    module.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = run_b(module, train_latent)
        loss = F.cross_entropy(logits[:, -1], labels) + F.mse_loss(
            logits[:, -1, 1], target_scores
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
        optimizer.step()
    wall_time = time.perf_counter() - wall_started
    cpu_time = time.process_time() - cpu_started
    module.eval()
    with torch.no_grad():
        final_logits, _ = run_b(module, train_latent)
        final_loss = (
            F.cross_entropy(final_logits[:, -1], labels)
            + F.mse_loss(final_logits[:, -1, 1], target_scores)
        ).item()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return TrainingReport(
        module.module_id,
        seed,
        "Adam",
        steps,
        initial_loss,
        final_loss,
        wall_time,
        cpu_time,
    )


def latent_metrics(predicted: Tensor, reference: Tensor) -> dict[str, float]:
    cosine = F.cosine_similarity(predicted, reference, dim=-1).mean().item()
    rmse = F.mse_loss(predicted, reference).sqrt()
    reference_rms = reference.square().mean().sqrt().clamp_min(1e-12)
    return {
        "mean_cosine": float(cosine),
        "normalized_rmse": float((rmse / reference_rms).item()),
    }


def latent_distribution(latent: Tensor) -> dict[str, float | int | bool]:
    flat = latent.reshape(-1, latent.shape[-1]).double()
    centered = flat - flat.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, flat.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-15)
    nonzero = probabilities > 0
    effective_rank = torch.exp(
        -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
    )
    variances = centered.square().mean(dim=0)
    variance_share = variances / variances.sum().clamp_min(1e-15)
    return {
        "finite": bool(torch.isfinite(latent).all().item()),
        "density": float((latent.abs() > 1e-6).float().mean().item()),
        "effective_rank": float(effective_rank.item()),
        "max_coordinate_variance_share": float(variance_share.max().item()),
    }


def accuracy(logits: Tensor, labels: Tensor) -> float:
    return float((logits[:, -1].argmax(dim=-1) == labels).float().mean().item())


def tensor_map_equal(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[key], right[key]) for key in left)


def module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(bytes(contiguous.reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


def state_digest(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(bytes(contiguous.reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()
