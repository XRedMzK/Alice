"""Small neural-module interface used by the bounded foundation runner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


TensorMap = dict[str, Tensor]


@dataclass(frozen=True)
class ExecutionEnvelope:
    """Structural edge metadata; it carries no semantic routing instruction."""

    tensor: Tensor
    interface_contract: str
    producer_instance: str
    producer_module_id: str
    producer_module_version: str
    output_port: str
    snapshot_id: str
    logical_step: int


@dataclass(frozen=True)
class ModuleOutput:
    tensors: TensorMap
    state: TensorMap


class NeuralModule(nn.Module, ABC):
    """A neural population with owned parameters and an explicit private state."""

    module_id: str
    module_version: str = "1.0.0"
    compatibility_version: str
    state_schema_version: str = "state/v1"
    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]
    input_contracts: Mapping[str, str]
    output_contracts: Mapping[str, str]
    input_features: Mapping[str, int]
    output_features: Mapping[str, int]
    state_features: Mapping[str, int]

    @abstractmethod
    def initial_state(
        self, batch_size: int, *, device: torch.device | str = "cpu"
    ) -> TensorMap:
        raise NotImplementedError

    @abstractmethod
    def forward_sequence(
        self, inputs: Mapping[str, Tensor], state: Mapping[str, Tensor]
    ) -> ModuleOutput:
        raise NotImplementedError

    def validate_inputs(self, inputs: Mapping[str, Tensor]) -> None:
        if set(inputs) != set(self.input_ports):
            raise ValueError(
                f"{self.module_id} expected ports {self.input_ports}, got {tuple(inputs)}"
            )
        for name, tensor in inputs.items():
            if tensor.dtype != torch.float32:
                raise TypeError(f"{self.module_id}.{name} requires float32")
            if tensor.ndim != 3:
                raise ValueError(f"{self.module_id}.{name} requires [batch,time,feature]")
            if tensor.shape[-1] != self.input_features[name]:
                raise ValueError(
                    f"{self.module_id}.{name} requires feature dimension "
                    f"{self.input_features[name]}"
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{self.module_id}.{name} requires contiguous tensors")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{self.module_id}.{name} contains non-finite values")

    def validate_state(
        self, state: Mapping[str, Tensor], *, batch_size: int
    ) -> None:
        if set(state) != set(self.state_features):
            raise ValueError(f"{self.module_id} returned or received an invalid state set")
        for name, tensor in state.items():
            if tensor.dtype != torch.float32:
                raise TypeError(f"{self.module_id}.{name} state requires float32")
            if tensor.ndim != 2 or tensor.shape != (batch_size, self.state_features[name]):
                raise ValueError(f"{self.module_id}.{name} state has an invalid shape")
            if not tensor.is_contiguous():
                raise ValueError(f"{self.module_id}.{name} state must be contiguous")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{self.module_id}.{name} state contains non-finite values")

    def validate_outputs(
        self,
        outputs: Mapping[str, Tensor],
        state: Mapping[str, Tensor],
        *,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        if set(outputs) != set(self.output_ports):
            raise ValueError(f"{self.module_id} returned an invalid output port set")
        for name, tensor in outputs.items():
            if tensor.dtype != torch.float32:
                raise TypeError(f"{self.module_id}.{name} output requires float32")
            if tensor.ndim != 3 or tensor.shape != (
                batch_size,
                sequence_length,
                self.output_features[name],
            ):
                raise ValueError(f"{self.module_id}.{name} output has an invalid shape")
            if not tensor.is_contiguous():
                raise ValueError(f"{self.module_id}.{name} output must be contiguous")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{self.module_id}.{name} output contains non-finite values")
        self.validate_state(state, batch_size=batch_size)

    @staticmethod
    def clone_state(state: Mapping[str, Tensor]) -> TensorMap:
        return {name: tensor.detach().clone() for name, tensor in state.items()}

    @staticmethod
    def parameter_bytes(module: nn.Module) -> int:
        return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def state_bytes(state: Mapping[str, Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state.values())


def storage_addresses(tensors: Mapping[str, Tensor]) -> set[int]:
    return {tensor.untyped_storage().data_ptr() for tensor in tensors.values()}


def assert_disjoint_states(states: Mapping[str, Mapping[str, Tensor]]) -> None:
    owners: dict[int, str] = {}
    for module_instance, state in states.items():
        for address in storage_addresses(state):
            previous = owners.get(address)
            if previous is not None:
                raise ValueError(
                    f"mutable state storage is shared by {previous} and {module_instance}"
                )
            owners[address] = module_instance
