"""Fixed, semantics-blind graph registry and runner."""

from __future__ import annotations

import copy
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from torch import Tensor

from .compatibility import BehavioralDecision, validate_behavioral_decision
from .neural import ExecutionEnvelope, NeuralModule, TensorMap, assert_disjoint_states
from .telemetry import TelemetryCollector


@dataclass(frozen=True)
class NodeDefinition:
    node_id: str
    module_id: str
    module_version: str


@dataclass(frozen=True)
class Endpoint:
    node: str
    port: str


@dataclass(frozen=True)
class EdgeDefinition:
    source: Endpoint
    target: Endpoint


@dataclass(frozen=True)
class GraphOutput:
    source: Endpoint
    name: str


@dataclass(frozen=True)
class ProviderAdmission:
    node: str
    role: str
    consumer_node: str
    fixture_id: str
    fixture_digest: str
    consumer_artifact_digest: str
    thresholds_digest: str


@dataclass(frozen=True)
class GraphDefinition:
    graph_id: str
    graph_version: str
    execution_mode: str
    nodes: tuple[NodeDefinition, ...]
    edges: tuple[EdgeDefinition, ...]
    outputs: tuple[GraphOutput, ...]
    provider_admissions: tuple[ProviderAdmission, ...]

    @classmethod
    def from_path(cls, path: str | Path) -> "GraphDefinition":
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            schema_path = Path(__file__).resolve().parents[2] / "configs/graph.schema.json"
            with schema_path.open("r", encoding="utf-8") as handle:
                schema = json.load(handle)
            Draft202012Validator(schema).validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError) as error:
            raise ValueError("graph JSON violates the closed Draft 2020-12 v2 schema") from error
        return cls(
            graph_id=raw["graph_id"],
            graph_version=raw["graph_version"],
            execution_mode=raw["execution_mode"],
            nodes=tuple(NodeDefinition(**node) for node in raw["nodes"]),
            edges=tuple(
                EdgeDefinition(Endpoint(**edge["source"]), Endpoint(**edge["target"]))
                for edge in raw["edges"]
            ),
            outputs=tuple(
                GraphOutput(source=Endpoint(**output["source"]), name=output["name"])
                for output in raw["outputs"]
            ),
            provider_admissions=tuple(
                ProviderAdmission(**admission)
                for admission in raw["provider_admissions"]
            ),
        )

    def node_ids(self) -> set[str]:
        return {node.node_id for node in self.nodes}

    def topological_order(self) -> tuple[str, ...]:
        node_ids = self.node_ids()
        if len(node_ids) != len(self.nodes):
            raise ValueError("graph node IDs must be unique")
        incoming = {node_id: 0 for node_id in node_ids}
        outgoing: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            if edge.target.node not in node_ids:
                raise ValueError(f"unknown target node {edge.target.node}")
            if edge.source.node != "$input":
                if edge.source.node not in node_ids:
                    raise ValueError(f"unknown source node {edge.source.node}")
                incoming[edge.target.node] += 1
                outgoing[edge.source.node].append(edge.target.node)
        declared_order = {node.node_id: index for index, node in enumerate(self.nodes)}
        ready = deque(
            sorted(
                (node_id for node_id, degree in incoming.items() if degree == 0),
                key=declared_order.get,
            )
        )
        result: list[str] = []
        while ready:
            current = ready.popleft()
            result.append(current)
            for successor in sorted(outgoing[current], key=declared_order.get):
                incoming[successor] -= 1
                if incoming[successor] == 0:
                    ready.append(successor)
        if len(result) != len(node_ids):
            raise ValueError("graph must be acyclic")
        return tuple(result)


class ModuleRegistry:
    def __init__(self) -> None:
        self._factories: dict[tuple[str, str], Callable[[], NeuralModule]] = {}
        self._manifests: dict[tuple[str, str], dict | None] = {}

    def register(
        self,
        module_id: str,
        module_version: str,
        factory: Callable[[], NeuralModule],
        *,
        manifest: dict | None = None,
    ) -> None:
        key = (module_id, module_version)
        if key in self._factories:
            raise ValueError(f"duplicate registry entry: {key}")
        if manifest is not None and (
            manifest.get("module_id"), manifest.get("module_version")
        ) != key:
            raise ValueError("registry manifest identity does not match its exact key")
        self._factories[key] = factory
        self._manifests[key] = copy.deepcopy(manifest) if manifest is not None else None

    def create(self, module_id: str, module_version: str) -> NeuralModule:
        key = (module_id, module_version)
        try:
            return self._factories[key]()
        except KeyError as error:
            raise KeyError(f"unregistered module version: {key}") from error

    def manifest(self, module_id: str, module_version: str) -> dict:
        key = (module_id, module_version)
        try:
            manifest = self._manifests[key]
        except KeyError as error:
            raise KeyError(f"unregistered module version: {key}") from error
        if manifest is None:
            raise KeyError(f"no manifest bound to registry entry: {key}")
        return copy.deepcopy(manifest)

@dataclass(frozen=True)
class GraphRun:
    outputs: TensorMap
    states: dict[str, TensorMap]
    trace: tuple[str, ...]
    envelopes: dict[str, ExecutionEnvelope]


class GraphRunner:
    def __init__(
        self,
        definition: GraphDefinition,
        modules: Mapping[str, NeuralModule],
        *,
        telemetry: TelemetryCollector,
        behavioral_decisions: Mapping[str, BehavioralDecision],
        allow_negative_test: bool = False,
    ) -> None:
        self.definition = definition
        self.modules = dict(modules)
        self.telemetry = telemetry
        if definition.execution_mode == "negative_test" and not allow_negative_test:
            raise PermissionError("negative-test graph cannot execute as an admitted graph")
        if definition.execution_mode not in {"admitted", "negative_test"}:
            raise ValueError(f"unknown graph execution mode {definition.execution_mode}")
        expected_nodes = definition.node_ids()
        if set(self.modules) != expected_nodes:
            raise ValueError("module instances must exactly match graph nodes")
        for node in definition.nodes:
            module = self.modules[node.node_id]
            if (module.module_id, module.module_version) != (
                node.module_id,
                node.module_version,
            ):
                raise ValueError(f"registry binding mismatch at node {node.node_id}")
        self._validate_behavioral_admissions(behavioral_decisions)
        self.order = definition.topological_order()
        self._validate_ports()

    def _validate_behavioral_admissions(
        self, decisions: Mapping[str, BehavioralDecision]
    ) -> None:
        declared = {item.node: item for item in self.definition.provider_admissions}
        if len(declared) != len(self.definition.provider_admissions):
            raise ValueError("provider admission nodes must be unique")
        if set(decisions) != set(declared):
            raise PermissionError(
                "behavioral decisions must exactly cover declared provider nodes"
            )
        for node_id, admission in declared.items():
            if node_id not in self.modules:
                raise ValueError(f"provider admission names unknown node {node_id}")
            decision = decisions[node_id]
            validate_behavioral_decision(decision)
            module = self.modules[node_id]
            if (decision.module_id, decision.module_version) != (
                module.module_id,
                module.module_version,
            ):
                raise PermissionError("behavioral decision identity mismatch")
            if decision.role != admission.role:
                raise PermissionError("behavioral decision role mismatch")
            from .synthetic import module_digest

            if decision.provider_artifact_digest != module_digest(module):
                raise PermissionError("behavioral decision provider artifact mismatch")
            if admission.consumer_node not in self.modules:
                raise ValueError("provider admission names unknown consumer node")
            if not any(
                edge.source.node == node_id
                and edge.target.node == admission.consumer_node
                for edge in self.definition.edges
            ):
                raise ValueError("provider admission consumer is not connected by an edge")
            if decision.consumer_artifact_digest != module_digest(
                self.modules[admission.consumer_node]
            ):
                raise PermissionError("behavioral decision consumer artifact mismatch")
            if decision.fixture_id != admission.fixture_id:
                raise PermissionError("behavioral decision fixture mismatch")
            if decision.fixture_digest != admission.fixture_digest:
                raise PermissionError("behavioral decision fixture digest mismatch")
            if decision.consumer_artifact_digest != admission.consumer_artifact_digest:
                raise PermissionError("behavioral decision frozen consumer mismatch")
            if decision.thresholds_digest != admission.thresholds_digest:
                raise PermissionError("behavioral decision threshold set mismatch")
            if self.definition.execution_mode == "admitted" and not decision.admitted:
                raise PermissionError("behavioral gate rejected provider for admitted graph")
            if self.definition.execution_mode == "negative_test" and (
                decision.admitted or not decision.negative_control_pass
            ):
                raise PermissionError("negative-test provider lacks passing negative evidence")

    def _validate_ports(self) -> None:
        incoming_ports: dict[str, set[str]] = defaultdict(set)
        for edge in self.definition.edges:
            target_module = self.modules[edge.target.node]
            if edge.target.port not in target_module.input_ports:
                raise ValueError(f"unknown target port {edge.target.node}.{edge.target.port}")
            if edge.target.port in incoming_ports[edge.target.node]:
                raise ValueError(f"input port wired more than once: {edge.target}")
            incoming_ports[edge.target.node].add(edge.target.port)
            if edge.source.node != "$input":
                source_module = self.modules[edge.source.node]
                if edge.source.port not in source_module.output_ports:
                    raise ValueError(f"unknown source port {edge.source.node}.{edge.source.port}")
                source_contract = source_module.output_contracts[edge.source.port]
                target_contract = target_module.input_contracts[edge.target.port]
                if source_contract != target_contract:
                    raise ValueError(
                        f"interface contract mismatch on {edge.source.node}.{edge.source.port}"
                        f" -> {edge.target.node}.{edge.target.port}"
                    )
        for node_id, module in self.modules.items():
            if incoming_ports[node_id] != set(module.input_ports):
                raise ValueError(f"not all inputs are wired for {node_id}")
        for output in self.definition.outputs:
            if output.source.node not in self.modules:
                raise ValueError(f"unknown graph output node {output.source.node}")
            if output.source.port not in self.modules[output.source.node].output_ports:
                raise ValueError(f"unknown graph output port {output.source.port}")

    def run(
        self,
        external_inputs: Mapping[str, Tensor | ExecutionEnvelope],
        states: Mapping[str, Mapping[str, Tensor]],
        *,
        logical_step: int = 0,
    ) -> GraphRun:
        if set(states) != set(self.modules):
            raise ValueError("state bundle must exactly match graph nodes")
        assert_disjoint_states(states)
        produced: dict[str, dict[str, ExecutionEnvelope]] = {}
        next_states: dict[str, TensorMap] = {}
        trace: list[str] = []
        for node_id in self.order:
            module = self.modules[node_id]
            node_inputs: TensorMap = {}
            for edge in self.definition.edges:
                if edge.target.node != node_id:
                    continue
                if edge.source.node == "$input":
                    try:
                        value = external_inputs[edge.source.port]
                    except KeyError as error:
                        raise ValueError(f"missing external input {edge.source.port}") from error
                    if isinstance(value, ExecutionEnvelope):
                        expected_contract = module.input_contracts[edge.target.port]
                        if value.interface_contract != expected_contract:
                            raise ValueError(
                                f"external envelope contract mismatch for {edge.target.node}."
                                f"{edge.target.port}"
                            )
                        value = value.tensor
                else:
                    envelope = produced[edge.source.node][edge.source.port]
                    expected_contract = module.input_contracts[edge.target.port]
                    if envelope.interface_contract != expected_contract:
                        raise ValueError(
                            f"runtime envelope contract mismatch for {edge.target.node}."
                            f"{edge.target.port}"
                        )
                    value = envelope.tensor
                node_inputs[edge.target.port] = value
            module.validate_inputs(node_inputs)
            exemplar = next(iter(node_inputs.values()))
            module.validate_state(states[node_id], batch_size=exemplar.shape[0])
            started = self.telemetry.begin()
            result = module.forward_sequence(node_inputs, states[node_id])
            module.validate_outputs(
                result.tensors,
                result.state,
                batch_size=exemplar.shape[0],
                sequence_length=exemplar.shape[1],
            )
            produced[node_id] = {
                port: ExecutionEnvelope(
                    tensor=tensor,
                    interface_contract=module.output_contracts[port],
                    producer_instance=node_id,
                    producer_module_id=module.module_id,
                    producer_module_version=module.module_version,
                    output_port=port,
                    snapshot_id=self.telemetry.snapshot_id,
                    logical_step=logical_step,
                )
                for port, tensor in result.tensors.items()
            }
            next_states[node_id] = result.state
            self.telemetry.observe(
                node_id,
                module,
                started,
                node_inputs,
                result.tensors,
                result.state,
                logical_step=logical_step,
            )
            trace.append(node_id)
        assert_disjoint_states(next_states)
        graph_outputs = {
            output.name: produced[output.source.node][output.source.port].tensor
            for output in self.definition.outputs
        }
        graph_envelopes = {
            output.name: produced[output.source.node][output.source.port]
            for output in self.definition.outputs
        }
        return GraphRun(graph_outputs, next_states, tuple(trace), graph_envelopes)
