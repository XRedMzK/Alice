from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from jsonschema import Draft202012Validator

from nmcm.checkpoint import (
    CheckpointError,
    load_module_shard,
    load_whole_checkpoint,
    save_module_shard,
    save_whole_checkpoint,
    sha256_file,
)
from nmcm.compatibility import evaluate_provider
from nmcm.concurrency import run_two_candidate_processes
from nmcm.contracts import ManifestError, load_manifest, validate_manifest
from nmcm.foundation import FoundationBundle, build_trained_bundle
from nmcm.graph import GraphDefinition, GraphRunner, ModuleRegistry
from nmcm.neural import assert_disjoint_states, storage_addresses
from nmcm.synthetic import (
    GatedRecurrentA2,
    IncompatibleA3,
    RecurrentA1,
    RecurrentB,
    RecurrentC,
    accuracy,
    latent_distribution,
    latent_metrics,
    module_digest,
    run_a,
    run_b,
    tensor_map_equal,
)
from nmcm.telemetry import TelemetryCollector
from conftest import LocalA1BResult, LocalAResult


ROOT = Path(__file__).resolve().parents[1]
MODULE_SCHEMA = ROOT / "configs/module_manifest.schema.json"
BASELINE = ROOT / "configs/foundation0_cpu.json"
THRESHOLDS = json.loads(BASELINE.read_text(encoding="utf-8"))["thresholds"]


def _collector(enabled: bool, suffix: str) -> TelemetryCollector:
    return TelemetryCollector(
        enabled=enabled,
        experiment_id=f"pytest-{suffix}",
        snapshot_id=f"snapshot-{suffix}",
    )


def _fresh_like(name: str, bundle: FoundationBundle):
    if name == "a1":
        return RecurrentA1(bundle.teacher)
    if name == "a2":
        return GatedRecurrentA2(bundle.teacher)
    if name == "a3":
        return IncompatibleA3(RecurrentA1(bundle.teacher))
    if name == "b":
        return RecurrentB()
    if name == "c":
        return RecurrentC()
    raise KeyError(name)


def _module_input(name: str, bundle: FoundationBundle) -> tuple[str, torch.Tensor]:
    heldout = bundle.splits["heldout"]
    with torch.no_grad():
        latent, _ = run_a(bundle.a1, heldout)
    if name in {"a1", "a2", "a3"}:
        return "stimulus", heldout.stimulus[:64]
    return "dense_latent", latent[:64]


def _a1_admission(bundle: FoundationBundle | LocalA1BResult):
    heldout = bundle.splits["heldout"]
    decision = evaluate_provider(
        module=bundle.a1,
        role="baseline",
        stimulus=heldout.stimulus,
        canonical_latent=heldout.latent,
        consumer=bundle.b,
        reference_logits=None,
        labels=heldout.labels,
        thresholds=THRESHOLDS,
        fixture_id="foundation0-heldout-seed4203/v1",
    )
    assert decision.admitted
    return {"a": decision}


@pytest.mark.cpu
def test_01_manifest_and_schema_validation() -> None:
    schema = json.loads(MODULE_SCHEMA.read_text(encoding="utf-8"))
    manifests = {
        path.stem: load_manifest(path, MODULE_SCHEMA)
        for path in sorted((ROOT / "configs/modules").glob("*.json"))
    }
    assert set(manifests) == {"a1", "a2", "a3", "b", "c"}
    from nmcm.synthetic import build_teacher

    teacher = build_teacher()
    a1 = RecurrentA1(teacher)
    modules = {
        "a1": a1,
        "a2": GatedRecurrentA2(teacher),
        "a3": IncompatibleA3(a1),
        "b": RecurrentB(teacher),
        "c": RecurrentC(),
    }
    for name, module in modules.items():
        assert manifests[name]["module_id"] == module.module_id
        assert manifests[name]["trainable_parameters"]["estimated_count"] == sum(
            parameter.numel() for parameter in module.parameters()
        )

    invalid = copy.deepcopy(manifests["a1"])
    invalid["output_latent_schema"] = {}
    with pytest.raises(ManifestError):
        validate_manifest(invalid, schema)

    for schema_path in (
        ROOT / "configs/graph.schema.json",
        ROOT / "configs/module_shard.schema.json",
        ROOT / "configs/telemetry_event.schema.json",
        ROOT / "configs/whole_checkpoint.schema.json",
    ):
        Draft202012Validator.check_schema(
            json.loads(schema_path.read_text(encoding="utf-8"))
        )


@pytest.mark.cpu
def test_02_registry_graph_and_semantics_blind_routing(tmp_path: Path) -> None:
    from nmcm.synthetic import build_teacher, build_splits

    teacher = build_teacher()
    splits = build_splits(teacher)
    a1_template = RecurrentA1(teacher, noise=0.0)
    b_template = RecurrentB(teacher, noise=0.0)
    registry = ModuleRegistry()
    manifest_a1 = load_manifest(ROOT / "configs/modules/a1.json", MODULE_SCHEMA)
    manifest_b = load_manifest(ROOT / "configs/modules/b.json", MODULE_SCHEMA)
    registry.register(
        "synthetic.a1",
        "1.0.0",
        lambda: copy.deepcopy(a1_template),
        manifest=manifest_a1,
    )
    registry.register(
        "synthetic.b",
        "1.1.0",
        lambda: copy.deepcopy(b_template),
        manifest=manifest_b,
    )
    assert registry.create("synthetic.a1", "1.0.0").module_id == "synthetic.a1"
    assert registry.manifest("synthetic.a1", "1.0.0")["module_id"] == "synthetic.a1"
    with pytest.raises(KeyError):
        registry.create("synthetic.a1", "9.9.9")

    graph_schema = json.loads((ROOT / "configs/graph.schema.json").read_text(encoding="utf-8"))
    for path in sorted((ROOT / "configs/graphs").glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        Draft202012Validator(graph_schema).validate(raw)
        GraphDefinition.from_path(path).topological_order()

    valid_graph = json.loads(
        (ROOT / "configs/graphs/foundation_a1b.json").read_text(encoding="utf-8")
    )
    old_schema_graph = copy.deepcopy(valid_graph)
    old_schema_graph["graph_schema"] = "nmcm.graph/v1"
    extra_field_graph = copy.deepcopy(valid_graph)
    extra_field_graph["undeclared"] = True
    for name, invalid_graph in (
        ("old-schema", old_schema_graph),
        ("extra-field", extra_field_graph),
    ):
        invalid_path = tmp_path / f"{name}.json"
        invalid_path.write_text(json.dumps(invalid_graph), encoding="utf-8")
        with pytest.raises(ValueError, match="closed Draft 2020-12 v2 schema"):
            GraphDefinition.from_path(invalid_path)

    definition = GraphDefinition.from_path(ROOT / "configs/graphs/foundation_a1b.json")
    stimulus = splits["heldout"].stimulus[:32]
    admitted = evaluate_provider(
        module=a1_template,
        role="baseline",
        stimulus=splits["heldout"].stimulus,
        canonical_latent=splits["heldout"].latent,
        consumer=b_template,
        reference_logits=None,
        labels=splits["heldout"].labels,
        thresholds=THRESHOLDS,
        fixture_id="foundation0-heldout-seed4203/v1",
    )
    assert admitted.admitted
    probe_admission = replace(
        definition.provider_admissions[0],
        fixture_digest=admitted.fixture_digest,
        consumer_artifact_digest=admitted.consumer_artifact_digest,
        thresholds_digest=admitted.thresholds_digest,
    )
    definition = replace(definition, provider_admissions=(probe_admission,))
    traces = []
    for index, value in enumerate((stimulus, -stimulus)):
        a = copy.deepcopy(a1_template)
        b = copy.deepcopy(b_template)
        runner = GraphRunner(
            definition,
            {"a": a, "b": b},
            telemetry=_collector(False, str(index)),
            behavioral_decisions={"a": admitted},
        )
        states = {"a": a.initial_state(32), "b": b.initial_state(32)}
        graph_run = runner.run({"x": value.contiguous()}, states)
        traces.append(graph_run.trace)
        assert graph_run.envelopes["dense_latent"].interface_contract == "dense_latent/v1"
    assert traces[0] == traces[1] == ("a", "b")

    bad_b = copy.deepcopy(b_template)
    bad_b.input_contracts = {"dense_latent": "rotated_basis/v1"}
    with pytest.raises(ValueError, match="interface contract mismatch"):
        GraphRunner(
            definition,
            {"a": copy.deepcopy(a1_template), "b": bad_b},
            telemetry=_collector(False, "bad-contract"),
            behavioral_decisions={"a": admitted},
        )
    with pytest.raises(ValueError, match="feature dimension"):
        a1_template.validate_inputs(
            {"stimulus": torch.zeros(2, 3, 15, dtype=torch.float32)}
        )


@pytest.mark.cpu
def test_03_private_state_ownership_and_isolation(
) -> None:
    from nmcm.synthetic import build_teacher, build_splits

    teacher = build_teacher()
    splits = build_splits(teacher)
    a1 = RecurrentA1(teacher)
    b = RecurrentB(teacher)
    c = RecurrentC()
    batch = 16
    states = {
        "a": a1.initial_state(batch),
        "b": b.initial_state(batch),
        "c": c.initial_state(batch),
    }
    assert_disjoint_states(states)
    all_addresses = [storage_addresses(state) for state in states.values()]
    assert all_addresses[0].isdisjoint(all_addresses[1])
    assert all_addresses[0].isdisjoint(all_addresses[2])
    b_before = b.clone_state(states["b"])
    a_before = a1.clone_state(states["a"])
    result = a1.forward_sequence(
        {"stimulus": splits["heldout"].stimulus[:batch]}, states["a"]
    )
    assert tensor_map_equal(states["a"], a_before)
    assert tensor_map_equal(states["b"], b_before)
    assert not tensor_map_equal(result.state, a_before)
    assert storage_addresses(result.state).isdisjoint(storage_addresses(states["a"]))


@pytest.mark.cpu
def test_04_a1_local_learning(a1_local_result: LocalAResult) -> None:
    report = a1_local_result.report
    assert report.steps > 0
    assert report.final_loss < report.initial_loss
    assert module_digest(a1_local_result.module) != a1_local_result.initial_digest
    with torch.no_grad():
        predicted, _ = run_a(a1_local_result.module, a1_local_result.splits["heldout"])
    metrics = latent_metrics(predicted, a1_local_result.splits["heldout"].latent)
    assert metrics["mean_cosine"] >= THRESHOLDS["a1_a2_mean_cosine_min"]
    assert metrics["normalized_rmse"] <= THRESHOLDS["a1_a2_normalized_rmse_max"]
    assert a1_local_result.loaded_module_ids == ("synthetic.a1",)


@pytest.mark.cpu
def test_05_a2_local_learning_and_interface(a2_local_result: LocalAResult) -> None:
    report = a2_local_result.report
    assert report.steps > 0
    assert report.final_loss < report.initial_loss
    assert module_digest(a2_local_result.module) != a2_local_result.initial_digest
    with torch.no_grad():
        a2_latent, _ = run_a(a2_local_result.module, a2_local_result.splits["heldout"])
    metrics = latent_metrics(a2_latent, a2_local_result.splits["heldout"].latent)
    assert metrics["mean_cosine"] >= THRESHOLDS["a1_a2_mean_cosine_min"]
    assert metrics["normalized_rmse"] <= THRESHOLDS["a1_a2_normalized_rmse_max"]
    a1_contract = RecurrentA1(a2_local_result.teacher)
    assert a1_contract.input_contracts == a2_local_result.module.input_contracts
    assert a1_contract.output_contracts == a2_local_result.module.output_contracts
    assert type(a1_contract) is not type(a2_local_result.module)
    assert a2_local_result.loaded_module_ids == ("synthetic.a2",)


@pytest.mark.cpu
@pytest.mark.integration
def test_06_a1_to_b_task_and_freeze(a1b_local_result: LocalA1BResult) -> None:
    foundation_bundle = a1b_local_result
    heldout = foundation_bundle.splits["heldout"]
    with torch.no_grad():
        latent, _ = run_a(foundation_bundle.a1, heldout)
        logits, _ = run_b(foundation_bundle.b, latent)
    assert accuracy(logits, heldout.labels) >= THRESHOLDS["b_a1_accuracy_min"]
    assert foundation_bundle.training_reports["b"].final_loss < foundation_bundle.training_reports["b"].initial_loss
    assert all(not parameter.requires_grad for parameter in foundation_bundle.b.parameters())
    assert module_digest(foundation_bundle.b) == foundation_bundle.frozen_b_digest
    assert "synthetic.a2" not in foundation_bundle.loaded_module_ids
    lesioned_b = copy.deepcopy(foundation_bundle.b)
    with torch.no_grad():
        lesioned_b.input_projection.weight.zero_()
        lesioned_b.recurrence.weight.zero_()
        lesioned_logits, _ = run_b(lesioned_b, latent)
        prefix = foundation_bundle.b.forward_sequence(
            {"dense_latent": latent[:64, :4].contiguous()},
            foundation_bundle.b.initial_state(64),
        )
        carried = foundation_bundle.b.forward_sequence(
            {"dense_latent": latent[:64, 4:].contiguous()}, prefix.state
        )
        reset = foundation_bundle.b.forward_sequence(
            {"dense_latent": latent[:64, 4:].contiguous()},
            foundation_bundle.b.initial_state(64),
        )
    assert accuracy(logits, heldout.labels) - accuracy(
        lesioned_logits, heldout.labels
    ) >= THRESHOLDS["targeted_ablation_accuracy_drop_min"]
    assert not torch.equal(carried.tensors["task_logits"], reset.tensors["task_logits"])


@pytest.mark.cpu
@pytest.mark.integration
def test_07_frozen_b_a1_to_a2_replacement(foundation_bundle: FoundationBundle) -> None:
    heldout = foundation_bundle.splits["heldout"]
    digest_before = module_digest(foundation_bundle.b)
    with torch.no_grad():
        a1_latent, _ = run_a(foundation_bundle.a1, heldout)
        a2_latent, _ = run_a(foundation_bundle.a2, heldout)
        a1_logits, _ = run_b(foundation_bundle.b, a1_latent)
        a2_logits, _ = run_b(foundation_bundle.b, a2_latent)
    accuracy_a1 = accuracy(a1_logits, heldout.labels)
    accuracy_a2 = accuracy(a2_logits, heldout.labels)
    assert accuracy_a2 >= THRESHOLDS["b_a2_accuracy_min"]
    assert abs(accuracy_a1 - accuracy_a2) <= THRESHOLDS[
        "b_a1_a2_accuracy_absolute_difference_max"
    ]
    assert foundation_bundle.replacement_b_optimizer_steps == 0
    assert module_digest(foundation_bundle.b) == digest_before == foundation_bundle.frozen_b_digest
    decision = evaluate_provider(
        module=foundation_bundle.a2,
        role="replacement",
        stimulus=heldout.stimulus,
        canonical_latent=heldout.latent,
        consumer=foundation_bundle.b,
        reference_logits=a1_logits,
        labels=heldout.labels,
        thresholds=THRESHOLDS,
        fixture_id="foundation0-heldout-seed4203/v1",
    )
    assert decision.admitted and all(decision.criteria.values())
    a2_graph = GraphDefinition.from_path(ROOT / "configs/graphs/foundation_a2b.json")
    GraphRunner(
        a2_graph,
        {"a": copy.deepcopy(foundation_bundle.a2), "b": copy.deepcopy(foundation_bundle.b)},
        telemetry=_collector(False, "a2-admitted"),
        behavioral_decisions={"a": decision},
    )
    changed_b = copy.deepcopy(foundation_bundle.b)
    with torch.no_grad():
        changed_b.state_score.weight.zero_()
    with pytest.raises(PermissionError, match="consumer artifact mismatch"):
        GraphRunner(
            a2_graph,
            {"a": copy.deepcopy(foundation_bundle.a2), "b": changed_b},
            telemetry=_collector(False, "a2-stale-consumer"),
            behavioral_decisions={"a": decision},
        )


@pytest.mark.cpu
@pytest.mark.integration
def test_08_a3_structural_positive_behavioral_negative(
    a1b_local_result: LocalA1BResult,
) -> None:
    foundation_bundle = a1b_local_result
    heldout = foundation_bundle.splits["heldout"]
    with torch.no_grad():
        a1_latent, _ = run_a(foundation_bundle.a1, heldout)
        a3_latent, _ = run_a(foundation_bundle.a3, heldout)
        a1_logits, _ = run_b(foundation_bundle.b, a1_latent)
        a3_logits, _ = run_b(foundation_bundle.b, a3_latent)
    assert torch.equal(a3_latent, -a1_latent)
    assert a3_latent.shape == a1_latent.shape
    assert a3_latent.dtype == a1_latent.dtype == torch.float32
    distribution_a1 = latent_distribution(a1_latent)
    distribution_a3 = latent_distribution(a3_latent)
    assert distribution_a1 == distribution_a3
    behavior = latent_metrics(a3_latent, heldout.latent)
    assert (
        behavior["mean_cosine"] < THRESHOLDS["a1_a2_mean_cosine_min"]
        or behavior["normalized_rmse"] > THRESHOLDS["a1_a2_normalized_rmse_max"]
    )
    accuracy_drop = accuracy(a1_logits, heldout.labels) - accuracy(a3_logits, heldout.labels)
    assert accuracy_drop >= THRESHOLDS["a3_b_accuracy_drop_min"]
    decision = evaluate_provider(
        module=foundation_bundle.a3,
        role="negative_control",
        stimulus=heldout.stimulus,
        canonical_latent=heldout.latent,
        consumer=foundation_bundle.b,
        reference_logits=a1_logits,
        labels=heldout.labels,
        thresholds=THRESHOLDS,
        fixture_id="foundation0-heldout-seed4203/v1",
    )
    assert not decision.admitted
    assert decision.negative_control_pass
    assert "synthetic.a2" not in foundation_bundle.loaded_module_ids
    negative_graph = GraphDefinition.from_path(ROOT / "configs/graphs/foundation_a3b.json")
    with pytest.raises(PermissionError, match="negative-test graph"):
        GraphRunner(
            negative_graph,
            {"a": copy.deepcopy(foundation_bundle.a3), "b": copy.deepcopy(foundation_bundle.b)},
            telemetry=_collector(False, "negative-not-admitted"),
            behavioral_decisions={"a": decision},
        )
    copied_as_admitted = replace(negative_graph, execution_mode="admitted")
    with pytest.raises(PermissionError, match="behavioral gate rejected"):
        GraphRunner(
            copied_as_admitted,
            {"a": copy.deepcopy(foundation_bundle.a3), "b": copy.deepcopy(foundation_bundle.b)},
            telemetry=_collector(False, "negative-mode-spoof"),
            behavioral_decisions={"a": decision},
        )
    manifest_a1 = load_manifest(ROOT / "configs/modules/a1.json", MODULE_SCHEMA)
    manifest_a3 = load_manifest(ROOT / "configs/modules/a3.json", MODULE_SCHEMA)
    assert manifest_a1["output_latent_schema"] == manifest_a3["output_latent_schema"]
    assert (
        manifest_a1["output_latent_schema"]["dense_latent"]["representation_contract_id"]
        == manifest_a3["output_latent_schema"]["dense_latent"]["representation_contract_id"]
    )


@pytest.mark.cpu
@pytest.mark.integration
def test_09_targeted_ablation_with_non_target_guard(
    foundation_bundle: FoundationBundle,
) -> None:
    heldout = foundation_bundle.splits["heldout"]
    b_digest_before = module_digest(foundation_bundle.b)
    with torch.no_grad():
        latent, a_state = run_a(foundation_bundle.a1, heldout)
        latent_a2, _ = run_a(foundation_bundle.a2, heldout)
        intact_logits, intact_state = run_b(foundation_bundle.b, latent)
        intact_a2_logits, _ = run_b(foundation_bundle.b, latent_a2)
        sham_logits, sham_state = run_b(foundation_bundle.b, latent.clone())
        ablated_logits, _ = run_b(foundation_bundle.b, torch.zeros_like(latent))
        ablated_a2_logits, _ = run_b(
            foundation_bundle.b, torch.zeros_like(latent_a2)
        )
        c_intact = foundation_bundle.c.forward_sequence(
            {"dense_latent": latent}, foundation_bundle.c.initial_state(latent.shape[0])
        )
        c_guard = foundation_bundle.c.forward_sequence(
            {"dense_latent": latent.clone()}, foundation_bundle.c.initial_state(latent.shape[0])
        )
    intact_accuracy = accuracy(intact_logits, heldout.labels)
    ablated_accuracy = accuracy(ablated_logits, heldout.labels)
    assert intact_accuracy - ablated_accuracy >= THRESHOLDS[
        "targeted_ablation_accuracy_drop_min"
    ]
    assert accuracy(intact_a2_logits, heldout.labels) - accuracy(
        ablated_a2_logits, heldout.labels
    ) >= THRESHOLDS["targeted_ablation_accuracy_drop_min"]
    assert ablated_accuracy == 0.5
    assert torch.equal(intact_logits, sham_logits)
    assert tensor_map_equal(intact_state, sham_state)
    assert torch.equal(c_intact.tensors["side_latent"], c_guard.tensors["side_latent"])
    assert tensor_map_equal(c_intact.state, c_guard.state)
    assert module_digest(foundation_bundle.b) == b_digest_before
    assert torch.isfinite(a_state["hidden"]).all().item()


@pytest.mark.cpu
@pytest.mark.integration
def test_10_add_remove_c_without_hidden_mutation(
    a1b_local_result: LocalA1BResult,
) -> None:
    foundation_bundle = a1b_local_result
    batch = 64
    stimulus = foundation_bundle.splits["heldout"].stimulus[:batch]
    definition_ab = GraphDefinition.from_path(ROOT / "configs/graphs/foundation_a1b.json")
    definition_abc = GraphDefinition.from_path(ROOT / "configs/graphs/foundation_a1bc.json")
    modules_ab = {"a": copy.deepcopy(foundation_bundle.a1), "b": copy.deepcopy(foundation_bundle.b)}
    modules_abc = {
        "a": copy.deepcopy(foundation_bundle.a1),
        "b": copy.deepcopy(foundation_bundle.b),
        "c": copy.deepcopy(foundation_bundle.c),
    }
    digests_before = {
        key: module_digest(module) for key, module in modules_abc.items() if key in {"a", "b"}
    }
    admission = _a1_admission(foundation_bundle)
    runner_ab = GraphRunner(
        definition_ab,
        modules_ab,
        telemetry=_collector(False, "ab"),
        behavioral_decisions=admission,
    )
    runner_abc = GraphRunner(
        definition_abc,
        modules_abc,
        telemetry=_collector(False, "abc"),
        behavioral_decisions=admission,
    )
    states_ab = {key: module.initial_state(batch) for key, module in modules_ab.items()}
    states_abc = {key: module.initial_state(batch) for key, module in modules_abc.items()}
    with torch.no_grad():
        run_ab = runner_ab.run({"x": stimulus}, states_ab)
        run_abc = runner_abc.run({"x": stimulus}, states_abc)
    assert torch.equal(run_ab.outputs["dense_latent"], run_abc.outputs["dense_latent"])
    assert torch.equal(run_ab.outputs["task_logits"], run_abc.outputs["task_logits"])
    assert tensor_map_equal(run_ab.states["a"], run_abc.states["a"])
    assert tensor_map_equal(run_ab.states["b"], run_abc.states["b"])
    assert digests_before == {
        key: module_digest(module) for key, module in modules_abc.items() if key in {"a", "b"}
    }
    assert digests_before == {key: module_digest(modules_ab[key]) for key in ("a", "b")}
    assert not torch.equal(
        run_abc.outputs["side_latent"], torch.zeros_like(run_abc.outputs["side_latent"])
    )

    latent = run_abc.outputs["dense_latent"]
    c = copy.deepcopy(foundation_bundle.c)
    with torch.no_grad():
        prefix = c.forward_sequence(
            {"dense_latent": latent[:, :4].contiguous()}, c.initial_state(batch)
        )
        carried = c.forward_sequence(
            {"dense_latent": latent[:, 4:].contiguous()}, prefix.state
        )
        reset = c.forward_sequence(
            {"dense_latent": latent[:, 4:].contiguous()}, c.initial_state(batch)
        )
        lesioned = c.forward_sequence(
            {"dense_latent": torch.zeros_like(latent)}, c.initial_state(batch)
        )
    assert not torch.equal(carried.tensors["side_latent"], reset.tensors["side_latent"])
    assert not torch.equal(run_abc.outputs["side_latent"], lesioned.tensors["side_latent"])
    assert "synthetic.a2" not in foundation_bundle.loaded_module_ids


@pytest.mark.cpu
def test_11_independent_module_shards_and_standalone_restore(
    foundation_bundle: FoundationBundle, tmp_path: Path
) -> None:
    modules = {
        "a1": foundation_bundle.a1,
        "a2": foundation_bundle.a2,
        "a3": foundation_bundle.a3,
        "b": foundation_bundle.b,
        "c": foundation_bundle.c,
    }
    records = {}
    tensor_paths = set()
    for name, module in modules.items():
        port, value = _module_input(name, foundation_bundle)
        prefix_value = value[:, :4].contiguous()
        suffix_value = value[:, 4:].contiguous()
        with torch.no_grad():
            prefix = module.forward_sequence(
                {port: prefix_value}, module.initial_state(value.shape[0])
            )
        records[name] = save_module_shard(
            tmp_path,
            name,
            module,
            prefix.state,
            snapshot_id="module-shards-1",
            logical_step=4,
        )
        tensor_paths.add(records[name]["tensor_file"])
        with torch.no_grad():
            expected = module.forward_sequence({port: suffix_value}, prefix.state)
        restored_module = _fresh_like(name, foundation_bundle)
        restored_state = load_module_shard(tmp_path, records[name], restored_module)
        with torch.no_grad():
            actual = restored_module.forward_sequence({port: suffix_value}, restored_state)
        assert tensor_map_equal(expected.tensors, actual.tensors)
        assert tensor_map_equal(expected.state, actual.state)
        assert records[name]["tensor_sha256"] == sha256_file(
            tmp_path / records[name]["tensor_file"]
        )
    assert len(tensor_paths) == len(modules)


@pytest.mark.cpu
@pytest.mark.integration
def test_12_whole_checkpoint_exact_continuation_and_fail_closed(
    foundation_bundle: FoundationBundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = 64
    stimulus = foundation_bundle.splits["heldout"].stimulus[:batch]
    definition = GraphDefinition.from_path(ROOT / "configs/graphs/foundation_a1b.json")
    modules = {"a": copy.deepcopy(foundation_bundle.a1), "b": copy.deepcopy(foundation_bundle.b)}
    admission = _a1_admission(foundation_bundle)
    runner = GraphRunner(
        definition,
        modules,
        telemetry=_collector(False, "whole-before"),
        behavioral_decisions=admission,
    )
    initial = {key: module.initial_state(batch) for key, module in modules.items()}
    with torch.no_grad():
        prefix = runner.run({"x": stimulus[:, :4].contiguous()}, initial)
    checkpoint_root = tmp_path / "published"
    checkpoint = save_whole_checkpoint(
        checkpoint_root,
        graph_id=definition.graph_id,
        graph_version=definition.graph_version,
        modules=modules,
        states=prefix.states,
        snapshot_id="whole-1",
        logical_step=4,
    )
    with torch.no_grad():
        expected = runner.run({"x": stimulus[:, 4:].contiguous()}, prefix.states)

    fresh_modules = {"a": RecurrentA1(foundation_bundle.teacher), "b": RecurrentB()}
    restored_states, manifest = load_whole_checkpoint(
        checkpoint,
        graph_id=definition.graph_id,
        graph_version=definition.graph_version,
        modules=fresh_modules,
    )
    restored_runner = GraphRunner(
        definition,
        fresh_modules,
        telemetry=_collector(False, "whole-after"),
        behavioral_decisions=admission,
    )
    with torch.no_grad():
        actual = restored_runner.run({"x": stimulus[:, 4:].contiguous()}, restored_states)
    assert tensor_map_equal(expected.outputs, actual.outputs)
    assert all(tensor_map_equal(expected.states[key], actual.states[key]) for key in expected.states)
    checkpoint_schema = json.loads(
        (ROOT / "configs/whole_checkpoint.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(checkpoint_schema).validate(manifest)
    assert sum(record["tensor_bytes"] for record in manifest["modules"]) > 0

    # Publishing a new snapshot never overwrites shards named by the old one.
    import nmcm.checkpoint as checkpoint_module

    published_bytes = checkpoint.read_bytes()
    original_save = checkpoint_module.save_module_shard
    calls = 0

    def fail_second_shard(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected interrupted save")
        return original_save(*args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(checkpoint_module, "save_module_shard", fail_second_shard)
        with pytest.raises(RuntimeError, match="injected interrupted save"):
            save_whole_checkpoint(
                checkpoint_root,
                graph_id=definition.graph_id,
                graph_version=definition.graph_version,
                modules=modules,
                states=prefix.states,
                snapshot_id="whole-2",
                logical_step=4,
            )
    assert checkpoint.read_bytes() == published_bytes
    load_whole_checkpoint(
        checkpoint,
        graph_id=definition.graph_id,
        graph_version=definition.graph_version,
        modules={"a": RecurrentA1(foundation_bundle.teacher), "b": RecurrentB()},
    )

    # Save rejects Draft-invalid identities, cross-module batch mismatch, and NaN
    # before publishing a whole-system manifest.
    with pytest.raises(CheckpointError, match="graph ID"):
        save_whole_checkpoint(
            tmp_path / "invalid-identity",
            graph_id=42,  # type: ignore[arg-type]
            graph_version=definition.graph_version,
            modules=modules,
            states=prefix.states,
            snapshot_id="invalid-identity",
            logical_step=4,
        )
    mixed_batch_states = {
        "a": {"hidden": prefix.states["a"]["hidden"][:2].clone()},
        "b": {"hidden": prefix.states["b"]["hidden"][:3].clone()},
    }
    with pytest.raises(CheckpointError, match="batch size"):
        save_whole_checkpoint(
            tmp_path / "invalid-batch",
            graph_id=definition.graph_id,
            graph_version=definition.graph_version,
            modules=modules,
            states=mixed_batch_states,
            snapshot_id="invalid-batch",
            logical_step=4,
        )
    non_finite_states = copy.deepcopy(prefix.states)
    non_finite_states["a"]["hidden"][0, 0] = float("nan")
    with pytest.raises(CheckpointError, match="non-finite"):
        save_whole_checkpoint(
            tmp_path / "invalid-finite",
            graph_id=definition.graph_id,
            graph_version=definition.graph_version,
            modules=modules,
            states=non_finite_states,
            snapshot_id="invalid-finite",
            logical_step=4,
        )

    # Draft 2020-12 additionalProperties=false is enforced on restore itself.
    extra_root = tmp_path / "extra-field"
    shutil.copytree(checkpoint_root, extra_root)
    extra_checkpoint = extra_root / "checkpoint.json"
    extra_manifest = json.loads(extra_checkpoint.read_text(encoding="utf-8"))
    extra_manifest["undeclared"] = True
    extra_checkpoint.write_text(
        json.dumps(extra_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="Draft 2020-12"):
        load_whole_checkpoint(
            extra_checkpoint,
            graph_id=definition.graph_id,
            graph_version=definition.graph_version,
            modules={"a": RecurrentA1(foundation_bundle.teacher), "b": RecurrentB()},
        )

    # A self-consistently re-hashed sidecar from another snapshot is rejected.
    mixed_root = tmp_path / "mixed"
    shutil.copytree(checkpoint_root, mixed_root)
    mixed_checkpoint = mixed_root / "checkpoint.json"
    mixed_manifest = json.loads(mixed_checkpoint.read_text(encoding="utf-8"))
    mixed_record = mixed_manifest["modules"][0]
    mixed_metadata_path = mixed_root / mixed_record["metadata_file"]
    mixed_metadata = json.loads(mixed_metadata_path.read_text(encoding="utf-8"))
    mixed_metadata["snapshot_id"] = "other-snapshot"
    mixed_metadata_path.write_text(
        json.dumps(mixed_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    mixed_record["metadata_sha256"] = sha256_file(mixed_metadata_path)
    mixed_checkpoint.write_text(
        json.dumps(mixed_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="mixed snapshot"):
        load_whole_checkpoint(
            mixed_checkpoint,
            graph_id=definition.graph_id,
            graph_version=definition.graph_version,
            modules={"a": RecurrentA1(foundation_bundle.teacher), "b": RecurrentB()},
        )

    # Even re-hashed tensor/metadata edits cannot violate the module state schema.
    from safetensors.torch import load_file, save_file

    shape_root = tmp_path / "shape"
    shutil.copytree(checkpoint_root, shape_root)
    shape_checkpoint = shape_root / "checkpoint.json"
    shape_manifest = json.loads(shape_checkpoint.read_text(encoding="utf-8"))
    shape_record = shape_manifest["modules"][0]
    shape_metadata_path = shape_root / shape_record["metadata_file"]
    shape_metadata = json.loads(shape_metadata_path.read_text(encoding="utf-8"))
    shape_tensor_path = shape_root / shape_record["tensor_file"]
    shape_tensors = load_file(str(shape_tensor_path), device="cpu")
    shape_tensors["private_state.hidden"] = torch.zeros(1, dtype=torch.float64)
    shape_temp = shape_tensor_path.with_suffix(".rewrite")
    save_file(shape_tensors, str(shape_temp))
    shape_temp.replace(shape_tensor_path)
    shape_metadata["tensor_sha256"] = sha256_file(shape_tensor_path)
    shape_metadata["tensor_bytes"] = shape_tensor_path.stat().st_size
    shape_metadata["private_state_spec"]["hidden"] = {
        "shape": [1],
        "dtype": "torch.float64",
    }
    shape_metadata_path.write_text(
        json.dumps(shape_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shape_record["tensor_sha256"] = shape_metadata["tensor_sha256"]
    shape_record["tensor_bytes"] = shape_metadata["tensor_bytes"]
    shape_record["metadata_sha256"] = sha256_file(shape_metadata_path)
    shape_checkpoint.write_text(
        json.dumps(shape_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(CheckpointError, match="Draft 2020-12|private-state schema mismatch"):
        load_whole_checkpoint(
            shape_checkpoint,
            graph_id=definition.graph_id,
            graph_version=definition.graph_version,
            modules={"a": RecurrentA1(foundation_bundle.teacher), "b": RecurrentB()},
        )

    # Corrupting the second shard cannot partially mutate the first active module.
    corrupt_record = manifest["modules"][1]
    corrupt_path = checkpoint_root / corrupt_record["tensor_file"]
    corrupt = bytearray(corrupt_path.read_bytes())
    corrupt[-1] ^= 0x01
    corrupt_path.write_bytes(bytes(corrupt))
    active_modules = {"a": RecurrentA1(foundation_bundle.teacher), "b": RecurrentB()}
    active_before = {key: module_digest(module) for key, module in active_modules.items()}
    with pytest.raises(CheckpointError, match="digest mismatch"):
        load_whole_checkpoint(
            checkpoint,
            graph_id=definition.graph_id,
            graph_version=definition.graph_version,
            modules=active_modules,
        )
    assert active_before == {key: module_digest(module) for key, module in active_modules.items()}


@pytest.mark.cpu
def test_13_telemetry_noninterference_and_schema(
    foundation_bundle: FoundationBundle, tmp_path: Path
) -> None:
    batch = 64
    stimulus = foundation_bundle.splits["heldout"].stimulus[:batch]
    definition = GraphDefinition.from_path(ROOT / "configs/graphs/foundation_a1b.json")
    off_modules = {"a": copy.deepcopy(foundation_bundle.a1), "b": copy.deepcopy(foundation_bundle.b)}
    on_modules = {"a": copy.deepcopy(foundation_bundle.a1), "b": copy.deepcopy(foundation_bundle.b)}
    off_collector = TelemetryCollector(
        enabled=False,
        experiment_id="pytest-telemetry-off",
        snapshot_id="snapshot-telemetry",
        candidate_id="synthetic.a1",
        local_status="PASS",
    )
    on_collector = TelemetryCollector(
        enabled=True,
        experiment_id="pytest-telemetry-on",
        snapshot_id="snapshot-telemetry",
        candidate_id="synthetic.a1",
        local_status="PASS",
    )
    admission = _a1_admission(foundation_bundle)
    off_runner = GraphRunner(
        definition,
        off_modules,
        telemetry=off_collector,
        behavioral_decisions=admission,
    )
    on_runner = GraphRunner(
        definition,
        on_modules,
        telemetry=on_collector,
        behavioral_decisions=admission,
    )
    off_states = {key: module.initial_state(batch) for key, module in off_modules.items()}
    on_states = {key: module.initial_state(batch) for key, module in on_modules.items()}
    torch.manual_seed(991)
    off_rng_before = torch.get_rng_state().clone()
    with torch.no_grad():
        off = off_runner.run({"x": stimulus}, off_states)
    off_rng_after = torch.get_rng_state().clone()
    torch.manual_seed(991)
    on_rng_before = torch.get_rng_state().clone()
    with torch.no_grad():
        on = on_runner.run({"x": stimulus}, on_states)
    on_rng_after = torch.get_rng_state().clone()
    assert tensor_map_equal(off.outputs, on.outputs)
    assert all(tensor_map_equal(off.states[key], on.states[key]) for key in off.states)
    assert {key: module_digest(value) for key, value in off_modules.items()} == {
        key: module_digest(value) for key, value in on_modules.items()
    }
    assert torch.equal(off_rng_before, off_rng_after)
    assert torch.equal(on_rng_before, on_rng_after)
    assert torch.equal(off_rng_after, on_rng_after)
    assert off_collector.events == []
    assert len(on_collector.events) == 2
    telemetry_path = tmp_path / "telemetry.jsonl"
    on_collector.write_jsonl(telemetry_path)
    schema = json.loads(
        (ROOT / "configs/telemetry_event.schema.json").read_text(encoding="utf-8")
    )
    lines = telemetry_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    for line in lines:
        event = json.loads(line)
        Draft202012Validator(schema).validate(event)
        assert event["gpu_time_ms"] is None
        assert event["active_vram_mib"] is None
        assert event["gpu_status"] == "NOT_RUN"
        assert event["local_status"] == "PASS"
        assert event["candidate_id"] == "synthetic.a1"
        assert event["cpu_time_ms"] >= 0
        assert event["process_peak_memory_bytes"] > 0
    assert [json.loads(line) for line in lines] == [asdict(event) for event in on_collector.events]


@pytest.mark.cpu
def test_14_full_serial_training_determinism(foundation_bundle: FoundationBundle) -> None:
    replay = build_trained_bundle()
    for name in ("a1", "a2", "a3", "b", "c"):
        assert module_digest(getattr(foundation_bundle, name)) == module_digest(getattr(replay, name))
    for name in ("a1", "a2", "b"):
        first = foundation_bundle.training_reports[name]
        second = replay.training_reports[name]
        assert first.initial_loss == second.initial_loss
        assert first.final_loss == second.final_loss
        assert first.steps == second.steps
    for split_name in ("train", "dev", "heldout"):
        first = foundation_bundle.splits[split_name]
        second = replay.splits[split_name]
        assert torch.equal(first.stimulus, second.stimulus)
        assert torch.equal(first.latent, second.latent)
        assert torch.equal(first.labels, second.labels)
        assert torch.equal(first.task_scores, second.task_scores)


@pytest.mark.cpu
@pytest.mark.integration
def test_15_two_concurrent_isolated_candidate_bundles(
    foundation_bundle: FoundationBundle, tmp_path: Path
) -> None:
    result = run_two_candidate_processes(
        foundation_bundle.a1,
        foundation_bundle.a2,
        foundation_bundle.b,
        foundation_bundle.splits["heldout"].stimulus[:32],
        tmp_path / "workers",
        repetitions=20,
    )
    assert result["distinct_pids"]
    assert result["distinct_artifact_roots"]
    assert result["payloads_immutable"]
    assert result["overlap_seconds"] > 0
    assert result["max_live"] == 2
    assert all(result["serial_parity"].values())
    assert all(Path(record["artifact_root"], "worker_result.json").is_file() for record in result["workers"].values())
