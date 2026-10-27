"""Static hidden-edge indexing preserves the native graph and execution."""
from types import SimpleNamespace as NS

import pytest

import native_toy
from heterollm_sim import planner
from heterollm_sim.reporting import run_scenario
from test_native_pp_capacity import pp_scenario


def scan_hidden_ids(scenario, layer):
    """The original lookup, retained only as this optimization's reference."""
    view = planner._execution_view(scenario)
    instances = view.layer_instances
    index = next(i for i, item in enumerate(instances) if item.layer_id == layer.layer_id)
    instance = instances[index]
    norm = next(op for op in view.operators if op.operator_id == instance.group_operator_id + ".norm1")
    residual = next(op for op in view.operators if op.operator_id == instance.group_operator_id + ".residual2")
    output_id = residual.output_tensor_ids[0]
    if index + 1 < len(instances) and instances[index + 1].group_operator_id == instance.group_operator_id:
        output_id += ":layer=" + instance.layer_id
    input_id = norm.input_tensor_ids[0]
    if index and instances[index - 1].group_operator_id == instance.group_operator_id:
        input_id = residual.output_tensor_ids[0] + ":layer=" + instances[index - 1].layer_id
    return input_id, output_id


def test_all_dense_and_moe_hidden_edges_match_scan_and_build_one_index_per_scenario(monkeypatch):
    scenarios = (native_toy.toy_scenario(layer_count=6), native_toy.moe_toy_scenario(layer_count=8, ep=2))
    expected = [{layer.layer_id: scan_hidden_ids(scenario, layer)
                 for layer in planner._execution_layers(scenario)} for scenario in scenarios]
    layers = [planner._execution_layers(scenario) for scenario in scenarios]
    original_view, calls = planner._execution_view, []

    def counted(scenario):
        calls.append(id(scenario))
        return original_view(scenario)

    monkeypatch.setattr(planner, "_execution_view", counted)
    first, second = scenarios
    with planner._compilation_scope(first):
        for _ in range(3):
            assert {layer.layer_id: planner._layer_hidden_tensor_ids(first, layer)
                    for layer in layers[0]} == expected[0]
        with planner._compilation_scope(second):
            for _ in range(3):
                assert {layer.layer_id: planner._layer_hidden_tensor_ids(second, layer)
                        for layer in layers[1]} == expected[1]
        assert {layer.layer_id: planner._layer_hidden_tensor_ids(first, layer)
                for layer in layers[0]} == expected[0]
    assert calls.count(id(first)) == calls.count(id(second)) == 1
    # A context is valid only for its own scenario; no process-global index leaks.
    with planner._compilation_scope(first):
        assert planner._layer_hidden_tensor_ids(second, layers[1][0]) == expected[1][layers[1][0].layer_id]


def test_index_keeps_expanded_repeated_group_hidden_edges(monkeypatch):
    scenario = native_toy.toy_scenario()
    view = NS(layer_instances=tuple(NS(layer_id=name, group_operator_id=group)
        for name, group in (("repeat-0", "dense-group"), ("repeat-1", "dense-group"),
                            ("moe-0", "moe-group"))),
        operators=tuple(NS(operator_id=group + suffix,
                           input_tensor_ids=(group + ".input",), output_tensor_ids=(group + ".output",))
                        for group in ("dense-group", "moe-group")
                        for suffix in (".norm1", ".residual2")))
    monkeypatch.setattr(planner, "_execution_view", lambda same: view)
    layers = tuple(NS(layer_id=item.layer_id) for item in view.layer_instances)
    expected = [scan_hidden_ids(scenario, layer) for layer in layers]
    assert expected[:2] == [("dense-group.input", "dense-group.output:layer=repeat-0"),
                           ("dense-group.output:layer=repeat-0", "dense-group.output")]
    with planner._compilation_scope(scenario):
        assert [planner._layer_hidden_tensor_ids(scenario, layer) for layer in layers] == expected
        assert [planner._layer_hidden_tensor_ids(scenario, layer) for layer in layers] == expected


@pytest.mark.parametrize("hardware", ["HBM4", "HBF+"])
def test_hidden_index_preserves_complete_pp_native_results(monkeypatch, hardware):
    scenario = pp_scenario(hardware, batch=4, microbatches=2)
    cached = run_scenario(scenario, retention_policy="aggregate").serving
    monkeypatch.setattr(planner, "_layer_hidden_tensor_ids", scan_hidden_ids)
    scanned = run_scenario(scenario, retention_policy="aggregate").serving
    assert cached.decode_measurement == scanned.decode_measurement
    assert cached.request_metrics == scanned.request_metrics
    assert cached.events == scanned.events
    assert cached.external_kv_ingress_metrics == scanned.external_kv_ingress_metrics
    assert [(batch.start_ns, batch.end_ns) for batch in cached.batches] == [
        (batch.start_ns, batch.end_ns) for batch in scanned.batches]
