import pytest

from heterollm_sim import ir, planner
from heterollm_sim.parallel import LogicalRank
from heterollm_sim.reference import build_reference_scenario


def test_direct_addresses_reuse_validated_view_without_changing_values(monkeypatch):
    scenario = build_reference_scenario()
    rank = LogicalRank(1, "gpu0", 0, 0, 0, "hbm0")
    metadata = [{"layer_id": f"layer-{index:03d}", "operator_id": "qkv"}
                for index in range(20)]
    expected = [planner._direct_memory_address(scenario, rank, "hbm0", 1024, item)
                for item in metadata]
    original = ir._model_graph_execution_payload
    payload_calls = []

    def counted_payload(graph):
        payload_calls.append(graph)
        return original(graph)

    monkeypatch.setattr(ir, "_model_graph_execution_payload", counted_payload)
    with planner._compilation_scope(scenario):
        actual = [planner._direct_memory_address(scenario, rank, "hbm0", 1024, item)
                  for item in metadata]

    assert actual == expected
    assert len(payload_calls) == 1


def test_new_compilation_revalidates_nested_graph_changes():
    scenario = build_reference_scenario()
    rank = LogicalRank(1, "gpu0", 0, 0, 0, "hbm0")
    with planner._compilation_scope(scenario):
        planner._direct_memory_address(scenario, rank, "hbm0", 1024)

    scenario.model.graph.operators[0].parameters["unsupported_cost_parameter"] = 1
    with planner._compilation_scope(scenario):
        with pytest.raises(ValueError, match="覆盖|参数|投影"):
            planner._direct_memory_address(scenario, rank, "hbm0", 1024)
