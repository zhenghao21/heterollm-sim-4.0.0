from types import SimpleNamespace

from heterollm_sim import reporting


def test_memory_layout_marks_missing_physical_shard_bytes_without_inventing_length(
    monkeypatch,
):
    placement = SimpleNamespace(
        parallel=SimpleNamespace(tp_degree=2, ep_degree=1),
        metadata={},
        tensor_bytes={},
        tensor_to_component={},
    )
    scenario = SimpleNamespace(placement=placement, hardware=SimpleNamespace(components=[]))
    result = SimpleNamespace(
        scenario=scenario,
        trace=SimpleNamespace(tasks=[]),
    )
    decision = {
        "weight_tensor_details": {
            "layer.weight": {
                "logical_tensor_id": "layer.weight",
                "logical_bytes": 1000,
            }
        },
        "rank_weight_shards": {
            "layer.weight": [
                {
                    "rank_id": 0,
                    "storage_component_id": "hbm0",
                    # physical_bytes intentionally absent
                },
                {
                    "rank_id": 1,
                    "storage_component_id": "hbm1",
                    "physical_bytes": 1.7,
                },
                {
                    "rank_id": 2,
                    "storage_component_id": "hbm2",
                    "physical_bytes": True,
                },
                {
                    "rank_id": 3,
                    "storage_component_id": "hbm3",
                    "physical_bytes": 256,
                },
            ]
        },
    }
    monkeypatch.setattr(reporting, "control_plane_decision", lambda _scenario: decision)

    layout, _lookup = reporting._memory_layout(result, segment_limit=10)

    segments = [
        segment
        for component_segments in layout["components"].values()
        for segment in component_segments
    ]
    assert [(segment["component_id"], segment["length_bytes"]) for segment in segments] == [("hbm3", 256)]
    assert layout["missing_physical_weight_shards"] == [
        {
            "logical_id": "layer.weight",
            "rank": 0,
            "component_id": "hbm0",
            "reason": "physical_bytes_missing_or_invalid",
        },
        {
            "logical_id": "layer.weight",
            "rank": 1,
            "component_id": "hbm1",
            "reason": "physical_bytes_missing_or_invalid",
        },
        {
            "logical_id": "layer.weight",
            "rank": 2,
            "component_id": "hbm2",
            "reason": "physical_bytes_missing_or_invalid",
        },
    ]
