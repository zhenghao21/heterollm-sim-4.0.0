"""Native publication preserves public bits, identities and rollback."""
import numpy as np
import pytest

from heterollm_sim import _nand_pack
from heterollm_sim.data_motion import PhysicalRuntimeContext
from heterollm_sim.memory_transfer import ResourceTimeline, _TIMELINE_MAPS
from heterollm_sim.memory_types import AccessRequest
from test_nand_dense_metrics import ordered_bits
from test_nand_generic_buffered_acceleration import PAGE, config


def public_bits(row):
    return tuple((name, ordered_bits(getattr(row, name))) for name in _TIMELINE_MAPS)


@pytest.mark.parametrize("mode", ["single_empty", "single_nonempty", "multi"])
def test_publication_keeps_bits_order_original_lists_and_rollback(mode):
    names = ("array", "channel", "host")
    multi = mode == "multi"
    histories = {} if mode == "single_empty" else {"channel": ["write", "read"] if multi else ["write"]}
    rows = [ResourceTimeline(
        ready_ns={"host": 7., "array": -0., "channel": 5.},
        busy_ns={"channel": .125, "array": .375, "host": .625},
        bytes_moved={"host": 2**54 + 1, "channel": 2**54 + 3},
        lane_available={"array": [-0.], "channel": [5., 9.] if multi else [5.], "host": [7.]},
        lane_directions={key: list(value) for key, value in histories.items()},
    ) for _ in range(2)]
    before = [public_bits(row) for row in rows]
    originals = [[row.lane_available[rid] for rid in names] for row in rows]
    tokens = [row._begin_owned_transaction() for row in rows]
    state = np.array([[-0., .1 + .2, .125, .875], [5., 7., 8., 9.], [6., 10., 11., 12.]])
    moved = np.array([0, 2**54 + 3, 1024], dtype=np.int64)
    directions = np.array([0, 2, 1], dtype=np.int64)
    lanes = np.array([-0., 5., 13., 6.] if multi else [-0., 5., 6.])
    lane_directions = np.array([0, 1, 2, 1] if multi else [0, 2, 1], dtype=np.int64)
    starts = np.array([0, 1, 3] if multi else [0, 1, 2], dtype=np.int64)
    counts = np.array([1, 2, 1] if multi else [1, 1, 1], dtype=np.int64)
    indexes = np.arange(3, dtype=np.int64)
    active, bus, labels = set(names), (0, 1, 1), [None, "read", "write"]
    for _ in range(2):
        for row in rows:
            row._reset_metric_recorder(0)
        _nand_pack.publish_active_timeline(rows[0], names, state, moved, directions, lanes,
            lane_directions, starts, counts, indexes, active, bus, labels, not multi, True)
        _nand_pack.publish_timeline(rows[1], names, state.tolist(), moved.tolist(), directions.tolist(),
            lanes.tolist(), lane_directions.tolist(), starts.tolist(), counts.tolist(), active, (),
            list(bus), labels, not multi)
        assert public_bits(rows[0]) == public_bits(rows[1])
        for row, old in zip(rows, originals):
            assert all(row.lane_available[rid] is value for rid, value in zip(names, old))
    for row, token, saved in zip(rows, tokens, before):
        token.rollback()
        assert public_bits(row) == saved


def test_read_batch_preserves_foreign_resources_and_restores_on_rollback():
    settings = config(channels=64, targets_per_channel=1, dies_per_target=1, planes_per_lun=2)
    runtime = PhysicalRuntimeContext(settings, "flash", capture_details=False)
    core = runtime.runtimes["flash"].core
    core.submit(AccessRequest("seed", "read", 0, 128 * PAGE))
    runtime.timeline.busy_ns["foreign-owner"] = .125
    runtime.timeline._touched.add("foreign-owner")
    before = public_bits(runtime.timeline)
    token = runtime.begin_transaction(owners=("flash",), _owned_timeline=True)
    requests = [AccessRequest(str(index), "read", (index + 1) * 128 * PAGE, 128 * PAGE)
                for index in range(8)]
    actual = core.submit_many(requests)
    assert all(row.counters.get("native_fullpage_pages") == 128 for row in actual)
    assert all("foreign-owner" not in row.counters["resource_busy_ns"] for row in actual)
    assert runtime.timeline.busy_ns["foreign-owner"] == .125
    runtime.restore(token)
    assert public_bits(runtime.timeline) == before
