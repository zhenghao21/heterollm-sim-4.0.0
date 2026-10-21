"""Typed cache handoff keeps the native two-slot object protocol unchanged."""
import importlib.util
from math import isfinite
from pathlib import Path

import numpy as np
import pytest

from heterollm_sim.nand_core import _ReadCacheSlot
from test_resource_ledger import _same


@pytest.fixture(params=("python", "compiled"))
def pack(request):
    folder = Path(__file__).resolve().parents[1] / "src/heterollm_sim"
    path = folder / "_nand_pack.py"
    name = "nand_pack_python_test"
    if request.param == "compiled":
        binaries = tuple(folder.glob("_nand_pack*.pyd")) + tuple(folder.glob("_nand_pack*.so"))
        if not binaries:
            pytest.skip("optional NAND packer has not been built")
        path, name = binaries[0], "heterollm_sim._nand_pack"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


KEYS = ((0, 0), (1, 0), (2, 0))
PREFIXES = ((0, 0, 0), (1, 0, 0), (2, 0, 0))


def cache():
    return {KEYS[0]: [_ReadCacheSlot((*PREFIXES[0], 1, 2), -0.0, .1 + .2, 2**62),
                      _ReadCacheSlot(None, .125, 1.25, 7)],
            KEYS[2]: [_ReadCacheSlot(), _ReadCacheSlot((*PREFIXES[2], 4, 5), 7.5, 19.25, 22)]}


def original_validate(cache, bank_keys, prefixes, slot_type, pages_per_block, finite):
    if type(cache) is not dict:
        return None
    rows, seen = [], set()
    for bank, prefix in zip(bank_keys, prefixes):
        slots = cache.get(bank, ())
        if bank in cache and (type(slots) is not list or len(slots) != 2):
            return None
        for slot in slots:
            if (type(slot) is not slot_type or id(slot) in seen
                    or vars(slot).keys() != {"page_key", "ready_ns", "reader_completion_ns", "last_use"}
                    or type(slot.last_use) is not int or not 0 <= slot.last_use <= 2**63 - 1
                    or any(type(value) not in (int, float) or not finite(value) or value < 0
                           for value in (slot.ready_ns, slot.reader_completion_ns))):
                return None
            seen.add(id(slot))
            tag = slot.page_key
            if tag is not None and (type(tag) is not tuple or len(tag) != len(prefix) + 2
                    or tag[:-2] != prefix or any(type(value) is not int or value < 0 for value in tag)
                    or tag[-1] >= pages_per_block
                    or tag[-2] * pages_per_block + tag[-1] > 2**63 - 1):
                return None
        rows.append(slots)
    return rows


@pytest.mark.parametrize("change", ("ordinary", "alias", "extra", "empty", "tag", "nan", "use"))
def test_cache_validation_preserves_decline_and_finite_callback_order(pack, change):
    source = cache()
    if change == "alias":
        source[KEYS[0]][1] = source[KEYS[0]][0]
    elif change == "extra":
        source[KEYS[0]][0].extension = 9
    elif change == "empty":
        source[KEYS[0]] = []
    elif change == "tag":
        source[KEYS[0]][0].page_key = (*PREFIXES[1], 1, 2)
    elif change == "nan":
        source[KEYS[0]][0].reader_completion_ns = float("nan")
    elif change == "use":
        source[KEYS[0]][0].last_use = True
    events = [[], []]
    outputs = []
    for index, function in enumerate((pack.validate_buffered_cache, original_validate)):
        def finite(value):
            events[index].append(value.hex() if type(value) is float else value)
            return isfinite(value)
        outputs.append(function(source, KEYS, PREFIXES, _ReadCacheSlot, 256, finite))
    _same(events[0], events[1])
    assert (outputs[0] is None) == (outputs[1] is None)
    if outputs[0] is not None:
        assert all(actual is expected for actual, expected in zip(*outputs))


def test_cache_array_copy_preserves_integer_tags_float_bits_and_missing_bank(pack):
    source = cache()
    rows = original_validate(source, KEYS, PREFIXES, _ReadCacheSlot, 256, isfinite)
    arrays = pack.pack_buffered_cache(rows, len(KEYS), 256)
    expected = (np.full((3, 2), -1, dtype=np.int64), np.zeros((3, 2)),
                np.zeros((3, 2)), np.zeros((3, 2), dtype=np.int64), np.zeros(3, dtype=np.bool_))
    tags, ready, readers, uses, present = expected
    for bank, slots in enumerate(rows):
        present[bank] = bool(slots)
        for index, slot in enumerate(slots):
            if slot.page_key is not None:
                tags[bank, index] = slot.page_key[-2] * 256 + slot.page_key[-1]
            ready[bank, index], readers[bank, index], uses[bank, index] = (
                slot.ready_ns, slot.reader_completion_ns, slot.last_use)
    for actual, reference in zip(arrays, expected):
        assert actual.dtype == reference.dtype
        assert actual.tobytes() == reference.tobytes()


def original_publish(core, cache, bank_keys, prefixes, pages_per_block,
                     tags, ready, readers, uses, active_banks, slot_type):
    tag_values, ready_values, reader_values, use_values = tags.tolist(), ready.tolist(), readers.tolist(), uses.tolist()
    for bank in np.flatnonzero(active_banks).tolist():
        key = bank_keys[bank]
        if core._nand_transactions:
            core._touch_cache_bank(key)
        slots = cache.get(key)
        if slots is None:
            slots = [slot_type(), slot_type()]
            cache[key] = slots
        for index, slot in enumerate(slots):
            tag = tag_values[bank][index]
            slot.page_key = None if tag < 0 else (*prefixes[bank], *divmod(tag, pages_per_block))
            slot.ready_ns, slot.reader_completion_ns, slot.last_use = (
                ready_values[bank][index], reader_values[bank][index], use_values[bank][index])


@pytest.mark.parametrize("fail", (False, True))
def test_cache_publication_preserves_callback_order_aliases_and_failure_prefix(pack, fail):
    sources = [cache(), cache()]
    arrays = pack.pack_buffered_cache(list((sources[0].get(key, ()) for key in KEYS)), 3, 256)
    tags, ready, readers, uses, present = arrays
    tags[1] = [123, 234]
    ready[:] += 100.
    readers[:] += 200.
    uses[:] += 9
    events = [[], []]
    for index, function in enumerate((pack.publish_buffered_cache, original_publish)):
        class Core:
            _nand_transactions = [object()]
            def _touch_cache_bank(self, key):
                events[index].append(key)
                if fail and key == KEYS[1]:
                    raise RuntimeError("authored cache checkpoint")
        arguments = (Core(), sources[index], KEYS, PREFIXES, 256, tags, ready, readers, uses,
                     np.ones(3, dtype=np.bool_), _ReadCacheSlot)
        old = sources[index][KEYS[0]][0]
        if fail:
            with pytest.raises(RuntimeError, match="authored cache checkpoint"):
                function(*arguments)
        else:
            function(*arguments)
        assert sources[index][KEYS[0]][0] is old
    _same(events[0], events[1])
    _same({key: [vars(slot) for slot in slots] for key, slots in sources[0].items()},
          {key: [vars(slot) for slot in slots] for key, slots in sources[1].items()})


def test_active_numeric_columns_publish_exact_float_bits_and_original_lane_objects(pack):
    from heterollm_sim.memory_transfer import ResourceTimeline
    names = ("array", "channel", "untouched", "host")
    timelines = [ResourceTimeline(lane_available={"array": [-0.], "channel": [7., 11.],
        "untouched": [19.]}, ready_ns={"untouched": 19.}, bytes_moved={"channel": 13}) for _ in range(2)]
    state = np.array([[-0., .1+.2, .125, .875], [5., 7., 8., 9.],
        [19., 0., 0., 0.], [6., 10., 11., 12.]], dtype=np.float64)
    moved = np.array([0, 2**54+3, 0, 1024], dtype=np.int64)
    directions = np.array([0, 2, 0, 1], dtype=np.int64)
    lanes = np.array([-0., 5., 13., 19., 6.], dtype=np.float64)
    lane_directions = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    starts = np.array([0, 1, 3, 4], dtype=np.int64)
    counts = np.array([1, 2, 1, 1], dtype=np.int64)
    indexes = np.array([0, 1, 3], dtype=np.int64)
    active = {names[index] for index in indexes}
    bus_mask = (0, 1, 0, 1)
    original_lanes = [timeline.lane_available["channel"] for timeline in timelines]
    pack.publish_timeline(timelines[0], names, state.tolist(), moved.tolist(), directions.tolist(),
        lanes.tolist(), lane_directions.tolist(), starts.tolist(), counts.tolist(), active, (),
        list(bus_mask), [None, "read", "write"], False)
    pack.publish_active_timeline(timelines[1], names, state, moved, directions, lanes,
        lane_directions, starts, counts, indexes, active, bus_mask, [None, "read", "write"], False)
    for field in ("ready_ns", "busy_ns", "bytes_moved", "directions", "last_intervals",
                  "lane_available", "lane_directions"):
        _same(getattr(timelines[0], field), getattr(timelines[1], field))
    assert all(timeline.lane_available["channel"] is original
               for timeline, original in zip(timelines, original_lanes))


def test_selected_packer_reads_only_proven_footprint_and_matches_full_columns(pack):
    from heterollm_sim.memory_transfer import ResourceTimeline
    names = ("a", "b", "unrelated", "host")
    timeline = ResourceTimeline(lane_available={name: [index+.125] for index, name in enumerate(names)},
        busy_ns={"a": .1+.2}, last_intervals={"a": (-0., .125)},
        directions={"b": "write"}, bytes_moved={"b": 2**54+3})
    selected = np.array([0, 1, 3], dtype=np.int64)
    actual = pack.pack_selected_single_lane(timeline, names, selected, 256, 4096)
    reference = pack.pack_single_lane(timeline, names, 256, 4096)
    for actual_column, reference_column in zip(actual, reference):
        assert actual_column[selected].tobytes() == reference_column[selected].tobytes()
    # Ordinary multi-lane/shared active objects continue to the complete
    # calendar packer; the selected fast path never flattens those calendars.
    timeline.lane_available["b"] = [2., 3.]
    assert pack.pack_selected_single_lane(timeline, names, selected, 256, 4096) is None


def test_actual_pd_resource_indexes_cover_256_arrays_112_channels_and_host(pack):
    arrays = np.arange(2800, dtype=np.int64)
    channels = np.arange(2800, 2912, dtype=np.int64)
    hosts = np.array([2912], dtype=np.int64)
    for values in (arrays, channels, hosts):
        values.flags.writeable = False
    indexes = pack.generic_request_resource_indexes(2700, 256, 112, 25, True, 1, 1,
        arrays, channels, hosts)
    assert indexes.dtype == np.int64
    assert len(indexes) == 369
    assert len(set(indexes)) == len(indexes)
    assert set(indexes[indexes < 2800]) == set(range(2700, 2800)) | set(range(156))
    assert set(indexes[indexes >= 2800]) == set(range(2800, 2913))
