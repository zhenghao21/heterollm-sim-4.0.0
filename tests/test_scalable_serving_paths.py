"""The critical-path tie-break compares key sequences exactly, without materializing them.

The oracle below is the previous implementation (full node-key tuples cached
per link, compared as tuples). It lives only in this file.
"""
from random import Random
import tracemalloc

import pytest

from heterollm_sim import live_execution_metrics, scalable_serving as serving
from heterollm_sim.contracts import ResourceDemand, RunManifest, TaskCategory, TaskSpec
from heterollm_sim.engine import ScheduleIR
from heterollm_sim.event_kernel import UnifiedEventKernel
from heterollm_sim.live_execution_metrics import LiveCohortMetrics
from heterollm_sim.schema_v4 import SIMULATION_SCHEMA_VERSION


class _OracleLink:
    __slots__ = ("previous", "duration_ns", "category", "node_key", "distance", "materialized_path")

    def __init__(self, previous, duration_ns, category, node_key):
        self.previous = previous
        self.duration_ns = duration_ns
        self.category = category
        self.node_key = node_key
        self.distance = (previous.distance if previous is not None else 0.0) + duration_ns
        self.materialized_path = None


def _oracle_materialize(link):
    if link is not None and link.materialized_path is not None:
        return link.materialized_path
    reverse_keys = []
    cursor = link
    prefix = ()
    while cursor is not None:
        if cursor.materialized_path is not None:
            prefix = cursor.materialized_path
            break
        if cursor.node_key is not None:
            reverse_keys.append(cursor.node_key)
        cursor = cursor.previous
    reverse_keys.reverse()
    result = prefix + tuple(reverse_keys)
    if link is not None:
        link.materialized_path = result
    return result


def _oracle_better(candidate, current):
    candidate_distance = candidate.distance if candidate is not None else 0.0
    current_distance = current.distance if current is not None else 0.0
    if candidate_distance > current_distance:
        return candidate
    if candidate_distance == current_distance:
        if _oracle_materialize(candidate) < _oracle_materialize(current):
            return candidate
    return current


def _oracle_extend(previous, duration_ns, category, node_key):
    return _oracle_better(_OracleLink(previous, duration_ns, category, node_key), None)


def _keys(link):
    keys = []
    while link is not None:
        keys.append(link.node_key)
        link = link.previous
    return tuple(reversed(keys))


def _forest(rng, size, *, alphabet, durations, shape="random"):
    """The same construction replayed on the new links and the oracle links."""
    new, old = [None], [None]
    for index in range(size):
        if shape == "trunk" and index and rng.random() < .6:
            parent = len(new) - 1
        elif shape == "deep" and index and rng.random() < .9:
            parent = max(0, len(new) - 1 - rng.randrange(3))
        else:
            parent = rng.randrange(len(new))
        key = "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 4)))
        duration = rng.choice(durations)
        category = rng.choice(list(TaskCategory))
        made = serving._extend_path(new[parent], duration, category, key)
        expected = _oracle_extend(old[parent], duration, category, key)
        assert (made is None) == (expected is None)
        new.append(made)
        old.append(expected)
    return new, old


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("shape,alphabet", [("random", "ab"), ("random", "abcxyz@."),
                                            ("trunk", "ab"), ("deep", "abc")])
def test_precedes_equals_tuple_order_of_the_key_sequences(seed, shape, alphabet):
    rng = Random(1000 * seed + len(alphabet))
    new, old = _forest(rng, 220, alphabet=alphabet, durations=(0., 0., 1., 2.), shape=shape)
    for link, oracle in zip(new, old):
        assert _keys(link) == _oracle_materialize(oracle)
    for _ in range(1500):
        i, j = rng.randrange(len(new)), rng.randrange(len(new))
        assert serving._path_precedes(new[i], new[j]) == (
            _oracle_materialize(old[i]) < _oracle_materialize(old[j])), (i, j)
        # The tie-break decision itself, on equal and unequal distances.
        chosen = serving._better_path(new[i], new[j])
        reference = _oracle_better(old[i], old[j])
        assert (chosen is None) == (reference is None)
        if chosen is not None:
            assert chosen is (new[i] if reference is old[i] else new[j])


def test_independently_built_equal_sequences_stay_equal():
    first = serving._extend_path(None, 1., TaskCategory.COMPUTE, "a")
    sibling = serving._extend_path(first, 1., TaskCategory.COMPUTE, "t@r")
    twin = serving._extend_path(first, 1., TaskCategory.COMPUTE, "t@r")
    assert sibling is not twin
    assert not serving._path_precedes(sibling, twin) and not serving._path_precedes(twin, sibling)
    # A whole duplicated chain, built from a second root with the same keys.
    left, right = None, None
    for key in ("a", "b", "c", "d", "e", "f", "g", "h", "i"):
        left = serving._extend_path(left, 1., TaskCategory.MEMORY, key)
        right = serving._extend_path(right, 1., TaskCategory.MEMORY, key)
    assert not serving._path_precedes(left, right) and not serving._path_precedes(right, left)
    assert serving._better_path(left, right) is right
    assert serving._better_path(right, left) is left
    # Equal up to a point, then different, and a proper prefix.
    longer = serving._extend_path(left, 1., TaskCategory.MEMORY, "z")
    other = serving._extend_path(right, 1., TaskCategory.MEMORY, "y")
    assert serving._path_precedes(other, longer) and not serving._path_precedes(longer, other)
    assert serving._path_precedes(right, longer) and not serving._path_precedes(longer, right)


def test_empty_and_zero_paths_follow_the_historical_rule():
    link = serving._PathLink(None, 0., TaskCategory.COMPUTE, "x")
    assert serving._path_precedes(None, link) and not serving._path_precedes(link, None)
    assert not serving._path_precedes(None, None)
    # A non-empty zero-distance path does not replace the empty path.
    assert serving._extend_path(None, 0., TaskCategory.COMPUTE, "x") is None
    assert serving._better_path(link, None) is None
    # The empty path sorts before any non-empty one at equal distance.
    assert serving._better_path(None, link) is None
    heavier = serving._PathLink(None, 1., TaskCategory.COMPUTE, "x")
    assert serving._better_path(None, heavier) is heavier and serving._better_path(heavier, None) is heavier
    assert serving._extend_path(None, 1., TaskCategory.COMPUTE, "x").depth == 1


def test_a_link_carries_two_extra_words_not_a_key_tuple():
    assert not hasattr(serving, "_materialize_path")
    assert not hasattr(serving._PathLink, "materialized_path")
    assert set(serving._PathLink.__slots__) == {
        "previous", "duration_ns", "category", "node_key", "distance", "depth", "jump"}


def test_deep_trunk_comparisons_need_no_per_link_memory():
    trunk = None
    for index in range(4000):
        trunk = serving._extend_path(trunk, 1., TaskCategory.COMPUTE, "n%05d" % index)
    leaves = [serving._PathLink(trunk, 1., TaskCategory.COMPUTE, "leaf%d" % (index % 7)) for index in range(4000)]
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        winners = {id(serving._better_path(leaves[i], leaves[i - 1])) for i in range(1, len(leaves))}
        grown = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert winners
    # 4000 materialized tuples of ~4000 keys would take over 120 MB.
    assert grown < 1024 * 1024


def _manifest():
    return RunManifest(schema_version=SIMULATION_SCHEMA_VERSION, run_id="paths", random_seed=0,
                       simulator_version="test", model_name="m", hardware_name="h", workload_name="w")


def _tie_heavy_tasks(seed, count=70):
    rng = Random(seed)
    names = sorted({"".join(rng.choice("abcd") for _ in range(rng.randrange(1, 4))) + str(i % 5)
                    for i in range(count * 2)})
    rng.shuffle(names)
    tasks = []
    for name in names[:count]:
        dependencies = tuple(task.task_id for task in rng.sample(tasks, min(len(tasks), rng.randrange(4))))
        demands = tuple(ResourceDemand(resource, rng.choice((0., 1., 2., 2.)), bytes_moved=3)
                        for resource in rng.sample(["a", "b", "c"], rng.randrange(4)))
        tasks.append(TaskSpec(name, "r", "work", rng.choice(list(TaskCategory)), dependencies=dependencies,
                              demands=demands, earliest_start_ns=rng.choice((0., 0., 1., 2.))))
    return tuple(tasks), {"a": 2, "b": 1, "c": 3}


def _patch_oracle(monkeypatch):
    monkeypatch.setattr(serving, "_better_path", _oracle_better)
    monkeypatch.setattr(serving, "_extend_path", _oracle_extend)
    monkeypatch.setattr(live_execution_metrics, "_better_path", _oracle_better)
    monkeypatch.setattr(live_execution_metrics, "_extend_path", _oracle_extend)


def _end_path_keys(tasks, capacities):
    kernel = UnifiedEventKernel.from_closed_graph(tasks, resource_capacities=capacities)
    live = LiveCohortMetrics()
    while kernel.has_active_tasks:
        live.observe(kernel.step())
    summary = live.summary()
    return {task_id: _keys(link) for task_id, link in live._end_paths.items()}, summary


@pytest.mark.parametrize("seed", range(10))
def test_cost_schedule_and_live_metrics_pick_the_same_critical_paths(seed, monkeypatch):
    tasks, capacities = _tie_heavy_tasks(seed)
    schedule = ScheduleIR(_manifest(), tasks, resource_capacities=capacities)
    new_cost = serving.execute_cost_schedule(schedule)
    new_keys, new_summary = _end_path_keys(tasks, capacities)
    with monkeypatch.context() as patch:
        _patch_oracle(patch)
        old_cost = serving.execute_cost_schedule(schedule)
        old_keys, old_summary = _end_path_keys(tasks, capacities)
    assert new_keys == old_keys
    for field in ("makespan_ns", "task_count", "bytes_moved", "resource_busy_ns", "category_time_ns",
                  "critical_path_category_ns", "coverage", "linear_state_bytes"):
        assert getattr(new_cost, field) == getattr(old_cost, field), field
        assert getattr(new_summary, field) == getattr(old_summary, field), field
    assert any(len(keys) > 3 for keys in new_keys.values())
