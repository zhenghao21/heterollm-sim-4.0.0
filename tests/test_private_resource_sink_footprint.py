"""Candidate-only footprint checks; do not run before installing its patch."""
from dataclasses import replace
import sys

import pytest

from heterollm_sim import data_motion
from heterollm_sim.memory_types import NandConfig
from test_private_task_resource_sink import (
    HBM_OWNER, NAND_OWNER, NativeTrace, bits, compare_physical_state, drain,
    make_fixture, observe_run, sink_mode, state_bits)


def test_singleton_owner_groups_still_enter_sink_and_match_full_state():
    reference, task, _ = make_fixture(count=1, scalar=False, foreign_groups=7)
    candidate, candidate_task, _ = make_fixture(count=1, scalar=False, foreign_groups=7)
    expected, _ = observe_run(reference, task, False)
    actual, trace = observe_run(candidate, candidate_task)
    assert trace.gates and all(hit and scope for hit, scope in trace.gates)
    assert trace.submits and all(sink and not fields for _, sink, fields in trace.submits)
    assert bits(actual) == bits(expected)
    assert state_bits(candidate) == state_bits(reference)
    compare_physical_state(candidate, reference)


@pytest.mark.parametrize("where", ["related_lane", "foreign_busy", "foreign_bytes", "foreign_last"])
def test_callback_operands_that_can_be_read_still_reject_private_sink(where):
    callbacks = []
    class Number(float):
        def __float__(self):
            callbacks.append("float")
            return super().__float__()
    class Count(int):
        def __int__(self):
            callbacks.append("int")
            return super().__int__()
    runs = []
    for enabled in (False, True):
        callbacks.clear()
        kernel, task, _ = make_fixture(count=1, scalar=False)
        timeline = kernel.physical_runtime.timeline
        if where == "related_lane":
            name = kernel.physical_runtime.runtimes[HBM_OWNER].core._hbm_numeric_state.ids[0]
            timeline.lane_available[name][0] = Number(timeline.lane_available[name][0])
        elif where == "foreign_busy":
            timeline.busy_ns["foreign:huge"] = Number(0.)
        elif where == "foreign_bytes":
            timeline.bytes_moved["foreign:huge"] = Count(2**80 + 3)
        else:
            timeline.last_intervals["foreign:huge"] = (Number(0.), .125)
        events, trace = observe_run(kernel, task, enabled)
        assert trace.gates and not any(hit for hit, _ in trace.gates)
        runs.append((kernel, events, state_bits(kernel), tuple(callbacks)))
    assert bits(runs[0][1]) == bits(runs[1][1])
    assert runs[0][2] == runs[1][2]
    assert runs[0][3] == runs[1][3]


def test_unrelated_foreign_lane_is_not_in_value_qualification_union():
    callbacks = []
    class Number(float):
        def __float__(self):
            callbacks.append("float")
            return super().__float__()
    runs = []
    for enabled in (False, True):
        callbacks.clear()
        kernel, task, _ = make_fixture(count=1, scalar=False, foreign_groups=7)
        kernel.physical_runtime.timeline.lane_available["foreign:huge"][0] = Number(.125)
        events, trace = observe_run(kernel, task, enabled)
        if enabled:
            assert trace.gates and all(hit and scope for hit, scope in trace.gates)
        runs.append((kernel, events, state_bits(kernel), tuple(callbacks)))
    assert bits(runs[0][1]) == bits(runs[1][1])
    assert runs[0][2] == runs[1][2]
    assert runs[0][3] == runs[1][3]
    compare_physical_state(runs[0][0], runs[1][0])


def test_live_values_changed_at_prepare_return_are_rechecked_before_sink():
    class Number(float):
        pass
    runs = []
    for enabled in (False, True):
        kernel, task, _ = make_fixture(count=1, scalar=False)
        timeline = kernel.physical_runtime.timeline
        owner = kernel.physical_runtime.runtimes[HBM_OWNER].core
        resource = owner._hbm_numeric_state.ids[0]

        class ChangedAfterPrepare(NativeTrace):
            def observe(self, frame, event, value):
                if event == "return" and frame.f_code is data_motion.prepare_physical_task.__code__:
                    timeline.ready_ns[resource] = Number(timeline.ready_ns[resource])
                super().observe(frame, event, value)

        # A local return observer leaves every native function identity and
        # the outer transaction proof unchanged, then authors a real edit.
        with sink_mode(enabled), ChangedAfterPrepare() as trace:
            events = drain(kernel, task)
        assert trace.gates and not any(hit for hit, _ in trace.gates)
        if enabled:
            assert all(scope for _, scope in trace.gates)
        runs.append((kernel, events, state_bits(kernel)))
    assert bits(runs[0][1]) == bits(runs[1][1])
    assert runs[0][2] == runs[1][2]


def test_nand_write_is_rejected_before_public_value_scans():
    kernel, task, _ = make_fixture(count=1, scalar=False, foreign_groups=7)
    rows = list(task.metadata["memory_accesses"])
    rows[0] = {**rows[0], "operation": "write"}
    task = replace(task, metadata={**task.metadata, "memory_accesses": tuple(rows)})
    early_returns = []
    class EarlyTrace(NativeTrace):
        def observe(self, frame, event, value):
            if event == "return" and frame.f_code is self.gate:
                early_returns.append(value is False and "timeline" not in frame.f_locals)
            super().observe(frame, event, value)
    with EarlyTrace() as trace:
        drain(kernel, task)
    assert trace.gates and not any(hit for hit, _ in trace.gates)
    assert early_returns and all(early_returns)


@pytest.mark.parametrize("change", ["prefix", "alias"])
def test_hbm_descriptor_changed_after_prepare_rejects_old_ids_and_preserves_callbacks(change):
    callbacks = []
    class Number(float):
        def __float__(self):
            callbacks.append("float")
            return super().__float__()
    runs = []
    for enabled in (False, True):
        callbacks.clear()
        kernel, task, _ = make_fixture(count=1, scalar=False)
        runtime, timeline = kernel.physical_runtime, kernel.physical_runtime.timeline
        hbm = runtime.runtimes[HBM_OWNER].core
        original_pack = hbm._hbm_numeric_state

        class ChangedAfterPrepare(NativeTrace):
            def observe(self, frame, event, value):
                if event == "return" and frame.f_code is data_motion.prepare_physical_task.__code__:
                    assert hbm._hbm_numeric_state is original_pack
                    if change == "prefix":
                        def changed(config):
                            return replace(config, metadata={**config.metadata,
                                "row_command_resource_prefix": "changed:hbm:row"})
                        hbm.config = changed(hbm.config)
                        runtime._owner_config_cache[HBM_OWNER] = tuple(changed(config)
                            for config in runtime._owner_config_cache[HBM_OWNER])
                        value.instructions = [(*item[:5], changed(item[5])) if item[4] == HBM_OWNER else item
                                              for item in value.instructions]
                        configs = value.metadata["physical_memory_configs"]
                        value.metadata = {**value.metadata, "physical_memory_configs": {
                            **configs, HBM_OWNER: changed(configs[HBM_OWNER])}}
                        target = "changed:hbm:row:0:0"
                        assert original_pack._config != hbm.config
                    else:
                        target = "foreign:hbm:alias-target"
                        timeline.resource_aliases[original_pack.ids[len(original_pack.channel_keys)]] = target
                        # Keep the NAND descriptor current, so the HBM alias
                        # check itself must reject this otherwise native task.
                        nand = runtime.runtimes[NAND_OWNER].core
                        nand._nand_numeric_state.scalar_resource_contract = nand._resident_resource_contract()
                    timeline.ready_ns[target] = Number(.125)
                    timeline.lane_available[target] = [Number(.125)]
                    timeline.busy_ns[target] = 0.
                    timeline.bytes_moved[target] = 0
                    timeline.last_intervals[target] = (0., .125)
                super().observe(frame, event, value)

        with sink_mode(enabled), ChangedAfterPrepare() as trace:
            events = drain(kernel, task)
        assert trace.gates and not any(hit for hit, _ in trace.gates)
        if enabled:
            assert all(scope for _, scope in trace.gates)
        runs.append((kernel, events, state_bits(kernel), tuple(callbacks)))
    assert bits(runs[0][1]) == bits(runs[1][1])
    assert runs[0][2:] == runs[1][2:]
    assert runs[0][3]


@pytest.mark.parametrize("field", ["directions", "lane_directions", "_intervals", "_interval_payloads"])
def test_foreign_authored_keys_reject_sink_without_qualification_equality_callbacks(field):
    runs = []
    for enabled in (False, True):
        callbacks = []
        class Key(str):
            __hash__ = str.__hash__
            def __eq__(self, other):
                callbacks.append("key equality")
                return str.__eq__(self, other)
        kernel, task, _ = make_fixture(count=1, scalar=False)
        timeline = kernel.physical_runtime.timeline
        values = getattr(timeline, field)
        values[Key("foreign:huge")] = "read" if field == "directions" else []
        callbacks.clear()
        events, trace = observe_run(kernel, task, enabled)
        assert trace.gates and not any(hit for hit, _ in trace.gates)
        if enabled:
            assert all(scope for _, scope in trace.gates)
        runs.append((kernel, events, state_bits(kernel), tuple(callbacks)))
    assert runs[0][3] == runs[1][3]
    assert bits(runs[0][1]) == bits(runs[1][1])
    assert runs[0][2] == runs[1][2]
