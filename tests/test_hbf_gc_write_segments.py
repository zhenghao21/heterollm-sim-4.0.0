"""A live-copy GC boundary must not scalarize the ordinary pages after it."""
from dataclasses import replace

import pytest

from heterollm_sim import _hbf_page_state
from heterollm_sim.hbf_staging import HbfStagingController
from test_hbf_page_state import PageIO, config


PAGE = 4096


def _fill_fragmented(controller, hardware):
    """Leave one retained page per full block, then exhaust the free pool."""
    lanes, block_pages = controller._lanes, controller.config.pages_per_block
    full_groups = (controller._block_count - 1) // lanes
    for group in range(full_groups):
        controller.write("K", 0, group * lanes * PAGE, lanes * PAGE, 0., hardware.io)
        controller.write("temporary", group, 0, lanes * (block_pages - 1) * PAGE,
                         0., hardware.io)
        controller.release("temporary", group)
    # Any final incomplete bank stripe is consumed with dead temporary pages.
    if controller._free_blocks or controller._active:
        count = len(controller._free_blocks) * block_pages + sum(
            block_pages - controller._used[block] for block in controller._active)
        controller.write("padding", 0, 0, count * PAGE, 0., hardware.io)
        controller.release("padding", 0)
    assert not controller._free_blocks and not controller._active


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("lanes,pages", [(1, 32), (3, 32), (4, 17), (1, 256)])
def test_segmented_writes_preserve_each_page_event_and_nested_rollback(
        monkeypatch, native, lanes, pages):
    monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
    bulk = HbfStagingController._write_full_pages
    outcomes = []
    for scalar in (True, False):
        monkeypatch.setattr(HbfStagingController, "_write_full_pages",
                            (lambda *args: None) if scalar else bulk)
        controller = HbfStagingController(config(channels=lanes, host_channels=lanes,
            pages_per_block=pages, blocks_per_plane=12), 4 * PAGE)
        hardware = PageIO()
        _fill_fragmented(controller, hardware)
        controller.read("K", 0, 0, ((controller._block_count - 1) // lanes) * lanes * PAGE,
                        0., hardware.io)
        before = controller.snapshot()
        outer = controller.begin_transaction()
        # The exhausted pool includes partially live victims. The remaining
        # target pages still follow their unchanged ordinary bulk stream.
        end = controller.write("workspace", 0, 0, (3 * pages + 7) * PAGE, 137., hardware.io)
        after_outer = controller.snapshot()
        inner = controller.begin_transaction()
        tail_end = controller.write("workspace", 0, 64, (pages + 2) * PAGE,
                                    3., hardware.io)  # Partial head/tail stays scalar.
        controller.rollback_transaction(inner)
        assert controller.snapshot() == after_outer
        controller.rollback_transaction(outer)
        assert controller.snapshot() == before
        end_again = controller.write("workspace", 0, 0, (3 * pages + 7) * PAGE,
                                     137., hardware.io)
        assert controller.stats["gc_copied_pages"] > 0
        outcomes.append((end, tail_end, end_again, controller.snapshot(), hardware.events))
    assert outcomes[0] == outcomes[1]


def test_bulk_decline_keeps_every_scalar_page_after_the_gc_boundary(monkeypatch):
    bulk = HbfStagingController._write_full_pages
    outcomes = []
    for scalar in (True, False):
        monkeypatch.setattr(HbfStagingController, "_write_full_pages",
                            (lambda *args: None) if scalar else bulk)
        controller = HbfStagingController(config(channels=1, host_channels=1,
            pages_per_block=32, blocks_per_plane=12), 4 * PAGE)
        hardware = PageIO()
        _fill_fragmented(controller, hardware)
        hardware.decline = True
        end = controller.write("workspace", 0, 0, 80 * PAGE, 137., hardware.io)
        outcomes.append((end, controller.snapshot(), hardware.events))
    assert outcomes[0] == outcomes[1]


def test_long_write_resumes_bulk_only_between_actual_live_gc_boundaries():
    controller = HbfStagingController(config(channels=1, host_channels=1,
        pages_per_block=32, blocks_per_plane=12), 4 * PAGE)
    hardware = PageIO()
    _fill_fragmented(controller, hardware)
    lengths = []

    def bulk(addresses, arrivals):
        lengths.append(len(addresses))
        return hardware.write_pages(addresses, arrivals)

    hardware.io.write_flash_pages = bulk
    controller.write("workspace", 0, 0, 80 * PAGE, 137., hardware.io)
    # Each 32-page target first receives one retained GC copy and the one
    # ordinary boundary write. Only its thirty remaining pages are batched.
    assert lengths == [30, 30, 17]
    assert controller.stats["gc_cycles"] == 3
    assert controller.stats["gc_copied_pages"] == 3


@pytest.mark.parametrize("override", ["subclass", "instance", "class"])
def test_custom_allocation_hooks_keep_the_original_gc_fallback_order(monkeypatch, override):
    bulk, allocate = HbfStagingController._write_full_pages, HbfStagingController._allocate
    outcomes = []
    for scalar in (True, False):
        monkeypatch.setattr(HbfStagingController, "_write_full_pages",
                            (lambda *args: None) if scalar else bulk)
        # Restore the class before building the next independent oracle.
        monkeypatch.setattr(HbfStagingController, "_allocate", allocate)
        controller = HbfStagingController(config(channels=1, host_channels=1,
            pages_per_block=32, blocks_per_plane=12), 4 * PAGE)
        hardware = PageIO()
        _fill_fragmented(controller, hardware)
        observations = []

        def observe(current, time, io, key):
            observations.append((key, len(hardware.events), current._pages.get(key)))
            return allocate(current, time, io, key)

        if override == "subclass":
            class AuthoredController(HbfStagingController):
                _allocate = observe
            controller.__class__ = AuthoredController
        elif override == "instance":
            controller._allocate = lambda time, io, key: observe(controller, time, io, key)
        else:
            monkeypatch.setattr(HbfStagingController, "_allocate", observe)
        end = controller.write("workspace", 0, 0, 80 * PAGE, 137., hardware.io)
        # Hook objects are external observations, not controller state.
        controller.__dict__.pop("_allocate", None)
        outcomes.append((end, observations, controller.snapshot(), hardware.events))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("native", [False, True])
def test_failure_after_committed_segment_restores_entire_runtime_before_retry(monkeypatch, native):
    from heterollm_sim import hbf_staging_runtime
    from heterollm_sim._io_execution import _PhysicalIO
    from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
    from test_hbf_staging_runtime import setup, task

    monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
    bulk = HbfStagingController._write_full_pages
    flash, flash_pages = _PhysicalIO.write_flash, _PhysicalIO.write_flash_pages
    outcomes = []
    for scalar in (True, False):
        monkeypatch.setattr(HbfStagingController, "_write_full_pages",
                            (lambda *args: None) if scalar else bulk)
        authored, binding = setup()
        authored = replace(authored, pages_per_block=32, capacity_bytes=16 * 32 * PAGE)
        runtime = PhysicalRuntimeContext(capture_details=False)
        for block in range(15):
            for buffer, generation, offset, count, size in (
                    ("K", 0, block * PAGE, PAGE, 80 * PAGE),
                    ("temporary", block, 0, 31 * PAGE, 31 * PAGE)):
                resolve_physical_task(task(authored, binding, "write", offset, count,
                    buffer=buffer, generation=generation, size=size), runtime, 137.)
            hbf_staging_runtime.release(runtime, "flash", "temporary", block)
            runtime.allocators["flash"].release("temporary", block)
        before = runtime.snapshot()
        physical_programs, committed_segments = 0, []

        def scalar_program(io, address, size, arrival):
            nonlocal physical_programs
            physical_programs += 1
            if physical_programs == 40:
                raise RuntimeError("injected fortieth physical program failure")
            return flash(io, address, size, arrival)

        def segmented_program(io, addresses, arrivals):
            nonlocal physical_programs
            remaining = 40 - physical_programs
            if len(addresses) >= remaining:
                # Preserve precisely the first 39 accepted physical programs,
                # including one complete prefix and a partial later segment.
                accepted = remaining - 1
                if accepted:
                    assert accepted >= 2
                    assert flash_pages(io, addresses[:accepted], arrivals[:accepted]) is not None
                physical_programs = 40
                raise RuntimeError("injected fortieth physical program failure")
            result = flash_pages(io, addresses, arrivals)
            if result is not None:
                physical_programs += len(addresses)
                committed_segments.append(len(addresses))
            return result

        work = task(authored, binding, "write", 0, 80 * PAGE, buffer="workspace", size=80 * PAGE)
        with monkeypatch.context() as fault:
            fault.setattr(_PhysicalIO, "write_flash", scalar_program)
            fault.setattr(_PhysicalIO, "write_flash_pages", segmented_program)
            with pytest.raises(RuntimeError, match="fortieth physical program"):
                resolve_physical_task(work, runtime, 709078321796.4534)
        assert physical_programs == 40
        assert committed_segments == ([] if scalar else [30])
        assert runtime.snapshot() == before
        row = resolve_physical_task(work, runtime, 709078321796.4534)
        outcomes.append((row.metadata["physical_completion_ns"],
                         row.metadata["physical_execution_by_owner"], runtime.snapshot()))
    (old_end, old_costs, old), (new_end, new_costs, new) = outcomes
    assert new_end == old_end and new_costs == old_costs
    assert new["timeline"] == old["timeline"]
    assert new["allocator_state"] == old["allocator_state"]
    assert new["hbf_staging"]["flash"]["saved_controller"] == old["hbf_staging"]["flash"]["saved_controller"]
    for owner in old["runtime_state"]:
        assert new["runtime_state"][owner]["core"] == old["runtime_state"][owner]["core"]
        assert new["runtime_state"][owner]["clock_ns"] == old["runtime_state"][owner]["clock_ns"]


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("capture_details", [False, True])
@pytest.mark.parametrize("arrival", [137., 709078321796.4534])
def test_segmented_gc_keeps_exact_host_nand_link_costs_and_calendars(
        monkeypatch, native, capture_details, arrival):
    from heterollm_sim import hbf_staging_runtime
    from heterollm_sim.data_motion import PhysicalRuntimeContext, resolve_physical_task
    from test_hbf_staging_runtime import setup, task

    monkeypatch.setattr(_hbf_page_state, "COMPILED", native)
    bulk = HbfStagingController._write_full_pages
    outcomes = []
    for scalar in (True, False):
        monkeypatch.setattr(HbfStagingController, "_write_full_pages",
                            (lambda *args: None) if scalar else bulk)
        authored, binding = setup()
        authored = replace(authored, pages_per_block=32, capacity_bytes=16 * 32 * PAGE)
        runtime = PhysicalRuntimeContext(capture_details=capture_details)
        ends, costs = [], []

        def execute(operation, offset, count, *, buffer="K", generation=0, size=80 * PAGE):
            row = resolve_physical_task(task(authored, binding, operation, offset, count,
                buffer=buffer, generation=generation, size=size), runtime, arrival)
            ends.append(row.metadata["physical_completion_ns"])
            costs.append(row.metadata["physical_execution_by_owner"])

        for block in range(15):
            execute("write", block * PAGE, PAGE)
            execute("write", 0, 31 * PAGE, buffer="temporary", generation=block, size=31 * PAGE)
            hbf_staging_runtime.release(runtime, "flash", "temporary", block)
            runtime.allocators["flash"].release("temporary", block)
        execute("read", 0, 15 * PAGE)
        for offset in (0, 2 * PAGE):
            execute("write", offset, 64 * PAGE, buffer="workspace")
            execute("read", 0, 64 * PAGE, buffer="workspace")
        controller = runtime.hbf_staging["flash"]["controller"]
        assert controller.stats["gc_copied_pages"] > 0
        outcomes.append((ends, costs, runtime.snapshot()))
    (old_ends, old_costs, old), (new_ends, new_costs, new) = outcomes
    assert new_ends == old_ends
    assert new_costs == old_costs
    assert new["timeline"] == old["timeline"]
    assert new["allocator_state"] == old["allocator_state"]
    assert new["hbf_staging"]["flash"]["saved_controller"] == old["hbf_staging"]["flash"]["saved_controller"]
    for owner in old["runtime_state"]:
        assert new["runtime_state"][owner]["core"] == old["runtime_state"][owner]["core"]
        assert new["runtime_state"][owner]["clock_ns"] == old["runtime_state"][owner]["clock_ns"]
