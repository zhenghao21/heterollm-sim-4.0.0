import numpy as np
import pytest

from heterollm_sim._dram_numeric import run_numeric
from heterollm_sim.component_presets import get_component_preset
from heterollm_sim.dram_core import DramCore, _mapping_cycle
from heterollm_sim.memory_types import AccessRequest, Operation, parse_physical_memory_config


class PythonReference(DramCore):
    def _execute_accepted(self, request):
        return self._execute_accelerated_accepted(
            request, request.address // self.config.burst_bytes,
            (request.address + request.byte_count - 1) // self.config.burst_bytes + 1)


def test_compiled_numeric_loop_preserves_native_state_across_read_write_probes():
    config = parse_physical_memory_config(get_component_preset(
        "gddr7-16gb-30_0-256bit").component.metadata["physical_memory_config"])
    reference = PythonReference(config, capture_details=False)
    lanes, banks = config.lane_count, config.bank_count
    count, resources = lanes * banks, lanes * banks + 2 * lanes
    cycle = _mapping_cycle(lanes, banks, config.bank_groups_per_rank, config.banks_per_group,
                           1, "dram:bank", "dram:command", "dram:data", None, 0, 0)
    ids = [cycle[bank * lanes + lane][0] for lane in range(lanes) for bank in range(banks)]
    ids += [f"dram:command:{lane}" for lane in range(lanes)]
    ids += [f"dram:data:{lane}" for lane in range(lanes)]
    bankrows = np.full(count, -1, dtype=np.int64)
    bankready, calendar, busy = np.zeros(count), np.zeros(resources), np.zeros(resources)
    moved, directions = np.zeros(resources, dtype=np.int64), np.zeros(resources, dtype=np.int8)
    start, end = np.zeros(resources), np.zeros(resources)
    geometry = np.array([lanes, banks, 1, config.row_bytes // 64, config.rows_per_bank, 1], dtype=np.int64)
    for operation, address, size, arrival in (
        (Operation.READ, 17, 64 * 1024 * 1024 - 31, 0.0),
        (Operation.WRITE, 71, 6 * 1024 * 1024 - 9, 0.0),
        (Operation.READ, 6 * 1024 * 1024 - 64, 8193, 1000000.0),
    ):
        read = operation is Operation.READ
        durations = np.array([
            64, config.burst_interval_ns, 64 / config.directional_lane_bandwidth_gb_s(operation),
            config.read_latency_ns if read else config.write_latency_ns,
            config.read_recovery_ns if read else config.write_recovery_ns,
            config.open_ns, config.close_ns, config.read_to_write_ns, config.write_to_read_ns,
        ], dtype=np.float64)
        touched, banktouched = np.zeros(resources, dtype=np.bool_), np.zeros(count, dtype=np.bool_)
        completion, hits, misses, conflicts, _ = run_numeric(
            address // 64, (address + size - 1) // 64 + 1, arrival, read, geometry, durations,
            False, bankrows, bankready, banktouched, calendar, busy, moved, directions, start, end, touched)
        result = reference.submit(AccessRequest("request", operation, address, size, arrival))
        assert completion == pytest.approx(result.completion_ns, rel=0, abs=1e-7)
        assert (hits, misses, conflicts) == tuple(result.counters[key] for key in ("row_hits", "row_misses", "row_conflicts"))
        for index, rid in enumerate(ids):
            if index < count and banktouched[index]:
                assert bankrows[index] == reference._banks[rid].open_row
                assert bankready[index] == pytest.approx(reference._banks[rid].ready_ns, rel=0, abs=1e-7)
            if touched[index]:
                assert calendar[index] == pytest.approx(reference.timeline.lane_available[rid][0], rel=0, abs=1e-7)
                assert busy[index] == pytest.approx(reference.timeline.busy_ns[rid], rel=0, abs=1e-7)
                assert (start[index], end[index]) == pytest.approx(reference.timeline.last_intervals[rid], rel=0, abs=1e-7)
                assert moved[index] == reference.timeline.bytes_moved.get(rid, 0)
        assert {ids[i] for i in np.flatnonzero(touched)} == reference.timeline._touched
