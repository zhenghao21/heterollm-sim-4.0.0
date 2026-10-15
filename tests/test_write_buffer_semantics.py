"""Capacity-controlled finite NAND write-buffer semantics."""

import unittest

from heterollm_sim.memory_types import AccessRequest, Operation, make_hbf_config
from heterollm_sim.nand_core import NandCore


PAGE = 16 * 1024


def _config(**kwargs):
    values = dict(
        channels=1,
        targets_per_channel=1,
        dies_per_target=1,
        luns_per_die=1,
        planes_per_lun=1,
        blocks_per_plane=1,
        pages_per_block=8,
        page_bytes=PAGE,
        host_granularity_bytes=4 * 1024,
        host_bandwidth_gb_s=10.0, host_read_bandwidth_gb_s=10.0, host_write_bandwidth_gb_s=10.0,
        internal_bandwidth_gb_s=10.0,
        page_read_ns=100.0,
        page_program_ns=200.0,
        write_buffer_bytes=PAGE,
        metadata={},
    )
    values.update(kwargs)
    return make_hbf_config(**values)


class WriteBufferSemanticsTests(unittest.TestCase):
    def test_small_writes_merge_and_reads_hit_staging_until_drain(self):
        core = NandCore(_config())
        first = core.submit(AccessRequest("w0", Operation.WRITE, 0, 4096))
        second = core.submit(AccessRequest("w1", Operation.WRITE, 4096, 4096))

        self.assertEqual(first.counters["write_completion"], "host_buffer_ack_pending_media_drain")
        self.assertEqual(first.pages_programmed, 0)
        self.assertEqual(second.merge_bytes, 4096)
        self.assertEqual(core.buffered_page_count, 1)
        self.assertEqual(core.pending_write_buffer_bytes, 8192)

        staged_read = core.submit(AccessRequest("r0", Operation.READ, 0, 8192))
        self.assertEqual(staged_read.staging_read_hits, 1)
        self.assertEqual(staged_read.physical_read_bytes, 0)
        self.assertNotIn("PAGE_READ", {stage.name for stage in staged_read.stages})

        drained = core.flush_write_buffer()
        self.assertEqual(drained["flushed_pages"], 1)
        self.assertEqual(drained["merge_bytes"], 4096)
        self.assertEqual(drained["physical_programs"], 1)
        self.assertEqual(drained["physical_write_bytes"], PAGE)
        self.assertEqual(drained["physical_read_bytes"], PAGE)
        self.assertGreater(drained["media_drain_completion_ns"], second.host_ack_completion_ns)
        self.assertEqual(core.pending_write_buffer_bytes, 0)

        media_read = core.submit(AccessRequest("r1", Operation.READ, 0, PAGE))
        self.assertEqual(media_read.staging_read_hits, 0)
        self.assertEqual(media_read.physical_read_bytes, PAGE)

    def test_full_buffer_auto_drains_before_next_host_ack(self):
        core = NandCore(_config(write_buffer_bytes=8192))
        core.submit(AccessRequest("w0", Operation.WRITE, 0, 4096))
        core.submit(AccessRequest("w1", Operation.WRITE, 4096, 4096))
        next_write = core.submit(AccessRequest("w2", Operation.WRITE, 8192, 4096))

        self.assertGreater(next_write.write_buffer_wait_ns, 0.0)
        self.assertEqual(next_write.counters["write_completion"], "host_buffer_ack_pending_media_drain")
        self.assertLessEqual(next_write.write_buffer_occupancy_bytes, 8192)

    def test_zero_buffer_capacity_uses_write_through(self):
        core = NandCore(_config(write_buffer_bytes=0))
        result = core.submit(AccessRequest("w", Operation.WRITE, 0, 4096))

        self.assertEqual(result.counters["write_completion"], "media_program_complete")
        self.assertEqual(result.pages_programmed, 1)
        self.assertEqual(core.buffered_page_count, 0)
        self.assertEqual(core.flush_write_buffer()["buffered_write_mode"], False)

    def test_partial_staging_cannot_supply_unwritten_bytes(self):
        core = NandCore(_config())
        core.submit(AccessRequest("w", Operation.WRITE, 0, 4096))
        untouched = core.submit(AccessRequest("r", Operation.READ, 8192, 4096))
        self.assertEqual(untouched.staging_read_hits, 0)
        self.assertEqual(untouched.physical_read_bytes, PAGE)

    def test_explicit_early_drain_waits_for_host_payload(self):
        core = NandCore(_config(front_ns=0.0, host_bandwidth_gb_s=0.1, host_read_bandwidth_gb_s=0.1, host_write_bandwidth_gb_s=0.1))
        write = core.submit(AccessRequest("w", Operation.WRITE, 0, PAGE))
        drained = core.flush_write_buffer(now_ns=0.0)
        self.assertGreaterEqual(drained["media_drain_start_ns"], write.completion_ns)

    def test_distinct_channels_drain_in_parallel(self):
        core = NandCore(_config(channels=2, write_buffer_bytes=2 * PAGE))
        write = core.submit(AccessRequest("w", Operation.WRITE, 0, 2 * PAGE))
        drained = core.flush_write_buffer(now_ns=write.completion_ns)
        self.assertAlmostEqual(drained["media_drain_completion_ns"] - write.completion_ns,
                               PAGE / core.config.internal_bandwidth_gb_s + core.config.page_program_ns)
        self.assertEqual(drained["pages_read"], 0)

    def test_auto_drain_includes_physical_work_in_triggering_request(self):
        core = NandCore(_config(write_buffer_bytes=4096))
        core.submit(AccessRequest("w0", Operation.WRITE, 0, 4096))
        second = core.submit(AccessRequest("w1", Operation.WRITE, PAGE, 4096))
        self.assertEqual(second.pages_programmed, 1)
        self.assertEqual(second.physical_read_bytes, PAGE)
        self.assertEqual(second.internal_transfer_bytes, 2 * PAGE)
        self.assertGreater(second.counters["resource_busy_ns"]["nand:array:0"], 0)

    def test_summary_large_read_matches_staged_detailed_read(self):
        config = _config(pages_per_block=256)
        detailed = NandCore(config)
        summary = NandCore(config, capture_details=False)
        for core in (detailed, summary):
            core.submit(AccessRequest("w", Operation.WRITE, 0, PAGE))
        request = AccessRequest("r", Operation.READ, 0, 128 * PAGE)
        left, right = detailed.submit(request), summary.submit(request)
        self.assertEqual(right.staging_read_hits, 1)
        self.assertEqual(left.physical_read_bytes, right.physical_read_bytes)
        self.assertEqual(left.completion_ns, right.completion_ns)


if __name__ == "__main__":
    unittest.main()
