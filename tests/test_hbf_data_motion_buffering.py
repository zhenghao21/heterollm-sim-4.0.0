"""Buffered write acknowledgements and transaction rollback stay observable."""
import unittest

from heterollm_sim.data_motion import AccessKind, PhysicalRuntimeContext, resolve_service
from heterollm_sim.ir import ComponentSpec
from heterollm_sim.memory_types import make_hbf_config


class BufferedServiceTests(unittest.TestCase):
    def test_ack_and_failed_transaction_do_not_publish_staged_mutations(self):
        config = make_hbf_config(capacity_bytes=16 * 16384, page_bytes=16384,
            write_buffer_bytes=16384, host_granularity_bytes=4096,
            metadata={})
        component = ComponentSpec("hbf", "hbf", capacity_bytes=config.capacity_bytes,
            read_bandwidth_gbps=800, write_bandwidth_gbps=80,
            metadata={"physical_memory_config": config})
        service = resolve_service(component)
        runtime = PhysicalRuntimeContext()
        result = service.price(AccessKind.WRITE, 4096, page_offset_bytes=0,
                               arrival_ns=0, runtime=runtime)
        self.assertEqual(result["write_completion"], "host_buffer_ack_pending_media_drain")
        snapshot = runtime.snapshot()
        core = runtime.runtimes[service.physical_owner].core
        before = core.pending_write_buffer_bytes
        service.price(AccessKind.WRITE, 4096, page_offset_bytes=4096,
                      arrival_ns=result["completion_ns"], runtime=runtime)
        self.assertGreater(core.pending_write_buffer_bytes, before)
        runtime.restore(snapshot)
        self.assertEqual(core.pending_write_buffer_bytes, before)
        self.assertEqual(next(iter(core._staged_pages.values())).logical_bytes, 4096)


if __name__ == "__main__":
    unittest.main()
