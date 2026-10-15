"""Serving-level regression coverage for physical paged KV migration."""

from dataclasses import replace
import unittest
from unittest.mock import patch

from heterollm_sim.config import kv_policy_from_dict
from heterollm_sim.ir import ComponentSpec, KVCachePolicy, LinkSpec, PortSpec
from heterollm_sim.memory_types import make_hbf_config
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serving import BatchCost, _OnlineRuntime, compile_serving_plan


class PagedPoolServingTests(unittest.TestCase):
    def _runtime(self):
        plan = compile_serving_plan(self._scenario_with_hbf())
        runtime = _OnlineRuntime(
            plan, lambda *_: BatchCost(0.0),
        )
        return runtime, runtime.states[plan.requests[0].request_id]

    def test_transfer_sync_preserves_physical_cohort_completion(self):
        runtime, request = self._runtime()
        self.assertTrue(runtime.ledger.resize(request, 3))
        self.assertTrue(runtime.ledger.offload(request))
        transfers = runtime.ledger.paged_pool.events
        completion = max(item.completion_ns for item in transfers)
        self.assertGreater(sum(item.duration_ns for item in transfers), completion)
        runtime._sync_paged_pool_events()
        self.assertEqual(runtime.now, completion)
        self.assertEqual(runtime.swap_transfer_time_ns, completion)
        observed = [event.timestamp_ns for event in runtime.events
                    if event.event_type == "kv_pool_transfer_complete"]
        self.assertEqual(observed, [item.completion_ns for item in transfers])

    def test_transfer_sync_survives_page_release(self):
        runtime, request = self._runtime()
        self.assertTrue(runtime.ledger.resize(request, 1))
        self.assertTrue(runtime.ledger.offload(request))
        runtime.ledger.discard_offload(request)
        runtime._sync_paged_pool_events()
        self.assertGreater(runtime.now, 0)

    def test_failed_shrink_restores_page_identity_and_physical_binding(self):
        runtime, request = self._runtime()
        self.assertTrue(runtime.ledger.resize(request, 2))
        pool = runtime.ledger.paged_pool
        executor = pool.transfer_executor
        identities = tuple(page.logical_page_id for page in pool.pages())
        addresses = tuple(executor.address(page) for page in pool.pages())
        physical = dict(runtime.physical_ledger.used_bytes)
        with patch.object(executor, "sync_pages", side_effect=RuntimeError("sync failed")):
            self.assertFalse(runtime.ledger.resize(request, 1))
        self.assertEqual(tuple(page.logical_page_id for page in pool.pages()), identities)
        self.assertEqual(tuple(executor.address(page) for page in pool.pages()), addresses)
        self.assertEqual(runtime.physical_ledger.used_bytes, physical)
        self.assertEqual(request.kv_pages, 2)

    def test_temporary_reservation_cannot_shrink_persistent_pages(self):
        runtime, request = self._runtime()
        self.assertTrue(runtime.ledger.resize(request, 2))
        self.assertTrue(runtime.ledger.reserve_temporary(request, 1))
        self.assertEqual(len(runtime.ledger.paged_pool.pages()), 2)
        self.assertEqual(request.kv_pages, 2)
        self.assertEqual(request.temporary_kv_pages, 0)

    def test_relay_component_survives_json_policy_normalization(self):
        policy = kv_policy_from_dict({
            "layout_mode": "paged_pool",
            "pool_components": ["hbm0", "hbf0"],
            "relay_components": ["hostmem0"],
        })
        self.assertEqual(policy.pool_components, ("hbm0", "hbf0"))
        self.assertEqual(policy.relay_components, ("hostmem0",))

    @staticmethod
    def _scenario_with_hbf():
        base = build_reference_scenario()
        capacity = 8 * 1024**3
        physical = make_hbf_config(
            capacity_bytes=capacity,
            channels=8,
            parallel_units=8,
            page_bytes=16 * 1024,
            host_bandwidth_gb_s=488.0,
            host_read_bandwidth_gb_s=488.0,
            host_write_bandwidth_gb_s=27.2,
        )
        hbf = ComponentSpec(
            component_id="hbf0",
            kind="hbf",
            ports=(
                PortSpec(
                    port_id="pcie0",
                    protocol="PCIe",
                    role="endpoint",
                    version="5.0",
                    lanes=16,
                    bandwidth_gbps=512.0,
                    payload="coherent_dma",
                ),
            ),
            capacity_bytes=capacity,
            read_bandwidth_gbps=488.0 * 8.0,
            write_bandwidth_gbps=27.2 * 8.0,
            metadata={
                "physical_memory_config": physical,
                "access_mode": "remote_flash",
            },
        )
        hardware = replace(
            base.hardware,
            components=base.hardware.components + (hbf,),
            links=base.hardware.links
            + (
                LinkSpec(
                    link_id="cpu-hbf-pcie",
                    source_component="cpu0",
                    source_port="pcie0",
                    target_component="hbf0",
                    target_port="pcie0",
                    protocol="PCIe",
                    version="5.0",
                    lanes=16,
                    bandwidth_gbps=512.0,
                    latency_ns=100.0,
                    payload="coherent_dma",
                ),
            ),
            require_connected=False,
        )
        policy = KVCachePolicy(
            cache_component="hbm0",
            offload_component="hbf0",
            tokens_per_page=16,
            dtype="int8",
            layout_mode="paged_pool",
            pool_components=("hbm0", "hbf0"),
            preemption_mode="swap",
        )
        return replace(
            base,
            hardware=hardware,
            placement=replace(
                base.placement,
                hardware_name=hardware.name,
                kv_policy=policy,
            ),
        )

    def test_paged_pool_offload_restore_uses_physical_executor(self):
        plan = compile_serving_plan(self._scenario_with_hbf())
        runtime = _OnlineRuntime(
            plan,
            lambda *_: BatchCost(0.0),
        )
        request = runtime.states[plan.requests[0].request_id]
        pool = runtime.ledger.paged_pool
        self.assertIsNotNone(pool)
        self.assertIsNotNone(pool.transfer_executor)

        self.assertTrue(runtime.ledger.resize(request, 1))
        page = pool.request_pages(request.spec.request_id)[0]
        self.assertEqual(page.owner_component, "hbm0")

        self.assertTrue(runtime.ledger.offload(request))
        runtime._sync_paged_pool_events()
        self.assertEqual(request.kv_pages, 0)
        self.assertEqual(request.swapped_pages, 1)
        self.assertEqual(page.owner_component, "hbf0")
        self.assertEqual(len(pool.transfer_executor.events), 1)

        self.assertTrue(runtime._resume_swap(request))
        self.assertEqual(request.kv_pages, 1)
        self.assertEqual(request.swapped_pages, 0)
        self.assertEqual(page.owner_component, "hbm0")
        self.assertEqual(len(pool.transfer_executor.events), 2)
        self.assertGreater(runtime.swap_transfer_time_ns, 0.0)

    def test_paged_pool_capacity_keeps_storage_tier_in_aggregate_budget(self):
        """The logical pool must not be clipped to the active HBM tier."""

        base = self._scenario_with_hbf()
        page_bytes = 16 * 1024  # reference int8 KV: 1024 B/token, 16 tokens/page
        hbm_bytes = 2 * page_bytes
        hbf_bytes = 8 * page_bytes

        hbm = base.hardware.get_component("hbm0")
        hbm_metadata = dict(hbm.metadata)
        hbm_physical = dict(hbm_metadata["physical_memory_config"])
        hbm_physical["capacity_bytes"] = hbm_bytes
        hbm_metadata["physical_memory_config"] = hbm_physical
        hbm = replace(hbm, capacity_bytes=hbm_bytes, metadata=hbm_metadata)

        hbf = base.hardware.get_component("hbf0")
        hbf_metadata = dict(hbf.metadata)
        hbf_physical = replace(
            hbf_metadata["physical_memory_config"], capacity_bytes=hbf_bytes
        )
        hbf_metadata["physical_memory_config"] = hbf_physical
        hbf = replace(hbf, capacity_bytes=hbf_bytes, metadata=hbf_metadata)

        hardware = replace(
            base.hardware,
            components=tuple(
                hbm
                if component.component_id == "hbm0"
                else hbf
                if component.component_id == "hbf0"
                else component
                for component in base.hardware.components
            ),
        )
        scenario = replace(
            base,
            hardware=hardware,
            placement=replace(base.placement, hardware_name=hardware.name),
        )

        plan = compile_serving_plan(scenario)
        self.assertEqual(plan.kv_policy.bytes_per_page, page_bytes)
        self.assertEqual(plan.kv_policy.capacity_pages, 10)
        self.assertEqual(plan.kv_policy.capacity_bytes, hbm_bytes + hbf_bytes)
        self.assertEqual(
            plan.kv_policy.capacity_bytes_by_component,
            {"hbm0": hbm_bytes, "hbf0": hbf_bytes},
        )

        runtime = _OnlineRuntime(
            plan,
            lambda *_: BatchCost(0.0),
        )
        resource = runtime._resource_snapshot([])
        self.assertEqual(resource["kv_capacity_pages"], 10)
        self.assertEqual(resource["kv_device_resident_capacity_pages"], 2)
        self.assertEqual(resource["kv_device_resident_capacity_bytes"], hbm_bytes)
        self.assertEqual(
            runtime._result().kv_metrics.effective_physical_capacity_bytes,
            hbm_bytes,
        )

    def test_relay_path_is_propagated_and_mixed_page_owners_are_charged(self):
        scenario = self._scenario_with_hbf()
        policy = replace(
            scenario.placement.kv_policy,
            relay_components=("hostmem0",),
        )
        scenario = replace(
            scenario,
            placement=replace(scenario.placement, kv_policy=policy),
        )
        plan = compile_serving_plan(scenario)
        runtime = _OnlineRuntime(
            plan,
            lambda *_: BatchCost(0.0),
        )
        executor = runtime._paged_pool_transfer_executor
        self.assertIsNotNone(executor)
        self.assertEqual(executor.relay_components, ("hostmem0",))
        request = runtime.states[plan.requests[0].request_id]
        pool = runtime.ledger.paged_pool
        self.assertTrue(runtime.ledger.resize(request, 2))
        pages = pool.request_pages(request.spec.request_id)
        self.assertEqual(len(pages), 2)
        self.assertTrue(pool.offload_page(pages[0].logical_page_id, "hbf0"))
        owners = [owner for owner, _access in pool.cache_accesses(
            request.spec.request_id,
            first_token=0,
            token_count=32,
        )]
        self.assertEqual(owners, ["hbf0", "hbm0"])
        read_before = executor.counters["read_bytes"]
        executor.read_page(pages[0])
        executor.read_page(pages[1])
        self.assertEqual(
            executor.counters["read_bytes"] - read_before,
            2 * pages[0].bytes,
        )
        self.assertGreater(executor.counters["relay_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
