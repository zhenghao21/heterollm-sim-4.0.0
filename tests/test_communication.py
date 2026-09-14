import unittest

from heterollm_sim.communication import (
    TopologyRouter,
    choose_collective_algorithm,
    plan_collective,
)
from heterollm_sim.ir import ComponentSpec, HardwareSpec, LinkSpec, PortSpec


def _port(port_id, protocol="nvlink"):
    return PortSpec(
        port_id=port_id,
        protocol=protocol,
        role="peer",
        bandwidth_gbps=100.0,
        max_links=4,
    )


def _coherent_dma_router(
    *,
    explicit=True,
    mismatched_span=False,
    payload_only=False,
    intermediate_kind="cpu",
    duplicate_dma_resource=False,
):
    endpoint_dma_metadata = (
        {"dma_bandwidth_gbps": 400.0, "dma_resource_id": "shared.dma"}
        if duplicate_dma_resource
        else {}
    )
    components = (
        ComponentSpec(
            "hostmem0",
            "host_memory",
            ports=(_port("ddr0", "DDR"),),
            read_bandwidth_gbps=640.0,
            write_bandwidth_gbps=640.0,
            metadata=endpoint_dma_metadata,
        ),
        ComponentSpec(
            "cpu0",
            intermediate_kind,
            ports=(_port("ddr0", "DDR"), _port("pcie0", "PCIe")),
        ),
        ComponentSpec(
            "gpu0",
            "gpu",
            ports=(_port("pcie0", "PCIe"), _port("vram0", "HBM")),
        ),
        ComponentSpec(
            "vram0",
            "hbm",
            ports=(_port("gpu0", "HBM"),),
            read_bandwidth_gbps=8_000.0,
            write_bandwidth_gbps=8_000.0,
            metadata=endpoint_dma_metadata,
        ),
    )

    def metadata(span_id):
        if not explicit:
            return {}
        return {
            "transfer_execution": "coherent_dma",
            "coherent_dma_span_id": span_id,
        }

    links = (
        LinkSpec(
            "cpu-hostmem-ddr",
            "cpu0",
            "ddr0",
            "hostmem0",
            "ddr0",
            "DDR",
            bandwidth_gbps=640.0,
            latency_ns=80.0,
            payload="coherent_dma" if payload_only else None,
            metadata=metadata("host-to-vram"),
        ),
        LinkSpec(
            "cpu-gpu-pcie",
            "cpu0",
            "pcie0",
            "gpu0",
            "pcie0",
            "PCIe",
            bandwidth_gbps=400.0,
            latency_ns=800.0,
            payload="coherent_dma" if payload_only else None,
            metadata=metadata("host-to-vram"),
        ),
        LinkSpec(
            "gpu-vram-hbm",
            "gpu0",
            "vram0",
            "vram0",
            "gpu0",
            "HBM",
            bandwidth_gbps=8_000.0,
            latency_ns=100.0,
            payload="coherent_dma" if payload_only else None,
            metadata=metadata(
                "other-span" if mismatched_span else "host-to-vram"
            ),
        ),
    )
    return TopologyRouter(HardwareSpec("coherent-dma", components, links))


class CommunicationTests(unittest.TestCase):
    def setUp(self):
        components = tuple(
            ComponentSpec(
                component_id="gpu{}".format(index),
                kind="gpu",
                ports=(_port("p0"), _port("p1")),
                read_bandwidth_gbps=200.0,
                write_bandwidth_gbps=160.0,
                metadata={"dma_bandwidth_gbps": 80.0},
            )
            for index in range(3)
        )
        links = (
            LinkSpec("l01", "gpu0", "p0", "gpu1", "p0", "nvlink", bandwidth_gbps=100.0, latency_ns=10.0),
            LinkSpec("l12", "gpu1", "p1", "gpu2", "p0", "nvlink", bandwidth_gbps=100.0, latency_ns=10.0),
            LinkSpec("l02", "gpu0", "p1", "gpu2", "p1", "nvlink", bandwidth_gbps=10.0, latency_ns=1.0),
        )
        self.router = TopologyRouter(HardwareSpec("mesh", components, links))

    def test_route_uses_payload_aware_lowest_time_path(self):
        small = self.router.route("gpu0", "gpu2", 1)
        large = self.router.route("gpu0", "gpu2", 1_000_000)
        self.assertEqual([hop.link_id for hop in small], ["l02"])
        self.assertEqual([hop.link_id for hop in large], ["l01", "l12"])
        self.assertIs(small, self.router.route("gpu0", "gpu2", 1))
        self.assertIs(large, self.router.route("gpu0", "gpu2", 1_000_000))

    def test_route_rejects_legacy_policy_aliases(self):
        for policy in ("shortest", "minimum_time"):
            with self.subTest(policy=policy), self.assertRaisesRegex(
                ValueError, "unsupported routing policy"
            ):
                self.router.route("gpu0", "gpu1", 1, policy=policy)

    def test_transfer_expands_endpoint_dma_and_link_phases(self):
        phases = self.router.transfer_phases("gpu0", "gpu1", 1024, name="kv")
        self.assertEqual(len(phases), 5)
        self.assertEqual(phases[0].metadata["event_kind"], "memory_read")
        self.assertEqual(phases[2].metadata["event_kind"], "transfer")
        self.assertEqual(phases[-1].metadata["event_kind"], "memory_write")
        self.assertTrue(all(phase.demands[0].service_ns > 0 for phase in phases))

    def test_explicit_complete_coherent_dma_span_collapses_to_one_phase(self):
        byte_count = 4096
        phases = _coherent_dma_router().transfer_phases(
            "hostmem0", "vram0", byte_count, name="weight_copy"
        )

        self.assertEqual(len(phases), 1)
        phase = phases[0]
        self.assertEqual(phase.metadata["transfer_execution"], "coherent_dma")
        self.assertEqual(phase.metadata["coherent_dma_span_id"], "host-to-vram")
        self.assertEqual(phase.metadata["bytes"], byte_count)
        self.assertEqual(
            phase.metadata["link_ids"],
            ("cpu-hostmem-ddr", "cpu-gpu-pcie", "gpu-vram-hbm"),
        )
        self.assertEqual(
            tuple(hop["source_component"] for hop in phase.metadata["route_hops"]),
            ("hostmem0", "cpu0", "gpu0"),
        )
        self.assertEqual(
            tuple(demand.resource_id for demand in phase.demands),
            (
                "component.hostmem0.read",
                "link.cpu-hostmem-ddr.hostmem0->cpu0",
                "link.cpu-gpu-pcie.cpu0->gpu0",
                "link.gpu-vram-hbm.gpu0->vram0",
                "component.vram0.write",
            ),
        )
        self.assertTrue(
            all(demand.bytes_moved == byte_count for demand in phase.demands)
        )
        self.assertEqual(
            phase.metadata["resource_directions"],
            {
                "component.hostmem0.read": "read",
                "link.cpu-hostmem-ddr.hostmem0->cpu0": "transfer",
                "link.cpu-gpu-pcie.cpu0->gpu0": "transfer",
                "link.gpu-vram-hbm.gpu0->vram0": "transfer",
                "component.vram0.write": "write",
            },
        )

    def test_strict_serialized_mode_keeps_each_dma_stage(self):
        router = _coherent_dma_router()
        phases = router.transfer_phases(
            "hostmem0",
            "vram0",
            4096,
            name="strict",
            coherent_dma_mode="strict_serialized",
        )
        # Strict mode exposes the physical read/DMA/link/DMA/write stages so
        # the planner can chain them and charge their summed latency.
        self.assertEqual(len(phases), 5)
        self.assertEqual(phases[0].metadata["event_kind"], "memory_read")
        self.assertEqual(phases[-1].metadata["event_kind"], "memory_write")
        self.assertTrue(
            all(phase.metadata.get("transfer_execution") != "coherent_dma" for phase in phases)
        )

    def test_router_rejects_unknown_coherent_dma_mode(self):
        with self.assertRaisesRegex(ValueError, "coherent_dma_mode"):
            TopologyRouter(_coherent_dma_router().hardware, coherent_dma_mode="foo")

    def test_payload_name_does_not_enable_coherent_dma(self):
        phases = _coherent_dma_router(
            explicit=False, payload_only=True
        ).transfer_phases("hostmem0", "vram0", 4096, name="legacy")

        self.assertEqual(len(phases), 5)
        self.assertNotIn("transfer_execution", phases[0].metadata)

    def test_incomplete_coherent_dma_span_fails_closed_to_legacy_serial(self):
        phases = _coherent_dma_router(mismatched_span=True).transfer_phases(
            "hostmem0", "vram0", 4096, name="incomplete"
        )

        self.assertEqual(len(phases), 5)
        self.assertEqual(
            [phase.metadata["event_kind"] for phase in phases],
            ["memory_read", "transfer", "transfer", "transfer", "memory_write"],
        )

    def test_coherent_dma_fails_closed_for_switch_or_duplicate_resource(self):
        cases = (
            (_coherent_dma_router(intermediate_kind="switch"), 5),
            (_coherent_dma_router(duplicate_dma_resource=True), 7),
        )
        for router, expected_phase_count in cases:
            with self.subTest(expected_phase_count=expected_phase_count):
                phases = router.transfer_phases(
                    "hostmem0", "vram0", 4096, name="fail_closed"
                )
                self.assertEqual(len(phases), expected_phase_count)
                self.assertTrue(
                    all(
                        phase.metadata.get("transfer_execution")
                        != "coherent_dma"
                        for phase in phases
                    )
                )

    def test_coherent_dma_fails_closed_for_fabric_switch_preset_kind(self):
        phases = _coherent_dma_router(
            intermediate_kind="fabric_switch"
        ).transfer_phases("hostmem0", "vram0", 4096, name="fabric_switch")

        self.assertEqual(len(phases), 5)
        self.assertTrue(
            all(
                phase.metadata.get("transfer_execution") != "coherent_dma"
                for phase in phases
            )
        )

    def test_coherent_dma_fails_closed_for_ethernet_switch_preset_kind(self):
        phases = _coherent_dma_router(
            intermediate_kind="ethernet_switch"
        ).transfer_phases("hostmem0", "vram0", 4096, name="ethernet_switch")

        self.assertEqual(len(phases), 5)
        self.assertTrue(
            all(
                phase.metadata.get("transfer_execution") != "coherent_dma"
                for phase in phases
            )
        )

    def test_zero_byte_transfer_is_a_resource_free_noop(self):
        self.assertEqual(
            self.router.transfer_phases("gpu0", "gpu1", 0, name="empty"),
            (),
        )
        with self.assertRaisesRegex(ValueError, "unknown route source component"):
            self.router.transfer_phases("missing", "gpu1", 0, name="empty")
        with self.assertRaisesRegex(ValueError, "unknown route target component"):
            self.router.transfer_phases("gpu0", "missing", 0, name="empty")

    def test_transfer_byte_counts_require_non_negative_integers(self):
        hop = self.router.route("gpu0", "gpu1", 1)[0]
        component = self.router.components["gpu0"]
        entry_points = (
            lambda value: hop.transfer_ns(value),
            lambda value: self.router.route("gpu0", "gpu1", value),
            lambda value: self.router.transfer_phases("gpu0", "gpu1", value),
            lambda value: TopologyRouter._endpoint_phase(
                component, value, read=True, name="read"
            ),
            lambda value: TopologyRouter._dma_phase(
                component, value, name="copy", direction="out"
            ),
        )
        for entry_point in entry_points:
            for invalid in (True, 1.5, -1):
                with self.subTest(entry_point=entry_point, invalid=invalid):
                    with self.assertRaisesRegex(
                        ValueError, "byte_count must be a non-negative integer"
                    ):
                        entry_point(invalid)

    def test_storage_endpoint_missing_required_direction_fails_closed(self):
        storage = ComponentSpec(
            component_id="hbf0",
            kind="hbf",
            read_bandwidth_gbps=128.0,
            write_bandwidth_gbps=0.0,
        )
        with self.assertRaisesRegex(ValueError, "positive write bandwidth"):
            TopologyRouter._endpoint_phase(
                storage, 4096, read=False, name="kv_offload"
            )

    def test_active_memory_may_use_the_declared_interface_without_endpoint_rate(self):
        hbm = ComponentSpec(
            component_id="hbm0",
            kind="hbm",
            read_bandwidth_gbps=0.0,
            write_bandwidth_gbps=0.0,
        )
        self.assertIsNone(
            TopologyRouter._endpoint_phase(
                hbm, 4096, read=True, name="rank_memory"
            )
        )

    def test_dma_metadata_without_bandwidth_is_not_silently_free(self):
        component = ComponentSpec(
            component_id="cpu0",
            kind="cpu",
            metadata={"dma_latency_ns": 250.0},
        )
        with self.assertRaisesRegex(ValueError, "no positive dma_bandwidth_gbps"):
            TopologyRouter._dma_phase(
                component, 4096, name="host_copy", direction="out"
            )

    def test_endpoint_queue_depth_only_overlaps_transaction_latency(self):
        serial = ComponentSpec(
            component_id="ssd_serial",
            kind="ssd",
            read_bandwidth_gbps=80.0,
            metadata={
                "read_latency_ns": 100.0,
                "transfer_granularity_bytes": 4096,
                "max_outstanding_requests": 1,
            },
        )
        queued = ComponentSpec(
            component_id="ssd_queued",
            kind="ssd",
            read_bandwidth_gbps=80.0,
            metadata={
                "read_latency_ns": 100.0,
                "transfer_granularity_bytes": 4096,
                "max_outstanding_requests": 4,
            },
        )
        byte_count = 8 * 4096
        serial_phase = TopologyRouter._endpoint_phase(
            serial, byte_count, read=True, name="read"
        )
        queued_phase = TopologyRouter._endpoint_phase(
            queued, byte_count, read=True, name="read"
        )
        self.assertIsNotNone(serial_phase)
        self.assertIsNotNone(queued_phase)
        self.assertEqual(serial_phase.metadata["latency_batches"], 8)
        self.assertEqual(queued_phase.metadata["latency_batches"], 2)
        self.assertAlmostEqual(
            serial_phase.demands[0].service_ns
            - queued_phase.demands[0].service_ns,
            600.0,
        )
        self.assertEqual(
            serial_phase.demands[0].bytes_moved,
            queued_phase.demands[0].bytes_moved,
        )

    def test_endpoint_and_link_metadata_fail_closed_on_invalid_numbers(self):
        invalid_endpoint = ComponentSpec(
            component_id="ssd-invalid",
            kind="ssd",
            read_bandwidth_gbps=80.0,
            metadata={"read_latency_ns": "unknown"},
        )
        with self.assertRaisesRegex(ValueError, "finite non-negative number"):
            TopologyRouter._endpoint_phase(
                invalid_endpoint, 4096, read=True, name="read"
            )

        components = (
            ComponentSpec("gpu-a", "gpu", ports=(_port("p0"),)),
            ComponentSpec("gpu-b", "gpu", ports=(_port("p0"),)),
        )
        invalid_link = LinkSpec(
            "invalid-energy",
            "gpu-a",
            "p0",
            "gpu-b",
            "p0",
            "nvlink",
            bandwidth_gbps=100.0,
            metadata={"energy_pj_per_byte": float("nan")},
        )
        with self.assertRaisesRegex(ValueError, "finite non-negative number"):
            TopologyRouter(HardwareSpec("invalid-link", components, (invalid_link,)))

    def test_ring_collective_has_explicit_rounds_and_routes(self):
        plan = plan_collective(
            self.router,
            "all_reduce",
            ("gpu0", "gpu1", "gpu2"),
            3_000_000,
            algorithm="ring",
        )
        self.assertEqual(plan.algorithm, "ring")
        self.assertEqual(len(plan.rounds), 4)
        self.assertEqual(len(plan.rounds[0].transfers), 3)
        self.assertTrue(all(item.route for item in plan.rounds[0].transfers))

    def test_collective_inputs_require_integer_bytes_and_participants(self):
        for invalid in (True, 1.5, -1):
            with self.subTest(tensor_bytes=invalid):
                with self.assertRaisesRegex(
                    ValueError, "tensor_bytes must be a non-negative integer"
                ):
                    plan_collective(
                        self.router,
                        "all_reduce",
                        ("gpu0", "gpu1"),
                        invalid,
                    )

        for invalid in (True, 1.5, 0, -1):
            with self.subTest(participant_count=invalid):
                with self.assertRaisesRegex(
                    ValueError, "participant_count must be a positive integer"
                ):
                    choose_collective_algorithm("auto", invalid, 1024)

    def test_collective_rejects_an_empty_participant_set(self):
        with self.assertRaisesRegex(ValueError, "participants must not be empty"):
            plan_collective(self.router, "all_reduce", (), 1024)

    def test_tree_collectives_preserve_reduce_and_shard_scatter_semantics(self):
        components = tuple(
            ComponentSpec(
                component_id="gpu{}".format(index),
                kind="gpu",
                ports=(_port("p0"), _port("p1")),
            )
            for index in range(4)
        )
        router = TopologyRouter(
            HardwareSpec(
                "tree-four",
                components,
                (
                    LinkSpec("l01", "gpu0", "p0", "gpu1", "p0", "nvlink"),
                    LinkSpec("l23", "gpu2", "p0", "gpu3", "p0", "nvlink"),
                    LinkSpec("l02", "gpu0", "p1", "gpu2", "p1", "nvlink"),
                ),
            )
        )

        plan = plan_collective(
            router,
            "all_reduce",
            ("gpu0", "gpu1", "gpu2", "gpu3"),
            10,
            algorithm="tree",
        )

        self.assertEqual(
            [
                tuple(
                    (transfer.source_component, transfer.target_component)
                    for transfer in round_spec.transfers
                )
                for round_spec in plan.rounds
            ],
            [
                (("gpu1", "gpu0"), ("gpu3", "gpu2")),
                (("gpu2", "gpu0"),),
                (("gpu0", "gpu2"),),
                (("gpu0", "gpu1"), ("gpu2", "gpu3")),
            ],
        )
        self.assertEqual(
            [
                tuple(transfer.byte_count for transfer in round_spec.transfers)
                for round_spec in plan.rounds
            ],
            [(12, 12), (12,), (10,), (10, 10)],
        )

        reduce_scatter = plan_collective(
            router,
            "reduce_scatter",
            ("gpu0", "gpu1", "gpu2", "gpu3"),
            10,
            algorithm="tree",
        )
        self.assertEqual(
            [
                tuple(
                    (
                        transfer.source_component,
                        transfer.target_component,
                        transfer.byte_count,
                    )
                    for transfer in round_spec.transfers
                )
                for round_spec in reduce_scatter.rounds
            ],
            [
                (("gpu1", "gpu0", 12), ("gpu3", "gpu2", 12)),
                (("gpu2", "gpu0", 12),),
                (("gpu0", "gpu2", 6),),
                (("gpu0", "gpu1", 3), ("gpu2", "gpu3", 3)),
            ],
        )
        self.assertEqual(
            sum(
                transfer.byte_count
                for round_spec in reduce_scatter.rounds
                for transfer in round_spec.transfers
            ),
            48,
        )

    def test_tree_reduce_scatter_handles_non_power_of_two_participants(self):
        plan = plan_collective(
            self.router,
            "reduce_scatter",
            ("gpu0", "gpu1", "gpu2"),
            10,
            algorithm="tree",
        )

        self.assertEqual(
            [
                tuple(
                    (
                        transfer.source_component,
                        transfer.target_component,
                        transfer.byte_count,
                    )
                    for transfer in round_spec.transfers
                )
                for round_spec in plan.rounds
            ],
            [
                (("gpu1", "gpu0", 12),),
                (("gpu2", "gpu0", 12),),
                (("gpu0", "gpu2", 4),),
                (("gpu0", "gpu1", 4),),
            ],
        )
        self.assertEqual(
            sum(
                transfer.byte_count
                for round_spec in plan.rounds
                for transfer in round_spec.transfers
            ),
            32,
        )

    def test_ring_all_to_all_covers_every_peer_and_rejects_tree(self):
        participants = tuple("gpu{}".format(index) for index in range(4))
        components = tuple(
            ComponentSpec(
                component_id=component_id,
                kind="gpu",
                ports=(_port("p0"),),
            )
            for component_id in participants
        )
        links = tuple(
            LinkSpec(
                "l{}{}".format(source, target),
                "gpu{}".format(source),
                "p0",
                "gpu{}".format(target),
                "p0",
                "nvlink",
                bandwidth_gbps=100.0,
                latency_ns=10.0,
            )
            for source in range(4)
            for target in range(source + 1, 4)
        )
        router = TopologyRouter(HardwareSpec("complete-four", components, links))

        plan = plan_collective(
            router,
            "all_to_all",
            participants,
            4096,
            algorithm="ring",
        )

        seen_targets = {source: set() for source in participants}
        self.assertEqual(plan.algorithm, "ring")
        self.assertEqual(len(plan.rounds), 3)
        for round_spec in plan.rounds:
            self.assertEqual(len(round_spec.transfers), 4)
            for transfer in round_spec.transfers:
                self.assertNotEqual(
                    transfer.source_component,
                    transfer.target_component,
                )
                seen_targets[transfer.source_component].add(
                    transfer.target_component
                )
        for source in participants:
            self.assertEqual(seen_targets[source], set(participants) - {source})

        with self.assertRaisesRegex(ValueError, "tree is not a valid all_to_all"):
            plan_collective(
                router,
                "all_to_all",
                participants,
                4096,
                algorithm="tree",
            )

    def test_invalid_or_disconnected_routes_fail(self):
        with self.assertRaisesRegex(ValueError, "unknown route"):
            self.router.route("missing", "gpu0", 1)


if __name__ == "__main__":
    unittest.main()
