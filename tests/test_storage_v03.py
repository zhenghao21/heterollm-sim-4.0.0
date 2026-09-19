import math
import unittest
from dataclasses import replace
from itertools import product
from types import SimpleNamespace

from heterollm_sim.communication import TopologyRouter
from heterollm_sim.config import hardware_from_dict
from heterollm_sim.control_plane_planner import PlacementPolicy, plan_runtime_placement
from heterollm_sim.ir import (
    SCHEMA_VERSION,
    ComponentSpec,
    HardwareSpec,
    KVCachePolicy,
    LinkSpec,
    ParallelSpec,
    PortSpec,
    RankMappingSpec,
)
from heterollm_sim.parallel import build_parallel_plan
from heterollm_sim.planner import (
    compile_scenario,
    compile_serving_cohort_schedule,
    estimate_serving_cohort_cost,
    validate_scenario,
)
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.serving import _kv_bytes_per_token
from heterollm_sim.topology import validate_topology


def _port(port_id, protocol, role, *, payload=None, bandwidth=128.0):
    return PortSpec(
        port_id=port_id,
        protocol=protocol,
        role=role,
        bandwidth_gbps=bandwidth,
        payload=payload,
    )


def _two_component_storage_hardware(kind, protocol):
    if protocol.lower() == "ucie":
        gpu_role = storage_role = "endpoint"
        payload = "streaming"
        gpu_package, gpu_die = "pkg0", "gpu_die"
        storage_package, storage_die = "pkg0", "storage_die"
    elif protocol.lower() == "cxl":
        gpu_role, storage_role = "host", "device"
        payload = None
        gpu_package = gpu_die = storage_package = storage_die = ""
    else:
        gpu_role, storage_role = "root", "endpoint"
        payload = None
        gpu_package = gpu_die = storage_package = storage_die = ""
    return HardwareSpec(
        name="storage-link",
        components=(
            ComponentSpec(
                "gpu0",
                "gpu",
                (_port("storage", protocol, gpu_role, payload=payload),),
                cost_profile_id="legacy-gpu",
                package_id=gpu_package,
                die_id=gpu_die,
            ),
            ComponentSpec(
                "storage0",
                kind,
                (_port("host", protocol, storage_role, payload=payload),),
                package_id=storage_package,
                die_id=storage_die,
                capacity_bytes=1 << 20,
                read_bandwidth_gbps=64.0,
                write_bandwidth_gbps=64.0,
            ),
        ),
        links=(
            LinkSpec(
                "storage-link",
                "gpu0",
                "storage",
                "storage0",
                "host",
                protocol,
                bandwidth_gbps=64.0,
                latency_ns=5.0,
                payload=payload,
            ),
        ),
    )


def _add_storage(scenario, kind, protocol):
    storage_id = "{}0".format(kind)
    gpu = scenario.hardware.get_component("gpu0")
    if protocol.lower() == "ucie":
        gpu_port = _port("{}-port".format(storage_id), protocol, "endpoint", payload="streaming")
        storage_port = _port("host", protocol, "endpoint", payload="streaming")
        storage = ComponentSpec(
            storage_id,
            kind,
            (storage_port,),
            package_id=gpu.package_id,
            die_id="{}_die".format(storage_id),
            capacity_bytes=1 << 30,
            read_bandwidth_gbps=64.0,
            write_bandwidth_gbps=32.0,
            metadata={
                "read_latency_ns": 100.0,
                "write_latency_ns": 200.0,
                "transfer_granularity_bytes": 4096,
            },
        )
        payload = "streaming"
    else:
        gpu_port = _port("{}-port".format(storage_id), protocol, "root")
        storage_port = _port("host", protocol, "endpoint")
        storage = ComponentSpec(
            storage_id,
            kind,
            (storage_port,),
            capacity_bytes=1 << 30,
            read_bandwidth_gbps=32.0,
            write_bandwidth_gbps=16.0,
            metadata={"read_latency_ns": 500.0, "write_latency_ns": 750.0},
        )
        payload = None
    updated_gpu = replace(gpu, ports=gpu.ports + (gpu_port,))
    components = tuple(
        updated_gpu if component.component_id == gpu.component_id else component
        for component in scenario.hardware.components
    ) + (storage,)
    link = LinkSpec(
        "gpu-{}".format(storage_id),
        gpu.component_id,
        gpu_port.port_id,
        storage_id,
        storage_port.port_id,
        protocol,
        bandwidth_gbps=32.0,
        latency_ns=20.0,
        payload=payload,
    )
    hardware = replace(
        scenario.hardware,
        components=components,
        links=scenario.hardware.links + (link,),
    )
    return replace(scenario, hardware=hardware), storage_id


def _multi_rank_cim_weight_backing_scenario():
    scenario = build_reference_scenario()
    hardware_name = "tp-pp-ep-cim-backing"
    tp_degree = 2
    pp_degree = 2
    ep_degree = 2
    world_size = tp_degree * pp_degree * ep_degree
    components = [
        scenario.hardware.get_component("cpu0"),
        scenario.hardware.get_component("hostmem0"),
        scenario.hardware.get_component("hbm0"),
    ]
    links = [
        next(
            link
            for link in scenario.hardware.links
            if link.link_id == "cpu-hostmem-ddr"
        ),
        next(
            link
            for link in scenario.hardware.links
            if link.link_id == "gpu-hbm0"
        ),
    ]
    hbf = ComponentSpec(
        "hbf0",
        "hbf",
        (
            PortSpec(
                "ucie",
                "UCIe",
                "endpoint",
                bandwidth_gbps=256.0,
                max_links=world_size * 2,
                payload="streaming",
            ),
        ),
        package_id="package0",
        die_id="hbf_die",
        capacity_bytes=1 << 30,
        read_bandwidth_gbps=128.0,
        metadata={"read_latency_ns": 25.0, "transfer_granularity_bytes": 64},
    )
    components.append(hbf)
    for index in range(world_size):
        gpu_id = "gpu{}".format(index)
        cim_id = "cim{}".format(index)
        memory_id = "sram{}".format(index)
        gpu_ports = (
            PortSpec(
                "nv",
                "NVLink",
                "endpoint",
                bandwidth_gbps=400.0,
                max_links=world_size,
            ),
            PortSpec(
                "cim",
                "UCIe",
                "endpoint",
                bandwidth_gbps=256.0,
                payload="streaming",
            ),
            PortSpec(
                "weights",
                "UCIe",
                "endpoint",
                bandwidth_gbps=128.0,
                payload="streaming",
            ),
            PortSpec(
                "memory",
                "UCIe",
                "endpoint",
                bandwidth_gbps=256.0,
                payload="streaming",
            ),
        )
        if index == 0:
            gpu_ports += (
                PortSpec(
                    "pcie0",
                    "PCIe",
                    "endpoint",
                    bandwidth_gbps=64.0,
                ),
                next(
                    port
                    for port in scenario.hardware.get_component("gpu0").ports
                    if port.port_id == "hbm0"
                ),
            )
        components.append(
            ComponentSpec(
                gpu_id,
                "gpu",
                gpu_ports,
                cost_profile_id="legacy-gpu",
                package_id="package0",
                die_id="{}_die".format(gpu_id),
                capacity_bytes=64 * 1024 * 1024,
                peak_ops_per_s=120_000_000_000_000.0,
            )
        )
        components.append(
            ComponentSpec(
                cim_id,
                "digital_sram_cim",
                (
                    PortSpec(
                        "gpu",
                        "UCIe",
                        "endpoint",
                        bandwidth_gbps=256.0,
                        payload="streaming",
                    ),
                    PortSpec(
                        "weights",
                        "UCIe",
                        "endpoint",
                        bandwidth_gbps=128.0,
                        payload="streaming",
                    ),
                ),
                cost_profile_id="legacy-cim",
                package_id="package0",
                die_id="{}_die".format(cim_id),
                capacity_bytes=512 * 1024 * 1024,
                write_bandwidth_gbps=128.0,
                metadata={"write_latency_ns": 10.0},
            )
        )
        components.append(
            ComponentSpec(
                memory_id,
                "sram",
                (
                    PortSpec(
                        "gpu",
                        "UCIe",
                        "endpoint",
                        bandwidth_gbps=256.0,
                        payload="streaming",
                    ),
                ),
                package_id="package0",
                die_id="{}_die".format(memory_id),
                capacity_bytes=64 * 1024 * 1024,
                read_bandwidth_gbps=256.0,
                write_bandwidth_gbps=256.0,
            )
        )
        links.extend(
            (
                LinkSpec(
                    "gpu{}-cim{}".format(index, index),
                    gpu_id,
                    "cim",
                    cim_id,
                    "gpu",
                    "UCIe",
                    bandwidth_gbps=256.0,
                    latency_ns=10.0,
                    payload="streaming",
                ),
                LinkSpec(
                    "weights-gpu{}".format(index),
                    "hbf0",
                    "ucie",
                    gpu_id,
                    "weights",
                    "UCIe",
                    bandwidth_gbps=128.0,
                    latency_ns=30.0,
                    payload="streaming",
                ),
                LinkSpec(
                    "weights-cim{}".format(index),
                    "hbf0",
                    "ucie",
                    cim_id,
                    "weights",
                    "UCIe",
                    bandwidth_gbps=128.0,
                    latency_ns=35.0,
                    payload="streaming",
                ),
                LinkSpec(
                    "gpu{}-memory{}".format(index, index),
                    gpu_id,
                    "memory",
                    memory_id,
                    "gpu",
                    "UCIe",
                    bandwidth_gbps=256.0,
                    latency_ns=10.0,
                    payload="streaming",
                ),
            )
        )
    for source in range(world_size):
        for target in range(source + 1, world_size):
            links.append(
                LinkSpec(
                    "gpu{}-gpu{}".format(source, target),
                    "gpu{}".format(source),
                    "nv",
                    "gpu{}".format(target),
                    "nv",
                    "NVLink",
                    bandwidth_gbps=400.0,
                    latency_ns=20.0,
                )
            )
    links.append(
        LinkSpec(
            "cpu-gpu-pcie",
            "cpu0",
            "pcie0",
            "gpu0",
            "pcie0",
            "PCIe",
            bandwidth_gbps=64.0,
            latency_ns=200.0,
        )
    )
    rank_mapping = []
    flat_rank = 0
    for pp_rank, ep_rank, tp_rank in product(
        range(pp_degree), range(ep_degree), range(tp_degree)
    ):
        rank_mapping.append(
            RankMappingSpec(
                rank=flat_rank,
                component_id="gpu{}".format(flat_rank),
                tp_rank=tp_rank,
                pp_rank=pp_rank,
                ep_rank=ep_rank,
                memory_component_id="sram{}".format(flat_rank),
                cim_component_id="cim{}".format(flat_rank),
            )
        )
        flat_rank += 1
    placement = replace(
        scenario.placement,
        hardware_name=hardware_name,
        parallel=ParallelSpec(
            tp_degree=tp_degree,
            pp_degree=pp_degree,
            ep_degree=ep_degree,
            rank_mapping=tuple(rank_mapping),
            layer_to_stage={"dense0": 0, "moe1": 1},
        ),
        tensor_to_component={},
        tensor_bytes={},
        kv_policy=KVCachePolicy(cache_component="sram0"),
    )
    request = replace(
        scenario.workload.requests[0],
        prompt_tokens=8,
        output_tokens=2,
    )
    workload = replace(
        scenario.workload,
        requests=(request,),
        mtp=None,
    )
    authoring = replace(
        scenario,
        hardware=HardwareSpec(
            hardware_name,
            tuple(
                replace(component, capacity_bytes=128 * 1024)
                if component.component_id.startswith("sram")
                else component
                for component in components
            ),
            tuple(links),
        ),
        placement=placement,
        workload=workload,
        weights_resident=False,
    )
    # This fixture adds eight GPU ranks; each needs its own explicit controller.
    authoring = replace(authoring, runtime_profile=replace(
        authoring.runtime_profile,
        gpu_controllers={"gpu{}".format(index): scenario.runtime_profile.gpu_controllers["gpu0"]
                         for index in range(world_size)},
    ))
    mapping = plan_runtime_placement(
        authoring,
        PlacementPolicy(allow_cold_cim_streaming=True),
    )
    return mapping.apply(authoring)


class SchemaV3Tests(unittest.TestCase):
    def test_old_component_and_link_schemas_are_rejected_not_migrated(self):
        self.assertEqual(SCHEMA_VERSION, "4.0.0")
        base = {
            "schema_version": "4.0.0",
            "name": "v3-storage",
            "components": [
                {
                    "schema_version": "4.0.0",
                    "component_id": "gpu0",
                    "kind": "gpu",
                },
                {
                    "schema_version": "4.0.0",
                    "component_id": "hbf0",
                    "kind": "hbf",
                },
            ],
            "links": [],
        }
        old_component = {
            **base,
            "components": [
                {**base["components"][0], "schema_version": "0.2"},
                base["components"][1],
            ],
        }
        with self.assertRaisesRegex(
            ValueError,
            r"component schema_version must be exactly 4\.0\.0; got 0\.2",
        ):
            hardware_from_dict(old_component)

        old_link = {
            **base,
            "links": [
                {
                    "schema_version": "0.3",
                    "link_id": "old-link",
                    "source_component": "gpu0",
                    "source_port": "p0",
                    "target_component": "hbf0",
                    "target_port": "p0",
                    "protocol": "UCIe",
                }
            ],
        }
        with self.assertRaisesRegex(
            ValueError,
            r"link schema_version must be exactly 4\.0\.0; got 0\.3",
        ):
            hardware_from_dict(old_link)

    def test_v3_hbf_remains_flash_and_non_v3_hardware_versions_fail(self):
        hardware = hardware_from_dict(
            {
                "schema_version": "4.0.0",
                "name": "v3-hbf",
                "components": [
                    {
                        "schema_version": "4.0.0",
                        "component_id": "hbf0",
                        "kind": "hbf",
                    }
                ],
                "links": [],
            }
        )
        self.assertEqual(hardware.schema_version, "4.0.0")
        self.assertEqual(hardware.get_component("hbf0").kind, "hbf")
        self.assertNotIn("schema_migrations", hardware.metadata)
        for version in ("0.2", "0.3", "9.0"):
            with self.subTest(schema_version=version):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"hardware schema_version must be exactly 4\.0\.0; got {version}",
                ):
                    hardware_from_dict(
                        {
                            "schema_version": version,
                            "name": "non-v3",
                            "components": [],
                            "links": [],
                        }
                    )


class StorageTopologyTests(unittest.TestCase):
    def test_hbf_is_ucie_only_and_ssd_is_pcie_or_cxl_only(self):
        self.assertTrue(validate_topology(_two_component_storage_hardware("hbf", "UCIe")).is_valid)
        hbf_wrong = validate_topology(_two_component_storage_hardware("hbf", "PCIe"))
        self.assertIn("hbf_non_ucie_protocol", hbf_wrong.error_codes)

        self.assertTrue(validate_topology(_two_component_storage_hardware("ssd", "PCIe")).is_valid)
        self.assertTrue(
            validate_topology(_two_component_storage_hardware("high_io_ssd", "CXL")).is_valid
        )
        ssd_wrong = validate_topology(_two_component_storage_hardware("ssd", "UCIe"))
        self.assertIn("ssd_non_storage_protocol", ssd_wrong.error_codes)


class EndpointTimingTests(unittest.TestCase):
    def test_endpoint_latency_and_flash_granularity_charge_physical_bytes(self):
        hardware = HardwareSpec(
            "timing",
            (
                ComponentSpec(
                    "storage",
                    "hbf",
                    (_port("out", "UCIe", "endpoint", payload="streaming"),),
                    package_id="p",
                    die_id="s",
                    read_bandwidth_gbps=8.0,
                    metadata={"read_latency_ns": 10.0, "transfer_granularity_bytes": 64},
                ),
                ComponentSpec(
                    "gpu",
                    "gpu",
                    (_port("in", "UCIe", "endpoint", payload="streaming"),),
                    cost_profile_id="legacy-gpu",
                    package_id="p",
                    die_id="g",
                    write_bandwidth_gbps=16.0,
                    metadata={"write_latency_ns": 5.0, "transfer_granularity_bytes": 64},
                ),
            ),
            (
                LinkSpec(
                    "l",
                    "storage",
                    "out",
                    "gpu",
                    "in",
                    "UCIe",
                    bandwidth_gbps=100.0,
                    payload="streaming",
                ),
            ),
        )
        phases = TopologyRouter(hardware).transfer_phases("storage", "gpu", 100)
        read = phases[0]
        write = phases[-1]
        self.assertEqual(read.demands[0].bytes_moved, 128)
        self.assertAlmostEqual(read.demands[0].service_ns, 148.0)
        self.assertEqual(read.metadata["transactions"], 2)
        self.assertAlmostEqual(write.demands[0].service_ns, 74.0)

    def test_dma_latency_and_energy_must_be_finite_and_non_negative(self):
        invalid_values = (
            ("dma_latency_ns", -1.0),
            ("dma_latency_ns", float("nan")),
            ("dma_latency_ns", float("inf")),
            ("dma_energy_pj_per_byte", -0.25),
            ("dma_energy_pj_per_byte", float("nan")),
            ("dma_energy_pj_per_byte", float("inf")),
        )
        for key, value in invalid_values:
            with self.subTest(key=key, value=value):
                hardware = HardwareSpec(
                    "dma-validation",
                    (
                        ComponentSpec(
                            "storage",
                            "hbf",
                            (_port("out", "UCIe", "endpoint", payload="streaming"),),
                            package_id="p",
                            die_id="s",
                            read_bandwidth_gbps=8.0,
                            metadata={
                                "dma_bandwidth_gbps": 16.0,
                                key: value,
                            },
                        ),
                        ComponentSpec(
                            "gpu",
                            "gpu",
                            (_port("in", "UCIe", "endpoint", payload="streaming"),),
                            cost_profile_id="legacy-gpu",
                            package_id="p",
                            die_id="g",
                            write_bandwidth_gbps=16.0,
                        ),
                    ),
                    (
                        LinkSpec(
                            "l",
                            "storage",
                            "out",
                            "gpu",
                            "in",
                            "UCIe",
                            bandwidth_gbps=100.0,
                            payload="streaming",
                        ),
                    ),
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "{} must be a finite non-negative number".format(key),
                ):
                    TopologyRouter(hardware).transfer_phases("storage", "gpu", 64)


class PlannerStorageTests(unittest.TestCase):
    def test_rank_memory_rejects_hbf_as_active_memory(self):
        scenario = build_reference_scenario()
        components = tuple(
            replace(
                component,
                kind="hbf",
                cost_profile_id=None,
            )
            if component.component_id == "hbm0"
            else component
            for component in scenario.hardware.components
        )
        scenario = replace(scenario, hardware=replace(scenario.hardware, components=components))
        with self.assertRaisesRegex(ValueError, "writable active memory"):
            build_parallel_plan(scenario)

    def test_ssd_is_valid_kv_offload_but_not_primary_cache(self):
        scenario, ssd_id = _add_storage(build_reference_scenario(), "ssd", "PCIe")
        placement = replace(
            scenario.placement,
            kv_policy=KVCachePolicy(cache_component="hbm0", offload_component=ssd_id),
        )
        offload_scenario = replace(scenario, placement=placement)
        self.assertTrue(validate_scenario(offload_scenario).is_valid)
        schedule = compile_scenario(offload_scenario)
        self.assertFalse(
            any(
                task.metadata.get("event_kind")
                in {"kv_offload", "kv_prefetch"}
                for task in schedule.tasks
            ),
            "static KV lowering must not mirror every append to offload storage",
        )

        primary = replace(
            placement,
            kv_policy=KVCachePolicy(cache_component=ssd_id, offload_component="hbm1"),
        )
        report = validate_scenario(replace(scenario, placement=primary))
        self.assertFalse(report.is_valid)
        self.assertTrue(any("可写的活动内存" in error for error in report.errors))

    def test_cold_model_weights_stream_through_v3_exact_and_online_paths(self):
        scenario, hbf_id = _add_storage(build_reference_scenario(), "hbf", "UCIe")
        tensor_mapping = dict(scenario.placement.tensor_to_component)
        tensor_mapping["model_weights"] = hbf_id
        tensor_bytes = dict(scenario.placement.tensor_bytes)
        tensor_bytes["model_weights"] = scenario.model.total_declared_weight_bytes
        placement = replace(
            scenario.placement,
            tensor_to_component=tensor_mapping,
            tensor_bytes=tensor_bytes,
        )
        configured = replace(
            scenario,
            placement=placement,
            weights_resident=False,
        )
        self.assertTrue(validate_scenario(configured).is_valid)

        schedule = compile_scenario(configured)
        weight_tasks = [
            task for task in schedule.tasks if task.metadata.get("event_kind") == "model_weight_read"
        ]
        self.assertTrue(weight_tasks)
        self.assertTrue(
            any(
                demand.resource_id == "component.{}.read".format(hbf_id)
                for task in weight_tasks
                for demand in task.demands
            )
        )
        self.assertFalse(
            any("attention_qk.model_weight_read" in task.name for task in schedule.tasks)
        )

        cohort = SimpleNamespace(
            cohort_id="online-0",
            kind="decode",
            items=(SimpleNamespace(request_id="r0", token_count=1, context_tokens=8),),
        )
        online = estimate_serving_cohort_cost(configured, cohort)
        self.assertIn(
            "component.{}.read".format(hbf_id),
            online["metadata"]["resource_busy_ns"],
        )

        unconfigured = replace(
            configured,
            weights_resident=True,
            placement=replace(
                configured.placement,
                tensor_to_component={
                    key: value
                    for key, value in configured.placement.tensor_to_component.items()
                    if key != "model_weights"
                },
                tensor_bytes={
                    key: value
                    for key, value in configured.placement.tensor_bytes.items()
                    if key != "model_weights"
                },
            ),
        )
        self.assertFalse(
            any(
                task.metadata.get("event_kind") == "model_weight_read"
                for task in compile_scenario(unconfigured).tasks
            )
        )

    def test_resident_weights_suppress_aggregate_and_detailed_offload_reads(self):
        scenario, hbf_id = _add_storage(
            build_reference_scenario(), "hbf", "UCIe"
        )
        tensor_mapping = {
            **scenario.placement.tensor_to_component,
            "model_weights": hbf_id,
            "dense0.attention_weights": hbf_id,
        }
        tensor_bytes = {
            **scenario.placement.tensor_bytes,
            "model_weights": scenario.model.total_declared_weight_bytes,
            "dense0.attention_weights": 1024,
        }
        configured = replace(
            scenario,
            placement=replace(
                scenario.placement,
                tensor_to_component=tensor_mapping,
                tensor_bytes=tensor_bytes,
            ),
            weights_resident=True,
        )
        report = validate_scenario(configured)
        self.assertTrue(report.is_valid, report.errors)

        cohort = SimpleNamespace(
            cohort_id="resident-online",
            kind="decode",
            items=(
                SimpleNamespace(
                    request_id="r0",
                    token_count=4,
                    context_tokens=8,
                ),
            ),
        )
        schedules = (
            compile_scenario(configured),
            compile_serving_cohort_schedule(configured, cohort),
        )
        for schedule in schedules:
            self.assertFalse(
                any(
                    task.metadata.get("weight_source_component") == hbf_id
                    and task.metadata.get("event_kind") == "model_weight_read"
                    for task in schedule.tasks
                )
            )
            audited = tuple(
                task
                for task in schedule.tasks
                if task.metadata.get("weight_source_kind") == "hbf"
                and task.metadata.get("operator_class") == "gemm"
            )
            self.assertTrue(audited)
            self.assertTrue(
                all(
                    task.metadata.get("weight_lifecycle_mode")
                    == "preloaded_resident"
                    and task.metadata.get("weight_backing_read_emitted") is False
                    and task.metadata.get("weight_backing_read_gate")
                    == "preloaded_offload_backing_suppressed"
                    for task in audited
                )
            )

    def test_cold_serving_reads_once_per_rank_gemm_not_per_batch_token(self):
        scenario, hbf_id = _add_storage(
            build_reference_scenario(), "hbf", "UCIe"
        )
        configured = replace(
            scenario,
            placement=replace(
                scenario.placement,
                tensor_to_component={
                    **scenario.placement.tensor_to_component,
                    "model_weights": hbf_id,
                },
                tensor_bytes={
                    **scenario.placement.tensor_bytes,
                    "model_weights": scenario.model.total_declared_weight_bytes,
                },
            ),
            weights_resident=False,
        )

        def backing_summary(schedule):
            compute = {}
            for task in schedule.tasks:
                if (
                    task.metadata.get("event_kind")
                    not in {"model_weight_read", "model_weight_access"}
                    and task.metadata.get("weight_lifecycle_mode")
                    == "cold_stream_per_use"
                    and task.metadata.get("weight_read_invocation_id")
                ):
                    invocation_id = task.metadata["weight_read_invocation_id"]
                    compute.setdefault(
                        invocation_id,
                        (
                            str(task.metadata.get("weight_tensor_id")),
                            int(task.metadata.get("weight_read_bytes", 0)),
                        ),
                    )
            source_reads = {}
            for task in schedule.tasks:
                if (
                    task.metadata.get("event_kind") == "model_weight_read"
                    and task.metadata.get("weight_source_component") == hbf_id
                    and task.metadata.get("component_id") == hbf_id
                ):
                    invocation_id = task.metadata["weight_read_invocation_id"]
                    self.assertNotIn(invocation_id, source_reads)
                    source_reads[invocation_id] = (
                        str(task.metadata.get("logical_weight_tensor")),
                        int(task.metadata.get("bytes", 0)),
                    )
            self.assertEqual(set(compute), set(source_reads))
            self.assertTrue(
                all(
                    source_reads[invocation_id][1] == expected_bytes
                    for invocation_id, (_tensor_id, expected_bytes) in compute.items()
                )
            )
            return set(compute), sorted(source_reads.values())

        def cohort(cohort_id, kind, token_counts):
            return SimpleNamespace(
                cohort_id=cohort_id,
                kind=kind,
                items=tuple(
                    SimpleNamespace(
                        request_id="{}-r{}".format(cohort_id, index),
                        phase=kind,
                        token_count=token_count,
                        context_tokens=8 + index,
                        main_tokens=1 if kind == "mtp" else None,
                        draft_tokens=(
                            token_count - 1 if kind == "mtp" else None
                        ),
                        verifier_tokens=(
                            token_count if kind == "mtp" else None
                        ),
                        committed_tokens=1 if kind == "mtp" else None,
                        expected_accepted_tokens=1.0,
                    )
                    for index, token_count in enumerate(token_counts)
                ),
            )

        decode_one = compile_serving_cohort_schedule(
            configured, cohort("decode-one", "decode", (1,))
        )
        decode_many = compile_serving_cohort_schedule(
            configured, cohort("decode-many", "decode", (3, 5))
        )
        first_ids, first_reads = backing_summary(decode_one)
        second_ids, second_reads = backing_summary(decode_many)
        self.assertEqual(
            [item for item in first_reads if item[0] != "moe1.expert_weights"],
            [item for item in second_reads if item[0] != "moe1.expert_weights"],
        )
        self.assertTrue(first_ids.isdisjoint(second_ids))

        mtp_one = compile_serving_cohort_schedule(
            configured, cohort("mtp-one", "mtp", (2,))
        )
        mtp_many = compile_serving_cohort_schedule(
            configured, cohort("mtp-many", "mtp", (2, 2, 2, 2))
        )
        _mtp_one_ids, mtp_one_reads = backing_summary(mtp_one)
        _mtp_many_ids, mtp_many_reads = backing_summary(mtp_many)
        self.assertEqual(
            [item for item in mtp_one_reads if item[0].startswith("mtp.")],
            [item for item in mtp_many_reads if item[0].startswith("mtp.")],
        )
        self.assertTrue(
            {
                "mtp.prediction_layer.000.weights",
                "mtp.aux_head.weights",
            }.issubset({tensor_id for tensor_id, _bytes in mtp_one_reads})
        )

    def test_detailed_weight_sources_override_aggregate_in_v3_lowering(self):
        scenario, hbf_id = _add_storage(
            build_reference_scenario(), "hbf", "UCIe"
        )
        tensor_mapping = dict(scenario.placement.tensor_to_component)
        tensor_mapping.update(
            {
                "model_weights": "hbm7",
                "dense0.attention_weights": hbf_id,
                "moe1.router_weights": hbf_id,
                "embedding_weights": hbf_id,
                "mtp.prediction_layer.000.weights": hbf_id,
                "mtp.aux_head.weights": hbf_id,
            }
        )
        tensor_bytes = dict(scenario.placement.tensor_bytes)
        tensor_bytes.update(
            {
                "model_weights": scenario.model.total_declared_weight_bytes,
                "dense0.attention_weights": 1024,
                "moe1.router_weights": 1024,
                "embedding_weights": scenario.model.embedding_weight_bytes,
                "mtp.prediction_layer.000.weights": 1024,
                "mtp.aux_head.weights": 1024,
            }
        )
        placement = replace(
            scenario.placement,
            tensor_to_component=tensor_mapping,
            tensor_bytes=tensor_bytes,
        )
        configured = replace(
            scenario,
            placement=placement,
            weights_resident=False,
        )
        report = validate_scenario(configured)
        self.assertTrue(report.is_valid, report.errors)

        parallel_schedule = compile_scenario(configured)
        detailed_reads = [
            task
            for task in parallel_schedule.tasks
            if task.metadata.get("weight_source_component") == hbf_id
            and task.metadata.get("logical_weight_tensor")
        ]
        logical_tensors = {
            task.metadata.get("logical_weight_tensor")
            for task in detailed_reads
        }
        self.assertTrue(
            {
                "dense0.attention_weights",
                "moe1.router_weights",
                "embedding_weights",
                "mtp.prediction_layer.000.weights",
                "mtp.aux_head.weights",
            }.issubset(logical_tensors),
            logical_tensors,
        )
        self.assertTrue(
            all(task.metadata.get("tensor") != "model_weights" for task in detailed_reads)
        )
        self.assertFalse(
            any(
                ("attention_qk" in task.name or "attention_pv" in task.name)
                and task.metadata.get("event_kind") == "model_weight_read"
                for task in parallel_schedule.tasks
            )
        )

    def test_control_plane_cim_physical_metadata_is_checked(self):
        scenario = build_reference_scenario()
        scenario = plan_runtime_placement(scenario).apply(scenario)
        baseline = validate_scenario(scenario)
        self.assertTrue(baseline.is_valid, baseline.errors)
        self.assertTrue(
            any(
                "warm CIM resident capacity is checked" in warning
                for warning in baseline.warnings_en
            )
        )

        tensor_id = "dense0.mlp_weights"
        padded_bytes = scenario.placement.tensor_bytes[tensor_id]
        metadata = dict(scenario.placement.metadata)
        control_plane = dict(metadata["control_plane"])
        decision = dict(control_plane["decision"])
        total_bytes = dict(decision["cim_total_physical_bytes"])
        total_bytes[tensor_id] = padded_bytes + 1
        details = {
            key: dict(value)
            for key, value in decision["weight_tensor_details"].items()
        }
        details[tensor_id]["total_physical_bytes"] = padded_bytes + 1
        decision["cim_total_physical_bytes"] = total_bytes
        decision["weight_tensor_details"] = details
        control_plane["decision"] = decision
        metadata["control_plane"] = control_plane
        report = validate_scenario(
            replace(
                scenario,
                placement=replace(scenario.placement, metadata=metadata),
            )
        )
        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("总物理字节" in error for error in report.errors)
        )

    def test_logical_weight_view_uses_backing_without_extra_capacity(self):
        scenario = build_reference_scenario()
        tensor_id = "lm_head_weights"
        request = scenario.workload.requests[0]
        page_tokens = scenario.placement.kv_policy.tokens_per_page
        live_tokens = request.prompt_tokens + request.output_tokens
        kv_capacity = (
            math.ceil(live_tokens / page_tokens)
            * page_tokens
            * _kv_bytes_per_token(scenario, None)
        )
        exact_capacity = (
            scenario.model.total_declared_weight_bytes
            + kv_capacity
        )
        hardware = replace(
            scenario.hardware,
            components=tuple(
                replace(component, capacity_bytes=exact_capacity)
                if component.component_id == "hbm0"
                else replace(component, capacity_bytes=1)
                if component.component_id == "cim0"
                else component
                for component in scenario.hardware.components
            ),
        )
        authoring = replace(
            scenario,
            hardware=hardware,
            weights_resident=False,
        )
        configured = plan_runtime_placement(authoring).apply(authoring)
        report = validate_scenario(configured)
        self.assertTrue(report.is_valid, report.errors)
        self.assertEqual(
            configured.placement.metadata["logical_weight_aliases"][tensor_id],
            "embedding_weights",
        )

        reads = [
            task
            for task in compile_scenario(configured).tasks
            if task.metadata.get("event_kind") == "model_weight_access"
            and task.metadata.get("weight_tensor_id") == tensor_id
        ]
        self.assertTrue(reads)
        self.assertTrue(
            all(task.metadata.get("tensor") == "embedding_weights" for task in reads)
        )

    def test_parallel_cold_cim_weights_route_from_backing_to_rank_cims(self):
        scenario = _multi_rank_cim_weight_backing_scenario()
        self.assertEqual(scenario.workload.scheduler.mode, "continuous")
        validation = validate_scenario(scenario)
        self.assertTrue(validation.is_valid, validation.errors)

        schedule = compile_scenario(scenario)
        model_weight_tasks = [
            task
            for task in schedule.tasks
            if task.metadata.get("event_kind") == "model_weight_read"
            and task.metadata.get("weight_source_component") == "hbf0"
        ]
        cim_weight_tasks = [
            task
            for task in model_weight_tasks
            if str(task.metadata.get("weight_target_component", "")).startswith("cim")
        ]
        gpu_weight_tasks = [
            task
            for task in model_weight_tasks
            if str(task.metadata.get("weight_target_component", "")).startswith("gpu")
        ]
        self.assertTrue(cim_weight_tasks)
        self.assertTrue(gpu_weight_tasks)
        cim_targets = {
            str(task.metadata["weight_target_component"])
            for task in cim_weight_tasks
        }
        self.assertTrue({"cim0", "cim1"}.issubset(cim_targets))
        self.assertTrue({"cim4", "cim5"}.issubset(cim_targets))
        self.assertTrue({"cim6", "cim7"}.issubset(cim_targets))
        cim_resources = {
            demand.resource_id
            for task in cim_weight_tasks
            for demand in task.demands
        }
        for target in ("cim0", "cim1", "cim4", "cim5", "cim6", "cim7"):
            self.assertIn(
                "link.weights-{}.hbf0->{}".format(target, target),
                cim_resources,
            )
        self.assertFalse(
            any(
                demand.resource_id.startswith("link.weights-gpu")
                for task in cim_weight_tasks
                for demand in task.demands
            )
        )

        cohort = SimpleNamespace(
            cohort_id="online-cim",
            kind="decode",
            items=(
                SimpleNamespace(request_id="r0", token_count=2, context_tokens=8),
            ),
        )
        online = estimate_serving_cohort_cost(scenario, cohort)
        busy = online["metadata"]["resource_busy_ns"]
        self.assertIn("link.weights-cim0.hbf0->cim0", busy)
        self.assertIn("link.weights-cim6.hbf0->cim6", busy)


if __name__ == "__main__":
    unittest.main()
