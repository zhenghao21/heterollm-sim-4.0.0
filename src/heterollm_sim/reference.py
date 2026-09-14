"""Bundled reference scenario that remains available after package install."""

from __future__ import annotations

from .config import FusionPolicy, InterconnectProfile, ScenarioConfig
from .cost_models import (
    CPUPipelineProfile,
    CPUProfile,
    CacheHierarchyProfile,
    CacheLevelProfile,
    DigitalSramCimProfile,
    GPUProfile,
    HBMProfile,
    HostMemoryProfile,
    HostOrchestrationProfile,
    TensorCoreProfile,
)
from .ir import (
    ComponentSpec,
    HardwareSpec,
    KVCachePolicy,
    LayerSpec,
    LinkSpec,
    MTPBranchSpec,
    MTPPolicy,
    ModelSpec,
    ParallelSpec,
    PlacementSpec,
    PortSpec,
    RankMappingSpec,
    RequestSpec,
    SchedulerSpec,
    WorkloadSpec,
    build_model_graph_from_layer_specs,
)


def build_reference_scenario() -> ScenarioConfig:
    hbm_port_bandwidth_gbps = 4096.0
    gpu_ports = [
        PortSpec(
            port_id="hbm{}".format(index),
            protocol="HBM",
            role="controller",
            version="3.0",
            lanes=16,
            bandwidth_gbps=hbm_port_bandwidth_gbps,
        )
        for index in range(8)
    ]
    gpu_ports.append(
        PortSpec(
            port_id="ucie0",
            protocol="UCIe",
            role="endpoint",
            version="1.1",
            lanes=64,
            bandwidth_gbps=2048.0,
            payload="streaming",
        )
    )
    gpu_ports.append(
        PortSpec(
            port_id="pcie0",
            protocol="PCIe",
            role="endpoint",
            version="5.0",
            lanes=16,
            bandwidth_gbps=512.0,
            payload="coherent_dma",
        )
    )
    components = [
        ComponentSpec(
            component_id="gpu0",
            kind="gpu",
            cost_profile_id="legacy-gpu",
            ports=tuple(gpu_ports),
            package_id="package0",
            die_id="gpu_die",
            capacity_bytes=64 * 1024 * 1024,
            peak_ops_per_s=120_000_000_000_000.0,
        )
    ]
    links = []
    components.extend(
        (
            ComponentSpec(
                component_id="cpu0",
                kind="cpu",
                cost_profile_id="legacy-cpu",
                ports=(
                    PortSpec(
                        port_id="pcie0",
                        protocol="PCIe",
                        role="root_complex",
                        version="5.0",
                        lanes=16,
                        bandwidth_gbps=512.0,
                        payload="coherent_dma",
                    ),
                    PortSpec(
                        port_id="ddr0",
                        protocol="DDR",
                        role="controller",
                        version="5.0",
                        lanes=64,
                        bandwidth_gbps=3276.8,
                    ),
                ),
                package_id="host0",
                die_id="cpu_die",
                capacity_bytes=48 * 1024**2,
            ),
            ComponentSpec(
                component_id="hostmem0",
                kind="host_memory",
                cost_profile_id="legacy-host-memory",
                ports=(
                    PortSpec(
                        port_id="ddr0",
                        protocol="DDR",
                        role="device",
                        version="5.0",
                        lanes=64,
                        bandwidth_gbps=3276.8,
                    ),
                ),
                package_id="host0",
                die_id="ddr_die",
                capacity_bytes=256 * 1024**3,
            ),
        )
    )
    links.extend(
        (
            LinkSpec(
                link_id="cpu-gpu-pcie",
                source_component="cpu0",
                source_port="pcie0",
                target_component="gpu0",
                target_port="pcie0",
                protocol="PCIe",
                version="5.0",
                lanes=16,
                bandwidth_gbps=512.0,
                latency_ns=800.0,
                payload="coherent_dma",
                metadata={"energy_pj_per_byte": 8.0},
            ),
            LinkSpec(
                link_id="cpu-hostmem-ddr",
                source_component="cpu0",
                source_port="ddr0",
                target_component="hostmem0",
                target_port="ddr0",
                protocol="DDR",
                version="5.0",
                lanes=64,
                bandwidth_gbps=3276.8,
                latency_ns=80.0,
                metadata={"energy_pj_per_byte": 12.0},
            ),
        )
    )
    for index in range(8):
        hbm_id = "hbm{}".format(index)
        components.append(
            ComponentSpec(
                component_id=hbm_id,
                kind="hbm",
                cost_profile_id="legacy-hbm",
                ports=(
                    PortSpec(
                        port_id="host",
                        protocol="HBM",
                        role="device",
                        version="3.0",
                        lanes=16,
                        bandwidth_gbps=hbm_port_bandwidth_gbps,
                    ),
                ),
                package_id="package0",
                die_id="{}_die".format(hbm_id),
                capacity_bytes=16 * 1024**3,
            )
        )
        links.append(
            LinkSpec(
                link_id="gpu-{}".format(hbm_id),
                source_component="gpu0",
                source_port=hbm_id,
                target_component=hbm_id,
                target_port="host",
                protocol="HBM",
                version="3.0",
                lanes=16,
                bandwidth_gbps=hbm_port_bandwidth_gbps,
                latency_ns=40.0,
                metadata={"energy_pj_per_byte": 4.0},
            )
        )
    components.append(
        ComponentSpec(
            component_id="cim0",
            kind="digital_sram_cim",
            cost_profile_id="legacy-cim",
            ports=(
                PortSpec(
                    port_id="ucie0",
                    protocol="UCIe",
                    role="endpoint",
                    version="1.1",
                    lanes=64,
                    bandwidth_gbps=2048.0,
                    payload="streaming",
                ),
            ),
            package_id="package0",
            die_id="cim_die",
            capacity_bytes=512 * 1024**2,
        )
    )
    links.append(
        LinkSpec(
            link_id="gpu-cim",
            source_component="gpu0",
            source_port="ucie0",
            target_component="cim0",
            target_port="ucie0",
            protocol="UCIe",
            version="1.1",
            lanes=64,
            bandwidth_gbps=2048.0,
            latency_ns=20.0,
            payload="streaming",
            metadata={"energy_pj_per_byte": 1.0},
        )
    )

    model_layers = (
        LayerSpec(
            layer_id="dense0",
            kind="dense",
            hidden_size=512,
            intermediate_size=1024,
            attention_heads=8,
            kv_heads=4,
            dtype="int8",
            quantization="w8a8",
            weight_bytes=2_621_440,
        ),
        LayerSpec(
            layer_id="moe1",
            kind="moe",
            hidden_size=512,
            intermediate_size=768,
            attention_heads=8,
            kv_heads=4,
            num_experts=4,
            experts_per_token=2,
            dtype="int8",
            quantization="w8a8",
            weight_bytes=5_767_168,
        ),
    )
    model_graph = build_model_graph_from_layer_specs(
        "reference-mixed-transformer",
        model_layers,
        architecture="decoder_only_transformer",
        vocabulary_size=32_000,
        max_sequence_length=32_768,
        mtp=MTPBranchSpec(
            prediction_layers=1,
            auxiliary_head=True,
            prediction_layer_weight_bytes=2_621_440,
            auxiliary_head_weight_bytes=16_384_000,
        ),
    )
    model = ModelSpec(
        name="reference-mixed-transformer",
        graph=model_graph,
    )
    placement = PlacementSpec(
        model_name=model.name,
        hardware_name="gpu-8hbm-cim-package",
        # V4 authoring describes resources and policy only.  The live CPU
        # control plane materializes operator/tensor placement for each run.
        op_to_component={},
        tensor_to_component={},
        tensor_bytes={},
        parallel=ParallelSpec(
            tp_degree=1,
            pp_degree=1,
            ep_degree=1,
            rank_mapping=(
                RankMappingSpec(
                    rank=0,
                    component_id="gpu0",
                    tp_rank=0,
                    pp_rank=0,
                    ep_rank=0,
                    memory_component_id="hbm0",
                    cim_component_id="cim0",
                ),
            ),
            layer_to_stage={"dense0": 0, "moe1": 0},
            collective_algorithm="auto",
            routing_policy="lowest_latency",
            allow_padding=True,
        ),
        kv_policy=KVCachePolicy(
            cache_component="hbm0",
            offload_component="hbm1",
            tokens_per_page=16,
            dtype="int8",
            offload_ratio=1.0,
            allocation_policy="lazy",
            preemption_mode="auto",
            prefetch_distance=0,
        ),
    )
    workload = WorkloadSpec(
        name="single-request-short-context",
        requests=(
            RequestSpec(
                request_id="request-0000",
                arrival_ns=0.0,
                prompt_tokens=64,
                output_tokens=4,
            ),
        ),
        random_seed=7,
        scheduler=SchedulerSpec(
            mode="continuous",
            max_num_seqs=4,
            max_num_batched_tokens=256,
            prefill_chunk_tokens=32,
            policy="decode_first",
            starvation_ns=5_000_000.0,
            preemption_enabled=True,
            preemption_granularity="boundary",
            preemption_policy="auto",
        ),
        mtp=MTPPolicy(
            method="head_based",
            candidate_tokens=4,
            acceptance_model="expected",
            acceptance_rate=0.65,
            proposal_cost_scale=0.15,
        ),
    )
    return ScenarioConfig(
        name="gpu-8hbm-ucie-sram-cim-reference",
        hardware=HardwareSpec(
            name="gpu-8hbm-cim-package",
            components=tuple(components),
            links=tuple(links),
        ),
        model=model,
        placement=placement,
        workload=workload,
        component_profiles={
            "gpu": {"legacy-gpu": GPUProfile(
            name="reference-gpu-int8",
            tensor_core=TensorCoreProfile(
                sm_count=120,
                tensor_cores_per_sm=4,
                frequency_ghz=1.5,
                mma_m=16,
                mma_n=16,
                mma_k=16,
                cycles_per_mma=49.152,
                supported_dtypes=("fp16", "bf16", "int8"),
                dtype_throughput_scale={
                    "fp16": 0.5,
                    "bf16": 0.5,
                    "int8": 1.0,
                },
                resource_id="gpu0.tensor_core",
            ),
            cache_hierarchy=CacheHierarchyProfile(
                levels=(
                    CacheLevelProfile(
                        name="l1_shared",
                        capacity_bytes=30 * 1024**2,
                        line_bytes=128,
                        hit_latency_ns=20.0,
                        bandwidth_gb_s=24_000.0,
                        associativity=16,
                        banks=120 * 32,
                        read_ports=2,
                        write_ports=1,
                        max_outstanding=32,
                        energy_pj_per_byte=0.15,
                        resource_id="gpu0.l1_shared",
                    ),
                    CacheLevelProfile(
                        name="l2",
                        capacity_bytes=50 * 1024**2,
                        line_bytes=128,
                        hit_latency_ns=120.0,
                        bandwidth_gb_s=12_000.0,
                        associativity=16,
                        banks=128,
                        read_ports=2,
                        write_ports=1,
                        max_outstanding=128,
                        energy_pj_per_byte=0.6,
                        resource_id="gpu0.l2",
                    ),
                )
            ),
            scalar_lanes_per_sm=128,
            scalar_ops_per_cycle=1.0,
            reduction_ops_per_cycle_per_sm=64.0,
            special_function_units_per_sm=16,
            special_function_ops_per_cycle=1.0,
            occupancy=0.85,
            attainable_efficiency=0.65,
            kernel_launch_ns=1000.0,
            tensor_energy_pj_per_op=0.2,
            scalar_energy_pj_per_op=0.35,
            special_function_energy_pj_per_op=1.2,
            launch_energy_pj=10_000.0,
            scalar_resource_id="gpu0.scalar",
            special_function_resource_id="gpu0.sfu",
            launch_resource_id="gpu0.frontend",
            )},
            "hbm": {"legacy-hbm": HBMProfile(
            bandwidth_gb_s=4096.0,
            efficiency=0.75,
            energy_pj_per_byte=4.0,
            resource_id="gpu0.hbm_fabric",
            )},
            "cpu": {"legacy-cpu": CPUProfile(
            pipeline=CPUPipelineProfile(
                core_count=16,
                frequency_ghz=3.2,
                simd_width_bits=512,
                decode_width=6,
                issue_width=8,
                retire_width=6,
                vector_fma_units_per_core=2,
                vector_alu_units_per_core=2,
                load_units_per_core=3,
                store_units_per_core=2,
                branch_units_per_core=2,
                special_function_units_per_core=1,
                special_function_cycles_per_vector=12.0,
                reorder_buffer_entries=352,
                load_store_queue_entries=192,
                memory_level_parallelism=16,
                branch_mispredict_ns=5.0,
                resource_id="cpu0.pipeline",
            ),
            cache_hierarchy=CacheHierarchyProfile(
                levels=(
                    CacheLevelProfile(
                        name="l1d",
                        capacity_bytes=16 * 48 * 1024,
                        line_bytes=64,
                        hit_latency_ns=1.0,
                        bandwidth_gb_s=3_000.0,
                        associativity=12,
                        banks=16 * 8,
                        read_ports=3,
                        write_ports=2,
                        max_outstanding=16,
                        energy_pj_per_byte=0.2,
                        resource_id="cpu0.l1d",
                    ),
                    CacheLevelProfile(
                        name="l2",
                        capacity_bytes=16 * 2 * 1024**2,
                        line_bytes=64,
                        hit_latency_ns=4.0,
                        bandwidth_gb_s=1_500.0,
                        associativity=16,
                        banks=16 * 8,
                        read_ports=2,
                        write_ports=1,
                        max_outstanding=32,
                        energy_pj_per_byte=0.8,
                        resource_id="cpu0.l2",
                    ),
                    CacheLevelProfile(
                        name="l3",
                        capacity_bytes=64 * 1024**2,
                        line_bytes=64,
                        hit_latency_ns=18.0,
                        bandwidth_gb_s=800.0,
                        associativity=16,
                        banks=64,
                        read_ports=2,
                        write_ports=1,
                        max_outstanding=64,
                        energy_pj_per_byte=2.0,
                        resource_id="cpu0.l3",
                    ),
                )
            ),
            attainable_efficiency=0.72,
            dispatch_ns=80.0,
            gemm_energy_pj_per_op=1.5,
            elementwise_energy_pj_per_op=1.0,
            reduction_energy_pj_per_op=1.2,
            special_function_energy_pj_per_op=4.0,
            dispatch_energy_pj=800.0,
            name="reference-host-cpu",
            )},
            "host_memory": {"legacy-host-memory": HostMemoryProfile(
            bandwidth_gb_s=409.6,
            efficiency=0.78,
            energy_pj_per_byte=12.0,
            resource_id="cpu0.memory",
            name="reference-ddr5",
            )},
            "cim": {"legacy-cim": DigitalSramCimProfile(
                name="reference-adc-free-digital-sram-cim",
                array_count=512,
                p_m=1,
                p_k=128,
                p_n=128,
                frequency_ghz=1.0,
                input_parallel_bits=4,
                weight_parallel_bits=4,
                cycles_per_eval=1,
                weight_capacity_bytes=512 * 1024**2,
                max_m_replication=4,
                load_bandwidth_gb_s=512.0,
                activation_bandwidth_gb_s=1024.0,
                output_bandwidth_gb_s=1024.0,
                noc_bandwidth_gb_s=2048.0,
                accumulator_outputs_per_cycle=4096.0,
                peripheral_elements_per_cycle=4096.0,
                load_latency_ns=20.0,
                noc_hop_latency_ns=2.0,
                noc_reduce_fan_in=4,
                peripheral_latency_ns=5.0,
                accumulator_bits=32,
                eval_energy_pj=1.0,
                load_energy_pj_per_byte=0.5,
                activation_energy_pj_per_byte=0.2,
                output_energy_pj_per_byte=0.2,
                noc_energy_pj_per_byte=0.1,
                accumulator_energy_pj_per_op=0.05,
                peripheral_energy_pj_per_element=0.1,
                array_resource_id="cim0.array",
                load_resource_id="cim0.load",
                activation_resource_id="cim0.activation",
                noc_resource_id="cim0.noc",
                accumulator_resource_id="cim0.accumulator",
                peripheral_resource_id="cim0.peripheral",
            )},
        },
        host_orchestration_profile=HostOrchestrationProfile(
            request_parse_ns=180.0,
            batch_fixed_ns=350.0,
            token_pack_ns=12.0,
            submission_ns=250.0,
            capacity_fixed_instructions=96,
            capacity_instructions_per_request=64,
            schedule_fixed_instructions=192,
            schedule_instructions_per_request=48,
            schedule_instructions_per_token=8,
            command_build_fixed_instructions=128,
            command_build_instructions_per_invocation=12,
            dma_queue_submission_ns=62.5,
            descriptor_bytes_per_request=96,
            token_bytes=4,
            dma_bandwidth_gb_s=48.0,
            dma_latency_ns=800.0,
            max_inflight_batches=4,
            pinned_memory=True,
            kv_page_lookup_ns=8.0,
            kv_descriptor_ns=4.0,
            kv_descriptor_bytes=32,
            cpu_component_id="cpu0",
            gpu_component_id="gpu0",
            scheduler_resource_id="cpu0.scheduler",
            pack_resource_id="cpu0.pack",
            dma_resource_id="cpu0.h2d_dma",
            submission_resource_id="gpu0.command_queue",
        ),
        fusion_policy=FusionPolicy(
            qkv_rope=True,
            flash_attention=True,
            gemm_epilogue_activation=True,
            residual_norm=True,
            max_fused_working_set_bytes=30 * 1024**2,
        ),
        cim_interconnect=InterconnectProfile(
            resource_id="ucie.gpu0-cim0",
            bandwidth_gb_s=256.0,
            latency_ns=20.0,
            energy_pj_per_byte=1.0,
        ),
        weights_resident=True,
        assumptions=(
            "Reference results are ANALYTICAL and are not calibrated to a named commercial chip.",
            "MoE routing is represented by a uniform aggregate active-expert model.",
            "CIM is ADC-free digital SRAM-CIM with warm resident weights.",
            "The workload is inference-only; no training graph or optimizer state is represented.",
            "Continuous batching admits, chunks, and preempts only at scheduler boundaries.",
        ),
    )


__all__ = ["build_reference_scenario"]
