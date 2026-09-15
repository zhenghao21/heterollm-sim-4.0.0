import inspect
import unittest
from dataclasses import replace

from heterollm_sim.cost_models import (
    CPUPipelineProfile,
    CPUProfile,
    CPUQuantizedDotCapability,
    CacheHierarchyProfile,
    CacheLevelProfile,
    DigitalSramCimProfile,
    ElementwiseWorkload,
    FusedAttentionKVPhysicalContract,
    FusedAttentionWorkload,
    GPUProfile,
    GPUQuantizedMatmulCapability,
    GemmWorkload,
    HBMProfile,
    HostGemmOffloadCapability,
    HostRecurrentOffloadCapability,
    HostMemoryProfile,
    MemoryWorkload,
    ReductionWorkload,
    TensorCoreProfile,
    TensorKernelWorkload,
    _cache_memory_demands,
    _cpu_instruction_schedule,
    _dma_setup_service,
    break_even_reuse,
    mma_output_tile_wave_proxy,
    estimate_cim_gemm,
    estimate_cpu_elementwise,
    estimate_cpu_gemm,
    estimate_cpu_memory,
    estimate_cpu_reduction,
    estimate_gpu_elementwise,
    estimate_gpu_fused_attention,
    estimate_gpu_gemm,
    estimate_gpu_memory,
    estimate_gpu_reduction,
    estimate_gpu_tensor_kernel,
)
from heterollm_sim.contracts import EvidenceStatus, OperatorClass


class DMASetupServiceTests(unittest.TestCase):
    def test_zero_bytes_require_no_transactions_or_controller_service(self):
        service = _dma_setup_service(
            0,
            batch_bytes=4096,
            queue_depth=8,
            max_outstanding=16,
            fixed_latency_ns=25.0,
            submission_ns_per_wave=3.0,
        )

        self.assertEqual(service.transaction_count, 0)
        self.assertEqual(service.queue_parallelism, 0)
        self.assertEqual(service.wave_count, 0)
        self.assertEqual(service.service_ns, 0.0)

    def test_one_wave_pays_one_submission_after_fixed_latency(self):
        service = _dma_setup_service(
            1025,
            batch_bytes=1024,
            queue_depth=4,
            max_outstanding=8,
            fixed_latency_ns=10.0,
            submission_ns_per_wave=3.0,
        )

        self.assertEqual(service.transaction_count, 2)
        self.assertEqual(service.queue_parallelism, 4)
        self.assertEqual(service.wave_count, 1)
        self.assertEqual(service.service_ns, 13.0)

    def test_outstanding_limit_creates_multiple_descriptor_waves(self):
        service = _dma_setup_service(
            10 * 1024,
            batch_bytes=1024,
            queue_depth=8,
            max_outstanding=3,
            fixed_latency_ns=10.0,
            submission_ns_per_wave=2.5,
        )

        self.assertEqual(service.transaction_count, 10)
        self.assertEqual(service.queue_parallelism, 3)
        self.assertEqual(service.wave_count, 4)
        self.assertEqual(service.service_ns, 20.0)

    def test_engine_parallelism_is_owned_by_the_event_resource_capacity(self):
        self.assertNotIn(
            "dma_engine_count",
            inspect.signature(_dma_setup_service).parameters,
        )


def fast_cim(**overrides):
    defaults = dict(
        array_count=1,
        p_m=1,
        p_k=4,
        p_n=4,
        frequency_ghz=1.0,
        input_parallel_bits=1,
        weight_parallel_bits=1,
        cycles_per_eval=1,
        weight_capacity_bytes=1 << 20,
        load_bandwidth_gb_s=1.0e9,
        activation_bandwidth_gb_s=1.0e9,
        output_bandwidth_gb_s=1.0e9,
        noc_bandwidth_gb_s=1.0e9,
        accumulator_outputs_per_cycle=1.0e9,
        peripheral_elements_per_cycle=1.0e9,
        accumulator_bits=64,
    )
    defaults.update(overrides)
    return DigitalSramCimProfile(**defaults)


def cache_hierarchy(
    *, capacity_bytes=1, bandwidth_gb_s=1.0e12, hit_latency_ns=0.0,
    resource_id="device.l1",
):
    return CacheHierarchyProfile(
        levels=(
            CacheLevelProfile(
                name="l1",
                capacity_bytes=capacity_bytes,
                line_bytes=1,
                hit_latency_ns=hit_latency_ns,
                bandwidth_gb_s=bandwidth_gb_s,
                resource_id=resource_id,
            ),
        )
    )


def gpu_profile(
    *, tensor_tops=100.0, scalar_gops=100.0, reduction_gops=100.0,
    sfu_gops=50.0, kernel_launch_ns=0.0, cache=None,
):
    tensor_core = TensorCoreProfile(
        sm_count=1,
        tensor_cores_per_sm=1,
        frequency_ghz=1.0,
        cycles_per_mma=8192.0 / (tensor_tops * 1000.0),
        supported_dtypes=("int8", "fp16", "fp32"),
        resource_id="gpu.tensor",
    )
    return GPUProfile(
        tensor_core=tensor_core,
        cache_hierarchy=cache or cache_hierarchy(resource_id="gpu.l1"),
        scalar_lanes_per_sm=1,
        scalar_ops_per_cycle=scalar_gops,
        reduction_ops_per_cycle_per_sm=reduction_gops,
        special_function_units_per_sm=1,
        special_function_ops_per_cycle=sfu_gops,
        kernel_launch_ns=kernel_launch_ns,
        scalar_resource_id="gpu.scalar",
        special_function_resource_id="gpu.sfu",
        launch_resource_id="gpu.frontend",
    )


def cpu_profile(*, pipeline=None, cache=None):
    return CPUProfile(
        pipeline=pipeline
        or CPUPipelineProfile(
            core_count=1,
            frequency_ghz=1.25,
            simd_width_bits=64,
            decode_width=1000,
            issue_width=1000,
            retire_width=1000,
            vector_fma_units_per_core=2,
            vector_alu_units_per_core=2,
            load_units_per_core=1000,
            store_units_per_core=1000,
            reorder_buffer_entries=100_000,
            load_store_queue_entries=100_000,
            memory_level_parallelism=100_000,
            resource_id="cpu.pipeline",
        ),
        cache_hierarchy=cache
        or cache_hierarchy(resource_id="cpu.l1"),
        attainable_efficiency=0.5,
    )


def quantized_dot_capability(
    *,
    name="avx2_q8_dot",
    effective_ops_per_instruction=32.0,
    activation_quantization_instructions_per_block=0.0,
):
    return CPUQuantizedDotCapability(
        name=name,
        supported_weight_formats=("q4_0",),
        source_activation_bits=(16,),
        dot_activation_bits=8,
        dot_weight_bits=8,
        accumulator_bits=32,
        effective_ops_per_instruction=effective_ops_per_instruction,
        dot_issue_instructions_per_cycle_per_core=2.0,
        auxiliary_ops_per_instruction=32.0,
        activation_quantization_block_elements=64,
        activation_quantization_instructions_per_block=(
            activation_quantization_instructions_per_block
        ),
        evidence="unit-test ISA contract",
    )


def gpu_quantized_matmul_capability(
    *,
    name="cuda_mmq_q8",
    supported_weight_formats=("iq3_s", "iq4_xs"),
    source_activation_bits=(16,),
    accumulator_bits=32,
):
    return GPUQuantizedMatmulCapability(
        name=name,
        supported_weight_formats=supported_weight_formats,
        source_activation_bits=source_activation_bits,
        internal_activation_bits=8,
        accumulator_bits=accumulator_bits,
        kernel_family="cuda_mmq",
        tensor_core_dtype="int8",
        evidence="unit-test CUDA MMQ source contract",
        provenance="llama.cpp test revision",
    )


class CacheDirectionalTrafficTests(unittest.TestCase):
    def estimate(self, *, write_back=True, write_allocate=True, read_bytes=128, write_bytes=128, reuse=4.0):
        level = CacheLevelProfile(
            name="l1", capacity_bytes=1024, line_bytes=16,
            hit_latency_ns=2.0, bandwidth_gb_s=8.0,
            read_ports=1, write_ports=1, max_outstanding=1,
            resource_id="cache.l1",
        )
        return _cache_memory_demands(
            hierarchy=CacheHierarchyProfile((level,), write_back=write_back, write_allocate=write_allocate),
            read_bytes=read_bytes, write_bytes=write_bytes,
            working_set_bytes=256, reuse_factor=reuse,
            streaming_fraction=0.0,
            backing_bandwidth_gb_s=4.0,
            backing_energy_pj_per_byte=0.25,
            backing_resource_id="memory",
        )

    def test_write_back_and_write_through_have_distinct_local_service(self):
        wb, wb_meta = self.estimate(write_back=True)
        wt, wt_meta = self.estimate(write_back=False)
        wb_row, wt_row = wb_meta["levels"][0], wt_meta["levels"][0]
        self.assertEqual(wb_row["dirty_writeback_bytes"], 128)
        self.assertEqual(wb_row["write_through_bytes"], 0)
        self.assertEqual(wt_row["dirty_writeback_bytes"], 0)
        self.assertEqual(wt_row["write_through_bytes"], 128)
        self.assertEqual(wb_meta["backing_write_bytes"], 128)
        self.assertEqual(wt_meta["backing_write_bytes"], 128)
        self.assertEqual(wb[0].service_ns, wt[0].service_ns)
        self.assertEqual(wb_row["dirty_bytes_retained"], 0)

    def test_no_write_allocate_bypasses_only_misses(self):
        allocated, allocated_meta = self.estimate(write_allocate=True)
        bypass, bypass_meta = self.estimate(write_allocate=False)
        row = bypass_meta["levels"][0]
        self.assertEqual(row["write_hit_bytes"], 96)
        self.assertEqual(row["write_bypass_bytes"], 32)
        self.assertEqual(row["write_allocate_bytes"], 0)
        self.assertEqual(allocated_meta["levels"][0]["write_allocate_bytes"], 32)
        self.assertLess(bypass[0].bytes_moved, allocated[0].bytes_moved)
        self.assertEqual(bypass_meta["backing_write_bytes"], 128)

    def test_directional_flow_conserves_every_declared_write(self):
        for write_back in (False, True):
            for write_allocate in (False, True):
                for read_bytes, write_bytes in ((0, 127), (127, 0), (127, 65), (0, 0)):
                    with self.subTest(wb=write_back, wa=write_allocate, reads=read_bytes, writes=write_bytes):
                        demands, metadata = self.estimate(
                            write_back=write_back, write_allocate=write_allocate,
                            read_bytes=read_bytes, write_bytes=write_bytes,
                        )
                        row = metadata["levels"][0]
                        self.assertEqual(row["read_hit_bytes"] + row["read_miss_bytes"], read_bytes)
                        self.assertEqual(row["write_hit_bytes"] + row["write_miss_bytes"], write_bytes)
                        self.assertEqual(row["write_bypass_bytes"] + row["write_through_bytes"] + row["dirty_writeback_bytes"], write_bytes)
                        self.assertEqual(metadata["backing_write_bytes"], write_bytes)
                        self.assertEqual(metadata["backing_bytes"], metadata["backing_read_bytes"] + metadata["backing_write_bytes"])
                        self.assertEqual(demands[-1].bytes_moved, metadata["backing_bytes"])

    def test_streaming_io_is_not_removed_by_any_write_policy(self):
        for wb in (False, True):
            for wa in (False, True):
                _, metadata = self.estimate(write_back=wb, write_allocate=wa, reuse=1.0)
                self.assertEqual(metadata["backing_bytes"], 256)
                self.assertFalse(metadata["cross_invocation_cache_state"])


class TileWaveProxyTests(unittest.TestCase):
    def test_wave_boundary_is_explicit_not_silently_smoothed(self):
        rows = [mma_output_tile_wave_proxy(
            16, 32, 16 * tiles,
            sm_count=32, tensor_cores_per_sm=4,
            mma_m=16, mma_n=16, mma_k=16, occupancy=1.0,
        ) for tiles in (127, 128, 129)]
        self.assertEqual([row["output_tile_wave_count"] for row in rows], [1, 1, 2])
        self.assertEqual([row["output_tile_wave_utilization"] for row in rows], [127 / 128, 1, 129 / 256])
        for row in rows:
            self.assertGreater(row["output_tile_wave_utilization"], 0)
            self.assertLessEqual(row["output_tile_wave_utilization"], 1)

    def test_serial_k_tiles_do_not_increase_output_parallelism(self):
        kwargs = dict(sm_count=4, tensor_cores_per_sm=4, mma_m=16, mma_n=16, mma_k=16, occupancy=0.75)
        one = mma_output_tile_wave_proxy(16, 16, 64, **kwargs)
        many = mma_output_tile_wave_proxy(16, 1024, 64, **kwargs)
        self.assertEqual(one["output_tile_wave_utilization"], many["output_tile_wave_utilization"])
        self.assertEqual(one["serial_k_tile_count"], 1)
        self.assertEqual(many["serial_k_tile_count"], 64)


class GPUCostModelTests(unittest.TestCase):
    def test_shared_gemm_counts_packed_conversion_on_scalar_resource_once(self):
        gpu = replace(gpu_profile(), scalar_energy_pj_per_op=3.0)
        hbm = HBMProfile(bandwidth_gb_s=1.0e12)
        workload = GemmWorkload(
            m=1, k=64, n=64, activation_bits=16, weight_bits=4,
            packed_weight_formats=("IQ4_XS",),
            packed_weight_transform_operations=4096,
            epilogue_operations=1024,
            epilogue_name="gate",
        )
        estimate = estimate_gpu_gemm(gpu, hbm, workload)
        phase = next(phase for phase in estimate.phases if phase.name == "gpu_gemm")
        scalars = [d for d in phase.demands if d.resource_id == gpu.scalar_resource_id]
        self.assertEqual(len(scalars), 1)
        self.assertEqual(scalars[0].work_units, 5120)
        self.assertEqual(scalars[0].service_ns, 5120 / gpu.elementwise_gops)
        self.assertEqual(scalars[0].energy_pj, 5120 * 3.0)
        without_transform = estimate_gpu_gemm(
            gpu, hbm, replace(workload, packed_weight_transform_operations=0)
        )
        self.assertEqual(estimate.metadata["minimum_io_bytes"], without_transform.metadata["minimum_io_bytes"])
        self.assertEqual(estimate.metadata["memory_service_ns"], without_transform.metadata["memory_service_ns"])
        self.assertEqual(estimate.metadata["compute_service_ns"], without_transform.metadata["compute_service_ns"])
        self.assertNotIn("fused_dequant_accounting", without_transform.metadata)

    def test_gpu_quantized_matmul_capability_is_optional_and_validated(self):
        self.assertEqual(gpu_profile().quantized_matmul_capabilities, ())
        capability = gpu_quantized_matmul_capability()
        declared = replace(
            gpu_profile(), quantized_matmul_capabilities=(capability,)
        )
        self.assertEqual(declared.quantized_matmul_capabilities, (capability,))

        invalid_overrides = (
            {"supported_weight_formats": ()},
            {"supported_weight_formats": ("IQ3_S", "iq3_s")},
            {"source_activation_bits": ()},
            {"source_activation_bits": (16, 16)},
            {"internal_activation_bits": 16},
            {"kernel_family": ""},
            {"evidence": ""},
            {"provenance": ""},
        )
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(capability, **overrides)

        with self.assertRaisesRegex(ValueError, "must be a tuple"):
            replace(declared, quantized_matmul_capabilities=[capability])
        with self.assertRaisesRegex(ValueError, "tensor dtype is not supported"):
            replace(
                declared,
                tensor_core=replace(
                    declared.tensor_core,
                    supported_dtypes=("fp16", "fp32"),
                ),
            )

    def test_packed_gpu_matmul_selects_capability_internal_int8_peak(self):
        base = gpu_profile(cache=cache_hierarchy(capacity_bytes=1))
        gpu = replace(
            base,
            tensor_core=replace(
                base.tensor_core,
                dtype_throughput_scale={
                    "int8": 2.0,
                    "fp16": 1.0,
                    "fp32": 0.5,
                },
            ),
            quantized_matmul_capabilities=(
                gpu_quantized_matmul_capability(),
            ),
        )
        workload = GemmWorkload(
            m=1,
            k=64,
            n=64,
            activation_bits=16,
            weight_bits=3,
            output_bits=16,
            accumulator_bits=32,
            packed_weight_formats=("IQ3_S",),
            weight_storage_bytes=1_792,
            weight_metadata_bytes=128,
        )

        estimate = estimate_gpu_gemm(
            gpu, HBMProfile(bandwidth_gb_s=1.0e12), workload
        )
        metadata = estimate.metadata
        capability = metadata["quantized_matmul_capability"]

        self.assertEqual(metadata["source_tensor_dtype"], "fp16")
        self.assertEqual(metadata["source_activation_bits"], 16)
        self.assertEqual(metadata["source_weight_bits"], 3)
        self.assertEqual(metadata["accumulator_bits"], 32)
        self.assertEqual(metadata["internal_tensor_dtype"], "int8")
        self.assertEqual(metadata["tensor_dtype"], "int8")
        self.assertEqual(metadata["kernel_family"], "cuda_mmq")
        self.assertEqual(capability["name"], "cuda_mmq_q8")
        self.assertEqual(capability["internal_activation_bits"], 8)
        self.assertEqual(capability["accumulator_bits"], 32)
        self.assertIn("CUDA MMQ", capability["evidence"])
        self.assertEqual(
            metadata["structural_peak_tops"],
            gpu.tensor_core.peak_tops("int8"),
        )
        self.assertEqual(workload.weight_bytes, 1_920)
        self.assertEqual(metadata["weight_bytes"], workload.weight_bytes)
        self.assertEqual(estimate.useful_ops, workload.operations)

    def test_gpu_quantized_matmul_mismatches_preserve_source_dtype_path(self):
        capability = gpu_quantized_matmul_capability()
        gpu = replace(
            gpu_profile(), quantized_matmul_capabilities=(capability,)
        )
        matching = GemmWorkload(
            m=1,
            k=64,
            n=64,
            activation_bits=16,
            weight_bits=4,
            accumulator_bits=32,
            packed_weight_formats=("IQ4_XS",),
        )
        mismatches = (
            replace(matching, activation_bits=32),
            replace(matching, accumulator_bits=16),
            replace(matching, packed_weight_formats=("Q4_0",)),
        )

        for workload in mismatches:
            with self.subTest(workload=workload):
                estimate = estimate_gpu_gemm(
                    gpu, HBMProfile(bandwidth_gb_s=1.0e12), workload
                )
                self.assertNotIn(
                    "quantized_matmul_capability", estimate.metadata
                )
                self.assertEqual(
                    estimate.metadata["tensor_dtype"],
                    "fp32" if workload.activation_bits == 32 else "fp16",
                )

    def test_gpu_quantized_matmul_legacy_and_ambiguous_matching_fail_closed(self):
        base = gpu_profile()
        capability = gpu_quantized_matmul_capability()
        declared = replace(
            base, quantized_matmul_capabilities=(capability,)
        )
        legacy_workload = GemmWorkload(
            m=1, k=64, n=64, activation_bits=16, weight_bits=4
        )
        hbm = HBMProfile(bandwidth_gb_s=1.0e12)
        self.assertEqual(
            estimate_gpu_gemm(base, hbm, legacy_workload),
            estimate_gpu_gemm(declared, hbm, legacy_workload),
        )

        ambiguous = replace(
            base,
            quantized_matmul_capabilities=(
                capability,
                replace(capability, name="second_cuda_mmq"),
            ),
        )
        with self.assertRaisesRegex(
            ValueError,
            "multiple GPU quantized-matmul capabilities match",
        ):
            estimate_gpu_gemm(
                ambiguous,
                hbm,
                replace(
                    legacy_workload,
                    packed_weight_formats=("IQ4_XS",),
                ),
            )

    def test_host_gemm_offload_capability_is_optional_and_validated(self):
        self.assertIsNone(gpu_profile().host_gemm_offload)

        capability = HostGemmOffloadCapability(
            minimum_m=32,
            evidence="runtime source audit",
        )
        self.assertEqual(
            replace(gpu_profile(), host_gemm_offload=capability).host_gemm_offload,
            capability,
        )

        for invalid_threshold in (0, -1, 1.5, True):
            with self.subTest(minimum_m=invalid_threshold):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    HostGemmOffloadCapability(
                        minimum_m=invalid_threshold,
                        evidence="runtime source audit",
                    )
        for invalid_evidence in ("", "   ", 42):
            with self.subTest(evidence=invalid_evidence):
                with self.assertRaisesRegex(ValueError, "non-empty text"):
                    HostGemmOffloadCapability(
                        minimum_m=32,
                        evidence=invalid_evidence,
                    )
        with self.assertRaisesRegex(ValueError, "HostGemmOffloadCapability"):
            replace(gpu_profile(), host_gemm_offload={"minimum_m": 32})

    def test_host_recurrent_offload_capability_is_strict_and_optional(self):
        self.assertIsNone(gpu_profile().host_recurrent_offload)
        capability = HostRecurrentOffloadCapability(
            op_offload=True,
            minimum_m=32,
            architecture="qwen3_5_hybrid_transformer",
            supported_ops=("rms_norm", "ssm_conv", "gated_delta_net"),
            query_width=2048,
            key_width=2048,
            value_width=6144,
            conv_kernel_size=4,
            state_dtype="fp32",
            evidence="runtime source audit",
            provenance="llama.cpp@revision",
        )
        self.assertEqual(
            replace(
                gpu_profile(), host_recurrent_offload=capability
            ).host_recurrent_offload,
            capability,
        )
        with self.assertRaisesRegex(ValueError, "op_offload must be boolean"):
            replace(capability, op_offload=1)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            replace(capability, minimum_m=0)
        with self.assertRaisesRegex(ValueError, "non-empty tuple"):
            replace(capability, supported_ops=())
        with self.assertRaisesRegex(ValueError, "must be unique"):
            replace(capability, supported_ops=("SSM_CONV", "ssm_conv"))
        with self.assertRaisesRegex(ValueError, "non-empty text"):
            replace(capability, provenance="")
        with self.assertRaisesRegex(
            ValueError, "HostRecurrentOffloadCapability"
        ):
            replace(
                gpu_profile(),
                host_recurrent_offload={"op_offload": True},
            )

    def test_gpu_roofline_is_memory_bound(self):
        workload = GemmWorkload(
            m=1,
            k=64,
            n=64,
            activation_bits=8,
            weight_bits=8,
            output_bits=8,
        )
        gpu = gpu_profile(tensor_tops=100.0)
        hbm = HBMProfile(bandwidth_gb_s=1.0, efficiency=1.0)

        estimate = estimate_gpu_gemm(gpu, hbm, workload)

        expected_bytes = 64 + 4096 + 64
        backing_bytes = estimate.metadata["cache"]["backing_bytes"]
        self.assertEqual(estimate.metadata["bound"], "memory")
        self.assertAlmostEqual(
            estimate.metadata["compute_service_ns"],
            estimate.metadata["issued_operations"] / 100_000.0,
        )
        self.assertEqual(backing_bytes, expected_bytes)
        self.assertEqual(
            estimate.metadata["cache"]["levels"][0]["accessed_bytes"],
            expected_bytes,
        )
        self.assertAlmostEqual(estimate.metadata["memory_service_ns"], backing_bytes)
        self.assertAlmostEqual(estimate.service_ns, backing_bytes)
        self.assertLess(estimate.utilization, 1.0)

    def test_generic_tensor_kernel_uses_scalar_and_sfu_resources(self):
        gpu = gpu_profile(tensor_tops=100.0, scalar_gops=50.0, sfu_gops=5.0)
        hbm = HBMProfile(bandwidth_gb_s=1000.0)
        workload = TensorKernelWorkload(
            operations=10_000,
            transcendental_operations=100,
            read_bytes=100,
            write_bytes=100,
        )

        estimate = estimate_gpu_tensor_kernel(gpu, hbm, workload)

        self.assertAlmostEqual(
            estimate.metadata["compute_service_ns"], 10_000 / 50.0
        )
        resource_ids = {
            demand.resource_id
            for demand in estimate.phase("gpu_elementwise").demands
        }
        self.assertIn(gpu.scalar_resource_id, resource_ids)
        self.assertIn(gpu.special_function_resource_id, resource_ids)
        self.assertNotIn(gpu.tensor_core.resource_id, resource_ids)
        self.assertEqual(
            estimate.metadata["evidence"], EvidenceStatus.ANALYTICAL.value
        )

    def test_typed_gpu_kernels_use_class_specific_gops(self):
        gpu = gpu_profile(
            tensor_tops=1000.0,
            scalar_gops=100.0,
            reduction_gops=10.0,
        )
        hbm = HBMProfile(bandwidth_gb_s=1.0e12)
        elementwise = ElementwiseWorkload(
            elements=1024, operations_per_element=2
        )
        reduction = ReductionWorkload(
            input_elements=1025,
            output_elements=1,
            operations_per_combine=2,
        )

        elementwise_estimate = estimate_gpu_elementwise(
            gpu, hbm, elementwise
        )
        reduction_estimate = estimate_gpu_reduction(gpu, hbm, reduction)

        self.assertEqual(elementwise.operations, reduction.operations)
        self.assertAlmostEqual(
            elementwise_estimate.metadata["compute_service_ns"],
            elementwise.operations / 100.0,
        )
        self.assertAlmostEqual(
            reduction_estimate.metadata["compute_service_ns"],
            reduction.operations / 10.0,
        )
        self.assertGreater(
            reduction_estimate.metadata["compute_service_ns"],
            elementwise_estimate.metadata["compute_service_ns"],
        )
        self.assertNotIn("attainable_tops", elementwise_estimate.metadata)
        self.assertNotIn("attainable_tops", reduction_estimate.metadata)

    def test_fused_gpu_phase_overlaps_compute_and_memory_demands(self):
        gpu = gpu_profile(
            tensor_tops=10.0,
            scalar_gops=4.0,
            reduction_gops=2.0,
            kernel_launch_ns=3.0,
        )
        hbm = HBMProfile(bandwidth_gb_s=2.0)
        workload = ElementwiseWorkload(
            elements=8,
            operations_per_element=2,
            input_bits=8,
            output_bits=8,
        )

        estimate = gpu.estimate_elementwise(hbm, workload)
        fused = estimate.phase("gpu_elementwise")

        self.assertEqual(len(fused.demands), 3)
        self.assertAlmostEqual(fused.metadata["compute_service_ns"], 4.0)
        self.assertAlmostEqual(fused.metadata["memory_service_ns"], 8.0)
        self.assertAlmostEqual(fused.service_ns, 8.0)
        self.assertAlmostEqual(estimate.service_ns, 3.0 + 8.0)
        self.assertEqual(estimate.bytes_moved, 32)
        self.assertEqual(
            estimate.metadata["operator_class"],
            OperatorClass.ELEMENTWISE.value,
        )
        self.assertEqual(
            estimate.metadata["evidence"], EvidenceStatus.ANALYTICAL.value
        )

    def test_gpu_memory_exposes_sram_and_hbm_demands_after_launch(self):
        gpu = gpu_profile(tensor_tops=1.0, kernel_launch_ns=2.0)
        hbm = HBMProfile(bandwidth_gb_s=4.0)
        workload = MemoryWorkload(read_bytes=12, write_bytes=4)

        estimate = estimate_gpu_memory(gpu, hbm, workload)
        memory_phase = estimate.phase("gpu_memory")

        self.assertEqual(len(memory_phase.demands), 2)
        self.assertEqual(memory_phase.demands[-1].resource_id, hbm.resource_id)
        self.assertAlmostEqual(memory_phase.service_ns, 4.0)
        self.assertAlmostEqual(estimate.service_ns, 6.0)
        self.assertEqual(estimate.metadata["bound"], "memory")
        self.assertEqual(estimate.utilization, 0.0)

    def test_tensor_tile_padding_is_explicit(self):
        workload = GemmWorkload(m=17, k=17, n=17)
        estimate = estimate_gpu_gemm(
            gpu_profile(tensor_tops=10.0),
            HBMProfile(bandwidth_gb_s=1.0e12),
            workload,
        )

        self.assertEqual(estimate.metadata["tile_count"], 8)
        self.assertEqual(estimate.metadata["issued_operations"], 8 * 8192)
        self.assertEqual(
            estimate.metadata["tile_utilization"],
            workload.operations / float(8 * 8192),
        )

    def test_gemm_hbm_bandwidth_uses_output_tile_wave_utilization(self):
        base_gpu = gpu_profile(
            cache=cache_hierarchy(
                capacity_bytes=1,
                bandwidth_gb_s=1.0e12,
                hit_latency_ns=0.0,
                resource_id="gpu.l1",
            )
        )
        gpu = replace(
            base_gpu,
            tensor_core=replace(
                base_gpu.tensor_core,
                sm_count=84,
                tensor_cores_per_sm=4,
            ),
            occupancy=0.85,
        )
        hbm = HBMProfile(
            bandwidth_gb_s=800.0,
            efficiency=0.9,
            energy_pj_per_byte=0.25,
        )
        workload_m1 = GemmWorkload(
            m=1,
            k=5120,
            n=10240,
            weight_storage_bytes=25_690_112,
        )
        workload_m128 = replace(workload_m1, m=128)

        estimate_m1 = estimate_gpu_gemm(gpu, hbm, workload_m1)
        estimate_m128 = estimate_gpu_gemm(gpu, hbm, workload_m128)
        bandwidth_m1 = estimate_m1.metadata["hbm_bandwidth"]
        bandwidth_m128 = estimate_m128.metadata["hbm_bandwidth"]

        self.assertEqual(workload_m1.weight_bytes, workload_m128.weight_bytes)
        self.assertAlmostEqual(estimate_m1.metadata["tile_utilization"], 0.0625)
        self.assertNotAlmostEqual(
            bandwidth_m1["output_tile_wave_utilization"],
            estimate_m1.metadata["tile_utilization"],
        )
        self.assertEqual(bandwidth_m1["model"], "mma_output_tile_wave_proxy_v1")
        self.assertFalse(bandwidth_m1["fallback_to_fixed_bandwidth"])
        self.assertEqual(bandwidth_m1["m_tile_count"], 1)
        self.assertEqual(bandwidth_m1["n_tile_count"], 640)
        self.assertEqual(bandwidth_m1["serial_k_tile_count"], 320)
        self.assertEqual(bandwidth_m1["independent_output_tile_count"], 640)
        self.assertEqual(bandwidth_m1["warp_equivalent_count"], 336)
        self.assertAlmostEqual(
            bandwidth_m1["resident_warp_equivalent_capacity"], 285.6
        )
        self.assertEqual(bandwidth_m1["parallel_tile_slots"], 285)
        self.assertEqual(bandwidth_m1["output_tile_wave_count"], 3)
        self.assertAlmostEqual(
            bandwidth_m1["output_tile_wave_utilization"], 640.0 / 855.0
        )
        self.assertEqual(bandwidth_m128["m_tile_count"], 8)
        self.assertEqual(
            bandwidth_m128["independent_output_tile_count"], 5120
        )
        self.assertEqual(bandwidth_m128["output_tile_wave_count"], 18)
        self.assertAlmostEqual(
            bandwidth_m128["output_tile_wave_utilization"], 5120.0 / 5130.0
        )
        self.assertFalse(bandwidth_m1["tile_utilization_applied_to_hbm"])
        self.assertIn("K tiles are serial", bandwidth_m1["parallelism_basis"])
        self.assertAlmostEqual(
            bandwidth_m1["peak_effective_hbm_bandwidth_gb_s"], 720.0
        )
        self.assertLess(
            bandwidth_m1["shape_effective_hbm_bandwidth_gb_s"],
            bandwidth_m128["shape_effective_hbm_bandwidth_gb_s"],
        )
        self.assertLessEqual(
            bandwidth_m128["shape_effective_hbm_bandwidth_gb_s"],
            bandwidth_m128["peak_effective_hbm_bandwidth_gb_s"],
        )

        for workload, estimate in (
            (workload_m1, estimate_m1),
            (workload_m128, estimate_m128),
        ):
            bandwidth = estimate.metadata["hbm_bandwidth"]
            cache = estimate.metadata["cache"]
            hbm_demand = estimate.phase("gpu_gemm").demands[-1]
            self.assertEqual(cache["backing_bytes"], workload.minimum_io_bytes)
            self.assertEqual(hbm_demand.bytes_moved, workload.minimum_io_bytes)
            self.assertAlmostEqual(
                hbm_demand.service_ns,
                workload.minimum_io_bytes
                / bandwidth["shape_effective_hbm_bandwidth_gb_s"],
            )
            self.assertAlmostEqual(
                hbm_demand.energy_pj,
                workload.minimum_io_bytes * hbm.energy_pj_per_byte,
            )
            self.assertEqual(
                estimate.phase("gpu_gemm").metadata["hbm_bandwidth"],
                bandwidth,
            )

    def test_cache_capacity_boundary_reduces_backing_traffic(self):
        workload = ElementwiseWorkload(
            elements=1024,
            reuse_factor=4.0,
            streaming_fraction=0.0,
        )
        hbm = HBMProfile(bandwidth_gb_s=1000.0)
        small = estimate_gpu_elementwise(
            gpu_profile(cache=cache_hierarchy(capacity_bytes=64, resource_id="gpu.l1")),
            hbm,
            workload,
        )
        fitting = estimate_gpu_elementwise(
            gpu_profile(
                cache=cache_hierarchy(
                    capacity_bytes=workload.effective_working_set_bytes,
                    resource_id="gpu.l1",
                )
            ),
            hbm,
            workload,
        )
        oversized = estimate_gpu_elementwise(
            gpu_profile(
                cache=cache_hierarchy(
                    capacity_bytes=2 * workload.effective_working_set_bytes,
                    resource_id="gpu.l1",
                )
            ),
            hbm,
            workload,
        )

        self.assertGreater(
            small.metadata["cache"]["backing_bytes"],
            fitting.metadata["cache"]["backing_bytes"],
        )
        self.assertEqual(
            fitting.metadata["cache"]["backing_bytes"],
            oversized.metadata["cache"]["backing_bytes"],
        )

    def test_fused_attention_counts_tensor_scalar_sfu_and_elided_scores(self):
        workload = FusedAttentionWorkload(
            batch_tokens=2,
            context_tokens=4,
            hidden_size=8,
        )
        gpu = gpu_profile()
        estimate = estimate_gpu_fused_attention(
            gpu, HBMProfile(bandwidth_gb_s=1000.0), workload
        )

        self.assertEqual(workload.tensor_operations, 256)
        self.assertEqual(workload.scalar_operations, 40)
        self.assertEqual(workload.transcendental_operations, 8)
        self.assertEqual(workload.score_matrix_bytes_elided, 16)
        self.assertEqual(estimate.metadata["tensor_operations"], 256)
        self.assertEqual(estimate.metadata["scalar_operations"], 40)
        self.assertEqual(estimate.metadata["special_function_operations"], 8)
        resources = {
            demand.resource_id
            for demand in estimate.phase("gpu_fused_attention").demands
        }
        self.assertTrue(
            {
                gpu.tensor_core.resource_id,
                gpu.scalar_resource_id,
                gpu.special_function_resource_id,
            }.issubset(resources)
        )

    def test_fused_attention_separates_query_compute_from_kv_storage(self):
        workload = FusedAttentionWorkload(
            batch_tokens=2,
            context_tokens=7,
            hidden_size=16,
            input_bits=16,
            output_bits=16,
            kv_hidden_size=4,
            kv_input_bits=8,
            kv_read_tokens=11,
        )
        estimate = estimate_gpu_fused_attention(
            gpu_profile(cache=cache_hierarchy(capacity_bytes=1)),
            HBMProfile(bandwidth_gb_s=1000.0),
            workload,
        )

        self.assertEqual(workload.tensor_operations, 4 * 2 * 7 * 16)
        self.assertEqual(workload.kv_read_bytes, 2 * 11 * 4)
        self.assertEqual(workload.read_bytes, 2 * 16 * 2 + 2 * 11 * 4)
        self.assertEqual(workload.write_bytes, 2 * 16 * 2)
        self.assertEqual(estimate.metadata["query_hidden_size"], 16)
        self.assertEqual(estimate.metadata["kv_hidden_size"], 4)
        self.assertEqual(estimate.metadata["kv_input_bits"], 8)
        self.assertEqual(estimate.metadata["kv_read_tokens"], 11)
        self.assertEqual(estimate.metadata["kv_read_bytes"], 88)
        self.assertEqual(estimate.metadata["kv_payload_bytes"], 88)
        self.assertEqual(estimate.metadata["kv_metadata_bytes"], 0)
        self.assertFalse(estimate.metadata["kv_physical_contract_applied"])

    def test_fused_attention_q4_kv_contract_is_compulsory_hbm_io(self):
        workload = FusedAttentionWorkload(
            batch_tokens=1,
            context_tokens=8192,
            hidden_size=5120,
            input_bits=16,
            output_bits=16,
            kv_hidden_size=1024,
            kv_input_bits=4,
            kv_read_tokens=8192,
            kv_physical_contract=FusedAttentionKVPhysicalContract(
                payload_bytes_per_token=1024,
                metadata_bytes_per_token=128,
                dequant_operations_per_token=2048,
                artifact_format="Q4_0",
            ),
        )
        hbm = HBMProfile(bandwidth_gb_s=1000.0)

        small_cache = estimate_gpu_fused_attention(
            gpu_profile(cache=cache_hierarchy(capacity_bytes=1)),
            hbm,
            workload,
        )
        large_cache = estimate_gpu_fused_attention(
            gpu_profile(cache=cache_hierarchy(capacity_bytes=1 << 40)),
            hbm,
            workload,
        )

        self.assertEqual(workload.query_read_bytes, 10_240)
        self.assertEqual(workload.write_bytes, 10_240)
        self.assertEqual(workload.kv_payload_bytes, 8_388_608)
        self.assertEqual(workload.kv_metadata_bytes, 1_048_576)
        self.assertEqual(workload.kv_read_bytes, 9_437_184)
        self.assertEqual(workload.kv_dequant_operations, 16_777_216)
        self.assertEqual(workload.read_bytes, 9_447_424)
        self.assertEqual(
            workload.read_bytes + workload.write_bytes,
            9_457_664,
        )

        metadata = small_cache.metadata
        phase_metadata = small_cache.phase("gpu_fused_attention").metadata
        self.assertEqual(metadata["kv_payload_bytes"], 8_388_608)
        self.assertEqual(metadata["kv_payload_bytes_per_token"], 1024)
        self.assertEqual(metadata["kv_scale_bytes"], 1_048_576)
        self.assertEqual(metadata["kv_scale_bytes_per_token"], 128)
        self.assertEqual(metadata["kv_dequant_operations"], 16_777_216)
        self.assertEqual(metadata["kv_dequant_operations_per_token"], 2048)
        self.assertEqual(metadata["minimum_io_bytes"], 9_457_664)
        self.assertEqual(
            metadata["cache"]["backing_bytes"],
            9_457_664,
        )
        self.assertEqual(
            large_cache.metadata["cache"]["backing_bytes"],
            9_457_664,
        )
        self.assertEqual(metadata["cache"]["reuse_factor"], 1.0)
        self.assertEqual(metadata["cache"]["streaming_fraction"], 1.0)
        self.assertEqual(
            metadata["memory_traffic_semantics"],
            "compulsory_minimum_io",
        )
        self.assertEqual(phase_metadata["kv_artifact_format"], "Q4_0")
        self.assertEqual(
            phase_metadata["softmax_scalar_operations"],
            8192 * 5,
        )
        self.assertEqual(
            metadata["scalar_operations"],
            8192 * 5 + 16_777_216,
        )
        self.assertEqual(len(small_cache.phases), 1)

    def test_gemm_storage_overrides_do_not_change_compute_geometry(self):
        workload = GemmWorkload(
            m=2,
            k=16,
            n=7,
            activation_bits=16,
            weight_bits=16,
            output_bits=16,
            weight_storage_bytes=28,
            output_storage_bytes=20,
        )
        estimate = estimate_gpu_gemm(
            gpu_profile(cache=cache_hierarchy(capacity_bytes=1)),
            HBMProfile(bandwidth_gb_s=1000.0),
            workload,
        )

        self.assertEqual(workload.operations, 2 * 2 * 16 * 7)
        self.assertEqual(workload.weight_bytes, 28)
        self.assertEqual(workload.output_bytes, 20)
        self.assertEqual(estimate.metadata["weight_bytes"], 28)
        self.assertEqual(estimate.metadata["output_bytes"], 20)


class CPUCostModelTests(unittest.TestCase):
    def setUp(self):
        self.cpu = cpu_profile()
        self.memory = HostMemoryProfile(bandwidth_gb_s=1.0e12)

    def test_fixed_q4_row_work_scales_independently_of_k_and_keeps_memory_io(self):
        capability = replace(
            quantized_dot_capability(), supported_weight_formats=("Q4_K",),
            source_dot_work_all_m=True, source_dot_work_max_k=2**31 - 1,
            source_dot_work={"Q4_K": {
                "block_elements": 256, "auxiliary_vector_ops": 52,
                "vector_loads": 14, "scalar_lut_loads": 2,
                "row_auxiliary_vector_ops": 16, "row_vector_loads": 9,
                "row_scalar_lut_loads": 0, "row_store_issue_ops": 8,
            }},
        )
        cpu = replace(self.cpu, quantized_dot_capabilities=(capability,))
        legacy = replace(cpu, quantized_dot_capabilities=(replace(
            capability, source_dot_work={}, source_dot_work_all_m=False,
            source_dot_work_max_k=None,
        ),))
        for m in (1, 3, 4, 8, 32, 128):
            for k in (256, 512):
                with self.subTest(m=m, k=k):
                    workload = GemmWorkload(
                        m=m, k=k, n=16, activation_bits=16, weight_bits=4,
                        packed_weight_formats=("Q4_K",),
                        packed_weight_transform_operations=8192,
                    )
                    result = estimate_cpu_gemm(cpu, self.memory, workload)
                    old = estimate_cpu_gemm(legacy, self.memory, workload)
                    schedule = result.metadata["instruction_schedule"]
                    rows, blocks = m * 16, m * 16 * k // 256
                    self.assertEqual(schedule["source_dot_rows"], rows)
                    self.assertEqual(schedule["source_dot_blocks"], blocks)
                    self.assertEqual(schedule["source_dot_row_totals"], {
                        "auxiliary_vector_ops": 16 * rows, "vector_loads": 9 * rows,
                        "scalar_lut_loads": 0, "store_issue_ops": 8 * rows,
                    })
                    self.assertEqual(schedule["packed_weight_transform_instructions"], 52 * blocks + 16 * rows)
                    self.assertEqual(schedule["load_instructions"], max(schedule["minimum_io_load_proxy"], 16 * blocks + 9 * rows))
                    self.assertEqual(schedule["store_instructions"], max(schedule["minimum_io_store_proxy"], 8 * rows))
                    self.assertEqual(schedule["compute_instructions"], 16 * blocks)
                    self.assertEqual(schedule["activation_elements"], m * k)
                    self.assertEqual(schedule["timing_completeness"], "partial")
                    self.assertEqual(result.bytes_moved, old.bytes_moved)
                    self.assertEqual(result.metadata["memory_service_ns"], old.metadata["memory_service_ns"])

        # Explicit zero row budgets preserve minimum output stores and remain
        # partial. The output-store proxy must never be added to source stores.
        zero_budget = dict(capability.source_dot_work["Q4_K"])
        zero_budget.update({key: 0 for key in zero_budget if key.startswith("row_")})
        zero_cpu = replace(cpu, quantized_dot_capabilities=(replace(capability, source_dot_work={"Q4_K": zero_budget}),))
        workload = GemmWorkload(m=1, k=256, n=16, activation_bits=16, weight_bits=4, packed_weight_formats=("Q4_K",))
        schedule = estimate_cpu_gemm(zero_cpu, self.memory, workload).metadata["instruction_schedule"]
        self.assertEqual(schedule["store_instructions"], schedule["minimum_io_store_proxy"])
        self.assertEqual(schedule["source_dot_rows"], 16)
        self.assertEqual(schedule["timing_completeness"], "partial")

        for k in (257, 2**31):
            with self.subTest(ineligible_k=k):
                schedule = estimate_cpu_gemm(cpu, self.memory, replace(workload, k=k)).metadata["instruction_schedule"]
                self.assertNotIn("source_dot_rows", schedule)
                self.assertNotIn("source_dot_blocks", schedule)
                self.assertEqual(schedule["timing_completeness"], "partial")
        mixed = replace(workload, packed_weight_formats=("Q4_K", "Q6_K"))
        self.assertEqual(estimate_cpu_gemm(cpu, self.memory, mixed), estimate_cpu_gemm(self.cpu, self.memory, mixed))

    def test_fixed_row_work_belongs_only_to_eligible_mixed_segments(self):
        capability = replace(
            quantized_dot_capability(), supported_weight_formats=("Q4_K", "Q5_K"),
            source_dot_work_max_m=7, source_dot_work_max_k=2**31 - 1,
            source_dot_work={"Q4_K": {
                "block_elements": 256, "auxiliary_vector_ops": 52,
                "vector_loads": 14, "scalar_lut_loads": 2,
                "row_auxiliary_vector_ops": 16, "row_vector_loads": 9,
                "row_scalar_lut_loads": 0, "row_store_issue_ops": 8,
            }},
        )
        cpu = replace(self.cpu, quantized_dot_capabilities=(capability,))
        workload = GemmWorkload(
            m=3, k=512, n=8, activation_bits=16, weight_bits=4,
            packed_weight_formats=("Q4_K", "Q5_K"),
            packed_weight_transform_operations=800,
            packed_weight_format_segments=(("Q4_K", 3, 300), ("Q5_K", 5, 500)),
        )
        schedule = estimate_cpu_gemm(cpu, self.memory, workload).metadata["instruction_schedule"]
        self.assertEqual(schedule["source_dot_rows"], 9)
        self.assertEqual(schedule["source_dot_blocks"], 18)
        self.assertEqual(schedule["source_dot_segment_totals"], {"auxiliary_vector_ops": 936, "vector_loads": 252, "scalar_lut_loads": 36})
        self.assertEqual(schedule["source_dot_row_totals"]["auxiliary_vector_ops"], 144)
        self.assertEqual(schedule["packed_weight_transform_operations"], 500)
        self.assertEqual(schedule["packed_weight_transform_instructions"], 936 + 144 + 16)
        self.assertEqual(schedule["source_dot_row_totals"]["store_issue_ops"], 72)
        for changed in (replace(workload, m=8), replace(workload, k=257), replace(workload, packed_weight_format_segments=())):
            schedule = estimate_cpu_gemm(cpu, self.memory, changed).metadata["instruction_schedule"]
            self.assertNotIn("source_dot_rows", schedule)
            self.assertEqual(schedule["timing_completeness"], "partial")

    def test_scalar_schedule_rejects_even_zero_dot_row_work(self):
        with self.assertRaisesRegex(ValueError, "plain 32-bit elementwise"):
            _cpu_instruction_schedule(
                self.cpu, operator_class=OperatorClass.ELEMENTWISE,
                operations=1, read_bytes=4, write_bytes=4, element_bits=32,
                dependency_depth=1, scalar_execution=True,
                source_dot_row_totals={"auxiliary_vector_ops": 0, "vector_loads": 0,
                                       "scalar_lut_loads": 0, "store_issue_ops": 0},
            )

    def test_source_dot_budget_scales_by_rows_without_multiplying_backing_io(self):
        capability = replace(
            quantized_dot_capability(), source_dot_work_max_m=7,
            supported_weight_formats=("q4_0", "iq4_xs"),
            source_dot_work={"q4_0": {"block_elements": 256,
                "auxiliary_vector_ops": 126, "vector_loads": 12, "scalar_lut_loads": 64}},
        )
        cpu = replace(self.cpu, quantized_dot_capabilities=(capability,))
        legacy = replace(cpu, quantized_dot_capabilities=(
            replace(capability, source_dot_work={}, source_dot_work_max_m=None),
        ))
        for m in (1, 3, 4, 7):
            workload = GemmWorkload(m=m, k=512, n=16, activation_bits=16,
                weight_bits=4, packed_weight_formats=("Q4_0",),
                packed_weight_transform_operations=8192)
            estimate = estimate_cpu_gemm(cpu, self.memory, workload)
            original = estimate_cpu_gemm(legacy, self.memory, workload)
            schedule = estimate.metadata["instruction_schedule"]
            self.assertEqual(schedule["source_dot_blocks"], m * 32)
            self.assertEqual(schedule["packed_weight_transform_instructions"], m * 32 * 126)
            self.assertEqual(schedule["source_scalar_lut_loads"], m * 32 * 64)
            self.assertEqual(schedule["compute_instructions"], original.metadata["instruction_schedule"]["compute_instructions"])
            self.assertEqual(schedule["timing_completeness"], "partial")
            self.assertEqual(estimate.metadata["memory_service_ns"], original.metadata["memory_service_ns"])
            self.assertEqual(estimate.bytes_moved, original.bytes_moved)

        for m, k, formats in ((8, 512, ("Q4_0",)), (4, 257, ("Q4_0",)),
                              (4, 512, ("Q4_0", "IQ4_XS")), (4, 512, ("IQ4_XS",))):
            workload = GemmWorkload(m=m, k=k, n=16, activation_bits=16,
                weight_bits=4, packed_weight_formats=formats)
            self.assertEqual(estimate_cpu_gemm(cpu, self.memory, workload),
                             estimate_cpu_gemm(legacy, self.memory, workload))

    def test_source_dot_budget_uses_only_audited_segments_of_mixed_projection(self):
        capability = replace(
            quantized_dot_capability(),
            supported_weight_formats=("q4_0", "q5_k"),
            source_dot_work_max_m=7,
            source_dot_work={"q4_0": {"block_elements": 256,
                "auxiliary_vector_ops": 126, "vector_loads": 12,
                "scalar_lut_loads": 64}},
        )
        cpu = replace(self.cpu, quantized_dot_capabilities=(capability,))
        workload = GemmWorkload(
            m=1, k=256, n=8, activation_bits=16, weight_bits=4,
            packed_weight_formats=("q4_0", "q5_k"),
            packed_weight_transform_operations=800,
            packed_weight_format_segments=(("Q4_0", 3, 300), ("Q5_K", 5, 500)),
        )

        schedule = estimate_cpu_gemm(cpu, self.memory, workload).metadata[
            "instruction_schedule"
        ]
        self.assertEqual(schedule["source_dot_blocks"], 3)
        self.assertIsNone(schedule["source_dot_work"])
        self.assertEqual(schedule["source_dot_segment_totals"], {
            "auxiliary_vector_ops": 378, "vector_loads": 36,
            "scalar_lut_loads": 192,
        })
        self.assertEqual(schedule["packed_weight_transform_instructions"], 394)
        self.assertEqual(schedule["source_vector_loads"], 36)
        self.assertEqual(schedule["source_scalar_lut_loads"], 192)
        self.assertEqual(schedule["timing_completeness"], "partial")

    def test_cpu_gops_unit_is_operations_per_nanosecond(self):
        workload = GemmWorkload(
            m=1,
            k=10,
            n=50,
            activation_bits=8,
            weight_bits=8,
            output_bits=8,
        )

        estimate = estimate_cpu_gemm(self.cpu, self.memory, workload)

        self.assertEqual(workload.operations, 1000)
        schedule = estimate.metadata["instruction_schedule"]
        self.assertEqual(schedule["compute_instructions"], 63)
        self.assertEqual(schedule["limiting_stage"], "execution")
        self.assertAlmostEqual(schedule["service_ns"], 51.2)
        self.assertAlmostEqual(
            estimate.metadata["throughput_compute_service_ns"], 100.0
        )
        self.assertAlmostEqual(estimate.metadata["compute_service_ns"], 100.0)
        self.assertEqual(
            estimate.metadata["operator_class"], OperatorClass.GEMM.value
        )

    def test_quantized_dot_capability_controls_schedule_and_throughput(self):
        pipeline = replace(self.cpu.pipeline, simd_width_bits=256)
        legacy_cpu = replace(self.cpu, pipeline=pipeline)
        avx2 = replace(
            legacy_cpu,
            quantized_dot_capabilities=(quantized_dot_capability(),),
        )
        vnni = replace(
            legacy_cpu,
            quantized_dot_capabilities=(
                quantized_dot_capability(
                    name="avx_vnni_q8_dot",
                    effective_ops_per_instruction=64.0,
                ),
            ),
        )
        packed_workload = GemmWorkload(
            m=1,
            k=256,
            n=256,
            activation_bits=16,
            weight_bits=4,
            output_bits=16,
            packed_weight_formats=("Q4_0",),
        )

        legacy = estimate_cpu_gemm(
            legacy_cpu,
            self.memory,
            replace(packed_workload, packed_weight_formats=()),
        )
        avx2_estimate = estimate_cpu_gemm(
            avx2, self.memory, packed_workload
        )
        vnni_estimate = estimate_cpu_gemm(
            vnni, self.memory, packed_workload
        )

        self.assertAlmostEqual(legacy_cpu.attainable_gemm_gops, 40.0)
        self.assertAlmostEqual(
            avx2.attainable_quantized_dot_gops(
                avx2.quantized_dot_capabilities[0]
            ),
            40.0,
        )
        self.assertAlmostEqual(
            vnni.attainable_quantized_dot_gops(
                vnni.quantized_dot_capabilities[0]
            ),
            80.0,
        )
        self.assertAlmostEqual(
            avx2_estimate.metadata["throughput_compute_service_ns"],
            legacy.metadata["throughput_compute_service_ns"],
        )
        self.assertAlmostEqual(
            vnni_estimate.metadata["throughput_compute_service_ns"],
            avx2_estimate.metadata["throughput_compute_service_ns"] / 2.0,
        )
        self.assertAlmostEqual(
            vnni_estimate.metadata["compute_service_ns"],
            avx2_estimate.metadata["compute_service_ns"] / 2.0,
        )
        self.assertAlmostEqual(
            vnni_estimate.metadata["compute_service_ns"],
            vnni_estimate.metadata["instruction_schedule"]["service_ns"],
        )
        self.assertEqual(
            avx2_estimate.metadata["instruction_schedule"][
                "effective_ops_per_instruction"
            ],
            32.0,
        )
        self.assertEqual(
            vnni_estimate.metadata["instruction_schedule"][
                "effective_ops_per_instruction"
            ],
            64.0,
        )

    def test_quantized_dot_keeps_storage_and_internal_widths_separate(self):
        cpu = replace(
            self.cpu,
            pipeline=replace(self.cpu.pipeline, simd_width_bits=256),
            quantized_dot_capabilities=(
                quantized_dot_capability(
                    activation_quantization_instructions_per_block=2.5
                ),
            ),
            elementwise_energy_pj_per_op=0.25,
        )
        workload = GemmWorkload(
            m=2,
            k=64,
            n=64,
            activation_bits=16,
            weight_bits=4,
            output_bits=16,
            packed_weight_formats=("q4_0",),
            packed_weight_transform_operations=65,
        )

        estimate = estimate_cpu_gemm(cpu, self.memory, workload)
        schedule = estimate.metadata["instruction_schedule"]

        self.assertEqual(workload.activation_bytes, 256)
        self.assertEqual(estimate.metadata["read_bytes"], 256 + 2048)
        self.assertEqual(schedule["source_activation_bits"], 16)
        self.assertEqual(schedule["dot_activation_bits"], 8)
        self.assertEqual(schedule["dot_weight_bits"], 8)
        self.assertEqual(schedule["accumulator_bits"], 32)
        self.assertEqual(schedule["packed_weight_transform_instructions"], 3)
        self.assertEqual(schedule["activation_quantization_blocks"], 2)
        self.assertEqual(schedule["activation_quantization_instructions"], 5)
        self.assertEqual(schedule["auxiliary_instructions"], 8)
        self.assertEqual(schedule["timing_completeness"], "complete")
        compute_demand = estimate.phase("cpu_gemm").demands[0]
        self.assertAlmostEqual(
            compute_demand.energy_pj,
            (65 + 5 * 32) * 0.25,
        )

    def test_unspecified_activation_quantization_cost_marks_partial_timing(self):
        cpu = replace(
            self.cpu,
            quantized_dot_capabilities=(
                quantized_dot_capability(
                    activation_quantization_instructions_per_block=None
                ),
            ),
        )
        workload = GemmWorkload(
            m=1,
            k=64,
            n=64,
            activation_bits=16,
            weight_bits=4,
            packed_weight_formats=("q4_0",),
        )

        schedule = estimate_cpu_gemm(
            cpu, self.memory, workload
        ).metadata["instruction_schedule"]

        self.assertGreater(schedule["activation_quantization_blocks"], 0)
        self.assertEqual(schedule["activation_quantization_instructions"], 0)
        self.assertFalse(schedule["activation_quantization_accounted"])
        self.assertEqual(schedule["timing_completeness"], "partial")

    def test_missing_quantized_dot_match_accounts_generic_unpack_work(self):
        capability_cpu = replace(
            self.cpu,
            quantized_dot_capabilities=(quantized_dot_capability(),),
        )
        legacy_workload = GemmWorkload(
            m=1,
            k=64,
            n=64,
            activation_bits=16,
            weight_bits=4,
        )
        unsupported_workload = replace(
            legacy_workload,
            packed_weight_formats=("unsupported_q4",),
            packed_weight_transform_operations=128,
        )

        without_format = estimate_cpu_gemm(
            capability_cpu, self.memory, legacy_workload
        )
        without_capability = estimate_cpu_gemm(
            self.cpu, self.memory, unsupported_workload
        )
        baseline = estimate_cpu_gemm(
            self.cpu, self.memory, legacy_workload
        )

        self.assertEqual(
            without_format.metadata["instruction_schedule"],
            baseline.metadata["instruction_schedule"],
        )
        # A generic CPU path must retain packed-weight unpack/dequant work even
        # when no ISA-specific quantized-dot capability matches.
        self.assertGreater(
            without_capability.metadata["instruction_schedule"]["auxiliary_instructions"],
            baseline.metadata["instruction_schedule"]["auxiliary_instructions"],
        )
        self.assertEqual(
            without_capability.metadata["instruction_schedule"]["packed_weight_transform_instructions"],
            128,
        )

    def test_ambiguous_quantized_dot_capabilities_are_rejected(self):
        first = quantized_dot_capability(name="first")
        cpu = replace(
            self.cpu,
            quantized_dot_capabilities=(
                first,
                replace(first, name="second"),
            ),
        )
        workload = GemmWorkload(
            m=1,
            k=64,
            n=64,
            activation_bits=16,
            weight_bits=4,
            packed_weight_formats=("q4_0",),
        )

        with self.assertRaisesRegex(
            ValueError, "multiple CPU quantized-dot capabilities match"
        ):
            estimate_cpu_gemm(cpu, self.memory, workload)

    def test_cpu_operator_classes_have_independent_throughput(self):
        elementwise = ElementwiseWorkload(elements=32)
        reduction = ReductionWorkload(
            input_elements=33, output_elements=1
        )

        ew_estimate = estimate_cpu_elementwise(
            self.cpu, self.memory, elementwise
        )
        reduction_estimate = estimate_cpu_reduction(
            self.cpu, self.memory, reduction
        )

        self.assertEqual(elementwise.operations, reduction.operations)
        self.assertEqual(
            ew_estimate.metadata["instruction_schedule"]["dependency_depth"],
            1,
        )
        self.assertEqual(
            reduction_estimate.metadata["instruction_schedule"]["dependency_depth"],
            6,
        )
        self.assertGreater(
            reduction_estimate.metadata["compute_service_ns"],
            ew_estimate.metadata["compute_service_ns"],
        )

    def test_cpu_memory_service_uses_decimal_gb_per_second_units(self):
        memory = HostMemoryProfile(
            bandwidth_gb_s=8.0,
            efficiency=0.5,
            energy_pj_per_byte=0.25,
        )
        workload = MemoryWorkload(read_bytes=24, write_bytes=8)

        estimate = estimate_cpu_memory(self.cpu, memory, workload)

        self.assertAlmostEqual(estimate.service_ns, 8.0)
        self.assertEqual(estimate.bytes_moved, 64)
        self.assertAlmostEqual(estimate.energy_pj, 8.0)
        self.assertEqual(
            estimate.metadata["operator_class"], OperatorClass.MEMORY.value
        )

    def test_frontend_rob_lsq_and_mlp_limits_are_audited(self):
        constrained_pipeline = replace(
            self.cpu.pipeline,
            decode_width=1,
            issue_width=1,
            retire_width=1,
            reorder_buffer_entries=2,
            load_store_queue_entries=2,
            memory_level_parallelism=1,
        )
        constrained = replace(self.cpu, pipeline=constrained_pipeline)
        workload = ElementwiseWorkload(elements=128, dependency_depth=4)

        estimate = estimate_cpu_elementwise(
            constrained, self.memory, workload
        )
        schedule = estimate.metadata["instruction_schedule"]

        self.assertGreater(schedule["rob_waves"], 1)
        self.assertGreater(schedule["lsq_waves"], 1)
        self.assertGreater(schedule["mlp_waves"], 1)
        self.assertIn(
            schedule["limiting_stage"],
            {
                "frontend",
                "dependency",
                "lsq_window",
                "memory_level_parallelism",
            },
        )


class TypedWorkloadContractTests(unittest.TestCase):
    def test_fixed_per_invocation_operations_are_counted_exactly(self):
        elementwise = ElementwiseWorkload(
            elements=8,
            operations_per_element=2,
            fixed_operations=3,
        )
        reduction = ReductionWorkload(
            input_elements=8,
            output_elements=2,
            operations_per_combine=2,
            fixed_operations=2,
        )

        self.assertEqual(elementwise.operations, 19)
        self.assertEqual(reduction.operations, 14)

    def test_shape_growth_is_monotonic_for_work_and_bytes(self):
        small_elementwise = ElementwiseWorkload(elements=7, input_bits=4)
        large_elementwise = replace(small_elementwise, elements=8)
        small_reduction = ReductionWorkload(input_elements=7)
        large_reduction = replace(small_reduction, input_elements=8)

        self.assertLess(
            small_elementwise.operations, large_elementwise.operations
        )
        self.assertLessEqual(
            small_elementwise.minimum_io_bytes,
            large_elementwise.minimum_io_bytes,
        )
        self.assertLess(small_reduction.operations, large_reduction.operations)
        self.assertLessEqual(
            small_reduction.minimum_io_bytes,
            large_reduction.minimum_io_bytes,
        )

    def test_invalid_shapes_bytes_and_profile_rates_are_rejected(self):
        invalid_constructors = (
            lambda: ElementwiseWorkload(elements=0),
            lambda: ElementwiseWorkload(elements=True),
            lambda: ReductionWorkload(input_elements=4, output_elements=5),
            lambda: MemoryWorkload(),
            lambda: MemoryWorkload(read_bytes=-1),
            lambda: ElementwiseWorkload(elements=1, fixed_operations=-1),
            lambda: ReductionWorkload(
                input_elements=2,
                output_elements=1,
                fixed_operations=-1,
            ),
            lambda: TensorCoreProfile(
                sm_count=1,
                tensor_cores_per_sm=1,
                frequency_ghz=float("nan"),
            ),
            lambda: gpu_profile(scalar_gops=float("inf")),
            lambda: replace(gpu_profile(), attainable_efficiency=True),
            lambda: CacheLevelProfile(
                name="bad",
                capacity_bytes=0,
                line_bytes=64,
                hit_latency_ns=1.0,
                bandwidth_gb_s=1.0,
            ),
            lambda: replace(cpu_profile().pipeline, issue_width=0),
            lambda: HostMemoryProfile(bandwidth_gb_s=float("inf")),
        )

        for constructor in invalid_constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()

    def test_packed_weight_contract_validation(self):
        invalid_constructors = (
            lambda: GemmWorkload(
                m=1, k=1, n=1, packed_weight_formats=["q4_0"]
            ),
            lambda: GemmWorkload(
                m=1,
                k=1,
                n=1,
                packed_weight_formats=("q4_0", "Q4_0"),
            ),
            lambda: GemmWorkload(
                m=1, k=1, n=1, packed_weight_formats=("",)
            ),
            lambda: GemmWorkload(
                m=1, k=1, n=1, packed_weight_transform_operations=1
            ),
        )

        for constructor in invalid_constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()


class CimCostModelTests(unittest.TestCase):
    def test_array_tiling_and_bit_slice_formula(self):
        workload = GemmWorkload(
            m=2,
            k=8,
            n=8,
            activation_bits=2,
            weight_bits=2,
            output_bits=8,
            accumulator_bits=32,
        )
        profile = fast_cim()

        estimate = estimate_cim_gemm(
            profile, workload, weights_resident=True
        )

        self.assertEqual(estimate.metadata["n_m"], 2)
        self.assertEqual(estimate.metadata["n_k"], 2)
        self.assertEqual(estimate.metadata["n_n"], 2)
        self.assertEqual(estimate.metadata["q_a"], 2)
        self.assertEqual(estimate.metadata["q_w"], 2)
        self.assertEqual(estimate.metadata["a_eff"], 1)
        # ceil((2 * 2 * 2) / 1) * (2 * 2) * 1 cycle at 1 GHz.
        self.assertEqual(estimate.metadata["array_cycles"], 32)
        self.assertAlmostEqual(
            estimate.phase("cim_array_eval").metadata["array_service_ns"],
            32.0,
        )

    def test_resident_weights_remove_load_phase(self):
        workload = GemmWorkload(m=1, k=32, n=32)
        profile = fast_cim(
            p_k=32,
            p_n=32,
            load_bandwidth_gb_s=1.0,
            load_latency_ns=25.0,
        )

        cold = profile.estimate_gemm(workload, weights_resident=False)
        resident = profile.estimate_gemm(workload, weights_resident=True)

        self.assertIn("weight_load", cold.phase_names)
        self.assertNotIn("weight_load", resident.phase_names)
        self.assertAlmostEqual(
            cold.service_ns - resident.service_ns,
            cold.phase("weight_load").service_ns,
        )
        self.assertGreater(cold.energy_pj, resident.energy_pj - 1.0e-12)

    def test_lower_precision_and_more_arrays_reduce_service(self):
        profile = fast_cim(
            p_k=64,
            p_n=16,
            array_count=1,
            accumulator_outputs_per_cycle=1.0e12,
        )
        low_precision = GemmWorkload(
            m=4,
            k=64,
            n=64,
            activation_bits=4,
            weight_bits=4,
            accumulator_bits=32,
        )
        high_precision = replace(
            low_precision, activation_bits=8, weight_bits=8
        )

        low = profile.estimate_gemm(low_precision, weights_resident=True)
        high = profile.estimate_gemm(high_precision, weights_resident=True)
        wider = replace(profile, array_count=4).estimate_gemm(
            low_precision, weights_resident=True
        )

        self.assertLess(low.service_ns, high.service_ns)
        self.assertLess(wider.service_ns, low.service_ns)
        self.assertLess(
            wider.metadata["array_cycles"], low.metadata["array_cycles"]
        )

    def test_accumulator_width_is_checked(self):
        workload = GemmWorkload(
            m=1,
            k=1024,
            n=1,
            activation_bits=8,
            weight_bits=8,
            accumulator_bits=16,
        )
        profile = fast_cim(
            p_k=1024, p_n=1, accumulator_bits=16
        )

        with self.assertRaisesRegex(ValueError, "accumulator width"):
            profile.estimate_gemm(workload, weights_resident=True)

    def test_padded_weight_capacity_is_checked(self):
        workload = GemmWorkload(
            m=1,
            k=5,
            n=5,
            activation_bits=2,
            weight_bits=2,
        )
        # Padding to 8 x 8 at two bits requires 16 bytes.
        profile = fast_cim(
            p_k=4, p_n=4, weight_capacity_bytes=15
        )

        with self.assertRaisesRegex(ValueError, "exceeds CIM capacity"):
            profile.estimate_gemm(workload, weights_resident=True)

    def test_cim_structural_counts_reject_fractional_values(self):
        profile = fast_cim()

        for field_name in (
            "array_count",
            "p_m",
            "p_k",
            "p_n",
            "input_parallel_bits",
            "weight_parallel_bits",
            "cycles_per_eval",
            "weight_capacity_bytes",
            "max_m_replication",
            "noc_reduce_fan_in",
            "accumulator_bits",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    replace(profile, **{field_name: 1.5})

        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            replace(profile, accumulator_guard_bits=0.5)

    def test_break_even_reuse(self):
        self.assertAlmostEqual(break_even_reuse(100.0, 60.0, 200.0), 5.0)
        self.assertIsNone(break_even_reuse(60.0, 60.0, 1.0))
        self.assertIsNone(break_even_reuse(50.0, 60.0, 1.0))


if __name__ == "__main__":
    unittest.main()
