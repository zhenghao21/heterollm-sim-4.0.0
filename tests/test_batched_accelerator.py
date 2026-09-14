import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from heterollm_sim import batched_accelerator
from heterollm_sim.batched_accelerator import (
    BatchedGemmInput,
    evaluate_batched_gemm,
)
from heterollm_sim.cost_models import GemmWorkload, HBMProfile, estimate_gpu_gemm
from heterollm_sim.reference import build_reference_scenario


def _cupy_is_usable():
    cupy, _ = batched_accelerator._probe_cupy()
    return cupy is not None


class BatchedAcceleratorTests(unittest.TestCase):
    def test_numpy_single_candidate_matches_scalar_gpu_model(self):
        workload = GemmWorkload(
            m=16,
            k=16,
            n=16,
            activation_bits=8,
            weight_bits=8,
            output_bits=16,
        )
        base_gpu = build_reference_scenario().gpu_profile
        tensor_core = replace(
            base_gpu.tensor_core,
            cycles_per_mma=(
                base_gpu.tensor_core.cycles_per_mma
                * base_gpu.peak_tops
                / 123.0
            ),
        )
        cache_level = replace(
            base_gpu.cache_hierarchy.levels[0],
            capacity_bytes=1,
            line_bytes=1,
            hit_latency_ns=0.0,
            bandwidth_gb_s=1.0e12,
        )
        gpu = replace(
            base_gpu,
            tensor_core=tensor_core,
            cache_hierarchy=replace(
                base_gpu.cache_hierarchy, levels=(cache_level,)
            ),
            attainable_efficiency=0.72,
            occupancy=1.0,
            kernel_launch_ns=37.0,
        )
        hbm = HBMProfile(bandwidth_gb_s=777.0, efficiency=0.81)
        scalar = estimate_gpu_gemm(gpu, hbm, workload)

        result = evaluate_batched_gemm(
            BatchedGemmInput(
                m=workload.m,
                k=workload.k,
                n=workload.n,
                activation_bits=workload.activation_bits,
                weight_bits=workload.weight_bits,
                output_bits=workload.output_bits,
                weight_metadata_bytes=workload.weight_metadata_bytes,
                peak_tops=gpu.peak_tops,
                compute_efficiency=gpu.attainable_efficiency,
                memory_bandwidth_gb_s=hbm.bandwidth_gb_s,
                memory_efficiency=hbm.efficiency,
                kernel_launch_ns=gpu.kernel_launch_ns,
                sm_count=gpu.tensor_core.sm_count,
                tensor_cores_per_sm=gpu.tensor_core.tensor_cores_per_sm,
                mma_m=gpu.tensor_core.mma_m,
                mma_n=gpu.tensor_core.mma_n,
                mma_k=gpu.tensor_core.mma_k,
                occupancy=gpu.occupancy,
            ),
            backend="numpy",
        )

        self.assertEqual(result.backend_used, "numpy")
        self.assertEqual(result.total_ns.shape, (1,))
        self.assertEqual(result.operations[0], workload.operations)
        self.assertEqual(result.gemm_io_bytes[0], workload.minimum_io_bytes)
        self.assertAlmostEqual(
            result.compute_ns[0], scalar.metadata["compute_service_ns"]
        )
        self.assertAlmostEqual(
            result.memory_ns[0], scalar.metadata["memory_service_ns"]
        )
        self.assertAlmostEqual(
            result.roofline_ns[0], scalar.metadata["roofline_service_ns"]
        )
        self.assertAlmostEqual(result.total_ns[0], scalar.service_ns)
        self.assertEqual(result.bound[0], scalar.metadata["bound"])
        scalar_bandwidth = scalar.metadata["hbm_bandwidth"]
        self.assertEqual(
            result.hbm_bandwidth_model, scalar_bandwidth["model"]
        )
        self.assertFalse(result.fallback_to_fixed_bandwidth[0])
        self.assertEqual(
            result.independent_output_tile_count[0],
            scalar_bandwidth["independent_output_tile_count"],
        )
        self.assertAlmostEqual(
            result.output_tile_wave_utilization[0],
            scalar_bandwidth["output_tile_wave_utilization"],
        )
        self.assertAlmostEqual(
            result.shape_effective_hbm_bandwidth_gb_s[0],
            scalar_bandwidth["shape_effective_hbm_bandwidth_gb_s"],
        )

    def test_vectorized_output_tile_wave_proxy_and_fixed_fallback(self):
        result = evaluate_batched_gemm(
            BatchedGemmInput(
                m=[1, 128, 1],
                k=5120,
                n=10240,
                peak_tops=100.0,
                memory_bandwidth_gb_s=800.0,
                memory_efficiency=0.9,
                sm_count=[84, 84, 0],
                tensor_cores_per_sm=4,
                mma_m=16,
                mma_n=16,
                mma_k=16,
                occupancy=0.85,
            ),
            backend="numpy",
        )

        self.assertEqual(
            result.hbm_bandwidth_model, "mma_output_tile_wave_proxy_v1"
        )
        np.testing.assert_array_equal(
            result.hbm_bandwidth_proxy_enabled, [True, True, False]
        )
        np.testing.assert_array_equal(
            result.fallback_to_fixed_bandwidth, [False, False, True]
        )
        np.testing.assert_allclose(result.m_tile_count, [1, 8, 0])
        np.testing.assert_allclose(result.n_tile_count, [640, 640, 0])
        np.testing.assert_allclose(result.serial_k_tile_count, [320, 320, 0])
        np.testing.assert_allclose(
            result.independent_output_tile_count, [640, 5120, 0]
        )
        np.testing.assert_allclose(result.warp_equivalent_count, [336, 336, 0])
        np.testing.assert_allclose(
            result.resident_warp_equivalent_capacity, [285.6, 285.6, 0]
        )
        np.testing.assert_allclose(result.parallel_tile_slots, [285, 285, 0])
        np.testing.assert_allclose(result.output_tile_wave_count, [3, 18, 0])
        np.testing.assert_allclose(
            result.output_tile_wave_utilization,
            [640.0 / 855.0, 5120.0 / 5130.0, 1.0],
        )
        np.testing.assert_allclose(
            result.peak_effective_hbm_bandwidth_gb_s, [720.0, 720.0, 720.0]
        )
        np.testing.assert_allclose(
            result.shape_effective_hbm_bandwidth_gb_s,
            [720.0 * 640.0 / 855.0, 720.0 * 5120.0 / 5130.0, 720.0],
        )
        np.testing.assert_allclose(
            result.memory_ns,
            result.gemm_io_bytes
            / result.shape_effective_hbm_bandwidth_gb_s,
        )

    def test_vectorized_components_and_stable_ranking(self):
        result = evaluate_batched_gemm(
            BatchedGemmInput(
                m=[1, 1, 1],
                k=64,
                n=64,
                peak_tops=[100.0, 100.0, 200.0],
                memory_bandwidth_gb_s=[1.0, 1.0, 2.0],
                activation_bits=8,
                weight_bits=8,
                output_bits=8,
                kernel_launch_ns=5.0,
                io_bytes=[100, 100, 0],
                io_bandwidth_gb_s=[10.0, 10.0, 0.0],
                io_latency_ns=[2.0, 2.0, 999.0],
                communication_bytes=[50, 50, 0],
                communication_bandwidth_gbps=[100.0, 100.0, 0.0],
                communication_latency_ns=[3.0, 3.0, 999.0],
                candidate_ids=("first", "second", "fast"),
            ),
            backend="numpy",
        )

        self.assertEqual(result.candidate_count, 3)
        np.testing.assert_allclose(result.io_ns, [12.0, 12.0, 0.0])
        np.testing.assert_allclose(result.communication_ns, [7.0, 7.0, 0.0])
        np.testing.assert_array_equal(result.sort_order, [2, 0, 1])
        np.testing.assert_array_equal(result.rank, [1, 2, 0])
        self.assertEqual(result.ranked_candidate_ids, ("fast", "first", "second"))
        self.assertFalse(result.total_ns.flags.writeable)

    def test_ten_thousand_candidates_keep_columnar_shape(self):
        count = 10_000
        result = evaluate_batched_gemm(
            BatchedGemmInput(
                m=np.arange(1, count + 1, dtype=np.int64),
                k=4096,
                n=4096,
                peak_tops=np.linspace(100.0, 500.0, count),
                memory_bandwidth_gb_s=1500.0,
            ),
            backend="numpy",
        )

        self.assertEqual(result.total_ns.shape, (count,))
        self.assertEqual(result.sort_order.shape, (count,))
        self.assertTrue(np.all(np.isfinite(result.total_ns)))

    def test_invalid_shapes_and_values_have_clear_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "形状无法广播"):
            evaluate_batched_gemm(
                BatchedGemmInput(
                    m=[1, 2],
                    k=64,
                    n=64,
                    peak_tops=[100.0, 200.0, 300.0],
                    memory_bandwidth_gb_s=1000.0,
                ),
                backend="numpy",
            )
        with self.assertRaisesRegex(ValueError, "m 必须全部大于 0"):
            evaluate_batched_gemm(
                BatchedGemmInput(
                    m=[1, 0],
                    k=64,
                    n=64,
                    peak_tops=100.0,
                    memory_bandwidth_gb_s=1000.0,
                ),
                backend="numpy",
            )
        with self.assertRaisesRegex(ValueError, "weight_bits 必须全部是整数"):
            evaluate_batched_gemm(
                BatchedGemmInput(
                    m=1,
                    k=64,
                    n=64,
                    peak_tops=100.0,
                    memory_bandwidth_gb_s=1000.0,
                    weight_bits=3.5,
                ),
                backend="numpy",
            )
        with self.assertRaisesRegex(ValueError, "io_bandwidth_gb_s 必须大于 0"):
            evaluate_batched_gemm(
                BatchedGemmInput(
                    m=1,
                    k=64,
                    n=64,
                    peak_tops=100.0,
                    memory_bandwidth_gb_s=1000.0,
                    io_bytes=1,
                ),
                backend="numpy",
            )
        with self.assertRaisesRegex(ValueError, "backend"):
            evaluate_batched_gemm(
                BatchedGemmInput(
                    m=1,
                    k=64,
                    n=64,
                    peak_tops=100.0,
                    memory_bandwidth_gb_s=1000.0,
                ),
                backend="tpu",
            )

    def test_auto_and_explicit_cupy_fall_back_when_unavailable(self):
        candidates = BatchedGemmInput(
            m=[1, 2],
            k=64,
            n=64,
            peak_tops=100.0,
            memory_bandwidth_gb_s=1000.0,
        )
        with mock.patch.object(
            batched_accelerator,
            "_probe_cupy",
            return_value=(None, "测试环境没有 CUDA"),
        ):
            automatic = evaluate_batched_gemm(candidates, backend="auto")
            explicit = evaluate_batched_gemm(candidates, backend="cupy")

        self.assertEqual(automatic.backend_requested, "auto")
        self.assertEqual(automatic.backend_used, "numpy")
        self.assertIn("回退到 NumPy CPU", automatic.diagnostics[0])
        self.assertIn("测试环境没有 CUDA", automatic.diagnostics[0])
        self.assertEqual(explicit.backend_requested, "cupy")
        self.assertEqual(explicit.backend_used, "numpy")
        self.assertIn("请求的 CuPy/CUDA 后端不可用", explicit.diagnostics[0])

    def test_real_missing_cupy_reason_does_not_expose_english_import_error(self):
        with mock.patch.object(
            batched_accelerator.importlib,
            "import_module",
            side_effect=ModuleNotFoundError("No module named 'cupy'"),
        ):
            cupy, reason = batched_accelerator._probe_cupy()

        self.assertIsNone(cupy)
        self.assertEqual(reason, "CuPy 未安装")
        self.assertNotIn("No module", reason)

    @unittest.skipUnless(_cupy_is_usable(), "CuPy/CUDA 不可用")
    def test_optional_cupy_matches_numpy(self):
        candidates = BatchedGemmInput(
            m=np.array([1, 8, 32, 128]),
            k=np.array([64, 256, 1024, 4096]),
            n=4096,
            peak_tops=np.array([80.0, 120.0, 160.0, 200.0]),
            compute_efficiency=0.7,
            memory_bandwidth_gb_s=np.array([800.0, 1000.0, 1200.0, 1400.0]),
            memory_efficiency=0.8,
            io_bytes=np.array([0, 1024, 2048, 4096]),
            io_bandwidth_gb_s=32.0,
            io_latency_ns=100.0,
            communication_bytes=np.array([16, 32, 64, 128]),
            communication_bandwidth_gbps=400.0,
            communication_latency_ns=25.0,
        )

        numpy_result = evaluate_batched_gemm(candidates, backend="numpy")
        cupy_result = evaluate_batched_gemm(candidates, backend="cupy")

        self.assertEqual(cupy_result.backend_used, "cupy")
        for name in (
            "operations",
            "gemm_io_bytes",
            "m_tile_count",
            "n_tile_count",
            "serial_k_tile_count",
            "independent_output_tile_count",
            "parallel_tile_slots",
            "output_tile_wave_count",
            "output_tile_wave_utilization",
            "shape_effective_hbm_bandwidth_gb_s",
            "compute_ns",
            "memory_ns",
            "roofline_ns",
            "io_ns",
            "communication_ns",
            "total_ns",
            "compute_utilization",
        ):
            np.testing.assert_allclose(
                getattr(cupy_result, name), getattr(numpy_result, name), rtol=1e-12
            )
        np.testing.assert_array_equal(
            cupy_result.fallback_to_fixed_bandwidth,
            numpy_result.fallback_to_fixed_bandwidth,
        )
        np.testing.assert_array_equal(
            cupy_result.sort_order, numpy_result.sort_order
        )


if __name__ == "__main__":
    unittest.main()
