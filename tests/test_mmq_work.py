from pathlib import Path
import sys
import unittest
from dataclasses import replace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from heterollm_sim.cost_models import (
    GemmWorkload,
    HBMProfile,
    HostMemoryProfile,
    estimate_cpu_gemm,
    estimate_cim_gemm,
    estimate_gpu_gemm,
)
from heterollm_sim.mmq_work import (
    MMVQ_MAX_BATCH_SIZE,
    UnsupportedMMQ,
    derive_mmq_work,
)
from test_cost_models import cpu_profile, gpu_profile
from heterollm_sim.cost_models import DigitalSramCimProfile


def work(*, m=128, k=768, n=128, fmt="Q4_K", sm=84, shared=101_376):
    return derive_mmq_work(
        m=m, k=k, n=n, weight_format=fmt, sm_count=sm,
        shared_memory_per_block=shared,
    )


def gemm(mmq):
    return GemmWorkload(
        mmq.m, mmq.k, mmq.n,
        activation_bits=16, weight_bits=4, output_bits=32,
        packed_weight_formats=(mmq.weight_format,),
        activation_storage_bytes=mmq.consumer_unique_bytes,
        output_storage_bytes=mmq.native_output_bytes,
        mmq_work=mmq,
    )


class MMQWorkTests(unittest.TestCase):
    def test_source_conversion_and_consumer_example(self):
        item = work()
        self.assertEqual(item.j, 128)
        self.assertEqual(item.jmax, 128)
        self.assertEqual(item.conversion_read_bytes, 4 * 128 * 768)
        self.assertEqual(item.conversion_write_bytes, 144 * 128 * 1024 // 128)
        self.assertEqual(item.conversion_operations, 20 * 128 * 1024 // 4)
        self.assertEqual(item.consumer_unique_bytes, 144 * 128 * 768 // 128)
        self.assertEqual(item.source_repeated_bytes, 144 * 128 * 768 // 128)
        self.assertEqual(item.native_output_bytes, 4 * 128 * 128)
        self.assertTrue(item.fixup_launch)
        self.assertEqual(item.partial_writer_count, 2)

    def test_shared_limit_selects_64_or_128_source_template(self):
        low = work(k=256, shared=49_152)
        high = work(k=256, shared=101_376)
        self.assertEqual((low.j, high.j), (64, 128))
        self.assertLessEqual(4 * low.j + 38_912 + 1024 * ((144 * low.j + 1023) // 1024), 49_152)

    def test_m6_m7_and_source_high_water_are_explicit(self):
        six = work(m=6, k=256)
        seven = work(m=7, k=256)
        self.assertEqual((six.jmax, seven.jmax), (0, 0))
        with self.assertRaises(UnsupportedMMQ):
            work(m=6, k=512)
        with self.assertRaises(UnsupportedMMQ):
            work(m=9, k=1024)
        with self.assertRaises(UnsupportedMMQ):
            work(m=MMVQ_MAX_BATCH_SIZE["Q4_K"], k=256)
        with self.assertRaises(UnsupportedMMQ):
            work(m=MMVQ_MAX_BATCH_SIZE["IQ4_XS"], k=256, fmt="IQ4_XS")
        with self.assertRaises(UnsupportedMMQ):
            work(m=128, k=2**35, fmt="Q5_0")
        with self.assertRaises(ValueError):
            work(m=True)
        self.assertEqual(
            set(MMVQ_MAX_BATCH_SIZE),
            {"IQ4_XS", "Q4_K", "Q5_0", "IQ3_S", "Q5_K", "Q8_0", "Q6_K"},
        )

    def test_fixup_p0_launch_and_no_launch_source_boundaries(self):
        p0_launch = work(k=256, n=128)
        p0_no_launch = work(k=256, n=128 * 84)
        partial = work(k=1024, n=128 * 8)
        self.assertEqual((p0_launch.partial_writer_count, p0_launch.fixup_launch), (0, True))
        self.assertEqual((p0_no_launch.u_tiles, p0_no_launch.partial_writer_count, p0_no_launch.fixup_launch), (84, 0, False))
        self.assertEqual((partial.u_tiles, partial.block_count, partial.partial_writer_count), (8, 84, 24))
        self.assertEqual(partial.main_partial_write_bytes, 4 * 24 * 128 * 128)

    def test_gemm_adds_partial_writes_without_changing_final_output(self):
        item = work(k=1024, n=128 * 8)
        mmq_gemm = gemm(item)
        plain = GemmWorkload(
            item.m, item.k, item.n, activation_bits=16, weight_bits=4,
            output_bits=32, packed_weight_formats=(item.weight_format,),
            activation_storage_bytes=item.consumer_unique_bytes,
            output_storage_bytes=item.native_output_bytes,
        )
        base_gpu = gpu_profile()
        gpu = replace(
            base_gpu,
            tensor_core=replace(base_gpu.tensor_core, sm_count=item.sm_count),
        )
        hbm = HBMProfile(bandwidth_gb_s=100)
        estimated = estimate_gpu_gemm(gpu, hbm, mmq_gemm)
        baseline = estimate_gpu_gemm(gpu, hbm, plain)
        self.assertEqual(mmq_gemm.output_bytes, item.native_output_bytes)
        self.assertEqual(estimated.metadata["output_bytes"], item.native_output_bytes)
        self.assertEqual(estimated.metadata["tensor_dtype"], "int8")
        self.assertEqual(estimated.metadata["mmq_source_work"], item.to_metadata())
        demand_delta = sum(
            demand.bytes_moved for demand in estimated.phases[-1].demands
        ) - sum(demand.bytes_moved for demand in baseline.phases[-1].demands)
        self.assertEqual(
            demand_delta,
            item.main_partial_write_bytes * (len(gpu.cache_hierarchy.levels) + 1),
        )
        self.assertEqual(
            len([phase for phase in estimated.phases if phase.name == "kernel_launch"]),
            len([phase for phase in baseline.phases if phase.name == "kernel_launch"]),
        )

    def test_mmq_work_is_rejected_by_cpu_and_cim(self):
        item = work(sm=1)
        workload = gemm(item)
        with self.assertRaisesRegex(ValueError, "GPU-only"):
            estimate_cpu_gemm(cpu_profile(), HostMemoryProfile(bandwidth_gb_s=100), workload)
        cim = DigitalSramCimProfile()
        with self.assertRaisesRegex(ValueError, "GPU-only"):
            estimate_cim_gemm(cim, workload)

    def test_normal_gemm_keeps_no_mmq_metadata(self):
        baseline = estimate_gpu_gemm(
            gpu_profile(), HBMProfile(bandwidth_gb_s=100),
            GemmWorkload(2, 16, 8, activation_bits=16, weight_bits=4),
        )
        self.assertNotIn("mmq_source_work", baseline.metadata)
        self.assertNotIn("mmq_source_work", baseline.phases[-1].metadata)


if __name__ == "__main__":
    unittest.main()
