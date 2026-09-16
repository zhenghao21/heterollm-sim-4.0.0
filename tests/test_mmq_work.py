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

    def test_qk32_tail_uses_true_source_stream_k_partition(self):
        for fmt in ("Q5_0", "Q8_0"):
            with self.subTest(fmt=fmt):
                tail = work(k=896, n=128 * 14, fmt=fmt)
                aligned = work(k=1024, n=128 * 14, fmt=fmt)
                self.assertEqual((tail.k, tail.k_padded, tail.k_execution), (896, 1024, 1024))
                self.assertEqual((tail.u_tiles, tail.block_count), (14, 84))
                self.assertEqual((tail.source_nonempty_block_count, tail.partial_writer_count), (42, 28))
                self.assertEqual((aligned.source_nonempty_block_count, aligned.partial_writer_count), (56, 42))
                self.assertEqual(tail.fixup_tile_indices, tuple(range(14)))
                self.assertEqual(tail.fixup_valid_elements, 14 * 128 * 128)
                self.assertEqual(tail.conversion_read_bytes, 4 * 128 * 896)
                self.assertEqual(tail.conversion_write_bytes, 144 * 128 * 1024 // 128)
                self.assertEqual(tail.consumer_unique_bytes, 144 * 128 * 1024 // 128)
                self.assertEqual(tail.consumer_extra_bytes, 144 * 128)
                self.assertEqual(tail.source_repeated_bytes, aligned.source_repeated_bytes)
                self.assertEqual(tail.logical_arithmetic_operations, 2 * 128 * 1792 * 896)
                self.assertEqual(tail.execution_arithmetic_operations, 2 * 128 * 1792 * 1024)
                self.assertEqual(tail.source_nominal_arithmetic_operations, aligned.source_nominal_arithmetic_operations)
                self.assertEqual(tail.stream_k_boundaries_qblocks[-1], 14 * 28)
                self.assertNotEqual(tail.stream_k_boundaries_qblocks, aligned.stream_k_boundaries_qblocks)

    def test_weight_tail_is_final_tensor_range_not_per_row_allocation(self):
        for fmt, block_bytes in (("Q5_0", 22), ("Q8_0", 34)):
            with self.subTest(fmt=fmt):
                tail = work(k=896, n=1792, fmt=fmt)
                logical = 1792 * 28 * block_bytes
                self.assertEqual(tail.logical_weight_bytes, logical)
                self.assertEqual(tail.weight_tail_read_bytes_per_row, 4 * block_bytes)
                self.assertEqual(tail.weight_read_high_water_bytes, logical + 4 * block_bytes)
                self.assertEqual(tail.weight_tail_range_bytes, (logical, logical + 4 * block_bytes))
                self.assertLess(tail.weight_read_high_water_bytes, 1792 * 32 * block_bytes)
                self.assertFalse(tail.to_metadata()["weight_tail_allocation_proven"])
                aligned = work(k=1024, n=1792, fmt=fmt)
                self.assertEqual(aligned.weight_tail_read_bytes_per_row, 0)
                self.assertEqual(aligned.weight_tail_range_bytes, (aligned.logical_weight_bytes,) * 2)

    def test_qk32_sub_block_tails_and_format_alignment(self):
        for fmt in MMVQ_MAX_BATCH_SIZE:
            with self.subTest(fmt=fmt):
                if fmt not in ("Q5_0", "Q8_0"):
                    with self.assertRaisesRegex(UnsupportedMMQ, "multiple of 256"):
                        work(k=896, fmt=fmt)
                    continue
                with self.assertRaisesRegex(UnsupportedMMQ, "multiple of 32"):
                    work(k=897, fmt=fmt)
                for k in (32, 64, 96, 160, 224, 288, 864, 928, 992):
                    tail = work(k=k, fmt=fmt)
                    self.assertEqual(tail.k_iterations, (k + 255) // 256)
                    self.assertEqual(tail.k_execution, 256 * ((k + 255) // 256))
                    self.assertEqual(tail.k_padded, 512 * ((k + 511) // 512))
                    self.assertEqual(tail.conversion_effective_bytes, 144 * 128 * ((k + 127) // 128))
                    self.assertEqual(tail.consumer_unique_bytes, 144 * 128 * tail.k_execution // 128)
                    self.assertLessEqual(tail.consumer_unique_bytes, tail.source_allocation_bytes)
                with self.assertRaisesRegex(UnsupportedMMQ, "high-water"):
                    work(m=9, k=896, fmt=fmt)

    def test_source_partition_loop_covers_every_tile_and_tail_once(self):
        # A direct transcription of process_tile span iteration, independent
        # of the aggregate traffic/partial writer formulas in derive_mmq_work.
        for k in (32, 224, 256, 288, 768, 800, 896, 992, 1024):
            for n in (128, 130, 128 * 14, 128 * 84):
                with self.subTest(k=k, n=n):
                    item = work(m=129, k=k, n=n, fmt="Q8_0")
                    B, R = k // 32, 8
                    starts = item.stream_k_boundaries_qblocks
                    loop_iterations = 0
                    partial_tiles = []
                    for start, end in zip(starts, starts[1:]):
                        cursor = start
                        while cursor < end:
                            tile, local_start = divmod(cursor, B)
                            local_stop = min(B, local_start + end - cursor)
                            loop_iterations += len(range(local_start, local_stop, R))
                            if local_stop < B:
                                partial_tiles.append(tile)
                            cursor += local_stop - local_start
                    self.assertEqual(loop_iterations, item.u_tiles * ((k + 255) // 256))
                    self.assertEqual(len(partial_tiles), item.partial_writer_count)
                    self.assertEqual(tuple(sorted(set(partial_tiles))), item.fixup_tile_indices)
                    self.assertEqual(item.source_nominal_arithmetic_operations, 2 * loop_iterations * item.i * item.j * 256)
                    self.assertTrue(all(start <= end for start, end in zip(starts, starts[1:])))
                    self.assertEqual((starts[0], starts[-1]), (0, item.u_tiles * B))

    def test_all_seven_aligned_formats_preserve_legacy_numeric_accounting(self):
        # Golden quantities captured before extending K coverage. They include
        # output tails, shared-memory selection, stream-K and fixup behavior.
        fields = (
            "i", "j", "jmax", "k_padded", "x_tiles", "y_tiles", "u_tiles", "k_iterations",
            "consumer_load_window_bytes", "consumer_extra_bytes", "source_repeated_bytes",
            "source_allocation_bytes", "block_count", "partial_writer_count", "fixup_valid_elements",
            "main_partial_write_bytes", "fixup_read_bytes", "fixup_write_bytes", "fixup_operations",
        )
        fixtures = (
            ((16, 256, 128, 101376), (128, 16, 16, 512, 1, 1, 1, 1, 3072, 768, 6144, 11520, 84, 0, 0, 0, 0, 0, 0)),
            ((128, 768, 128, 101376), (128, 128, 128, 1024, 1, 1, 1, 3, 18432, 0, 110592, 165888, 84, 2, 16384, 131072, 196608, 65536, 49152)),
            ((129, 768, 130, 49152), (128, 64, 128, 1024, 3, 2, 6, 3, 9216, 9072, 331776, 167040, 84, 12, 16770, 393216, 460296, 67080, 115074)),
            ((128, 1024, 1792, 101376), (128, 128, 128, 1024, 1, 14, 14, 4, 18432, 0, 2064384, 165888, 84, 42, 229376, 2752512, 3670016, 917504, 917504)),
        )
        for fmt in MMVQ_MAX_BATCH_SIZE:
            for (m, k, n, shared), expected in fixtures:
                with self.subTest(fmt=fmt, m=m, k=k, n=n, shared=shared):
                    item = work(m=m, k=k, n=n, shared=shared, fmt=fmt)
                    metadata = item.to_metadata()
                    self.assertEqual(tuple(metadata[field] for field in fields), expected)
                    self.assertEqual(item.conversion_effective_bytes, 144 * m * k // 128)
                    self.assertEqual(item.conversion_read_bytes, 4 * m * k)
                    self.assertEqual(item.conversion_write_bytes, 144 * m * item.k_padded // 128)
                    self.assertEqual(item.conversion_operations, (20 if fmt in ("Q4_K", "Q5_K") else 14) * m * item.k_padded // 4)
                    self.assertEqual(item.k_execution, k)
                    self.assertEqual(item.execution_arithmetic_operations, item.logical_arithmetic_operations)
                    self.assertEqual(item.weight_tail_read_bytes_per_row, 0)

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
