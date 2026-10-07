"""Source geometry for ordinary contiguous F16-weight CUDA vector products.

Source: llama.cpp d3146f2b56c2db4711ac8391871c9e529d1946d7,
ggml/src/ggml-cuda/mmvf.cu: ggml_cuda_should_use_mmvf,
launch_mul_mat_vec_f_cuda and mul_mat_vec_f.  This supplies work/launch
geometry, never native latency or a measured instruction issue rate.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MMVFWork:
    k: int
    n: int
    block_threads: int
    cta_count: int
    fused_gate: bool
    weight_bytes: int
    activation_bytes: int
    output_bytes: int
    conversion_operations: int
    reduction_operations: int

    def __post_init__(self):
        if (type(self.k) is not int or type(self.n) is not int
                or self.k <= 0 or self.n <= 0 or self.k % 2
                or type(self.fused_gate) is not bool or (self.fused_gate and self.n % 2)):
            raise ValueError("invalid MMVF source dimensions")
        rows = self.n // 2 if self.fused_gate else self.n
        if (self.block_threads != _block_threads(self.k) or self.cta_count != rows
                or self.weight_bytes != 2 * self.k * self.n
                or self.activation_bytes != 4 * self.k or self.output_bytes != 4 * rows
                or self.conversion_operations != self.n * (self.k + 2 * self.block_threads)
                or self.reduction_operations != self.n * (
                    6 * self.block_threads + (160 if self.block_threads > 32 else 0))):
            raise ValueError("MMVF source work differs from CUDA loop/grid geometry")

    @property
    def signature(self):
        return "mmvf:f16:f32:contiguous:m1:gate={}".format(int(self.fused_gate))

    @property
    def shared_bytes(self):
        return 128 * (2 if self.fused_gate else 1)

    def audit(self):
        return {
            "status": "source_conditional", "source_revision": "d3146f2b56c2db4711ac8391871c9e529d1946d7",
            "source": "ggml-cuda/mmvf.cu:ggml_cuda_should_use_mmvf+launch_mul_mat_vec_f_cuda+mul_mat_vec_f",
            "dispatch_signature": self.signature, "cta_count": self.cta_count,
            "block_threads": self.block_threads, "dynamic_shared_bytes": self.shared_bytes,
            "input_storage_dtype": "f32", "weight_storage_dtype": "f16", "output_storage_dtype": "f32",
            "native_default_accumulation": "half2 partial sums, then float warp/block reduction",
            "conversion_operations": self.conversion_operations,
            "reduction_operations": self.reduction_operations,
            "rate_source": "existing declared GPU scalar capacity; not native measurement",
            "registers_measured": False, "native_dispatch_observed": False,
        }


def _block_threads(k):
    threads = 32
    iterations = (k + 63) // 64
    for candidate in range(64, 257, 32):
        count = (k + 2 * candidate - 1) // (2 * candidate)
        if count < iterations:
            threads, iterations = candidate, count
    return threads


def derive_mmvf_work(k, n, *, fused_gate=False):
    if type(k) is not int or type(n) is not int or k <= 0 or n <= 0 or k % 2:
        raise ValueError("MMVF needs positive dimensions and an even contiguous K")
    if type(fused_gate) is not bool or (fused_gate and n % 2):
        raise ValueError("fused MMVF needs two equal physical output widths")
    threads = _block_threads(k)
    rows = n // 2 if fused_gate else n
    # Every partial half2 sum is converted to float and summed; each thread
    # participates in five warp reduction rounds, with one further warp
    # reduction when the block contains multiple warps. Both gate/up sums
    # are included through N, whereas the fused launch has only N/2 CTAs.
    reductions = n * (threads + 5 * threads + (160 if threads > 32 else 0))
    return MMVFWork(k, n, threads, rows, fused_gate, 2 * k * n, 4 * k,
                    4 * rows, k * n + 2 * threads * n, reductions)
