"""Independent exact-Q8 probe for MMVQ or MMQ using native GGML DLLs.

Inputs are exactly representable after the source F32 -> Q8_1 conversion.
Numerics are compared against double-precision dots of the *packed* weights,
not the original weights. Host graph wall is recorded but never used as the
CUDA kernel calibration target; the collector extracts device intervals.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from probe_synthetic_attention import loaded_modules

FORMATS = {'Q4_K': 12, 'Q5_K': 13, 'Q6_K': 14, 'IQ4_XS': 23,
           'Q5_0': 6, 'Q8_0': 8}


def exact_q8_input(m, k, rng):
    """d=2^-7 and integral q: quantization and half(ds) lose no information."""
    values = rng.integers(-127, 128, (m, k), dtype=np.int16)
    blocks = values.reshape(m, -1, 32)
    blocks[..., 0] = 127
    # Keep the half-precision sum exactly representable as well. Pair opposite
    # lanes; each block sum is 127 plus one integer in [-127,127].
    blocks[..., 2:32:2] = -blocks[..., 1:31:2]
    return (values.astype(np.float32) / 128).copy()


def run(args):
    root = Path(__file__).resolve().parents[1]
    folder = root / 'source/llama.cpp-native-thread-control/build-native-thread-control/bin'
    dll_dirs = [os.add_dll_directory(str(folder)), os.add_dll_directory('E:/cuda/bin')]
    base = C.CDLL(str(folder / 'ggml-base.dll'))
    cuda = C.CDLL(str(folder / 'ggml-cuda.dll'))
    ptr, size, i64 = C.c_void_p, C.c_size_t, C.c_int64

    def fn(lib, name, ret, types):
        f = getattr(lib, name); f.restype = ret; f.argtypes = types
        return f

    class Init(C.Structure):
        _fields_ = [('mem_size', size), ('mem_buffer', ptr), ('no_alloc', C.c_bool)]

    class Traits(C.Structure):
        _fields_ = [('type_name', C.c_char_p), ('block', i64), ('interleave', i64),
                    ('type_size', size), ('quantized', C.c_bool),
                    ('to_float', ptr), ('from_float_ref', ptr)]

    typ = FORMATS[args.quant]
    traits = fn(base, 'ggml_get_type_traits', C.POINTER(Traits), [C.c_int])(typ).contents
    if args.k % traits.block or not traits.to_float:
        raise ValueError('K must align to a supported physical quantization block')
    backend = fn(cuda, 'ggml_backend_cuda_init', ptr, [C.c_int])(0)
    ctx = fn(base, 'ggml_init', ptr, [Init])(Init(8 * 1024 * 1024, None, True))
    if not backend or not ctx:
        raise RuntimeError('CUDA backend/context initialization failed')
    buffer = None
    try:
        tensor = fn(base, 'ggml_new_tensor_2d', ptr, [ptr, C.c_int, i64, i64])
        weight = tensor(ctx, typ, args.k, args.n)
        activation = tensor(ctx, 0, args.k, args.m)
        result = fn(base, 'ggml_mul_mat', ptr, [ptr, ptr, ptr])(ctx, weight, activation)
        graph = fn(base, 'ggml_new_graph_custom', ptr, [ptr, size, C.c_bool])(ctx, 32, False)
        fn(base, 'ggml_build_forward_expand', None, [ptr, ptr])(graph, result)
        buffer = fn(base, 'ggml_backend_alloc_ctx_tensors', ptr, [ptr, ptr])(ctx, backend)
        if not buffer or not fn(base, 'ggml_backend_supports_op', C.c_bool, [ptr, ptr])(backend, result):
            raise RuntimeError('CUDA allocation/operation support gate failed')
        rng = np.random.default_rng(args.seed)
        weights = rng.uniform(-1, 1, (args.n, args.k)).astype(np.float32)
        data = exact_q8_input(args.m, args.k, rng)
        row_bytes = args.k // traits.block * traits.type_size
        packed = np.empty((args.n, row_bytes), np.uint8)
        importance = np.ones(args.k, np.float32)
        quantize = fn(base, 'ggml_quantize_chunk', size, [C.c_int, ptr, ptr, i64, i64, i64, ptr])
        count = quantize(typ, weights.ctypes.data, packed.ctypes.data, 0, args.n, args.k, importance.ctypes.data)
        if count != packed.nbytes:
            raise RuntimeError('packed weight byte count mismatch')
        del weights
        put = fn(base, 'ggml_backend_tensor_set', None, [ptr, ptr, size, size])
        put(weight, packed.ctypes.data, 0, packed.nbytes)
        put(activation, data.ctypes.data, 0, data.nbytes)
        get = fn(base, 'ggml_backend_tensor_get', None, [ptr, ptr, size, size])
        compute = fn(base, 'ggml_backend_graph_compute', C.c_int, [ptr, ptr])
        decode = C.CFUNCTYPE(None, ptr, ptr, i64)(traits.to_float)
        indices = np.unique(np.linspace(0, args.m * args.n - 1, 32, dtype=np.int64))
        reference = []
        decoded = np.empty(args.k, np.float32)
        for index in indices:
            col = int(index % args.n); row = int(index // args.n)
            decode(packed[col].ctypes.data, decoded.ctypes.data, args.k)
            reference.append(np.dot(decoded.astype(np.float64), data[row].astype(np.float64)))
        reference = np.asarray(reference, np.float64)

        def check():
            actual = np.empty((args.m, args.n), np.float32)
            get(result, actual.ctypes.data, 0, actual.nbytes)
            error = np.abs(actual.ravel()[indices] - reference)
            return {'passed': bool(np.isfinite(actual).all() and np.all(error <= args.atol + args.rtol * np.abs(reference))),
                    'max_absolute_error': float(error.max()), 'atol': args.atol, 'rtol': args.rtol,
                    'reference': 'packed_weight_double_dot_exactly_representable_Q8_1_input'}

        before = loaded_modules()
        if compute(backend, graph):
            raise RuntimeError('first compute failed')
        first = check()
        if not first['passed']:
            raise RuntimeError('first-call numerical gate failed: ' + json.dumps(first))
        runs = []
        for index in range(args.warmup + args.repeats):
            start = time.perf_counter_ns(); status = compute(backend, graph); end = time.perf_counter_ns()
            if status:
                raise RuntimeError('graph compute failed: ' + str(status))
            runs.append({'phase': 'warmup' if index < args.warmup else 'formal', 'elapsed_ns': end - start})
        final = check(); after = loaded_modules()
        if before != after or not final['passed']:
            raise RuntimeError('final numerical/module gate failed')
        return {'schema': 'synthetic-exact-q8-mmq/v1', 'status': 'measured',
                'M': args.m, 'N': args.n, 'K': args.k, 'weight_format': args.quant,
                'first_call_correctness': first, 'final_correctness': final,
                'modules_stable': True, 'loaded_modules_after': after,
                'graph_compute_calls': 1 + args.warmup + args.repeats,
                'input_dtype': 'F32_exact_Q8_1_representable', 'output_dtype': 'F32',
                'layout': 'ordinary_contiguous_2d', 'seed': args.seed,
                'warmup': args.warmup, 'repeats': args.repeats, 'runs': runs,
                'input_sha256': hashlib.sha256(data.tobytes()).hexdigest(),
                'packed_weight_sha256': hashlib.sha256(packed.tobytes()).hexdigest(),
                'timing_scope': 'host_graph_wall_not_kernel', 'target_llm_timing_used': False}
    finally:
        if buffer:
            fn(base, 'ggml_backend_buffer_free', None, [ptr])(buffer)
        fn(base, 'ggml_free', None, [ptr])(ctx)
        fn(base, 'ggml_backend_free', None, [ptr])(backend)
        del dll_dirs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--quant', choices=FORMATS, required=True)
    for field in ('m', 'n', 'k'):
        p.add_argument('--' + field, type=int, required=True)
    p.add_argument('--seed', type=int, default=20260930)
    p.add_argument('--warmup', type=int, default=32)
    p.add_argument('--repeats', type=int, default=40)
    p.add_argument('--atol', type=float, default=.001)
    p.add_argument('--rtol', type=float, default=.001)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.m <= 1024 or min(args.n, args.k) <= 0 or args.k % 32 or args.warmup < 0 or args.repeats < 2:
        p.error('invalid probe geometry/repetitions')
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['final_correctness']), flush=True)


if __name__ == '__main__':
    main()
