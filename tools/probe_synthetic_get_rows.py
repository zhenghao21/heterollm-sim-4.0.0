"""Synthetic CPU GET_ROWS numerical/timing probe; no GGUF or LLM timings.

The measured boundary is the synchronous backend graph call (one GET_ROWS),
including its planning/dispatch overhead, not isolated conversion throughput.
"""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time
import numpy as np
from probe_synthetic_attention import loaded_modules

FORMATS = {'F32': (0, 1, 4), 'F16': (1, 1, 2), 'Q4_K': (12, 256, 144), 'Q6_K': (14, 256, 210)}


def synthetic_rows(fmt, width, table_rows, rng):
    _, block, size = FORMATS[fmt]
    if width % block:
        raise ValueError('width must align to physical quantization block')
    if fmt in ('F16', 'F32'):
        data = (rng.standard_normal((table_rows, width)) * .1).astype('<f2' if fmt == 'F16' else '<f4')
        return data.view(np.uint8).reshape(table_rows, -1)
    data = rng.integers(0, 256, (table_rows, width // block, size), dtype=np.uint8)
    if fmt == 'Q4_K':
        data[..., :2] = np.frombuffer(np.float16(.03125).tobytes(), np.uint8)
        data[..., 2:4] = np.frombuffer(np.float16(.015625).tobytes(), np.uint8)
    else:
        data[..., 192:208] = rng.integers(-16, 16, data[..., 192:208].shape, dtype=np.int8).view(np.uint8)
        data[..., 208:] = np.frombuffer(np.float16(.015625).tobytes(), np.uint8)
    return data.reshape(table_rows, -1)


def decode_rows(fmt, packed, width):
    """Independent NumPy interpretation of packed blocks, not DLL to_float."""
    if fmt in ('F16', 'F32'):
        return packed.copy().view('<f2' if fmt == 'F16' else '<f4').reshape(-1, width).astype(np.float32)
    size = FORMATS[fmt][2]
    blocks = packed.reshape(-1, size)
    output = np.empty((len(blocks), 256), dtype=np.float32)
    if fmt == 'Q4_K':
        d = blocks[:, :2].copy().view('<f2').ravel().astype(np.float32)
        dm = blocks[:, 2:4].copy().view('<f2').ravel().astype(np.float32)
        scales = blocks[:, 4:16].astype(np.int32)
        for group in range(8):
            if group < 4:
                scale, offset = scales[:, group] & 63, scales[:, group+4] & 63
            else:
                scale = (scales[:, group+4] & 15) | ((scales[:, group-4] >> 6) << 4)
                offset = (scales[:, group+4] >> 4) | ((scales[:, group] >> 6) << 4)
            values = blocks[:, 16 + (group//2)*32:48 + (group//2)*32].astype(np.int32)
            values = values & 15 if group % 2 == 0 else values >> 4
            output[:, group*32:(group+1)*32] = (d*scale)[:, None]*values - (dm*offset)[:, None]
    else:
        d = blocks[:, 208:].copy().view('<f2').ravel().astype(np.float32)
        scales = blocks[:, 192:208].copy().view(np.int8)
        for half in range(2):
            ql = blocks[:, half*64:half*64+64].astype(np.int32)
            qh = blocks[:, 128+half*32:160+half*32].astype(np.int32)
            for group in range(4):
                low = ql[:, (group%2)*32:(group%2+1)*32]
                low = low & 15 if group < 2 else low >> 4
                value = (low | (((qh >> (2*group)) & 3) << 4)) - 32
                scale = scales[:, half*8+group*2:half*8+group*2+2].repeat(16,axis=1)
                output[:, half*128+group*32:half*128+(group+1)*32] = d[:, None]*scale*value
    return output.reshape(-1, width)


def run_case(base, cpu, fmt, width, rows, table_rows, warmup, repeats):
    ptr, size, i64 = C.c_void_p, C.c_size_t, C.c_int64
    def fn(lib, name, ret, args):
        function = getattr(lib, name); function.restype = ret; function.argtypes = args
        return function
    class Init(C.Structure):
        _fields_ = [('mem_size', size), ('mem_buffer', ptr), ('no_alloc', C.c_bool)]
    backend = fn(cpu, 'ggml_backend_cpu_init', ptr, [])()
    ctx = fn(base, 'ggml_init', ptr, [Init])(Init(8*1024*1024, None, True))
    if not backend or not ctx:
        raise RuntimeError('CPU backend/context allocation failed')
    buffer = None
    try:
        fn(cpu, 'ggml_backend_cpu_set_n_threads', None, [ptr, C.c_int])(backend, 16)
        tensor = fn(base, 'ggml_new_tensor_2d', ptr, [ptr, C.c_int, i64, i64])
        weight = tensor(ctx, FORMATS[fmt][0], width, table_rows)
        indices = tensor(ctx, 26, rows, 1)  # GGML_TYPE_I32 from locked ggml.h.
        result = fn(base, 'ggml_get_rows', ptr, [ptr, ptr, ptr])(ctx, weight, indices)
        graph = fn(base, 'ggml_new_graph_custom', ptr, [ptr, size, C.c_bool])(ctx, 32, False)
        fn(base, 'ggml_build_forward_expand', None, [ptr, ptr])(graph, result)
        buffer = fn(base, 'ggml_backend_alloc_ctx_tensors', ptr, [ptr, ptr])(ctx, backend)
        if not buffer:
            raise RuntimeError('tensor allocation failed')
        rng = np.random.default_rng(20260929)
        data = synthetic_rows(fmt, width, table_rows, rng)
        ids = rng.integers(0, table_rows, rows, dtype=np.int32)
        # Both out-of-order and repeated lookups exercise indexing/aliasing.
        if rows > 1:
            ids[-1] = ids[0]
        put = fn(base, 'ggml_backend_tensor_set', None, [ptr, ptr, size, size])
        put(weight, data.ctypes.data, 0, data.nbytes)
        put(indices, ids.ctypes.data, 0, ids.nbytes)
        compute = fn(base, 'ggml_backend_graph_compute', C.c_int, [ptr, ptr])
        actual = np.empty((rows, width), dtype=np.float32)
        get = fn(base, 'ggml_backend_tensor_get', None, [ptr, ptr, size, size])
        expected = decode_rows(fmt, data[ids], width)
        def validate():
            get(result, actual.ctypes.data, 0, actual.nbytes)
            if not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
                raise RuntimeError('independent GET_ROWS numerical validation failed')
            return float(np.max(np.abs(actual-expected)))
        elapsed = []
        for index in range(warmup + repeats):
            start = time.perf_counter_ns(); status = compute(backend, graph); stop = time.perf_counter_ns()
            if status:
                raise RuntimeError(f'graph compute failed: {status}')
            if index == 0:
                validate()
            if index >= warmup:
                elapsed.append(stop-start)
        error = validate()
        return {'format': fmt, 'width': width, 'rows': rows, 'table_rows': table_rows,
                'selected_row_bytes': rows*data.shape[1], 'index_bytes': ids.nbytes,
                'output_bytes': actual.nbytes, 'table_capacity_bytes': data.nbytes,
                'correctness': {'passed': True, 'max_abs': error, 'atol': 1e-6, 'rtol': 1e-6},
                'durations_ns': elapsed, 'median_ns': statistics.median(elapsed),
                'stddev_ns': statistics.stdev(elapsed), 'requested_threads': 16,
                'cpu_get_rows_tasks': 1, 'cache_protocol': 'hot_same_rows',
                'measurement_boundary': 'synchronous_cpu_backend_graph_wall',
                'kernel_only_timing': False}
    finally:
        if buffer:
            fn(base,'ggml_backend_buffer_free',None,[ptr])(buffer)
        fn(base,'ggml_free',None,[ptr])(ctx)
        fn(base,'ggml_backend_free',None,[ptr])(backend)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('refusing to overwrite measurement')
    root = Path(__file__).resolve().parents[1]
    folder = root/'source/llama.cpp-semantic/build-semantic-hosttrace/bin'
    dll_dir = os.add_dll_directory(str(folder))
    base = C.CDLL(str(folder/'ggml-base.dll')); cpu = C.CDLL(str(folder/'ggml-cpu.dll'))
    required = ('ggml-base.dll','ggml-cpu.dll')
    before = loaded_modules(required)
    result = {'schema': 'synthetic-cpu-get-rows/v1', 'source_kind': 'independent_synthetic_operator',
              'target_llm_latency_used': False, 'production_qualified': False,
              'cpu': {'platform': platform.platform(), 'processor': platform.processor(), 'logical_processors': os.cpu_count()},
              'source_sha256': {}, 'cases': []}
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        result['cpu']['name'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    for relative in ('ggml/include/ggml.h', 'ggml/src/ggml-quants.c', 'ggml/src/ggml-cpu/ggml-cpu.c', 'ggml/src/ggml-cpu/ops.cpp'):
        file = root/'source/llama.cpp-semantic'/relative
        result['source_sha256'][relative] = hashlib.sha256(file.read_bytes()).hexdigest()
    result['probe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for fmt in ('Q4_K','Q6_K','F16'):
        for rows, width, role in [(1,1024,'train'),(1,4096,'train'),(64,1024,'train'),(64,4096,'train'),(8,2048,'holdout')]:
            case = run_case(base, cpu, fmt, width, rows, 4096, 8, 40)
            case['role'] = role; result['cases'].append(case)
            print(fmt, rows, width, case['median_ns'], flush=True)
    after = loaded_modules(required)
    if before != after:
        raise RuntimeError('GGML module identity changed during measurement')
    result.update(loaded_modules=after, modules_stable=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    dll_dir.close()


if __name__ == '__main__':
    main()
