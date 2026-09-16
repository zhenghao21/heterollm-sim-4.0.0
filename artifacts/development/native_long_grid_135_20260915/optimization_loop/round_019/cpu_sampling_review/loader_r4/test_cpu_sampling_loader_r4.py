"""R19 R4 loader-binding regressions; no compile, LoadLibrary, model, GPU, or timing probe."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[7]
PROBE = ROOT / 'artifacts/development/native_long_grid_135_20260915/optimization_loop/round_019/cpu_sampling_probe_r4'
SPEC = importlib.util.spec_from_file_location('cpu_sampling_probe_entry_r4', PROBE / 'entry.py')
entry = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(entry)

IDENTITY = {
    'cpu_brand': 'Synthetic loader test CPU',
    'cpuid_signature': 1234,
    'active_processor_group_count': 1,
    'active_processor_count_group0': 16,
    'group': 0,
    'logical_cpu': 0,
    'thread_affinity_mask': 1,
}
FREQUENCY_FIELDS = ['os_max_mhz', 'os_reported_current_mhz', 'os_limit_mhz']
_, _, _, _, _, LOADER = entry.inputs()


def observation():
    return {'group': 0, 'logical_processor': 0, 'thread_affinity_mask': 1, 'os_max_mhz': 5000, 'os_reported_current_mhz': 4500, 'os_limit_mhz': 100}


def stage(name):
    before, after = observation(), observation()
    return {'stage': name, 'numeric_quality': 'exact_pass', 'warmup_calls': 16, 'steady_repeats': 64, 'first_use_ticks': 1, 'steady_raw_ticks': [100] * 64, 'reported_frequency_fields_checked': FREQUENCY_FIELDS, 'reported_frequency_stable': True, 'timing_usable': True, 'diagnostic_only': False, 'frequency_changed_diagnostic_only': False, 'cpu_before': before, 'cpu_after': after}


def valid_document():
    cases = []
    for vocabulary in (32768, 131072, 262144):
        for pattern in ('monotone_ascending', 'deterministic_random_permutation'):
            cases.append({'vocabulary_size': vocabulary, 'pattern': pattern, 'split': 'holdout' if vocabulary == 131072 else 'train', 'candidate_record_bytes': 12, 'top_k': 1, 'first_use_not_pooled_with_steady': True, 'stages': [stage('candidate_loop'), stage('original_dll_topk_apply')]})
    expected = entry._expected_loaded_module_paths(LOADER)
    return {'status': 'complete', 'process_index': 0, 'process_id': 991, 'logical_cpu': 0, 'GPU_context_created': False, 'GPU_context_created_by_probe_calls': False, 'no_probe_gpu_api_calls': True, 'dependency_dlls_loaded': True, 'dependency_loading_is_not_GPU_measured': True, 'no_dllmain_context_claim': True, 'model_loaded': False, 'full_sampler_chain_measured': False, 'actual_cpu_identity': copy.deepcopy(IDENTITY), 'qpc_frequency': 10_000_000, 'observer_empty_bracket_ticks': [1] * 64, 'loaded_native_modules': [{'name': name, 'actual_path': path} for name, path in expected.items()], 'cases': cases}


def test_real_loader_binding_is_readonly_verified_and_has_frozen_eight_module_order():
    assert LOADER['dependency_load_order'] == ['cudart64_12.dll', 'cublasLt64_12.dll', 'cublas64_12.dll', 'ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll', 'ggml.dll', 'llama.dll']
    assert [Path(item['path']).name for item in LOADER['runtime_modules']] == ['llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll']
    assert [Path(item['path']).name for item in LOADER['cuda_dependency_modules']] == ['cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll']
    assert Path(LOADER['cuda_directory']).as_posix().lower().endswith('/e:/cuda/bin') is False
    assert Path(LOADER['cuda_directory']).drive.lower() == 'e:'


def test_matching_actual_loaded_paths_pass_without_real_dll_load():
    result = entry.validate_result(valid_document(), 0, 0, IDENTITY, LOADER)
    assert result['stable_stages'] == 12


def test_foreign_loaded_module_path_or_missing_module_is_rejected():
    foreign = valid_document()
    foreign['loaded_native_modules'][0]['actual_path'] = str(PROBE / 'foreign.dll')
    with pytest.raises(ValueError, match='loaded module path differs'):
        entry.validate_result(foreign, 0, 0, IDENTITY, LOADER)
    missing = valid_document()
    missing['loaded_native_modules'].pop()
    with pytest.raises(ValueError, match='loaded module set differs'):
        entry.validate_result(missing, 0, 0, IDENTITY, LOADER)


def test_scoped_loader_source_preserves_order_error_detail_and_lifetime():
    source = (PROBE / 'cpu_sampling_probe.cpp').read_text(encoding='utf-8')
    assert 'AddDllDirectory' in source and 'RemoveDllDirectory' in source
    assert 'FormatMessageW' in source and 'Win32Error=' in source
    constructor = source.index('explicit NativeSampler')
    assert source.index('preload(cuda_dir / name, name)', constructor) < source.index('preload(runtime_dir / name, name)', constructor) < source.index('LoadLibraryExW(path.c_str()', constructor)
    destructor = source[source.index('~NativeSampler()'):source.index('NativeSampler(const NativeSampler&)')]
    assert destructor.index('free_sampler(sampler)') < destructor.index('FreeLibrary(library)') < destructor.index('release_loader()')
    assert 'SetEnvironmentVariable' not in source
    assert 'dependency_loading_is_not_GPU_measured' in source


def test_entry_only_sets_explicit_annotation_environment_and_passes_frozen_directories():
    source = (PROBE / 'entry.py').read_text(encoding='utf-8')
    assert "child_environment = {**os.environ, 'LLAMA_TRACE_ANNOTATIONS': '0'}" in source
    assert "'--native-runtime-dir', loader['runtime_directory']" in source
    assert "'--cuda-dependency-dir', loader['cuda_directory']" in source
    assert "'PATH':" not in source



