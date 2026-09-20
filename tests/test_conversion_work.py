"""Source-only conversion geometry/DAG checks; no GPU or timing fit."""
from pathlib import Path
import json
import pytest

from heterollm_sim.conversion_work import (
    SOURCE_SHA256, ConversionSourceContract, UnsupportedConversion,
    derive_conversion_work, verify_conversion_source_tree,
)

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_FIXTURE = ROOT / 'tests/fixtures/baseline_evidence/kernel_launch_geometry.json'


def contract(**changes):
    params = dict(compute_capability=1200, highest_compiled_arch=1200, warp_size=32,
                  source_hashes=dict(SOURCE_SHA256), runtime_binary_sha256='a' * 64,
                  ordinary_contiguous_2d=True)
    params.update(changes)
    return ConversionSourceContract(**params)


def work(m=1, k=896, path='MMVQ_Q8_1', fmt='Q5_0'):
    return derive_conversion_work(m=m, k=k, path=path, weight_format=fmt, contract=contract())


@pytest.mark.parametrize('m', [1, 2, 4, 8])
@pytest.mark.parametrize('fmt', ['Q5_0', 'Q8_0'])
@pytest.mark.parametrize('k', [32, 480, 512, 544, 896, 1024, 1056])
def test_mmvq_grid_io_and_partial_operations(m, fmt, k):
    w = work(m=m, k=k, fmt=fmt)
    kp = ((k + 511) // 512) * 512
    assert w.grid == (kp // 256, m, 1)
    assert w.block == (256, 1, 1)
    assert w.cta_count == m * kp // 256
    assert w.launched_threads == m * kp and w.launched_warps == m * kp // 32
    assert w.padding_only_threads == m * (kp - k)
    assert w.read_bytes == 4 * m * k
    assert w.write_bytes == 36 * m * kp // 32
    assert w.partial_scalar_operations == 11 * m * kp
    counts = w.source_expression_counts
    assert counts['abs'] == m * kp
    assert counts['float_max'] == counts['float_add'] == 5 * m * kp
    assert counts['shuffle'] == 10 * m * kp
    # Known-zero padding blocks skip normalize/round, while source d=amax/127 remains.
    assert counts['division'] == m * kp + m * k
    assert counts['roundf'] == counts['int8_cast'] == m * k
    assert counts['fp16_cast'] == 2 * m * kp // 32


@pytest.mark.parametrize('path,fmt', [('MMQ_D4', 'Q5_0'), ('MMQ_D4', 'Q8_0'),
    ('MMQ_DS4', 'Q4_0'), ('MMQ_DS4', 'Q4_1'), ('MMQ_DS4', 'Q5_1')])
@pytest.mark.parametrize('m,k', [(1, 32), (16, 896), (32, 1024), (64, 544)])
def test_mmq_layout_specific_grid_dag_and_io(path, fmt, m, k):
    w = work(m=m, k=k, path=path, fmt=fmt)
    kp = ((k + 511) // 512) * 512
    t = m * kp // 4
    assert w.grid == (m, kp // 512, 1) and w.block == (128, 1, 1)
    assert w.cta_count == m * kp // 512 and w.launched_threads == t
    assert w.read_bytes == 4 * m * k and w.write_bytes == 144 * m * kp // 128
    assert w.partial_scalar_operations == (20 if path == 'MMQ_DS4' else 14) * t
    counts = w.source_expression_counts
    assert counts['abs'] == 4 * t and counts['float_max'] == 6 * t
    assert counts['division'] == 2 * t and counts['multiply'] == 4 * t
    assert counts['roundf'] == counts['int8_cast'] == 4 * t
    assert counts['shuffle'] == (6 if path == 'MMQ_DS4' else 3) * t
    assert counts.get('float_add', 0) == (6 * t if path == 'MMQ_DS4' else 0)
    assert counts.get('fp16_cast', 0) == (t // 4 if path == 'MMQ_DS4' else 0)


@pytest.mark.parametrize('path,fmt', [('MMVQ_Q8_1', 'Q5_0'), ('MMQ_D4', 'Q8_0'), ('MMQ_DS4', 'Q5_1')])
def test_dag_is_acyclic_and_input_stores_close_byte_totals(path, fmt):
    w = work(path=path, fmt=fmt)
    seen = set()
    for node in w.dag:
        assert node.node_id not in seen
        assert all(dep in seen for dep in node.dependencies)
        assert type(node.count) is int and node.count > 0
        seen.add(node.node_id)
    assert sum(n.count for n in w.dag if n.operation.startswith('store_')) == w.write_bytes
    assert sum(n.count for n in w.dag if n.operation == 'read_F32') == w.read_bytes
    assert all(n.to_metadata()['cost_cycles'] is None for n in w.dag)


def test_cta_resource_upper_bound_not_memory_efficiency():
    assert work().active_sm_upper_bound(84) == 4
    assert work(m=2).active_sm_upper_bound(84) == 8
    assert work(m=4).active_sm_upper_bound(84) == 16
    w = work(m=64, path='MMQ_D4')
    assert w.cta_count == 128 and w.active_sm_upper_bound(84) == 84
    assert w.to_metadata()['hbm_utilization_multiplier'] is None
    assert w.to_metadata()['measured_occupancy'] is None
    with pytest.raises(UnsupportedConversion): w.active_sm_upper_bound(True)


def test_layout_not_inferred_from_model_or_token_batch():
    # Caller must state observed/source-qualified path. Same M/K does not force same layout.
    a = work(m=4, path='MMVQ_Q8_1')
    b = work(m=4, path='MMQ_D4')
    assert a.write_bytes == b.write_bytes and a.read_bytes == b.read_bytes
    assert a.grid != b.grid and a.dag != b.dag
    assert a.to_metadata()['kernel_launch_count'] == 1
    assert a.to_metadata()['host_synchronization_count_added'] == 0
    assert a.to_metadata()['native_dispatch_proven'] is False
    assert a.to_metadata()['cost_model_applied'] is False
    assert 'model_name' not in a.to_metadata()


@pytest.mark.parametrize('kwargs', [
    {'compute_capability': 890}, {'highest_compiled_arch': 900}, {'warp_size': 64},
    {'ordinary_contiguous_2d': False}, {'ordinary_contiguous_2d': 1}, {'input_dtype': 'F16'},
    {'channels': 2}, {'samples': 2}, {'channels': True}, {'scatter': True}, {'has_ids': True},
    {'native_fp4': True}, {'scatter': 0}, {'runtime_binary_sha256': ''}, {'source_hashes': {}},
])
def test_source_contract_rejects_unknown_backend_semantics(kwargs):
    with pytest.raises(UnsupportedConversion): contract(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'m': True}, {'m': 0}, {'k': 897}, {'k': 0}, {'m': 9}, {'fmt': 'Q4_K'},
    {'path': 'MMQ_DS4', 'fmt': 'Q5_0'}, {'path': 'MMQ_D4', 'fmt': 'Q5_1'},
    {'path': 'D2S6'}, {'fmt': 'q5_0'}, {'path': 'MMQ_D4', 'm': 2147483648},
])
def test_unproven_shape_path_format_refuses(kwargs):
    with pytest.raises(UnsupportedConversion): work(**kwargs)


def test_runtime_identity_is_frozen_and_not_taken_from_measurement():
    c = contract()
    with pytest.raises(TypeError): c.source_hashes['ggml-cuda/quantize.cu'] = 'b' * 64
    w = derive_conversion_work(m=4, k=896, weight_format='Q5_0', path='MMVQ_Q8_1', contract=c)
    assert w.runtime_binary_sha256 == c.runtime_binary_sha256
    assert w.to_metadata()['binary_source_equivalence_proven'] is False
    assert not any(key.endswith('_ns') for key in w.to_metadata())


def test_locked_source_hashes_if_source_checkout_available():
    source = ROOT / 'source/llama.cpp-semantic/ggml/src'
    if not (source / 'ggml-cuda/quantize.cu').is_file():
        pytest.skip('local immutable native source not present in this checkout')
    assert dict(verify_conversion_source_tree(source)) == dict(SOURCE_SHA256)


def test_completed_conversion_launch_geometry_crosscheck_if_evidence_available():
    # Static geometry only: no timing data or active collector dependencies.
    cases = json.loads(GEOMETRY_FIXTURE.read_text(encoding='utf-8'))['conversion_cases']
    assert len(cases) == 9
    checked = 0
    for doc in cases:
        if any(c.get('issues') for c in doc['calls']):
            continue
        config = doc['config']
        label = 'MMVQ_Q8_1' if config['expected_source_path'] == 'MMVQ_Q8_1_HALF' else 'MMQ_D4'
        w = work(m=config['M'], k=config['K'], fmt=config['quant'], path=label)
        for call in doc['calls']:
            conversion = [x['kernel'] for x in call['kernel_launch_pairs'] if x['role'].startswith('conversion_')]
            assert len(conversion) == 1
            k = conversion[0]
            assert tuple(k[name] for name in ('gridX', 'gridY', 'gridZ')) == w.grid
            assert tuple(k[name] for name in ('blockX', 'blockY', 'blockZ')) == w.block
            checked += 1
    assert checked >= 36
