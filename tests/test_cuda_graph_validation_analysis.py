"""Paired results cannot hide missing lifecycle work or platform mismatches."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
try:
    spec = importlib.util.spec_from_file_location('cuda_graph_validation_analysis', TOOLS / 'analyze_cuda_graph_validation.py')
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
finally:
    sys.path.remove(str(TOOLS))


def lifecycle_case():
    event_lists = [
        ['ordinary_submit'],
        ['capture', 'instantiate', 'update', 'first_launch_submit'],
        ['destroy_graph', 'capture', 'update_failure', 'destroy_exec', 'instantiate', 'first_launch_submit'],
        ['replay_submit'],
    ]
    lifecycle = {'remaining_compiled_invocations': 0,
        'transitions': [{'events': events, 'body_executions': 1, 'capture_executes_body': False,
                         'pricing_ready': True, 'unresolved_update_count': 0} for events in event_lists],
        'event_counts': dict(Counter(e for events in event_lists for e in events))}
    env = {'GGML_CUDA_DISABLE_GRAPHS': None, 'GGML_CUDA_GRAPH_OPT': None}
    native = {'configuration': {'effective_env': env}}
    diagnostic = {'status': 'completed', 'cuda_graphs_mode_requested': 'on',
        'ctx_checkpoints_requested': 0, 'effective_env': env, 'latency_values_retained': False,
        'backend_calls': 4, 'actual_events': {'direct': 1, 'replay': 3, 'capture_begin': 2,
        'capture_end': 2, 'instantiate': 1, 'reinstantiate': 1, 'update': 2,
        'update_failed': 1, 'destroy_graph': 1, 'destroy_exec': 1}}
    return lifecycle, diagnostic, native


def test_native_first_launch_and_update_failure_match_simulator_events():
    assert analysis.lifecycle_evidence_errors(*lifecycle_case(), 'on') == []


@pytest.mark.parametrize('mutation', ['count', 'missing_rebuild', 'unresolved', 'body_twice', 'wrong_env'])
def test_graph_validation_rejects_incomplete_or_mismatched_lifecycle(mutation):
    lifecycle, diagnostic, native = deepcopy(lifecycle_case())
    if mutation == 'count':
        diagnostic['backend_calls'] = 5
    elif mutation == 'missing_rebuild':
        diagnostic['actual_events']['reinstantiate'] = 0
    elif mutation == 'unresolved':
        lifecycle['transitions'][2]['pricing_ready'] = False
    elif mutation == 'body_twice':
        lifecycle['transitions'][1]['body_executions'] = 2
    else:
        diagnostic['effective_env'] = {'GGML_CUDA_DISABLE_GRAPHS': '0'}
    assert analysis.lifecycle_evidence_errors(lifecycle, diagnostic, native, 'on')


def test_graph_off_also_requires_actual_diagnostic_without_replay():
    lifecycle, diagnostic, native = lifecycle_case()
    assert analysis.lifecycle_evidence_errors(lifecycle, None, native, 'off')
    diagnostic['cuda_graphs_mode_requested'] = 'off'
    assert analysis.lifecycle_evidence_errors(lifecycle, diagnostic, native, 'off')


def test_current_local_identity_binds_existing_import_and_measurement_contract():
    base = Path(__file__).parents[1] / 'docs/cuda_graph_validation_2026-10-08'
    scenario = analysis.read(base / 'scenario_qwen3_0_6b_f16_graph_on.json')
    native = analysis.read(base / 'native_qwen3_0_6b_f16_graph_on.json')
    assert analysis.identity_errors(base, scenario, native, 'qwen3_0_6b_f16') == []
    scenario['model']['metadata']['metadata']['gguf_sha256'] = 'different-existing-import'
    assert analysis.identity_errors(base, scenario, native, 'qwen3_0_6b_f16')
