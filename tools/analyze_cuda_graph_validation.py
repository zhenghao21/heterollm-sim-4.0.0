"""Compare browser-submitted Graph experiments with matching native samples."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from analyze_frontend_validation import (
    comparable_hardware, comparable_model, hardware, paired_input_contract_errors,
    physical_participation, positive, requests,
)
from render_cuda_graph_validation_report import CASES, METRICS, ROOT, native_metrics


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lifecycle_evidence_errors(lifecycle, diagnostic, native, mode):
    """Reconcile every predicted event with separate server diagnostics."""
    errors = []
    if not isinstance(diagnostic, dict) or diagnostic.get('status') != 'completed':
        return ['native Graph mode has no completed server diagnostic verification']
    if (diagnostic.get('cuda_graphs_mode_requested') != mode
            or diagnostic.get('ctx_checkpoints_requested') != 0
            or diagnostic.get('effective_env') != native['configuration'].get('effective_env')
            or diagnostic.get('latency_values_retained') is not False):
        errors.append('native diagnostic configuration differs from paired timing')
    if not isinstance(lifecycle, dict) or lifecycle.get('remaining_compiled_invocations') != 0:
        return errors + ['source-compiled invocation program was not fully consumed']
    transitions = lifecycle.get('transitions')
    if not isinstance(transitions, (list, tuple)) or len(transitions) != diagnostic.get('backend_calls'):
        return errors + ['simulation/native CUDA backend invocation counts differ']
    counts = Counter(event for transition in transitions for event in transition.get('events', ()))
    if dict(counts) != lifecycle.get('event_counts'):
        errors.append('CUDA lifecycle summary does not match realized transitions')
    if any(transition.get('body_executions') != 1 or transition.get('capture_executes_body') is not False
           or transition.get('pricing_ready') is not True or transition.get('unresolved_update_count') != 0
           for transition in transitions):
        errors.append('CUDA lifecycle contains unresolved costs or duplicated device work')
    actual = diagnostic.get('actual_events', {})
    launches = actual.get('replay', 0)
    first_launches = actual.get('instantiate', 0) + actual.get('reinstantiate', 0)
    expected = {
        'ordinary_submit': actual.get('direct', 0), 'capture': actual.get('capture_end', 0),
        'instantiate': first_launches, 'update_failure': actual.get('update_failed', 0),
        'update': actual.get('update', 0) - actual.get('update_failed', 0),
        'first_launch_submit': first_launches, 'replay_submit': launches - first_launches,
        'destroy_graph': actual.get('destroy_graph', 0), 'destroy_exec': actual.get('destroy_exec', 0),
    }
    if (actual.get('capture_begin', 0) != actual.get('capture_end', 0)
            or actual.get('direct', 0) + launches != diagnostic.get('backend_calls')
            or any(value < 0 for value in expected.values())):
        errors.append('native Graph diagnostic event counts are inconsistent')
    if (mode == 'off' and launches != 0) or (mode == 'on' and launches <= 0):
        errors.append('native diagnostic Graph execution differs from requested mode')
    for event in counts.keys() | expected.keys():
        if counts.get(event, 0) != expected.get(event, 0):
            errors.append(f'simulation/native lifecycle event count differs: {event}')
    return errors


def identity_errors(base, scenario, native, slug):
    """Check the existing GGUF identity contract and recorded platform aliases."""
    errors = []
    preparation_path = base / 'preparation.json'
    if not preparation_path.is_file():
        return ['missing GGUF preparation identity record']
    preparation = read(preparation_path)
    cases = [case for case in preparation.get('cases', ()) if case.get('case_id') == slug]
    if len(cases) != 1:
        return ['missing or ambiguous GGUF preparation identity']
    model = cases[0]['model']
    if scenario['model'].get('metadata', {}).get('metadata', {}).get('gguf_sha256') != model.get('sha256'):
        errors.append('submitted GGUF full identity differs from imported native model')
    if model.get('path') != native['identity'].get('model_path'):
        errors.append('native model path differs from GGUF import identity')
    revision = preparation.get('source_commit')
    build = native['identity'].get('server_build_info', '')
    if not isinstance(revision, str) or not revision[:7] in build or 'dirty' in build:
        errors.append('native build differs from pinned clean source revision')
    gpu_count = 0
    for component in hardware(scenario)['components']:
        if component['kind'] == 'cpu' and component.get('metadata', {}).get('component_preset_id') != 'amd-ryzen-9-9950x3d':
            errors.append('native experiment CPU differs from measured host platform')
        if component['kind'] != 'gpu':
            continue
        gpu_count += 1
        kernel = component['execution_profile']['parameters']['kernel_model']
        measured, binding = kernel.get('runtime_calibration', {}), kernel.get('runtime_calibration_binding', {})
        expected = {'kernel_hardware_id': 'nvidia-rtx-5080',
                    'kernel_architecture': 'sm120-source-rule-analytical',
                    'measured_hardware_id': 'NVIDIA GeForce RTX 5080',
                    'measured_architecture': 'sm_120',
                    'measured_runtime_id': 'CUDA-12080-driver-617.14'}
        if any(binding.get(key) != value for key, value in expected.items()):
            errors.append('runtime measurement platform binding differs from native experiment')
        for key in ('hardware_id', 'runtime_id', 'architecture'):
            if binding.get('kernel_' + key) != kernel.get(key) or binding.get('measured_' + key) != measured.get(key):
                errors.append('runtime measurement identity alias does not bind both profiles: ' + key)
        if (measured.get('cpu_id') != 'AMD Ryzen 9 9950X3D 16-Core Processor'
                or measured.get('qualified') is not False
                or kernel.get('runtime_measurement_mode') != 'experimental_exact_structure'):
            errors.append('experimental runtime measurement scope or host identity differs')
    if gpu_count != 1:
        errors.append('native experiment requires exactly one paired GPU')
    return errors


def compare(base, slug, mode):
    stem = f'ui_{slug}_graph_{mode}'
    paths = {key: base / name for key, name in {
        'submission': stem + '_submission.json', 'result': stem + '_result.json',
        'created': stem + '_job_created.json',
        'prepared': f'scenario_{slug}_graph_{mode}.json',
        'native': f'native_{slug}_graph_{mode}.json',
    }.items()}
    row = {'case_id': slug, 'graph_mode': mode, 'status': 'pending',
           'missing_files': [path.name for path in paths.values() if not path.is_file()]}
    if row['missing_files']:
        return row
    data = {key: read(path) for key, path in paths.items()}
    scenario, job, native = data['submission']['scenario'], data['result'], data['native']
    row['job_id'] = job.get('job_id')
    if job.get('status') != 'completed':
        row.update(status=job.get('status', 'invalid_result'), error=job.get('error'))
        return row
    errors = paired_input_contract_errors(scenario, data['prepared'])
    errors.extend(identity_errors(base, scenario, native, slug))
    if job.get('job_id') != data['created'].get('job_id'):
        errors.append('browser-created job identity differs from result')
    if comparable_model(scenario['model']) != comparable_model(data['prepared']['model']):
        errors.append('submitted model differs from prepared GGUF graph')
    if comparable_hardware(hardware(scenario)) != comparable_hardware(hardware(data['prepared'])):
        errors.append('submitted hardware differs from prepared physical contract')
    config = native['configuration']
    runtime = scenario['profiles']['llama_cpp']
    for sk, nk in {'context': 'context', 'batch': 'batch', 'ubatch': 'ubatch',
                   'threads': 'threads', 'threads_batch': 'threads_batch',
                   'gpu_layers': 'gpu_layers_requested', 'parallel': 'parallel',
                   'kv_type_k': 'cache_type_k', 'kv_type_v': 'cache_type_v'}.items():
        if runtime.get(sk) != config.get(nk):
            errors.append(f'native/simulation runtime setting differs: {sk}')
    if runtime.get('flash_attn') != (config.get('flash_attn') == 'on'):
        errors.append('Flash Attention differs')
    if config.get('cache_prompt') is not False or config.get('speculative_decoding') is not False:
        errors.append('native must disable prompt cache and speculative decoding')
    if config.get('ctx_checkpoints_requested') != 0 or scenario['workload']['metadata'].get('native_ctx_checkpoints') != 0:
        errors.append('controlled comparison must explicitly disable context checkpoints on both sides')
    if scenario['workload'].get('mtp') is not None:
        errors.append('simulation must disable speculative decoding')
    if scenario['workload']['metadata'].get('native_model_path') != config.get('model_path'):
        errors.append('native model file differs')
    if native.get('identity', {}).get('model_path') != config.get('model_path') or native.get(
            'native_server_props', {}).get('model_path') != config.get('model_path'):
        errors.append('native loaded model identity differs from recorded command')
    if config.get('effective_env', {}).get('GGML_CUDA_GRAPH_OPT') not in (None, '0'):
        errors.append('unpaired CUDA Graph optimizer must remain disabled')
    observed = native.get('actual_server_context', {})
    if observed.get('props_n_ctx') != runtime['context'] or observed.get('startup_n_ctx_slot') != runtime['context']:
        errors.append('actual native context differs')
    metadata = scenario['workload']['metadata']
    program = metadata.get('cuda_graph_structural_program', {})
    if program.get('graph_enabled') is not (mode == 'on'):
        errors.append('structural program Graph mode differs')
    for component in hardware(scenario)['components']:
        if component['kind'] != 'gpu':
            continue
        kernel = component['execution_profile']['parameters']['kernel_model']
        if kernel.get('graph_enabled') is not (mode == 'on'):
            errors.append('GPU kernel Graph mode differs')
    try:
        native_stats = native_metrics(paths['native'], mode)
    except (ValueError, KeyError) as exc:
        errors.append(str(exc))
        native_stats = {}
    report = job['report']
    submitted = {req['request_id']: req for req in scenario['workload']['requests']}
    realized = {req['request_id']: req for req in requests(report)}
    if set(submitted) != set(realized) or any(req.get('status') != 'finished' for req in realized.values()):
        errors.append('not every explicit startup/warmup/measured request completed')
    measured_id = metadata.get('cuda_graph_comparison_request_id')
    expected, measured = submitted.get(measured_id, {}), realized.get(measured_id, {})
    if (expected.get('prompt_tokens'), expected.get('output_tokens'), measured.get('visible_output_tokens')) != (512, 128, 128):
        errors.append('measured request must produce 128 tokens from 512 input tokens')
    for key, _ in METRICS:
        if not positive(measured.get(key)):
            errors.append('missing positive measured-request ' + key)
    if report.get('measurement_semantics', {}).get('latency', {}).get('primary_boundary') != 'engine':
        errors.append('comparison requires engine boundaries excluding queue and HTTP time')
    summary = report.get('summary', {})
    if (not positive(summary.get('batch_count'))
            or summary.get('physical_live_batch_count') != summary.get('batch_count')):
        errors.append('not every cohort ran the persistent physical kernel')
    participation = physical_participation(scenario, report)
    errors.extend(participation['errors'])
    lifecycle = find_lifecycle(report)
    diagnostic_path = base / f'native_graph_diagnostic_{slug}_{mode}.json'
    diagnostic = read(diagnostic_path) if diagnostic_path.is_file() else None
    errors.extend(lifecycle_evidence_errors(lifecycle, diagnostic, native, mode))
    if diagnostic and Path(diagnostic.get('paired_timing_case', '')).name != paths['native'].name:
        errors.append('native Graph diagnostic belongs to a different timing case')
    row.update(validation_errors=errors, status='configuration_mismatch' if errors else 'compared',
               simulation_file=paths['result'].name, native_file=paths['native'].name,
               compared_request_id=measured_id, lifecycle=lifecycle,
               physical_participation=participation,
               prediction_qualification='experimental_exact_structure_microbenchmarks',
               generalization_qualified=False,
               measurement_boundary='engine request start to first/last visible token; excludes warmups and queue',
               target_llm_latency_used_for_calibration=False)
    if not errors:
        row['metrics'] = {}
        for key, _ in METRICS:
            sim, stats = measured[key], native_stats[key]
            delta = sim - stats['median_ns']
            row['metrics'][key] = {'simulation_ns': sim, 'native': stats,
                'signed_error_ns': delta, 'absolute_error_ns': abs(delta),
                'signed_error_percent': delta / stats['median_ns'] * 100,
                'absolute_error_percent': abs(delta) / stats['median_ns'] * 100,
                'native_mad_ns': statistics.median(abs(s[key] - stats['median_ns']) for s in native['samples'])}
    return row


def find_lifecycle(value):
    if isinstance(value, dict):
        if 'llama_cuda_graph_lifecycle' in value:
            return value['llama_cuda_graph_lifecycle']
        for child in value.values():
            found = find_lifecycle(child)
            if found is not None:
                return found
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/cuda_graph_validation_2026-10-08')
    parser.add_argument('--case')
    args = parser.parse_args()
    for slug, _ in CASES:
        if args.case and args.case != slug:
            continue
        for mode in ('off', 'on'):
            row = compare(args.output, slug, mode)
            (args.output / f'comparison_{slug}_graph_{mode}.json').write_text(
                json.dumps(row, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(slug, mode, row['status'], row.get('validation_errors', []))


if __name__ == '__main__':
    main()
