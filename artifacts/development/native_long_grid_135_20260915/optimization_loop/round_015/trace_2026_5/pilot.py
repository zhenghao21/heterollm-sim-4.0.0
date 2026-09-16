"""One fixed synthetic GEMM trace and one direct control; no LLM inputs."""
from __future__ import annotations
import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
from pathlib import Path
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / 'operator_microbench_v2'
ORIGINAL = BASE / 'trace_pilot'
NSYS = Path(r'F:\codex_project\37_LLMsim\tools\nsys_cli\target-windows-x64\nsys.exe')
NVML = Path(r'C:\Windows\System32\nvml.dll')
ENV_KEYS = ('GGML_OP_OFFLOAD_MIN_BATCH', 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_FORCE_MMQ',
    'GGML_CUDA_FORCE_CUBLAS', 'CUDA_VISIBLE_DEVICES', 'GGML_SCHED_DEBUG', 'OMP_NUM_THREADS',
    'GGML_NO_IQ_PANEL', 'LLAMA_TRACE_ANNOTATIONS', 'GGML_CPU_DISABLE_FUSION',
    'GGML_CUDA_DISABLE_FUSION', 'GGML_CUDA_CUBLAS_COMPUTE_TYPE', 'CUDA_LAUNCH_BLOCKING',
    'CUDA_MODULE_LOADING', 'CUDA_CACHE_DISABLE', 'CUDA_INJECTION64_PATH', 'NVTX_INJECTION64_PATH')
CREATE_FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value, *, exclusive=False):
    with path.open('x' if exclusive else 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def ref(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def qpc():
    value = ct.c_longlong()
    if not ct.windll.kernel32.QueryPerformanceCounter(ct.byref(value)):
        raise OSError('QPC unavailable')
    return value.value


def qpf():
    value = ct.c_longlong()
    if not ct.windll.kernel32.QueryPerformanceFrequency(ct.byref(value)):
        raise OSError('QPF unavailable')
    return value.value


class GpuTelemetry:
    def __init__(self):
        self.lib = ct.CDLL(str(NVML))
        self.check(self.lib.nvmlInit_v2())
        self.handle = ct.c_void_p()
        self.check(self.lib.nvmlDeviceGetHandleByIndex_v2(ct.c_uint(0), ct.byref(self.handle)))

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError('NVML status ' + str(code))

    def string(self, name, device=True):
        buf = ct.create_string_buffer(128)
        args = [self.handle] if device else []
        self.check(getattr(self.lib, name)(*args, buf, ct.c_uint(len(buf))))
        return buf.value.decode('utf-8', errors='replace')

    def scalar(self, name, *args, wide=False):
        value = ct.c_ulonglong() if wide else ct.c_uint()
        fn = getattr(self.lib, name, None)
        if fn is None:
            return {'status': 'symbol_unavailable'}
        code = fn(self.handle, *[ct.c_uint(a) for a in args], ct.byref(value))
        return {'status': code, 'value': value.value if code == 0 else None}

    def identity(self):
        return {'uuid': self.string('nvmlDeviceGetUUID'), 'name': self.string('nvmlDeviceGetName'),
                'driver_version': self.string('nvmlSystemGetDriverVersion', False), 'nvml_library': ref(NVML)}

    def sample(self):
        return {'utc': utc(), 'qpc_ticks': qpc(),
            'graphics_mhz': self.scalar('nvmlDeviceGetClockInfo', 0),
            'sm_mhz': self.scalar('nvmlDeviceGetClockInfo', 1),
            'memory_mhz': self.scalar('nvmlDeviceGetClockInfo', 2),
            'pstate': self.scalar('nvmlDeviceGetPerformanceState'),
            'temperature_c': self.scalar('nvmlDeviceGetTemperature', 0),
            'power_mw': self.scalar('nvmlDeviceGetPowerUsage'),
            'clocks_throttle_reasons': self.scalar('nvmlDeviceGetCurrentClocksThrottleReasons', wide=True)}

    def close(self):
        self.lib.nvmlShutdown()


def verify_build():
    manifest = json.loads((BASE / 'build_manifest.json').read_text(encoding='utf-8-sig'))
    items = [manifest['executable'], *manifest['inputs'], *manifest['dependencies']]
    for item in items:
        actual = ref(item['path'])
        if actual['sha256'] != item['sha256'] or actual['bytes'] != item['bytes']:
            raise ValueError('frozen build identity changed: ' + item['path'])
    return manifest


def capture_info(args, destination):
    result = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace',
                            timeout=30, creationflags=CREATE_FLAGS)
    destination.write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('preflight command failed: ' + str(args[1:]))
    return ref(destination)


def prepare():
    if (HERE / 'protocol.json').exists():
        raise RuntimeError('protocol already fixed; do not overwrite')
    manifest = verify_build()
    original = json.loads((ORIGINAL / 'protocol.json').read_text(encoding='utf-8-sig'))
    identity = original['gpu_identity']
    state = {'status': 'not_sampled_during_prepare', 'gpu_workload_launched': False}
    tool_info = {
        'version': capture_info([str(NSYS), '--version'], HERE / 'nsys_version.txt'),
        'profile_help': capture_info([str(NSYS), 'profile', '--help'], HERE / 'nsys_profile_help.txt'),
        'export_help': capture_info([str(NSYS), 'export', '--help'], HERE / 'nsys_export_help.txt')}
    msvc = Path(r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\include')
    supplementary_headers = [Path(r'E:\cuda\include\cuda_runtime_api.h'),
        Path(r'E:\cuda\include\nvtx3\nvToolsExt.h'), msvc / '__msvc_chrono.hpp', msvc / 'intrin.h']
    params = ['--device', 'cuda', '--quant', 'Q4_K', '--m', '64', '--n', '4096', '--k', '1024',
        '--threads', '16', '--cuda-index', '0', '--warmup', '3', '--repeats', '20',
        '--samples', '256', '--seed', '20260914', '--atol', '0.05', '--rtol', '0.03',
        '--evict-mib', '128', '--run', '--nvtx']
    env = {key: None for key in ENV_KEYS}
    env.update({'GGML_CUDA_DISABLE_GRAPHS': '1', 'CUDA_VISIBLE_DEVICES': identity['uuid'],
                'OMP_NUM_THREADS': '16', 'LLAMA_TRACE_ANNOTATIONS': '0'})
    source = Path(manifest['source_root'])
    protocol = {
        'schema': 'synthetic-gemm-trace-pilot-protocol/v2', 'fixed_before_execution_utc': utc(),
        'scope': 'one independent synthetic GGML MUL_MAT; no GGUF, LLM request, or target LLM latency',
        'attempt_limits': {'profile': 1, 'direct_control': 1, 'each_subprocess_wait_checkpoint_seconds': 180,
                           'kill_on_deadline': False, 'launch_next_stage_while_previous_running': False,
                           'automatic_retries': 0},
        'execution_order': ['profile', 'direct_control'], 'argv_common': params,
        'shape': {'M': 64, 'N': 4096, 'K': 1024, 'weight_type': 'Q4_K', 'input_type': 'F32', 'output_type': 'F32'},
        'correctness': {'absolute_tolerance': 0.05, 'relative_tolerance': 0.03, 'samples': 256,
            'criterion': 'abs(actual-reference)<=atol+rtol*abs(reference); all outputs finite',
            'reference': 'dequantized synthetic packed weights, double accumulation and F32 input',
            'check_times': ['after first call', 'after final call'], 'tolerance_may_change_after_run': False},
        'cache': {'requested_eviction_mib': 128, 'effective_rule': 'max(requested bytes,4*device L2)',
            'operation': 'untimed F32 scale read/write sweep before every invocation',
            'outside_nvtx_and_qpc_interval': True, 'guaranteed_dram_transactions': False},
        'nsys': {'executable': ref(NSYS), 'information': tool_info,
            'profile_options': ['--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none', '--backtrace=none',
                '--cuda-graph-trace=node', '--kill=false', '--duration=60', '--wait=primary', '--stop-on-exit=true',
                '--stats=false', '--export=none', '--force-overwrite=false']},
        'environment_explicit': env, 'gpu_identity': identity, 'gpu_state_before_protocol': state,
        'gpu_identity_provenance': {'type': 'expected_device_from_original_pilot', 'source': ref(ORIGINAL / 'protocol.json'),
            'runtime_actual_identity_required': True, 'specification_is_not_current_hardware_evidence': True},
        'comparison': {'original_protocol': ref(ORIGINAL / 'protocol.json'), 'synthetic_workload_and_runtime_unchanged': True, 'profiler_tool_changed': True,
            'recipe_common_argv_identical': params == original['argv_common'],
            'environment_identical': env == original['environment_explicit'],
            'safety_control_change': 'Explicit --kill=false plus non-destructive wait checkpoint; no numeric recipe change.',
            'old_measurements_remain_diagnostic_and_untouched': True},
        'timing': {'host_wall': 'existing per-call steady_clock nanoseconds since process_start; graph_compute includes synchronization',
            'nvtx': 'GENERIC_GEMM|phase=...|index=...|format=...|M=...|N=...|K=...; push before start_ns, pop after end_ns',
            'cuda': 'union of observed CUDA kernel intervals attributable to each external NVTX range',
            'host_and_cuda_are_not_additive': True, 'graph_synchronize_source': ref(source / 'ggml/src/ggml-backend.cpp'),
            'qpc_frequency_hz': qpf(), 'per_call_ns_preserved_verbatim': True,
            'absolute_application_qpc_ticks_available': False,
            'qpc_limitation': 'Existing executable exports QPC-derived relative ns only; process_start QPC epoch is absent. Do not invent absolute ticks or align clocks by guessed offsets.',
            'runner_telemetry_qpc_ticks': 'raw QPC ticks around launch, exit, and NVML samples; not substitutes for in-process call timestamps'},
        'clocks': {'locked_by_pilot': False, 'control_status': 'unlocked_diagnostic_only', 'sampling_period_seconds': 0.05,
            'temperature_controlled': False, 'clock_samples_may_not_bracket_every_short_kernel': True},
        'build_manifest': ref(BASE / 'build_manifest.json'), 'frozen_executable': manifest['executable'],
        'all_existing_manifest_refs_verified': True,
        'build_closure': {'original_manifest_modified': False, 'supplementary_current_header_refs': [ref(p) for p in supplementary_headers],
            'complete_compiler_header_dependency_closure': False,
            'limitation': 'Original build did not capture transitive CUDA/NVTX/MSVC/Windows SDK headers. Current supplemental hashes do not retrospectively prove the complete original include closure.'},
        'runner': ref(Path(__file__)),
        'extractor': ref(HERE / 'analyze.py'),
        'attribution_reference': ref(NSYS.parent / 'reports' / 'nvtx_gpu_proj_trace.py'),
        'tool_inventory': ref(HERE / 'tool_inventory.json'),
        'tool_signatures': ref(HERE / 'tool_signatures.json'),
        'python': {'executable': ref(sys.executable), 'version': sys.version},
        'calibration_eligible': False, 'llm_latency_fitting': False, 'less_than_5_percent_claim_allowed': False,
        'publication': {'keep_local_only': ['nsys-rep','qdstrm','sqlite','exe','obj','dll','raw event exports'],
                        'small_reviewable_evidence': ['protocol.json','result_summary.json','command receipts','runner/parser code']}}
    if not protocol['comparison']['recipe_common_argv_identical'] or not protocol['comparison']['environment_identical']:
        raise ValueError('tool-only comparison changed synthetic recipe/environment')
    if '2026.5.1' not in (HERE / 'nsys_version.txt').read_text(encoding='utf-8-sig'):
        raise ValueError('unexpected Nsight version')
    write_json(HERE / 'protocol.json', protocol, exclusive=True)
    freeze = {'schema': 'synthetic-trace-full-freeze/v1', 'utc': utc(),
        'files': [ref(HERE / name) for name in ('protocol.json', 'pilot.py', 'analyze.py', 'test_analyze.py',
            'test_pilot.py', 'inspect_schema.py', 'prepare_tool_identity.ps1', 'tool_inventory.json', 'tool_signatures.json',
            'nsys_version.txt', 'nsys_profile_help.txt', 'nsys_export_help.txt', 'README.md')],
        'external_build_manifest': protocol['build_manifest'], 'python': protocol['python']['executable']}
    write_json(HERE / 'freeze_manifest.json', freeze, exclusive=True)
    verify_freeze()
    print(json.dumps({'prepared': True, 'protocol': str(HERE / 'protocol.json'), 'gpu': identity['name'],
                      'uuid': identity['uuid'], 'application_invocations_per_process': 24}, ensure_ascii=False))


def verify_freeze():
    freeze = json.loads((HERE / 'freeze_manifest.json').read_text(encoding='utf-8'))
    if not freeze.get('files'):
        raise ValueError('empty freeze file map')
    refs = [*freeze['files'], freeze['external_build_manifest'], freeze['python']]
    protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
    refs.extend([protocol['gpu_identity']['nvml_library'], protocol['comparison']['original_protocol']])
    inventory = json.loads((HERE / 'tool_inventory.json').read_text(encoding='utf-8'))
    refs.extend(inventory['files'])
    if not inventory['files']:
        raise ValueError('empty Nsight tool inventory')
    for item in refs:
        if not item.get('path') or not item.get('sha256') or item.get('bytes') is None:
            raise ValueError('incomplete freeze identity')
        actual = ref(item['path'])
        if actual['sha256'] != item['sha256'] or actual['bytes'] != item['bytes']:
            raise ValueError('frozen identity changed: ' + item['path'])
    actual_paths = {str(path.resolve()) for path in NSYS.parent.rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.suffix.lower() != '.pyc'}
    frozen_paths = {item['path'] for item in inventory['files']}
    if actual_paths != frozen_paths:
        raise ValueError('Nsight tool dependency inventory changed')
    manifest = verify_build()
    return {'passed': True, 'utc': utc(), 'local_external_file_count': len(refs), 'tool_files': len(inventory['files']),
            'manifest_item_count': 1 + len(manifest['inputs']) + len(manifest['dependencies'])}


def wait_without_killing(proc, receipt, checkpoint_seconds=180):
    try:
        receipt['returncode'] = proc.wait(timeout=checkpoint_seconds)
        receipt['status'] = 'completed' if receipt['returncode'] == 0 else 'failed'
        receipt['timed_out'] = False
    except subprocess.TimeoutExpired:
        receipt['status'] = 'still_running'
        receipt['timed_out'] = True
        receipt['returncode'] = None
        receipt['process_left_running'] = True
        receipt['checkpoint_utc'] = utc()
        receipt['next_action'] = 'Observe recorded PID and files; no retry, no next stage, no inferred exit code.'
    return receipt


def measure(label, command, environment, identity, *, telemetry=True):
    run_dir = HERE / label
    run_dir.mkdir(exist_ok=False)
    receipt = {'schema': 'synthetic-gemm-pilot-process/v1', 'label': label, 'argv': command,
        'protocol_sha256': ref(HERE / 'protocol.json')['sha256'], 'utc_started': utc(),
        'launch_qpc_ticks': qpc(), 'qpc_frequency_hz': qpf(), 'wait_checkpoint_seconds': 180, 'kill_on_deadline': False,
        'explicit_environment': {key: environment.get(key) for key in ENV_KEYS}}
    write_json(run_dir / 'command_receipt.json', receipt, exclusive=True)
    rows = []
    stop = threading.Event()
    gpu = GpuTelemetry() if telemetry else None
    if gpu is not None:
        actual_identity = gpu.identity()
        receipt['gpu_identity_actual'] = actual_identity
        if any(actual_identity[key] != identity[key] for key in ('uuid', 'driver_version')):
            receipt['status'] = 'identity_failed'
            write_json(run_dir / 'command_receipt.json', receipt)
            gpu.close()
            raise ValueError('selected GPU UUID or driver changed')
    def collect():
        while not stop.is_set():
            try:
                rows.append(gpu.sample())
            except Exception as exc:
                rows.append({'utc': utc(), 'error': str(exc)})
            stop.wait(0.05)
    thread = threading.Thread(target=collect, daemon=True) if gpu else None
    if thread:
        thread.start()
    try:
        with (run_dir / 'stdout.txt').open('wb') as out, (run_dir / 'stderr.txt').open('wb') as err:
            proc = subprocess.Popen(command, cwd=str(BASE), env=environment, stdout=out, stderr=err,
                                    creationflags=CREATE_FLAGS)
            receipt['pid'] = proc.pid
            wait_without_killing(proc, receipt)
    finally:
        if receipt.get('status') == 'still_running':
            receipt['checkpoint_qpc_ticks'] = qpc()
        else:
            receipt['exit_qpc_ticks'] = qpc()
            receipt['utc_finished'] = utc()
        stop.set()
        if thread:
            thread.join(timeout=2)
        if gpu:
            rows.append(gpu.sample())
            gpu.close()
        write_json(run_dir / 'telemetry.json', {'gpu_identity': receipt.get('gpu_identity_actual'), 'expected_gpu_identity': identity, 'samples': rows,
            'clock_locked': False, 'period_seconds': 0.05,
            'coverage': 'checkpoint_only_process_continues' if receipt.get('status') == 'still_running' else 'process_complete'})
        write_json(run_dir / 'command_receipt.json', receipt)
    receipt['freeze_after_checkpoint_or_completion'] = verify_freeze()
    write_json(run_dir / 'command_receipt.json', receipt)
    return receipt


def persist_execution(receipts):
    complete = len(receipts) == 3 and all(row.get('status') == 'completed' for row in receipts)
    write_json(HERE / 'execution_receipts.json', {'schema': 'synthetic-gemm-trace-pilot-execution/v2',
        'protocol': ref(HERE / 'protocol.json'), 'processes': receipts,
        'native_llm_executed': False, 'calibration_eligible': False, 'attempts_complete': complete})
    return complete


def run():
    before = verify_freeze()
    write_json(HERE / 'pre_execution_verification.json', before, exclusive=True)
    protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
    if ref(Path(__file__))['sha256'] != protocol['runner']['sha256']:
        raise ValueError('runner changed after protocol; do not mutate fixed protocol')
    manifest = verify_build()
    if ref(BASE / 'build_manifest.json')['sha256'] != protocol['build_manifest']['sha256']:
        raise ValueError('original build manifest changed')
    if (HERE / 'profile').exists() or (HERE / 'direct_control').exists():
        raise RuntimeError('one attempt already started; automatic rerun forbidden')
    environment = os.environ.copy()
    for key, value in protocol['environment_explicit'].items():
        environment.pop(key, None)
        if value is not None:
            environment[key] = value
    environment['PATH'] = manifest['native_bin'] + r';E:\cuda\bin;' + environment.get('PATH', '')
    exe = str(manifest['executable']['path'])
    output = str(HERE / 'profile' / 'microbench.json')
    profile = [str(NSYS), 'profile', *protocol['nsys']['profile_options'],
        '--output=' + str(HERE / 'profile' / 'gemm'), exe, *protocol['argv_common'], '--output', output]
    receipts = [measure('profile', profile, environment, protocol['gpu_identity'])]
    persist_execution(receipts)
    if receipts[-1].get('status') != 'completed':
        print(json.dumps({'completed': False, 'checkpoint': receipts[-1]}, ensure_ascii=False))
        return
    # The control uses precisely the same synthetic parameters, NVTX calls,
    # eviction policy and explicit runtime environment without Nsight injection.
    direct = [exe, *protocol['argv_common'], '--output', str(HERE / 'direct_control' / 'microbench.json')]
    receipts.append(measure('direct_control', direct, environment, protocol['gpu_identity']))
    persist_execution(receipts)
    if receipts[-1].get('status') != 'completed':
        print(json.dumps({'completed': False, 'checkpoint': receipts[-1]}, ensure_ascii=False))
        return
    report = HERE / 'profile' / 'gemm.nsys-rep'
    if report.is_file():
        command = [str(NSYS), 'export', '--type=sqlite', '--force-overwrite=false',
            '--output=' + str(HERE / 'profile' / 'gemm.sqlite'), str(report)]
        receipts.append(measure('export_sqlite', command, environment, protocol['gpu_identity'], telemetry=False))
        persist_execution(receipts)
        if receipts[-1].get('status') != 'completed':
            print(json.dumps({'completed': False, 'checkpoint': receipts[-1]}, ensure_ascii=False))
            return
    for label in ('profile', 'direct_control'):
        path = HERE / label / 'microbench.json'
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            write_json(HERE / label / 'qpc_derived_call_timestamps.json', {
                'source': ref(path), 'clock': data.get('timing_contract', {}).get('clock'),
                'per_call_raw_absolute_qpc_ticks_available': False,
                'runs_preserved_verbatim': data.get('runs', [])})
    complete = persist_execution(receipts)
    write_json(HERE / 'post_execution_verification.json', verify_freeze(), exclusive=True)
    print(json.dumps({'completed': complete, 'process_exit_codes': {r['label']: r.get('returncode') for r in receipts},
        'report_exists': report.is_file(), 'sqlite_exists': (HERE / 'profile' / 'gemm.sqlite').is_file()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run', 'verify', 'status'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'run':
        run()
    elif args.mode == 'verify':
        print(json.dumps(verify_freeze(), ensure_ascii=False))
    else:
        path = HERE / 'execution_receipts.json'
        print(path.read_text(encoding='utf-8') if path.exists() else json.dumps({'status': 'prepared_not_run'}))
