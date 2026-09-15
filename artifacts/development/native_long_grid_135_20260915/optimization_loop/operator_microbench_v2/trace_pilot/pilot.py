"""One fixed synthetic GEMM trace and one direct control; no LLM inputs."""
from __future__ import annotations
import argparse
import ctypes as ct
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
NSYS = Path(r'C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe')
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
    gpu = GpuTelemetry()
    try:
        identity, state = gpu.identity(), gpu.sample()
    finally:
        gpu.close()
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
        'schema': 'synthetic-gemm-trace-pilot-protocol/v1', 'fixed_before_execution_utc': utc(),
        'scope': 'one independent synthetic GGML MUL_MAT; no GGUF, LLM request, or target LLM latency',
        'attempt_limits': {'profile': 1, 'direct_control': 1, 'each_subprocess_timeout_seconds': 180,
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
                '--cuda-graph-trace=node', '--duration=60', '--wait=primary', '--stop-on-exit=true',
                '--stats=false', '--export=none', '--force-overwrite=false']},
        'environment_explicit': env, 'gpu_identity': identity, 'gpu_state_before_protocol': state,
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
        'calibration_eligible': False, 'llm_latency_fitting': False, 'less_than_5_percent_claim_allowed': False,
        'publication': {'keep_local_only': ['nsys-rep','qdstrm','sqlite','exe','obj','dll','raw event exports'],
                        'small_reviewable_evidence': ['protocol.json','result_summary.json','command receipts','runner/parser code']}}
    write_json(HERE / 'protocol.json', protocol, exclusive=True)
    print(json.dumps({'prepared': True, 'protocol': str(HERE / 'protocol.json'), 'gpu': identity['name'],
                      'uuid': identity['uuid'], 'application_invocations_per_process': 24}, ensure_ascii=False))


def kill_owned_tree(proc):
    import psutil
    try:
        owner = psutil.Process(proc.pid)
        children = owner.children(recursive=True)
    except psutil.NoSuchProcess:
        return []
    killed = []
    for child in reversed(children):
        try:
            child.kill()
            killed.append(child.pid)
        except psutil.NoSuchProcess:
            pass
    if proc.poll() is None:
        proc.kill()
        killed.append(proc.pid)
    return killed


def measure(label, command, environment, identity, *, telemetry=True):
    run_dir = HERE / label
    run_dir.mkdir(exist_ok=False)
    receipt = {'schema': 'synthetic-gemm-pilot-process/v1', 'label': label, 'argv': command,
        'protocol_sha256': ref(HERE / 'protocol.json')['sha256'], 'utc_started': utc(),
        'launch_qpc_ticks': qpc(), 'qpc_frequency_hz': qpf(), 'timeout_seconds': 180,
        'explicit_environment': {key: environment.get(key) for key in ENV_KEYS}}
    write_json(run_dir / 'command_receipt.json', receipt, exclusive=True)
    rows = []
    stop = threading.Event()
    gpu = GpuTelemetry() if telemetry else None
    if gpu is not None and gpu.identity()['uuid'] != identity['uuid']:
        gpu.close()
        raise ValueError('selected GPU changed')
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
            try:
                receipt['returncode'] = proc.wait(timeout=180)
                receipt['timed_out'] = False
            except subprocess.TimeoutExpired:
                receipt['timed_out'] = True
                receipt['killed_owned_pids'] = kill_owned_tree(proc)
                receipt['returncode'] = proc.wait(timeout=10)
    finally:
        receipt['exit_qpc_ticks'] = qpc()
        receipt['utc_finished'] = utc()
        stop.set()
        if thread:
            thread.join(timeout=2)
        if gpu:
            rows.append(gpu.sample())
            gpu.close()
        write_json(run_dir / 'telemetry.json', {'gpu_identity': identity, 'samples': rows,
            'clock_locked': False, 'period_seconds': 0.05})
        write_json(run_dir / 'command_receipt.json', receipt)
    return receipt


def run():
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
    # The control uses precisely the same synthetic parameters, NVTX calls,
    # eviction policy and explicit runtime environment without Nsight injection.
    direct = [exe, *protocol['argv_common'], '--output', str(HERE / 'direct_control' / 'microbench.json')]
    receipts.append(measure('direct_control', direct, environment, protocol['gpu_identity']))
    report = HERE / 'profile' / 'gemm.nsys-rep'
    if report.is_file():
        command = [str(NSYS), 'export', '--type=sqlite', '--force-overwrite=false',
            '--output=' + str(HERE / 'profile' / 'gemm.sqlite'), str(report)]
        receipts.append(measure('export_sqlite', command, environment, protocol['gpu_identity'], telemetry=False))
    for label in ('profile', 'direct_control'):
        path = HERE / label / 'microbench.json'
        if path.exists():
            data = json.loads(path.read_text(encoding='utf-8'))
            write_json(HERE / label / 'qpc_derived_call_timestamps.json', {
                'source': ref(path), 'clock': data.get('timing_contract', {}).get('clock'),
                'per_call_raw_absolute_qpc_ticks_available': False,
                'runs_preserved_verbatim': data.get('runs', [])})
    write_json(HERE / 'execution_receipts.json', {'schema': 'synthetic-gemm-trace-pilot-execution/v1',
        'protocol': ref(HERE / 'protocol.json'), 'processes': receipts,
        'native_llm_executed': False, 'calibration_eligible': False, 'attempts_complete': True})
    print(json.dumps({'completed': True, 'process_exit_codes': {r['label']: r.get('returncode') for r in receipts},
        'report_exists': report.is_file(), 'sqlite_exists': (HERE / 'profile' / 'gemm.sqlite').is_file()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'run'))
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else run)()
