"""Root-reviewed, serial GPU execution within existing research authorization.
Build and host-only tests never call this entry point.
"""
from pathlib import Path
import argparse, datetime, json, os, re, subprocess, sys
import build, analyze
HERE = Path(__file__).resolve().parent
LOADED_CODE_REFS = [build.ref(p) for p in (__file__, build.__file__, analyze.__file__)]
OS_KEYS = ('SystemRoot', 'WINDIR', 'COMSPEC', 'TEMP', 'TMP', 'LOCALAPPDATA', 'APPDATA',
           'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'ProgramData', 'ProgramFiles', 'ProgramFiles(x86)')
DISPATCH_PREFIXES = ('CUDA_', 'GGML_', 'CUBLAS_', 'CUDNN_', 'CUPTI_', 'NVIDIA_', 'NV_',
                     'NVTX_', 'NSYS_', 'NCU_', 'LD_', 'OMP_', 'KMP_', 'MKL_', 'OPENBLAS_',
                     'HOST_PROBE_', 'CAPTURE_', 'HIP_', 'ROCR_')
EXPLICIT_FLAGS = {'GGML_CUDA_DISABLE_GRAPHS': '1'}
TOOLCHAIN_ONLY = re.compile(r'CUDA_(?:PATH(?:_V[0-9]+_[0-9]+)?|HOME|ROOT)')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def environment_for(protocol, inherited=None):
    source = dict(os.environ if inherited is None else inherited)
    lookup = {k.upper(): v for k, v in source.items()}
    rejected = sorted(k for k, v in source.items()
                      if (k.upper().startswith(DISPATCH_PREFIXES) or
                          any(s in k.upper() for s in ('INJECTION', 'PROFILER', 'PRELOAD')))
                      and not TOOLCHAIN_ONLY.fullmatch(k.upper())
                      and (k.upper() not in EXPLICIT_FLAGS or v != EXPLICIT_FLAGS[k.upper()]))
    if rejected:
        raise ValueError('Inherited dispatch/profiling controls rejected (names only): ' + ', '.join(rejected))
    env = {k: lookup[k.upper()] for k in OS_KEYS if k.upper() in lookup}
    system = lookup.get('SYSTEMROOT', r'C:\Windows')
    env['SystemRoot'] = system
    dirs = list(dict.fromkeys(str(Path(r['path']).parent) for r in protocol['target_modules']))
    env['PATH'] = os.pathsep.join(dirs + [str(Path(system)/'System32'), system])
    env.update(EXPLICIT_FLAGS)
    env['HOST_PROBE_AUTHORIZED_GPU_RUN'] = '1'
    policy = {'policy': 'explicit_minimal_OS_and_locked_DLL_PATH/v1', 'dll_directories': dirs,
              'configured_flags': EXPLICIT_FLAGS, 'inherited_environment_verified': False,
              'runtime_dispatch_qualified_by_environment': False,
              'omitted_parent_variable_names': sorted(k for k in source if k.upper() not in
                 {x.upper() for x in OS_KEYS} | set(EXPLICIT_FLAGS)),
              'child_environment': env}
    return env, policy


def process_conflicts(records, current_pid=None):
    current_pid = os.getpid() if current_pid is None else current_pid
    conflicts = []
    for item in records:
        if item['pid'] == current_pid:
            continue
        name = (item.get('name') or '').lower()
        args = item.get('cmdline')
        text = (' '.join(args) if isinstance(args, list) else args or '').replace('\\', '/').lower()
        python = name.startswith(('python', 'pypy'))
        native = (name.startswith(('llama-', 'mmvq-', 'mmvq_', 'launch-recorder', 'host-submission-target'))
                  or name in {'test-backend-ops.exe', 'nsys.exe', 'ncu.exe'}
                  or 'microbench' in name or ('probe' in name and name.endswith('.exe')))
        simulation = any(v in text for v in ('predict_stable_native_dataset.py', 'run_candidate.py',
                      'native_grid_predict.py', 'evaluate_identity_repair.py', 'evaluate_candidate.py',
                      'native_llama_compare.py', '-m heterollm_sim', 'run_estimation.py',
                      'run_generalization_matrix.py', 'run_kernel_microbench', '--run-recorder-only',
                      '--run-correctness', 'run_wrapper_correctness', 'run_probe.py'))
        project_python = python and ('37_llmsim' in text or 'heterollm' in text)
        # Only recognized host entry points are exempt; native/probe detection still wins.
        safe_host = (python and not simulation and (
            bool(re.search(r'(?:^|[/ ])build\.py\s+(?:verify|build|host-test)(?:\s|$)', text.replace(chr(34), '').replace(chr(39), ''))) or
            ('pytest' in text and 'test_host_probe.py' in text)))
        unknown = python and not args
        if native or (python and simulation) or unknown or (project_python and not safe_host):
            conflicts.append({'pid': item['pid'], 'name': item.get('name'), 'native_or_probe': native,
                              'simulation_or_probe_runner': python and simulation,
                              'unknown_python_commandline': unknown,
                              'unclassified_project_python': project_python and not safe_host and not simulation})
    return conflicts


def process_snapshot():
    system = os.environ.get('SystemRoot', r'C:\Windows')
    powershell = Path(system)/'System32/WindowsPowerShell/v1.0/powershell.exe'
    script = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process -ErrorAction Stop | Select-Object ProcessId,Name,CommandLine) | ConvertTo-Json -Compress"
    result = subprocess.run([str(powershell), '-NoProfile', '-NonInteractive', '-Command', script],
                            capture_output=True, encoding='utf-8-sig', check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    rows = json.loads(result.stdout)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        raise ValueError('Process inventory unavailable')
    records = [{'pid': int(r['ProcessId']), 'name': r['Name'], 'cmdline': r['CommandLine']} for r in rows]
    return {'created_utc': now(), 'inventory_count': len(records), 'conflicts': process_conflicts(records),
            'scope': 'project native/simulator/probe; recognized compiler and host-only tests allowed',
            'continuous_exclusivity_verified': False, 'background_GPU_isolation_verified': False,
            'desktop_GPU_process_absence_required': False}


def identity_snapshot(manifest):
    for expected in LOADED_CODE_REFS:
        if build.ref(expected['path']) != expected:
            raise ValueError('Loaded runner/analyzer/build helper code changed')
    before_ref = build.ref(manifest)
    m = json.loads(Path(manifest).read_text(encoding='utf-8'))
    inputs = build.verify_inputs()
    if m['inputs'] != inputs:
        raise ValueError('Build/protocol inputs changed')
    compiled = []
    for expected in m['headers'] + [m['target_executable']]:
        actual = build.ref(expected['path'])
        if actual != expected:
            raise ValueError('Compiled target/header changed')
        compiled.append(actual)
    if before_ref != build.ref(manifest):
        raise ValueError('Build manifest changed during identity check')
    return {'build_ref': before_ref, 'protocol_ref': build.ref(HERE/'protocol.json'),
            'inputs': inputs, 'compiled_refs': compiled, 'loaded_code_refs': LOADED_CODE_REFS}


def validated_finish(directory, protocol, manifest, mode, topology):
    finish = json.loads((directory/'finish.json').read_text(encoding='utf-8'))
    if (finish.get('status') != 'validated' or finish.get('returncode') != 0 or
            finish.get('inputs_unchanged') is not True or finish.get('process_gates_passed') is not True):
        raise ValueError('prior failed execution/identity gate is preserved and blocks this sequence')
    if build.ref(directory/'start.json') != finish['start_ref']:
        raise ValueError('prior start changed')
    start = json.loads((directory/'start.json').read_text(encoding='utf-8'))
    current = identity_snapshot(manifest)
    if finish['identity_before'] != start['identity_before'] or finish['identity_before'] != finish['identity_after'] or current != finish['identity_after']:
        raise ValueError('qualification identity changed')
    if build.ref(directory/'record.json') != finish['record_ref']:
        raise ValueError('prior record changed')
    if finish['build_ref'] != current['build_ref'] or finish['protocol_ref'] != current['protocol_ref']:
        raise ValueError('qualification identity changed')
    record = json.loads((directory/'record.json').read_text(encoding='utf-8'))
    analyze.validate_record(record, protocol, mode, topology, finish['protocol_ref']['sha256'])
    return record


def prerequisites(mode, topology, index, manifest):
    frozen = identity_snapshot(manifest)
    protocol = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    m = json.loads(Path(manifest).read_text(encoding='utf-8'))
    if mode == 'path':
        if index != 1:
            raise ValueError('path qualification has one immutable attempt per topology')
    else:
        if index not in range(1, 6):
            raise ValueError('five preregistered timing processes')
        for t in ('chain', 'fanout'):
            validated_finish(HERE/'runs'/('path_'+t+'_0001'), protocol, manifest, 'path', t)
        if topology == 'fanout':
            for i in range(1, 6):
                validated_finish(HERE/'runs'/('timing_chain_%04d' % i), protocol, manifest, 'timing', 'chain')
        for i in range(1, index):
            validated_finish(HERE/'runs'/('timing_'+topology+'_%04d' % i), protocol, manifest, 'timing', topology)
    if identity_snapshot(manifest) != frozen:
        raise ValueError('Prerequisite identities changed during verification')
    return protocol, m, frozen


def launch_process(argv, env, directory):
    # Popen/wait deliberately has no kill-on-exception context manager or timeout.
    with (directory/'stdout.log').open('xb') as out, (directory/'stderr.log').open('xb') as err:
        child = subprocess.Popen(argv, env=env, cwd=HERE, stdout=out, stderr=err,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        try:
            build.write_new(directory/'child_started.json', {'created_utc': now(), 'pid': child.pid})
            code = child.wait()
            build.write_new(directory/'child_exited.json', {'created_utc': now(), 'pid': child.pid,
                             'returncode': code, 'natural_terminal_observed': True})
            return code
        except BaseException as error:
            build.write_new(directory/'child_observation_error.json', {'created_utc': now(), 'pid': child.pid,
                'observed_returncode': child.poll(), 'termination_requested': False, 'error_type': type(error).__name__})
            raise


def execute(a):
    # Before an immutable attempt exists, read-only prerequisite rejection is not an execution.
    protocol, m, frozen = prerequisites(a.mode, a.topology, a.index, a.manifest)
    directory = HERE/'runs'/(a.mode+'_'+a.topology+'_%04d' % a.index)
    directory.mkdir(parents=True, exist_ok=False)
    finish = {'schema': 'host-submission-run-finish/v2', 'status': 'failed', 'returncode': None,
              'inputs_unchanged': False, 'process_gates_passed': False, 'record_ref': None,
              'identity_before': frozen, 'identity_after': None, 'start_ref': None,
              'build_ref': frozen['build_ref'], 'protocol_ref': frozen['protocol_ref'],
              'termination_requested': False, 'cost_model_admitted': False, 'errors': [], 'mode': a.mode, 'topology': a.topology, 'process_index': a.index}
    error = None
    try:
        env, policy = environment_for(protocol)
        before = identity_snapshot(a.manifest)
        if before != frozen:
            raise ValueError('Prerequisite identities changed before execution')
        finish.update(identity_before=before, build_ref=before['build_ref'], protocol_ref=before['protocol_ref'])
        argv = [m['target_executable']['path'], a.mode, a.topology, str(directory/'record.json'), before['protocol_ref']['sha256']]
        start = {'schema': 'host-submission-run-start/v2', 'created_utc': now(), 'mode': a.mode,
                 'topology': a.topology, 'process_index': a.index, 'identity_before': before,
                 'build_ref': before['build_ref'], 'protocol_ref': before['protocol_ref'], 'argv': argv,
                 'environment_policy': policy, 'independent_process': True, 'target_LLM_run': False,
                 'native_target_latency_used': False, 'cost_model_admitted': False}
        build.write_new(directory/'start.json', start)
        finish['start_ref'] = build.ref(directory/'start.json')
        finish['process_before'] = process_snapshot()
        if finish['process_before']['conflicts']:
            raise ValueError('native/simulator/probe process conflict before launch')
        finish['returncode'] = launch_process(argv, env, directory)
        if finish['returncode'] != 0:
            raise RuntimeError('target returned failure')
        finish['record_ref'] = build.ref(directory/'record.json')
        record = json.loads((directory/'record.json').read_text(encoding='utf-8'))
        analyze.validate_record(record, protocol, a.mode, a.topology, before['protocol_ref']['sha256'])
    except BaseException as exc:
        error = exc
        finish['errors'].append({'phase': 'execution', 'type': type(exc).__name__, 'message': str(exc)})
    finally:
        # Every post-check is independent; run identity verification last.
        try:
            finish['process_after'] = process_snapshot()
            finish['process_gates_passed'] = ('process_before' in finish and
                not finish['process_before']['conflicts'] and not finish['process_after']['conflicts'])
            if not finish['process_gates_passed']:
                raise ValueError('native/simulator/probe process conflict or missing pre-check')
        except BaseException as exc:
            finish['errors'].append({'phase': 'post_process', 'type': type(exc).__name__, 'message': str(exc)})
        # Preserve raw failed records too; absence/hash failure is explicit in the terminal receipt.
        try:
            raw_ref = build.ref(directory/'record.json')
            if finish['record_ref'] is not None and raw_ref != finish['record_ref']:
                raise ValueError('record changed after validation')
            finish['record_ref'] = raw_ref
        except BaseException as exc:
            finish['errors'].append({'phase': 'post_record', 'type': type(exc).__name__, 'message': str(exc)})
        try:
            after = identity_snapshot(a.manifest)
            finish['identity_after'] = after
            finish['inputs_unchanged'] = finish['identity_before'] is not None and after == finish['identity_before']
            if not finish['inputs_unchanged']:
                raise ValueError('pre/post identity differs')
            if finish['start_ref'] is not None and build.ref(directory/'start.json') != finish['start_ref']:
                finish['inputs_unchanged'] = False
                raise ValueError('start receipt changed')
        except BaseException as exc:
            finish['inputs_unchanged'] = False
            finish['errors'].append({'phase': 'post_identity', 'type': type(exc).__name__, 'message': str(exc)})
        if not finish['errors'] and finish['returncode'] == 0:
            finish['status'] = 'validated'
        finish['finished_utc'] = now()
        build.write_new(directory/'finish.json', finish)
    if finish['status'] != 'validated':
        raise RuntimeError('failed attempt preserved, no retry or performance acceptance: '+str(directory)) from error
    print(json.dumps({'result': str(directory), 'cost_model_admitted': False}))
    return finish


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('path', 'timing'))
    parser.add_argument('--topology', choices=('chain', 'fanout'), required=True)
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--authorize-gpu-execution', action='store_true')
    a = parser.parse_args()
    if not a.authorize_gpu_execution:
        raise ValueError('Root-reviewed serial execution flag required within existing research authorization')
    execute(a)


if __name__ == '__main__':
    main()
