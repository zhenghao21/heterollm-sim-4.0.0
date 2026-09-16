"""Pinned predecessor gates plus successor/freeze runner names; no GPU execution."""
from pathlib import Path
import datetime,json,os,re,subprocess

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
                      'native_grid_predict.py', 'run_controls.py', 'run_wrapper_capture.py', 'run_wrapper_correctness_v2.py', 'run_target_capture_ex.py', 'freeze_candidate.py', 'verify_frozen_preflight.py', 'evaluate_identity_repair.py', 'evaluate_candidate.py',
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
    child = subprocess.Popen([str(powershell), '-NoProfile', '-NonInteractive', '-Command', script],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8-sig',
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    try:
        output, error = child.communicate()
    except BaseException as exc:
        raise RuntimeError('process inventory observation interrupted; helper left running pid=' + str(child.pid)) from exc
    if child.returncode != 0:
        raise ValueError('process inventory failed; returncode=' + str(child.returncode))
    rows = json.loads(output)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows:
        raise ValueError('Process inventory unavailable')
    records = [{'pid': int(r['ProcessId']), 'name': r['Name'], 'cmdline': r['CommandLine']} for r in rows]
    return {'created_utc': now(), 'inventory_count': len(records), 'conflicts': process_conflicts(records),
            'scope': 'project native/simulator/probe; recognized compiler and host-only tests allowed',
            'continuous_exclusivity_verified': False, 'background_GPU_isolation_verified': False,
            'desktop_GPU_process_absence_required': False}
