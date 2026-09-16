import shlex
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

def process_classification(row):
    name=(row.get('name') or '').lower()
    args=row.get('cmdline')
    if isinstance(args,str):
        try: args=[x.strip('"') for x in shlex.split(args,posix=False)]
        except ValueError: args=[]
    args=args or []
    tokens=[str(x).replace('\\','/').lower() for x in args]
    text=' '.join(tokens)
    python=name.startswith(('python','pypy'))
    if (name=='gpu-operator-timing.exe' or name.startswith(('llama-','mmvq-','mmvq_','launch-recorder','host-submission-target'))
        or 'microbench' in name or ('probe' in name and name.endswith('.exe')) or name in ('nsys.exe','ncu.exe')):
        return 'blocked_native'
    if not python:return 'other'
    if '--multiprocessing-fork' in tokens or 'multiprocessing.spawn' in text:return 'spawn_requires_parent'
    if not tokens:return 'unknown_python'
    scripts=[(i,x.rsplit('/',1)[-1]) for i,x in enumerate(tokens) if x.endswith('.py')]
    for i,script in scripts:
        mode=tokens[i+1] if i+1<len(tokens) else ''
        if script in ('runner.py','diagnostic.py') and mode in ('run','resume','extend','evaluate'):
            return 'blocked_execution'
    if any(x in text for x in ('run_candidate.py','predict_stable_native_dataset.py','heterollm_sim',
                              'run_probe.py','run_wrapper_correctness','run_controls.py','run_wrapper_capture.py',
                              'run_target_capture_ex.py','freeze_candidate.py','verify_frozen_preflight.py',
                              'evaluate_identity_repair.py','evaluate_candidate.py','native_llama_compare.py',
                              'run_generalization_matrix.py','native_grid_predict.py','run_estimation.py',
                              'run_kernel_microbench','--authorize-large-file-read')):
        return 'blocked_project'
    # A complete explicit command is required for host-only exemptions; never trust '-c' snippets.
    if '-c' not in tokens:
        if any(script=='diagnostic.py' and i+1<len(tokens) and tokens[i+1]=='self-test' for i,script in scripts):
            return 'allowed_host'
        if any(script=='build.py' and i+1<len(tokens) and tokens[i+1] in ('verify','build','host-test','check') for i,script in scripts):
            return 'allowed_host'
        tests=[script for i,script in scripts if script.startswith('test_')]
        if ('pytest' in tokens or any(x.endswith('/pytest.exe') for x in tokens)) and tests and set(tests)<={'test_identity_diagnostic.py','test_host_probe.py','test_controls.py'}:
            return 'allowed_host'
    return 'unknown_project_python' if '37_llmsim' in text or 'gpu_operator_timing' in text else 'other_python'


def process_conflicts(rows,current_pid=None):
    current_pid=os.getpid() if current_pid is None else current_pid
    by_pid={row['pid']:row for row in rows};bad=[]
    def classify(pid,seen):
        if pid in seen:return 'unknown_parent_cycle'
        row=by_pid.get(pid)
        if row is None:return 'unknown_parent_missing'
        kind=process_classification(row)
        if kind!='spawn_requires_parent':return kind
        parent=row.get('ppid')
        if not parent:return 'unknown_spawn_parent'
        owner=classify(parent,seen|{pid})
        if owner=='allowed_host':return 'allowed_host_spawn'
        if owner=='allowed_host_spawn':return owner
        return 'blocked_spawn_parent' if owner.startswith('blocked_') else 'unknown_spawn_owner'
    for row in rows:
        if row['pid']==current_pid:continue
        kind=classify(row['pid'],set())
        if kind.startswith(('blocked_','unknown_')):
            bad.append({'pid':row['pid'],'ppid':row.get('ppid'),'name':row.get('name'),'reason':kind})
    return bad


def process_snapshot():
    system = os.environ.get('SystemRoot', r'C:\Windows')
    powershell = Path(system)/'System32/WindowsPowerShell/v1.0/powershell.exe'
    script = "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; @(Get-CimInstance Win32_Process -ErrorAction Stop | Select-Object ProcessId,ParentProcessId,Name,CommandLine) | ConvertTo-Json -Compress"
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
    records = [{'pid': int(r['ProcessId']), 'ppid': int(r['ParentProcessId']), 'name': r['Name'], 'cmdline': r['CommandLine']} for r in rows]
    return {'created_utc': now(), 'inventory_count': len(records), 'conflicts': process_conflicts(records),
            'scope': 'project native/simulator/probe; recognized compiler and host-only tests allowed',
            'continuous_exclusivity_verified': False, 'background_GPU_isolation_verified': False,
            'desktop_GPU_process_absence_required': False}
