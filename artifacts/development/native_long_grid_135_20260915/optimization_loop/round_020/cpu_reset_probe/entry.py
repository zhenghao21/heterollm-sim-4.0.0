"""Static/prepare entry. Measurement needs root authorization and a frozen CPU identity."""
from pathlib import Path
import argparse, hashlib, json, math, os, platform, re, statistics, struct, subprocess, sys

P = Path(__file__).resolve().parent
ROOT = P.parents[5]
VS = Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat')
CL = Path('C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe')
INCLUDES = [ROOT / 'source/llama.cpp-semantic/include', ROOT / 'source/llama.cpp-semantic/ggml/include']
REQUIRED_EXPORTS = ('llama_sampler_init_top_k', 'llama_sampler_apply', 'llama_sampler_free')
IDENTITY_FILE = P / 'source_identity_reset_r1.json'
IDENTITY_KEYS = ('cpu_brand', 'cpuid_signature', 'active_processor_group_count', 'active_processor_count_group0', 'group', 'logical_cpu', 'thread_affinity_mask')


def ref(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def verify(expected):
    actual = ref(expected['path'])
    if actual['sha256'] != expected['sha256'] or ('bytes' in expected and actual['bytes'] != expected['bytes']):
        raise ValueError('frozen identity mismatch: ' + expected['path'])
    return actual


def write_new(path, doc):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(doc, handle, indent=2, allow_nan=False)
        handle.write('\n')


def pe_identity(path):
    """Read PE export/import directories from bytes; never LoadLibrary or execute."""
    blob = Path(path).read_bytes()
    u16 = lambda offset: struct.unpack_from('<H', blob, offset)[0]
    u32 = lambda offset: struct.unpack_from('<I', blob, offset)[0]
    if blob[:2] != b'MZ':
        raise ValueError('PE DOS header missing')
    pe = u32(0x3c)
    if blob[pe:pe + 4] != b'PE\0\0' or u16(pe + 4) != 0x8664:
        raise ValueError('x64 native PE required')
    optional = pe + 24
    if u16(optional) != 0x20b:
        raise ValueError('PE32+ required')
    section_count = u16(pe + 6)
    table = optional + u16(pe + 20)
    sections = [(u32(table + index * 40 + 12), max(u32(table + index * 40 + 8), u32(table + index * 40 + 16)), u32(table + index * 40 + 20)) for index in range(section_count)]

    def offset(rva):
        for start, size, raw in sections:
            if start <= rva < start + size:
                return raw + rva - start
        raise ValueError('unmapped PE RVA')

    def cstr(rva):
        start = offset(rva)
        end = blob.find(b'\0', start)
        if end < 0:
            raise ValueError('unterminated PE string')
        return blob[start:end].decode('ascii')

    export_rva, export_size = struct.unpack_from('<II', blob, optional + 112)
    exports = {}
    if export_rva:
        directory = offset(export_rva)
        functions, names = u32(directory + 20), u32(directory + 24)
        function_array, name_array, ordinal_array = offset(u32(directory + 28)), offset(u32(directory + 32)), offset(u32(directory + 36))
        for index in range(names):
            name = cstr(u32(name_array + 4 * index))
            ordinal = u16(ordinal_array + 2 * index)
            if ordinal >= functions:
                raise ValueError('invalid PE export ordinal')
            function_rva = u32(function_array + 4 * ordinal)
            if name in REQUIRED_EXPORTS:
                exports[name] = {'function_rva': function_rva, 'forwarded': export_rva <= function_rva < export_rva + export_size}
    import_rva = u32(optional + 120)
    imports = []
    if import_rva:
        index = offset(import_rva)
        while any(blob[index:index + 20]):
            imports.append(cstr(u32(index + 12)))
            index += 20
    return {'machine': 'x86_64', 'required_sampler_exports': exports, 'imports': imports, 'read_only_PE_inspection': True, 'DLL_loaded': False}



def _loader_bindings(identity):
    loader = identity.get('loader_r4')
    if not isinstance(loader, dict):
        raise ValueError('r4 loader binding required')
    runtime = loader.get('runtime_modules')
    cuda = loader.get('cuda_dependency_modules')
    if not isinstance(runtime, list) or not isinstance(cuda, list) or len(runtime) != 5 or len(cuda) != 3:
        raise ValueError('five runtime and three CUDA dependency bindings required')
    expected_runtime = ('llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll')
    expected_cuda = ('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll')
    if tuple(Path(item['path']).name for item in runtime) != expected_runtime or tuple(Path(item['path']).name for item in cuda) != expected_cuda:
        raise ValueError('loader module names/order mismatch')
    runtime_dir = Path(loader.get('runtime_directory', '')).resolve(strict=True)
    cuda_dir = Path(loader.get('cuda_directory', '')).resolve(strict=True)
    if any(Path(item['path']).resolve().parent != runtime_dir for item in runtime) or any(Path(item['path']).resolve().parent != cuda_dir for item in cuda):
        raise ValueError('loader module lies outside frozen directory')
    proof = verify(loader['r18_setup_proof_ref'])
    setup = None
    for line in Path(proof['path']).read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        if record.get('record') == 'setup':
            setup = record
            break
    if setup is None:
        raise ValueError('R18 setup proof record missing')
    loaded = {item.get('name'): item for item in setup.get('loaded_modules_before', [])}
    for item in [*runtime[1:], *cuda]:
        expected = verify(item)
        observed = loaded.get(Path(item['path']).name)
        if not isinstance(observed, dict) or Path(observed.get('path', '')).resolve() != Path(expected['path']).resolve() or observed.get('sha256') != expected['sha256']:
            raise ValueError('R18 setup proof does not bind ggml/CUDA loader module: ' + Path(item['path']).name)
    order = loader.get('dependency_load_order')
    if order != ['cudart64_12.dll', 'cublasLt64_12.dll', 'cublas64_12.dll', 'ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll', 'ggml.dll', 'llama.dll']:
        raise ValueError('frozen dependency load order mismatch')
    return {'runtime_directory': str(runtime_dir), 'cuda_directory': str(cuda_dir), 'runtime_modules': runtime, 'cuda_dependency_modules': cuda, 'dependency_load_order': order, 'r18_setup_proof_ref': proof}


def _expected_loaded_module_paths(loader):
    return {Path(item['path']).name: str(Path(item['path']).resolve()) for item in [*loader['runtime_modules'], *loader['cuda_dependency_modules']]}

def inputs():
    identity = json.loads(IDENTITY_FILE.read_text(encoding='utf-8'))
    protocol = json.loads((P / 'protocol.json').read_text(encoding='utf-8'))
    refs = [verify(identity['source_policy_contract'])]
    refs.extend(verify(item) for item in identity['native_sources'].values())
    refs.extend(verify(item) for item in identity['native_runtime_modules'])
    for path, digest in identity['loader_dependency_sha256'].items():
        refs.append(verify({'path': path, 'sha256': digest}))
    lineage = identity['historical_header_build_lineage']
    refs.extend(verify(item) for item in lineage['receipts'])
    refs.extend(verify(item['snapshot_ref']) for item in lineage['historical_header_bindings'])
    loader = _loader_bindings(identity)
    refs.extend(verify(item) for item in [*loader['runtime_modules'], *loader['cuda_dependency_modules']])
    refs.append(loader['r18_setup_proof_ref'])
    native = Path(next(item['path'] for item in loader['runtime_modules'] if Path(item['path']).name == 'llama.dll'))
    pe = pe_identity(native)
    if set(pe['required_sampler_exports']) != set(REQUIRED_EXPORTS) or any(value['forwarded'] for value in pe['required_sampler_exports'].values()):
        raise ValueError('original DLL sampler exports unavailable/forwarded')
    source = (P / 'cpu_reset_probe.cpp').read_text(encoding='utf-8')
    common = Path(identity['native_sources']['common_sampling']['path']).read_text(encoding='utf-8')
    normalized = lambda value: re.sub(r'\s+', '', value)
    body = identity['source_loop_origin']['body']
    statements = [body[:body.index('cur_p =')], body[body.index('cur_p ='):]]
    if not all(normalized(part) in normalized(common) and normalized(part) in normalized(source) for part in statements):
        raise ValueError('candidate loop no longer matches locked common source statements')
    for literal in ('VOCABS{32768, 131072, 262144}', 'WARMUP_CALLS = 16, STEADY_REPEATS = 64, PROCESS_COUNT = 6', 'INPUT_SEED = 20260916', 'cur_p = { cur.data(), cur.size(), -1, false };'):
        if literal not in source:
            raise ValueError('source/protocol constant mismatch')
    if protocol['vocabulary_sizes'] != [32768, 131072, 262144] or protocol['processes'] != 6 or protocol['steady_raw_repeats_per_stage_case_process'] != 64:
        raise ValueError('protocol domain changed')
    if protocol.get('loader_dependency_policy', {}).get('environment', {}).get('LLAMA_TRACE_ANNOTATIONS') != '0':
        raise ValueError('explicit annotation environment policy missing')
    if protocol['reset_ab']['process_plan'] != process_plan():
        raise ValueError('preregistered alternating arm plan changed')
    refs.extend(ref(P / name) for name in ('cpu_reset_probe.cpp', 'entry.py', 'protocol.json', IDENTITY_FILE.name))
    return identity, protocol, list({item['path']: item for item in refs}.values()), pe, native, loader


def compile_source(root_idle, root_authorized):
    if not root_idle or not root_authorized:
        raise ValueError('root compile authorization and idle confirmation required')
    identity, protocol, before, pe, native, loader = inputs()
    if (P / 'build_manifest.json').exists():
        raise ValueError('build already frozen; new revision needed')
    attempt = 1 + len(list(P.glob('compile.*.log')))
    stem = f'compile.{attempt:04d}'
    exe, obj, asm = P / f'cpu-reset-probe.{attempt:04d}.exe', P / f'cpu-reset-probe.{attempt:04d}.obj', P / f'cpu-reset-probe.{attempt:04d}.asm'
    if any(path.exists() for path in (exe, obj, asm)):
        raise ValueError('refuse existing compile attempt')
    command = '"' + str(CL) + '" /nologo /showIncludes /std:c++17 /EHsc /O2 /fp:strict /MD /utf-8 /GL- /FAs ' + ' '.join('/I"' + str(path) + '"' for path in INCLUDES) + ' "' + str(P / 'cpu_reset_probe.cpp') + '" /Fe:"' + str(exe) + '" /Fo:"' + str(obj) + '" /Fa"' + str(asm) + '" /link PowrProf.lib'
    cmd, log = P / (stem + '.cmd'), P / (stem + '.log')
    with cmd.open('x', encoding='utf-8') as handle:
        handle.write('@echo off\ncall "' + str(VS) + '" >nul\nif errorlevel 1 exit /b %errorlevel%\n' + command + '\n')
    with (P / (stem + '.source.cpp')).open('xb') as handle:
        handle.write((P / 'cpu_reset_probe.cpp').read_bytes())
    with log.open('xb') as output:
        result = subprocess.run(['cmd.exe', '/c', str(cmd)], cwd=P, stdout=output, stderr=subprocess.STDOUT, check=False, timeout=180)
    if result.returncode:
        raise RuntimeError('compile failed; preserve ' + str(log))
    if inputs()[2] != before:
        raise ValueError('compile input drift')
    raw = log.read_bytes()
    try:
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('gb18030')
    headers = set()
    for line in text.splitlines():
        match = re.search(r'^.*?:.*?:[ \t]+([A-Za-z]:[\\/].*)$', line) or re.search(r'including file:[ \t]+(.*)$', line)
        if match:
            headers.add(Path(match.group(1).strip()))
    if len(headers) < 30:
        raise ValueError('compile header inventory incomplete')
    build = {'schema': 'cpu-reset-probe-build/v4', 'inputs': before, 'compiler': ref(CL), 'environment_setup': ref(VS), 'command': command, 'log': ref(log), 'exe': ref(exe), 'object': ref(obj), 'assembly_listing': ref(asm), 'compile_headers': [ref(path) for path in sorted(headers)], 'native_exports': pe, 'loader_binding': loader, 'probe_executed': False, 'native_DLL_rebuilt': False, 'candidate_binary_equivalence_proven': False}
    write_new(P / 'build_manifest.json', build)
    print(json.dumps({'compiled': True, 'exe': build['exe'], 'headers': len(headers), 'probe_executed': False}))


def verified_build():
    identity, protocol, before, pe, native, loader = inputs()
    build = json.loads((P / 'build_manifest.json').read_text(encoding='utf-8'))
    if build['inputs'] != before or build.get('loader_binding') != loader:
        raise ValueError('frozen probe build inputs/loader binding changed')
    for key in ('compiler', 'environment_setup', 'exe', 'object', 'assembly_listing'):
        verify(build[key])
    for item in build['compile_headers']:
        verify(item)
    return build, protocol, native, loader


def _identity_from_document(value, cpu):
    if not isinstance(value, dict) or set(IDENTITY_KEYS) - set(value):
        raise ValueError('actual CPU identity fields missing')
    if not isinstance(value['cpu_brand'], str) or not value['cpu_brand'].strip():
        raise ValueError('CPU brand missing')
    for key in IDENTITY_KEYS[1:]:
        if type(value[key]) is not int or value[key] < 0:
            raise ValueError('invalid CPU identity field: ' + key)
    if value['active_processor_group_count'] != 1 or value['group'] != 0 or value['logical_cpu'] != cpu:
        raise ValueError('unsupported or wrong CPU identity topology')
    if value['active_processor_count_group0'] <= cpu or value['active_processor_count_group0'] > 64:
        raise ValueError('CPU identity count cannot support selected logical CPU')
    if value['thread_affinity_mask'] != 1 << cpu:
        raise ValueError('actual thread affinity differs from selected CPU')
    return {key: value[key] for key in IDENTITY_KEYS}


def _stage_frequency_state(stage, expected_identity):
    before, after = stage.get('cpu_before'), stage.get('cpu_after')
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise ValueError('stage CPU observations missing')
    for observation in (before, after):
        for key in ('group', 'logical_processor', 'thread_affinity_mask'):
            if type(observation.get(key)) is not int or observation[key] < 0:
                raise ValueError('invalid stage CPU observation: ' + key)
        for key in ('os_reported_current_mhz', 'os_max_mhz', 'os_limit_mhz'):
            if type(observation.get(key)) is not int or observation[key] <= 0:
                raise ValueError('frequency observation unavailable or nonpositive: ' + key)
        if observation['group'] != expected_identity['group'] or observation['logical_processor'] != expected_identity['logical_cpu'] or observation['thread_affinity_mask'] != expected_identity['thread_affinity_mask']:
            raise ValueError('stage CPU affinity identity drift')
    fields = ['os_max_mhz', 'os_reported_current_mhz', 'os_limit_mhz']
    stable = all(before[key] == after[key] for key in fields)
    if stage.get('reported_frequency_fields_checked') != fields or stage.get('reported_frequency_stable') is not stable or stage.get('timing_usable') is not stable or stage.get('diagnostic_only') is stable or stage.get('frequency_changed_diagnostic_only') is stable:
        raise ValueError('stage frequency status inconsistent with all reported fields')
    return stable


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError('percentile requires samples')
    rank = (len(ordered) - 1) * fraction
    lower, upper = int(math.floor(rank)), int(math.ceil(rank))
    return float(ordered[lower]) if lower == upper else float(ordered[lower]) + (float(ordered[upper]) - float(ordered[lower])) * (rank - lower)


def _stage_timing_quality(stage, observer_ticks, case_key):
    ticks = stage['steady_raw_ticks']
    steady_median = float(statistics.median(ticks))
    p10, p90 = _percentile(ticks, 0.10), _percentile(ticks, 0.90)
    observer_median = float(statistics.median(observer_ticks))
    reasons = []
    if not math.isfinite(steady_median) or steady_median <= 0:
        reasons.append('steady_median_nonpositive')
    if not math.isfinite(p10) or p10 <= 0 or not math.isfinite(p90) or p90 / p10 > 1.5:
        reasons.append('steady_p90_p10_ratio_exceeds_1_5')
    if not math.isfinite(observer_median) or observer_median < 0 or steady_median <= 0 or observer_median / steady_median > 0.01:
        reasons.append('observer_median_over_steady_median_exceeds_1_percent')
    return {'case_key': case_key, 'stage': stage['stage'], 'steady_median_ticks': steady_median, 'steady_p10_ticks': p10, 'steady_p90_ticks': p90, 'steady_p90_p10_ratio': None if p10 <= 0 else p90 / p10, 'observer_median_ticks': observer_median, 'observer_to_steady_median_ratio': None if steady_median <= 0 else observer_median / steady_median, 'steady_dispersion_pass': 'steady_p90_p10_ratio_exceeds_1_5' not in reasons, 'observer_overhead_pass': 'observer_median_over_steady_median_exceeds_1_percent' not in reasons, 'timing_usable': not reasons, 'diagnostic_only': bool(reasons), 'reasons': reasons}


def process_plan():
    return [{'process_index':i, 'reset_policy':('memcpy_baseline' if i < 3 else 'source_candidate_loop')} for i in (0,3,1,4,2,5)]


def validate_result(doc, index, cpu, expected_identity, loader=None):
    expected_arm = next((item['reset_policy'] for item in process_plan() if item['process_index']==index), None)
    if expected_arm is None or doc.get('schema') != 'cpu-reset-independent-probe/v1':
        raise ValueError('unregistered process or measurement schema')
    if doc.get('status') != 'complete' or doc.get('process_index') != index or doc.get('logical_cpu') != cpu:
        raise ValueError('process identity or status mismatch')
    if doc.get('GPU_context_created') is not False or doc.get('model_loaded') is not False or doc.get('full_sampler_chain_measured') is not False:
        raise ValueError('probe scope mismatch')
    actual_identity = _identity_from_document(doc.get('actual_cpu_identity'), cpu)
    if actual_identity != expected_identity:
        raise ValueError('actual CPU identity differs from frozen identity')
    if loader is not None:
        expected_paths = _expected_loaded_module_paths(loader)
        actual_modules = doc.get('loaded_native_modules')
        if not isinstance(actual_modules, list) or {item.get('name') for item in actual_modules} != set(expected_paths):
            raise ValueError('loaded module set differs from frozen loader binding')
        for item in actual_modules:
            if Path(item.get('actual_path', '')).resolve() != Path(expected_paths[item['name']]).resolve():
                raise ValueError('loaded module path differs from frozen loader binding: ' + item['name'])
        if doc.get('dependency_dlls_loaded') is not True or doc.get('dependency_loading_is_not_GPU_measured') is not True or doc.get('no_probe_gpu_api_calls') is not True:
            raise ValueError('dependency/GPU scope flags missing')
    if type(doc.get('qpc_frequency')) is not int or doc['qpc_frequency'] <= 0:
        raise ValueError('QPC frequency missing')
    observer = doc.get('observer_empty_bracket_ticks', [])
    if len(observer) != 64 or any(type(value) is not int or value < 0 for value in observer):
        raise ValueError('observer raw repeats incomplete')
    expected = {(vocabulary, pattern) for vocabulary in (32768, 131072, 262144) for pattern in ('monotone_ascending', 'deterministic_random_permutation')}
    cases = doc.get('cases', [])
    if len(cases) != 6 or {(case['vocabulary_size'], case['pattern']) for case in cases} != expected:
        raise ValueError('six unique cases required')
    stable_stages, stage_quality = 0, []
    for case in cases:
        if case.get('split') != 'quality_diagnosis':
            raise ValueError('all sizes are disclosed diagnostic data')
        if case.get('reset_policy') != expected_arm or case.get('reset_outside_clock_window') is not True:
            raise ValueError('wrong arm or reset timing boundary')
        if case.get('candidate_record_bytes') != 12 or case.get('top_k') != 1 or case.get('first_use_not_pooled_with_steady') is not True:
            raise ValueError('numeric contract drift')
        if len(case['stages']) != 1 or case['stages'][0]['stage'] != 'original_dll_topk_apply':
            raise ValueError('independent stage records missing')
        case_key = (expected_arm, case['vocabulary_size'], case['pattern'])
        for stage in case['stages']:
            if stage.get('numeric_quality') != 'exact_pass' or stage.get('warmup_calls') != 16 or stage.get('steady_repeats') != 64:
                raise ValueError('numeric/loop gate failed')
            ticks = stage.get('steady_raw_ticks', [])
            if len(ticks) != 64 or any(type(value) is not int or value < 0 for value in ticks) or type(stage.get('first_use_ticks')) is not int or stage['first_use_ticks'] < 0:
                raise ValueError('raw sample record invalid')
            stable_stages += int(_stage_frequency_state(stage, expected_identity))
            stage_quality.append(_stage_timing_quality(stage, observer, case_key))
    return {'stable_stages': stable_stages, 'total_stages': 6, 'stage_quality': stage_quality, 'case_process_records': 6}


def local_os_correlation():
    return {'os_name': os.name, 'platform': platform.platform(aliased=True), 'python_os_cpu_count': os.cpu_count(), 'used_as_expected_hardware_identity': False}


def capture_identity(args):
    if not args.root_idle_confirmed or not args.root_identity_authorized:
        raise ValueError('root identity authorization and idle confirmation required')
    if args.cpu_index is None or not 0 <= args.cpu_index < 64:
        raise ValueError('explicit logical CPU index0..63 required')
    if not args.identity_freeze or re.fullmatch(r'[a-zA-Z0-9_-]+', args.identity_freeze) is None:
        raise ValueError('new safe identity freeze name required')
    build, protocol, native, loader = verified_build()
    destination = P / args.identity_freeze
    destination.mkdir(exist_ok=False)
    result, stdout, stderr = destination / 'identity.json', destination / 'identity.stdout.txt', destination / 'identity.stderr.txt'
    argv = [build['exe']['path'], '--identity-root-approved', '--identity-only', '--cpu-index', str(args.cpu_index), '--output', str(result)]
    with stdout.open('xb') as out, stderr.open('xb') as err:
        completed = subprocess.run(argv, cwd=P, stdout=out, stderr=err, check=False, timeout=60)
    receipt = {'argv': argv, 'explicit_environment': {'LLAMA_TRACE_ANNOTATIONS': '0'}, 'returncode': completed.returncode, 'stdout_ref': ref(stdout), 'stderr_ref': ref(stderr), 'result_ref': ref(result) if result.exists() else {'status': 'missing'}}
    write_new(destination / 'identity.receipt.json', receipt)
    verified_build()
    if completed.returncode:
        raise RuntimeError('identity capture failed; preserve receipt and do not measure')
    document = json.loads(result.read_text(encoding='utf-8'))
    if document.get('schema') != 'cpu-reset-probe-actual-cpu-identity/v1' or document.get('status') != 'complete' or document.get('GPU_context_created') is not False or document.get('model_loaded') is not False:
        raise ValueError('identity-only scope/result mismatch')
    actual = _identity_from_document(document.get('actual_cpu_identity'), args.cpu_index)
    frozen = {'schema': 'cpu-reset-probe-actual-cpu-freeze/v1', 'build_manifest_ref': ref(P / 'build_manifest.json'), 'protocol_ref': ref(P / 'protocol.json'), 'identity_result_ref': ref(result), 'cpu_index': args.cpu_index, 'actual_cpu_identity': actual, 'local_os_correlation': local_os_correlation(), 'local_os_correlation_is_not_expected_hardware_identity': True, 'GPU_executed': False, 'model_loaded': False, 'timing_values_produced': False}
    write_new(destination / 'actual_cpu_identity_freeze.json', frozen)
    print(json.dumps({'identity_freeze': str(destination / 'actual_cpu_identity_freeze.json'), 'cpu_index': args.cpu_index, 'timing_values_produced': False}))


def read_identity_freeze(path, cpu):
    path = Path(path).resolve(strict=True)
    document = json.loads(path.read_text(encoding='utf-8'))
    if document.get('schema') != 'cpu-reset-probe-actual-cpu-freeze/v1' or document.get('cpu_index') != cpu:
        raise ValueError('wrong CPU identity freeze')
    if document.get('build_manifest_ref') != ref(P / 'build_manifest.json') or document.get('protocol_ref') != ref(P / 'protocol.json'):
        raise ValueError('CPU identity freeze belongs to a different build or protocol')
    if document.get('local_os_correlation_is_not_expected_hardware_identity') is not True:
        raise ValueError('local OS correlation must not become expected hardware identity')
    return _identity_from_document(document.get('actual_cpu_identity'), cpu), ref(path)


def summarize_quality(results, identity_ref):
    if len(results) != 6 or len({item['document']['process_id'] for item in results}) != 6 or [item['document']['process_index'] for item in results] != [p['process_index'] for p in process_plan()]:
        raise ValueError('six independent alternating-arm process records required for quality')
    stable = sum(item['frequency']['stable_stages'] for item in results)
    total = sum(item['frequency']['total_stages'] for item in results)
    case_process_total = sum(item['frequency']['case_process_records'] for item in results)
    stages = [quality for item in results for quality in item['frequency']['stage_quality']]
    if total != 36 or len(stages) != 36 or case_process_total != 36:
        raise ValueError('all 36 stage records and 36 case-process records required for quality')
    groups = {}
    for stage in stages:
        groups.setdefault((stage['case_key'], stage['stage']), []).append(stage)
    cross_process = []
    for key, items in sorted(groups.items()):
        if len(items) != 3:
            raise ValueError('three process medians required for every case/stage')
        medians = [item['steady_median_ticks'] for item in items]
        reference = float(statistics.median(medians))
        deviation = None if reference <= 0 else max(abs(value - reference) for value in medians) / reference
        passed = reference > 0 and math.isfinite(reference) and deviation is not None and math.isfinite(deviation) and deviation <= 0.05
        cross_process.append({'case_key': key, 'stage': key[1], 'process_medians_ticks': medians, 'median_of_process_medians_ticks': reference, 'max_abs_deviation_over_median': deviation, 'pass': passed})
    if len(groups) != 12:
        raise ValueError('12 arm/case groups required')
    frequency_ok = stable == total
    steady_ok = all(stage['steady_dispersion_pass'] for stage in stages)
    observer_ok = all(stage['observer_overhead_pass'] for stage in stages)
    cross_ok = all(item['pass'] for item in cross_process)
    timing_usable = frequency_ok and steady_ok and observer_ok and cross_ok
    reasons = []
    if not frequency_ok: reasons.append('frequency_drift')
    if not steady_ok: reasons.append('steady_dispersion')
    if not observer_ok: reasons.append('observer_overhead')
    if not cross_ok: reasons.append('cross_process_median_deviation')
    return {'schema': 'cpu-reset-probe-quality/v1', 'exact_numeric_and_structure_pass': True, 'process_count': 6, 'case_process_records_total': case_process_total, 'stage_records_total': total, 'steady_samples': 2304, 'identity_freeze_ref': identity_ref, 'identity_consistent': True, 'frequency_stable_stages': stable, 'frequency_changed_stages': total - stable, 'steady_dispersion_pass_stages': sum(stage['steady_dispersion_pass'] for stage in stages), 'observer_overhead_pass_stages': sum(stage['observer_overhead_pass'] for stage in stages), 'cross_process_case_stage_groups_total': len(cross_process), 'cross_process_case_stage_groups_pass': sum(item['pass'] for item in cross_process), 'cross_process_medians': cross_process, 'timing_usable': timing_usable, 'diagnostic_only': not timing_usable, 'accepted_for_timing_evidence': timing_usable, 'timing_quality_failure_reasons': reasons, 'fit_performed': False, 'holdout_used_for_fit': False}


def compiled_observation_schema(build):
    # Validate the C++ emitter before any timing series can start. The fixture
    # never loads the native DLL, pins a CPU or records performance values.
    completed = subprocess.run([build['exe']['path'], '--observation-schema-fixture'],
                               check=True, capture_output=True, text=True)
    fixture = json.loads(completed.stdout)
    if fixture.get('schema') != 'cpu-observation-schema-fixture/v1' or fixture.get('synthetic') is not True or fixture.get('DLL_loaded') is not False or fixture.get('GPU_calls') is not False:
        raise ValueError('compiled observation schema fixture scope mismatch')
    observation = fixture.get('cpu_observation')
    if not isinstance(observation, dict) or any(type(observation.get(k)) is not int for k in ('group', 'logical_processor', 'thread_affinity_mask', 'os_max_mhz', 'os_reported_current_mhz', 'os_limit_mhz')):
        raise ValueError('compiled observation serializer differs from integer schema')
    expected = {'group':0, 'logical_cpu':0, 'thread_affinity_mask':1}
    stage = {'cpu_before':observation, 'cpu_after':observation,
             'reported_frequency_fields_checked':['os_max_mhz','os_reported_current_mhz','os_limit_mhz'],
             'reported_frequency_stable':True, 'timing_usable':True,
             'diagnostic_only':False, 'frequency_changed_diagnostic_only':False}
    if _stage_frequency_state(stage, expected) is not True:
        raise ValueError('compiled observation fixture failed validator')
    return {'fixture':fixture, 'exe_ref':build['exe'], 'verified_before_series':True,
            'synthetic_not_hardware_evidence':True}


def measure(args):
    if not args.root_idle_confirmed or not args.root_measure_authorized:
        raise ValueError('root measurement authorization and idle confirmation required')
    if args.cpu_index is None or not 0 <= args.cpu_index < 64:
        raise ValueError('explicit logical CPU index0..63 required')
    if not args.series or re.fullmatch(r'[a-zA-Z0-9_-]+', args.series) is None:
        raise ValueError('new safe series name required')
    if args.cpu_identity_freeze is None:
        raise ValueError('frozen actual CPU identity required')
    expected_identity, identity_ref = read_identity_freeze(args.cpu_identity_freeze, args.cpu_index)
    build, protocol, native, loader = verified_build()
    schema_validation = compiled_observation_schema(build)
    if list(P.glob('*/run_freeze.json')):
        raise ValueError('one preregistered series only; no retry after observation')
    destination = P / args.series
    destination.mkdir(exist_ok=False)
    plan = {'schema': 'cpu-reset-probe-run-freeze/v4', 'build_manifest_ref': ref(P / 'build_manifest.json'), 'protocol_ref': ref(P / 'protocol.json'), 'actual_cpu_identity_freeze_ref': identity_ref, 'actual_cpu_identity': expected_identity, 'local_os_correlation_is_not_expected_hardware_identity': True, 'cpu_index': args.cpu_index, 'process_indices': [p['process_index'] for p in process_plan()], 'process_plan': process_plan(), 'case_order': protocol['case_order'], 'holdout_vocabulary_sizes': [], 'all_sizes_quality_diagnosis_only': True, 'allocation_and_reset_semantics': protocol['stages'], 'clock_policy': protocol['clock'], 'loops': {'warmup': 16, 'steady': 64}, 'loader_binding': loader, 'runtime_environment': {'LLAMA_TRACE_ANNOTATIONS': '0'}, 'root_authorized': True, 'compiled_schema_validation':schema_validation, 'no_auto_retry': True}
    write_new(destination / 'run_freeze.json', plan)
    results = []
    for arm_item in process_plan():
        index = arm_item['process_index']
        verified_build()
        result, stdout, stderr = destination / f'process_{index}.json', destination / f'process_{index}.stdout.txt', destination / f'process_{index}.stderr.txt'
        argv = [build['exe']['path'], '--measure-root-approved', '--native-dll', str(native), '--native-runtime-dir', loader['runtime_directory'], '--cuda-dependency-dir', loader['cuda_directory'], '--reset-policy', arm_item['reset_policy'], '--process-index', str(index), '--cpu-index', str(args.cpu_index), '--output', str(result)]
        child_environment = {**os.environ, 'LLAMA_TRACE_ANNOTATIONS': '0'}
        with stdout.open('xb') as out, stderr.open('xb') as err:
            with subprocess.Popen(argv, cwd=P, stdout=out, stderr=err, env=child_environment) as child:
                child_code = child.wait()
                launched_pid = child.pid
        receipt = {'argv': argv, 'explicit_environment': {'LLAMA_TRACE_ANNOTATIONS': '0'}, 'returncode': child_code, 'stdout_ref': ref(stdout), 'stderr_ref': ref(stderr), 'launched_pid': launched_pid, 'reset_policy': arm_item['reset_policy'], 'run_freeze_ref': ref(destination / 'run_freeze.json'), 'result_ref': ref(result) if result.exists() else {'status': 'missing'}}
        write_new(destination / f'process_{index}.receipt.json', receipt)
        verified_build()
        if child_code:
            raise RuntimeError('probe process failed; no retry or replacement in this series')
        document = json.loads(result.read_text(encoding='utf-8'))
        if document.get('process_id') != launched_pid:
            raise ValueError('raw process PID differs from launcher')
        results.append({'document': document, 'frequency': validate_result(document, index, args.cpu_index, expected_identity, loader)})
    if len({item['document']['process_id'] for item in results}) != 6:
        raise ValueError('independent process identity check failed')
    quality = summarize_quality(results, identity_ref)
    quality['result_refs'] = [ref(destination / f'process_{index}.json') for index in (0,3,1,4,2,5)]
    write_new(destination / 'quality.json', quality)
    if not quality['timing_usable']:
        raise RuntimeError('timing-quality gate failure preserved as diagnostic-only; series is not accepted for timing evidence: ' + ','.join(quality['timing_quality_failure_reasons']))
    print(json.dumps({'series': str(destination), 'numeric_quality_pass': True, 'timing_usable': True, 'fit_performed': False}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('inspect-r4', 'compile', 'capture-identity', 'measure'))
    parser.add_argument('--root-idle-confirmed', action='store_true')
    parser.add_argument('--root-compile-authorized', action='store_true')
    parser.add_argument('--root-identity-authorized', action='store_true')
    parser.add_argument('--root-measure-authorized', action='store_true')
    parser.add_argument('--cpu-index', type=int)
    parser.add_argument('--series')
    parser.add_argument('--identity-freeze')
    parser.add_argument('--cpu-identity-freeze', type=Path)
    args = parser.parse_args()
    if args.mode == 'inspect-r4':
        identity, protocol, refs, pe, native, loader = inputs()
        report = {'schema': 'cpu-reset-probe-readonly-inspection/v4', 'source_equivalent_loop_statements_checked': True, 'native_PE': pe, 'input_refs': refs, 'identity_freeze_required_before_measure': True, 'all_frequency_fields_gate_timing_use': True, 'loader_binding': loader, 'GPU_executed': False, 'DLL_loaded': False, 'compiled': False, 'probe_executed': False}
        write_new(P / 'readonly_inspection_r4.json', report)
        print(json.dumps({'source_statements_checked': True, 'PE_exports': pe, 'compiled': False, 'probe_executed': False}))
    elif args.mode == 'compile':
        compile_source(args.root_idle_confirmed, args.root_compile_authorized)
    elif args.mode == 'capture-identity':
        capture_identity(args)
    else:
        measure(args)


if __name__ == '__main__':
    main()



