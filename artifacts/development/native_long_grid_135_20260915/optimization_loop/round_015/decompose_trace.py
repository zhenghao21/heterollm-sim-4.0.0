"""Read-only decomposition of the frozen 24-call Nsight Systems trace; no execution."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics

PROCESS_MASK = 0xFFFFFFFFFF000000
TABLES = {
    'nvtx': ('NVTX_EVENTS', {'start', 'end', 'globalTid', 'text'}),
    'cuda_kernels': ('CUPTI_ACTIVITY_KIND_KERNEL', {'start', 'end', 'globalPid', 'correlationId', 'deviceId', 'streamId'}),
    'cuda_runtime': ('CUPTI_ACTIVITY_KIND_RUNTIME', {'start', 'end', 'globalTid', 'correlationId', 'nameId', 'returnValue', 'eventClass'}),
    'cuda_sync': ('CUPTI_ACTIVITY_KIND_SYNCHRONIZATION', {'start', 'end', 'globalPid', 'correlationId', 'streamId', 'syncType'}),
    'cuda_memcpy': ('CUPTI_ACTIVITY_KIND_MEMCPY', {'start', 'end', 'globalPid', 'correlationId', 'deviceId'}),
}


def file_ref(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def intervals_union(intervals):
    """Disjoint half-open intervals, without double-counting overlaps or touching edges."""
    values = []
    for start, end in intervals:
        if type(start) is not int or type(end) is not int or end < start:
            raise ValueError('invalid integer interval')
        if end > start:
            values.append((start, end))
    merged = []
    for start, end in sorted(values):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def union_ns(intervals):
    return sum(end - start for start, end in intervals_union(intervals))


def clipped(intervals, begin, end):
    return intervals_union((max(start, begin), min(stop, end))
                           for start, stop in intervals if start < end and stop > begin)


def intersection_ns(left, right):
    a, b = intervals_union(left), intervals_union(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def gaps(intervals, begin, end):
    cursor = begin
    result = []
    for start, stop in clipped(intervals, begin, end):
        if start > cursor:
            result.append((cursor, start))
        cursor = stop
    if cursor < end:
        result.append((cursor, end))
    return result


def event_intervals(events):
    return [(row['start'], row['end']) for row in events]


def stats(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'minimum_ns': values[0], 'median_ns': statistics.median(values),
            'p95_nearest_rank_ns': values[math.ceil(.95 * len(values)) - 1],
            'maximum_ns': values[-1], 'mean_ns': statistics.mean(values)}


def correlation_key(row, host=False):
    return ((row['globalTid'] & PROCESS_MASK) if host else row['globalPid'], row['correlationId'])


def verify_rows(raw, sql_rows, label):
    by_id = {row['evidence_rowid']: row for row in raw}
    if len(by_id) != len(raw) or len(raw) != len(sql_rows):
        raise ValueError('row coverage mismatch: ' + label)
    for row in sql_rows:
        supplied = by_id.get(row['evidence_rowid'], {})
        if any(supplied.get(key) != value for key, value in row.items()):
            raise ValueError('row differs from SQLite: ' + label)


def decompose_call(call, raw, strings, event_classes):
    marker = call['nvtx_range']
    begin, end, tid = marker['start'], marker['end'], marker['globalTid']
    pid = tid & PROCESS_MASK
    source_nvtx = next((r for r in raw['nvtx'] if r['evidence_rowid'] == marker['evidence_rowid']), None)
    if source_nvtx is None or any(source_nvtx[k] != marker[k] for k in ('start', 'end', 'globalTid')):
        raise ValueError('mapped NVTX differs from SQLite/raw')
    apis = [r for r in raw['cuda_runtime'] if r['globalTid'] == tid and begin <= r['start'] and r['end'] <= end]
    if {r['evidence_rowid'] for r in apis} != {r['evidence_rowid'] for r in call['runtime_api_events']}:
        raise ValueError('mapped API coverage differs from trace')
    raw_api = {r['evidence_rowid']: r for r in apis}
    if any(raw_api[r['evidence_rowid']] != r for r in call['runtime_api_events']):
        raise ValueError('mapped API differs from raw')
    by_correlation = defaultdict(list)
    for api in apis:
        by_correlation[correlation_key(api, host=True)].append(api)
    kernels = [k for k in raw['cuda_kernels'] if correlation_key(k) in by_correlation]
    if {k['evidence_rowid'] for k in kernels} != {k['evidence_rowid'] for k in call['kernel_events']}:
        raise ValueError('mapped kernel correlation differs from trace')
    if len(kernels) != 3 or any(k['start'] < begin or k['end'] > end for k in kernels):
        raise ValueError('expected three contained kernels per fixed invocation')
    raw_kernel = {k['evidence_rowid']: k for k in kernels}
    if any(raw_kernel[k['evidence_rowid']] != k for k in call['kernel_events']):
        raise ValueError('mapped kernel differs from raw')
    launches = [a for a in apis if 'LaunchKernel' in strings[a['nameId']]]
    sync = [a for a in apis if 'Synchronize' in strings[a['nameId']]]
    others = [a for a in apis if a not in launches and a not in sync]
    if len(launches) != 3 or len(sync) != 1:
        raise ValueError('unexpected launch/synchronization call count')
    correlated_sync = [s for s in raw['cuda_sync'] if correlation_key(s) in {correlation_key(a, True) for a in sync}]
    if len(correlated_sync) != len(sync):
        raise ValueError('synchronization activity correlation incomplete/ambiguous')
    kernel_intervals = event_intervals(kernels)
    first, last = min(k['start'] for k in kernels), max(k['end'] for k in kernels)
    merged = intervals_union(kernel_intervals)
    gap_intervals = gaps(kernel_intervals, first, last)
    launch_intervals, sync_intervals = event_intervals(launches), event_intervals(sync)
    api_intervals = event_intervals(apis)
    prefix, tail = [(begin, first)], [(last, end)]
    partition = {'before_first_kernel_ns': first - begin, 'mapped_kernel_union_ns': union_ns(kernel_intervals),
                 'between_mapped_kernels_ns': union_ns(gap_intervals), 'after_last_kernel_ns': end - last}
    if sum(partition.values()) != end - begin:
        raise ValueError('NVTX partition does not close')
    kernel_details = []
    for kernel in sorted(kernels, key=lambda x: (x['start'], x['evidence_rowid'])):
        launch = by_correlation[correlation_key(kernel)]
        if len(launch) != 1 or launch[0] not in launches:
            raise ValueError('kernel lacks unique launch API on NVTX thread/process')
        launch = launch[0]
        kernel_details.append({'kernel': kernel, 'launch_api': launch,
            'kernel_start_offset_from_nvtx_ns': kernel['start'] - begin,
            'kernel_end_offset_from_nvtx_ns': kernel['end'] - begin,
            'launch_api_duration_ns': launch['end'] - launch['start'],
            'launch_start_to_kernel_start_ns': kernel['start'] - launch['start'],
            'launch_return_to_kernel_start_signed_ns': kernel['start'] - launch['end'],
            'interval_interpretation': 'Observed aligned trace timestamps; not a pure queue, scheduler, CPU or profiler overhead measurement.'})
    devices = {k['deviceId'] for k in kernels}
    device_activity = [k for k in [*raw['cuda_kernels'], *raw['cuda_memcpy']] if k['deviceId'] in devices]
    external = [k for k in device_activity if not ('short_name_text' in k and k['evidence_rowid'] in raw_kernel)]
    clipped_api = clipped(api_intervals, begin, end)
    sync_union = union_ns(sync_intervals)
    sync_gpu = intersection_ns(sync_intervals, kernel_intervals)
    sync_prefix = intersection_ns(sync_intervals, prefix)
    sync_gaps = intersection_ns(sync_intervals, gap_intervals)
    sync_tail = intersection_ns(sync_intervals, tail)
    if sync_union != sync_prefix + sync_gpu + sync_gaps + sync_tail:
        raise ValueError('sync partition does not close')
    values = {'nvtx_duration_ns': end - begin, **partition, 'kernel_span_ns': last - first,
              'host_api_union_ns': union_ns(clipped_api), 'host_launch_api_union_ns': union_ns(launch_intervals),
              'host_sync_api_union_ns': sync_union, 'host_other_api_union_ns': union_ns(event_intervals(others)),
              'host_api_gpu_overlap_ns': intersection_ns(api_intervals, kernel_intervals),
              'host_sync_gpu_overlap_ns': sync_gpu, 'host_sync_before_first_kernel_ns': sync_prefix,
              'host_sync_between_kernels_ns': sync_gaps, 'host_sync_after_last_kernel_ns': sync_tail,
              'after_sync_to_nvtx_end_ns': end - max(a['end'] for a in sync),
              'captured_kernel_or_api_union_ns': union_ns([*kernel_intervals, *clipped_api]),
              'nvtx_uncovered_by_mapped_kernels_or_host_api_ns': union_ns(gaps([*kernel_intervals, *clipped_api], begin, end)),
              'captured_external_device_activity_in_nvtx_ns': union_ns(clipped(event_intervals(external), begin, end)),
              'captured_device_activity_inside_mapped_gaps_ns': intersection_ns(event_intervals(device_activity), gap_intervals),
              'correlated_sync_activity_union_ns': union_ns(event_intervals(correlated_sync))}
    return {'phase': call['phase'], 'index': call['index'], 'clock_domain': 'nsys_sqlite_nanoseconds',
            'nvtx_range': marker, 'globalPid_from_globalTid': pid, 'device_ids': sorted(devices),
            'stream_ids': sorted({k['streamId'] for k in kernels}), 'metrics': values,
            'nvtx_disjoint_partition_ns': partition,
            'inter_kernel_gaps': [{'start': a, 'end': b, 'duration_ns': b - a} for a, b in gap_intervals],
            'kernel_launch_pairs': kernel_details, 'host_api_events_verbatim': apis,
            'host_api_event_classes': {str(a['eventClass']): event_classes.get(a['eventClass']) for a in apis},
            'synchronization_activity_events_verbatim': correlated_sync,
            'original_qpc_record_verbatim_not_aligned_or_subtracted': call['qpc_derived_run_verbatim']}


def make_report(trace_root):
    trace_root = Path(trace_root).resolve(strict=True)
    paths = {'mapped': trace_root / 'mapped_calls.json', 'raw': trace_root / 'profile/raw_events.json',
             'sqlite': trace_root / 'profile/gemm.sqlite', 'summary': trace_root / 'result_summary.json',
             'protocol': trace_root / 'protocol.json', 'script': Path(__file__).resolve()}
    refs = {key: file_ref(path) for key, path in paths.items()}
    mapped, raw, summary = [json.loads(paths[key].read_text(encoding='utf-8')) for key in ('mapped', 'raw', 'summary')]
    attribution_path = Path(mapped['attribution_reference']['path'])
    refs['attribution_reference'] = file_ref(attribution_path)
    if refs['attribution_reference']['sha256'] != mapped['attribution_reference']['sha256']:
        raise ValueError('vendor correlation attribution source changed')
    for source in (mapped, raw, summary):
        if source['source_sqlite']['sha256'] != refs['sqlite']['sha256']:
            raise ValueError('SQLite input identity mismatch')
    if mapped['protocol']['sha256'] != refs['protocol']['sha256'] or summary['source_protocol']['sha256'] != refs['protocol']['sha256']:
        raise ValueError('protocol input identity mismatch')
    if summary['collection_status'] != 'completed' or summary['structural_extraction_status'] != 'passed':
        raise ValueError('requires completed structurally validated source')
    schema = {}
    with sqlite3.connect(paths['sqlite'].as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for key, (table, required) in TABLES.items():
            columns = [r['name'] for r in db.execute('PRAGMA table_info("' + table + '")')]
            if not required.issubset(columns):
                raise ValueError('required actual schema missing: ' + table)
            schema[table] = columns
            verify_rows(raw[key], [dict(r) for r in db.execute('SELECT rowid AS evidence_rowid,* FROM "' + table + '"')], table)
        strings = {r['id']: r['value'] for r in db.execute('SELECT * FROM StringIds')}
        classes = {r['id']: r['label'] for r in db.execute('SELECT * FROM ENUM_NSYS_EVENT_CLASS')}
        profiler_events = {t: [dict(r) for r in db.execute('SELECT rowid AS evidence_rowid,* FROM "' + t + '"')]
                           for t in ('CUPTI_ACTIVITY_KIND_OVERHEAD', 'PROFILER_OVERHEAD')}
    rows = [decompose_call(call, raw, strings, classes) for call in mapped['calls']]
    keys = [(row['phase'], row['index']) for row in rows]
    expected = {('first_call', 0), *(('warmup', i) for i in range(3)), *(('formal', i) for i in range(20))}
    if len(keys) != 24 or set(keys) != expected:
        raise ValueError('fixed 24-call scope mismatch')
    for row in rows:
        begin, end = row['nvtx_range']['start'], row['nvtx_range']['end']
        for table, events in profiler_events.items():
            matching = [e for e in events if e['globalTid'] == row['nvtx_range']['globalTid'] and e['start'] < end and e['end'] > begin]
            row[table.lower() + '_overlapping_events_verbatim'] = matching
            row['metrics'][table.lower() + '_within_nvtx_union_ns'] = union_ns(clipped(event_intervals(matching), begin, end))
    aggregate = {}
    for phase in ('all', 'first_call', 'warmup', 'formal'):
        members = rows if phase == 'all' else [r for r in rows if r['phase'] == phase]
        aggregate[phase] = {metric: stats([r['metrics'][metric] for r in members]) for metric in rows[0]['metrics']}
    kernel_names = defaultdict(list)
    for row in rows:
        if row['phase'] == 'formal':
            for pair in row['kernel_launch_pairs']:
                k = pair['kernel'];kernel_names[k['short_name_text']].append(k['end'] - k['start'])
    after = {key: file_ref(ref['path']) for key, ref in refs.items()}
    if refs != after:
        raise ValueError('source changed during read-only decomposition')
    return {'schema': 'synthetic-trace-interval-decomposition/v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'source_refs': refs, 'source_identity_unchanged_after_analysis': True,
            'scope': {'all_calls': 24, 'formal_calls': 20, 'mapped_kernels': 72, 'captured_kernels': len(raw['cuda_kernels']),
                      'nonmapped_kernels': len(raw['cuda_kernels']) - 72, 'captured_api_events': len(raw['cuda_runtime']),
                      'shape': summary['shape'], 'no_original_record_removed_or_modified': True},
            'actual_sqlite_schema_checked': schema,
            'correlation_rule': {'join': 'kernel.correlationId == api.correlationId AND kernel.globalPid == (api.globalTid & 0xFFFFFFFFFF000000)',
                                 'nvtx_ownership': 'API start/end contained in range and matching globalTid; kernel containment separately checked'},
            'timing_policy': {'interval_unit': 'ns', 'arithmetic_clock': 'one exported Nsight Systems SQLite timeline',
                              'application_qpc_clock_not_aligned': True, 'profile_control_calls_not_paired': True,
                              'host_gpu_durations_never_added': True, 'pure_host_overhead_inferred': False,
                              'gap_meaning': 'Absence of captured kernel/memcpy activity in this trace only; not proof of device-wide hardware idleness.'},
            'aggregates': aggregate, 'formal_kernel_duration_by_name': {k: stats(v) for k, v in kernel_names.items()},
            'profile_control_distribution_context_verbatim': {k: summary[k] for k in ('formal_host_profile', 'formal_host_direct_control', 'profiling_comparison')},
            'source_limits_verbatim': summary['limits'], 'calls': rows,
            'limitations': ['Launch-return to kernel-start intervals include unresolved scheduling, driver, device and tracing effects; no cause decomposition.',
                            'Stream synchronization API intervals overlap GPU work and cannot be added to it.',
                            'Tail after last kernel is a traced boundary interval, not a pure host cost; launch and sync service include instrumentation.',
                            'No aligned application raw QPC ticks; no subtraction from application wall or profile-versus-control pairs.',
                            'Single synthetic Q4_K M64 N4096 K1024 case, one profiled process, unlocked clocks, 20 dependent formal calls; no LLM latency or accuracy claim.',
                            'Profiler overhead event coverage is partial; absence of an explicit record does not prove zero profiling perturbation.'],
            'new_evidence': ['Current-driver successful kernel records and exact process/correlation ownership for every measured invocation.',
                             'Actual traced GPU active intervals, inter-kernel gaps, launch API intervals and synchronization overlap/tail.',
                             'Formal long-span call delay placement before/between/after kernels without removing outliers.'],
            'calibration_eligible': False, 'native_llm_executed': False, 'core_model_modified': False}


def markdown_report(document):
    formal = document['aggregates']['formal']
    lines = ['# R15 已捕获跟踪的区间拆分', '',
             '已核验 SQLite、原始事件及既有调用映射；保留全部 24 次调用，正式样本为 20 次。未启动 GPU、原生推理或仿真，也未改动冻结跟踪目录。', '',
             '所有下表数字来自同一份 Nsight Systems 跟踪时间轴，单位为微秒。API 与 GPU 区间有重叠，不能相加。正式样本的各列中位数是分别计算的，不能把中位数相加重构某次调用。', '',
             '| 正式样本指标 | 最小 | 中位数 | P95（最近秩） | 最大 |', '|---|---:|---:|---:|---:|']
    names = [('nvtx_duration_ns', 'NVTX 调用区间'), ('mapped_kernel_union_ns', '三个 GPU 内核区间并集'),
             ('kernel_span_ns', '首个内核开始至最后内核结束'), ('before_first_kernel_ns', 'NVTX 开始至首个内核开始'),
             ('between_mapped_kernels_ns', '映射内核之间无内核区间'), ('after_last_kernel_ns', '最后内核结束至 NVTX 结束'),
             ('host_launch_api_union_ns', '主机 launch API 区间并集'), ('host_sync_api_union_ns', '主机同步 API 区间'),
             ('host_sync_gpu_overlap_ns', '同步 API 与 GPU 内核重叠'), ('host_sync_before_first_kernel_ns', '同步 API 内、首个内核开始前'),
             ('host_sync_after_last_kernel_ns', '同步 API 内、最后内核结束后'), ('after_sync_to_nvtx_end_ns', '同步 API 返回至 NVTX 结束'),
             ('nvtx_uncovered_by_mapped_kernels_or_host_api_ns', '未被这些内核或 API 覆盖的区间')]
    for key, label in names:
        s = formal[key]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{s[k]/1000:.3f}' for k in ('minimum_ns', 'median_ns', 'p95_nearest_rank_ns', 'maximum_ns')) + ' |')
    lines += ['', '## 真实内核与同步尾部', '',
              '| 内核（每个正式调用各一个） | 中位数 µs | 最小–最大 µs |', '|---|---:|---:|']
    for name, s in document['formal_kernel_duration_by_name'].items():
        lines.append(f"| {name} | {s['median_ns']/1000:.3f} | {s['minimum_ns']/1000:.3f}–{s['maximum_ns']/1000:.3f} |")
    lines += ['', '下面逐次表是一组互不重叠、总和严格等于 NVTX 区间的拆分。空隙只表示这份跟踪未捕获到相关 GPU 内核；在本数据中已同时检查其他捕获内核和 memcpy 是否覆盖这些空隙。不能据此断言设备全局空闲。', '',
              '| 阶段 / 序号 | NVTX µs | 首内核前 µs | 内核并集 µs | 内核间 µs | 末内核后 µs | 同步尾部 µs |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in document['calls']:
        m = row['metrics']
        keys = ('nvtx_duration_ns', 'before_first_kernel_ns', 'mapped_kernel_union_ns', 'between_mapped_kernels_ns', 'after_last_kernel_ns', 'host_sync_after_last_kernel_ns')
        lines.append(f"| {row['phase']} / {row['index']} | " + ' | '.join(f'{m[k]/1000:.3f}' for k in keys) + ' |')
    outliers = sorted((r for r in document['calls'] if r['phase'] == 'formal'), key=lambda r: r['metrics']['nvtx_duration_ns'], reverse=True)[:3]
    lines += ['', '正式调用的最长三项：']
    for row in outliers:
        m = row['metrics']
        lines.append(f"- formal/{row['index']}：NVTX {m['nvtx_duration_ns']/1000:.3f} µs，其中首内核之前 {m['before_first_kernel_ns']/1000:.3f} µs、GPU 内核并集 {m['mapped_kernel_union_ns']/1000:.3f} µs、最后内核之后 {m['after_last_kernel_ns']/1000:.3f} µs；这是时间位置描述，不是原因判定。")
    lines += ['', '## 可以确认与仍未知', '',
              '- 96 个捕获内核中，72 个按 correlationId 与完整进程标识关联到 24 个 NVTX 调用；其余 24 个保留在原始证据中，为测量区间外的 eviction 调用。SQLite 的 CUDA_RUNTIME 表实际同时含 CUDA runtime 与 CUDA driver（eventClass 0/1），这里没有假设独立 DRIVER 表。',
              '- 当前驱动下已获得每次调用的三个内核、各自 launch API 和同步活动，并能确认长调用的时间主要落在哪段区间。该证据关闭了此前“仅有应用总时长、无法观察实际 GPU 活动”的缺口。',
              '- 同步等待覆盖 GPU 工作以及部分首内核前区间；最后内核之后的同步尾部单独列出。不能把同步 API 时长等同纯主机开销，也不能在 GPU 内核时长之外再整体加一次同步时长。',
              '- 原应用 QPC 相对计时未与跟踪时间轴对齐，不做跨域时间相减。profile 和 direct control 是一次有顺序且未锁频的进程对，保留它们的分布统计，但不按相同调用序号配对减去开销。',
              '- launch 返回到内核开始之间的延迟来源仍未知：调度、驱动、设备状态和跟踪扰动未分别测量。显式 profiler overhead 记录不能代表所有扰动；未记录不等于不存在。',
              '- 单一合成形状与 20 次相关调用不能外推 LLM 场景、HBM 流量或独立精度验收；未拟合延迟、未改核心模型，也不作误差 <5% 的声明。', '',
              '## 证据与复现', '',
              'decomposition.json 保存全部 24 次的原始 NVTX、API、内核与同步事件、精确行号、关联条件、分区及输入前后 SHA256 校验。脚本仅以 mode=ro 打开 SQLite，输出禁止覆盖。', '',
              '输入 SHA256：']
    for key, ref in document['source_refs'].items():
        lines.append(f"- {key}: `{ref['sha256']}`（{ref['bytes']} bytes）")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    trace = args.trace_root.resolve(strict=True)
    outputs = [args.output.resolve(), args.markdown.resolve()]
    if outputs[0] == outputs[1] or any(path.exists() or path.is_relative_to(trace) for path in outputs):
        raise ValueError('output must be new and outside frozen trace')
    document = make_report(trace)
    text = markdown_report(document)
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open('x', encoding='utf-8') as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
    with outputs[1].open('x', encoding='utf-8') as stream:
        stream.write(text)
    print(json.dumps({'scope': document['scope'], 'output': str(outputs[0]), 'source_identity_unchanged': True}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
