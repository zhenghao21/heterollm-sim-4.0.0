"""Render measured Graph comparisons without inventing missing predictions."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ('qwen3_0_6b_f16', 'Qwen3 0.6B · F16'),
    ('qwen3_1_7b_q8_0', 'Qwen3 1.7B · Q8_0'),
    ('qwen3_4b_q4_k_m', 'Qwen3 4B · Q4_K_M'),
    ('qwen3_8b_q4_k_m', 'Qwen3 8B · Q4_K_M'),
    ('qwen3_8_27b_mixed', 'Qwen3.8 27B · IQ3_S / IQ4_XS'),
]
METRICS = [('engine_ttft_ns', '首 token'), ('engine_tpot_ns', '后续每 token'), ('engine_e2e_ns', '请求总耗时')]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def native_metrics(path, mode):
    record = read(path)
    config = record['configuration']
    env = config['effective_env']
    if (record.get('status') != 'completed' or len(record['samples']) != 5
            or len(record['warmups']) != 2 or config.get('cuda_graphs_mode_requested') != mode
            or (mode == 'off' and env.get('GGML_CUDA_DISABLE_GRAPHS') != '1')
            or (mode == 'on' and env.get('GGML_CUDA_DISABLE_GRAPHS') is not None)):
        raise ValueError('native Graph mode/sample contract mismatch: ' + path.name)
    if any(sample.get('prompt_n') != 512 or sample.get('visible_output_tokens') != 128
           or sample.get('cache_n') != 0 or sample.get('status') != 'completed'
           for sample in record['samples']):
        raise ValueError('native token contract mismatch: ' + path.name)
    result = {}
    for key, _ in METRICS:
        values = [sample[key] for sample in record['samples']]
        median = statistics.median(values)
        result[key] = {'median_ns': median, 'min_ns': min(values), 'max_ns': max(values),
                       'mad_ns': statistics.median(abs(value - median) for value in values)}
    return result


def graph_speedup(modes):
    """Keep fractional speedup and percentage-point prediction error distinct."""
    native_off = modes['off']['native']['engine_e2e_ns']['median_ns']
    native_on = modes['on']['native']['engine_e2e_ns']['median_ns']
    observed = (1 - native_on / native_off) * 100
    result = {'native_e2e_reduction_percent': observed,
              'native_speedup_ratio': native_off / native_on,
              'simulation_e2e_reduction_percent': None, 'simulation_speedup_ratio': None,
              'signed_reduction_error_percentage_points': None,
              'absolute_reduction_error_percentage_points': None}
    comparisons = [modes[mode]['comparison'] for mode in ('off', 'on')]
    if all(item and item.get('status') == 'compared' for item in comparisons):
        predicted_off, predicted_on = (item['metrics']['engine_e2e_ns']['simulation_ns']
                                       for item in comparisons)
        predicted = (1 - predicted_on / predicted_off) * 100
        result.update(simulation_e2e_reduction_percent=predicted,
                      simulation_speedup_ratio=predicted_off / predicted_on,
                      signed_reduction_error_percentage_points=predicted - observed,
                      absolute_reduction_error_percentage_points=abs(predicted - observed))
    return result


def host_cost_summary(path):
    """Read additive service amounts without converting them into E2E bounds."""
    if not path.is_file():
        return None
    data = read(path)
    if (data.get('schema') != 'heterollm.cuda-graph-host-cost-sensitivity/v1'
            or data.get('scope') != 'all_completed_report_invocations_including_startup_and_warmup'
            or data.get('end_to_end_sensitivity_computed') is not False):
        raise ValueError('host cost summary scope mismatch: ' + path.name)
    return {key: data[key] for key in ('scope', 'invocation_count', 'event_count',
            'event_counts', 'host_service_totals_ns', 'prediction_qualified',
            'end_to_end_sensitivity_computed')} | {'source_file': path.name}


def render(base):
    rows, result, speedups, speedup_data, host_rows = [], [], [], [], []
    for slug, name in CASES:
        modes = {}
        for mode in ('off', 'on'):
            native = native_metrics(base / f'native_{slug}_graph_{mode}.json', mode)
            prediction_file = base / f'comparison_{slug}_graph_{mode}.json'
            comparison = read(prediction_file) if prediction_file.is_file() else None
            # Only an explicitly completed, configuration-checked comparison
            # may display a prediction. Native enablement alone is not proof.
            compared = comparison is not None and comparison.get('status') == 'compared'
            host = host_cost_summary(base / f'host_cost_sensitivity_{slug}_graph_{mode}.json')
            modes[mode] = {'native': native, 'comparison': comparison, 'host_cost_sensitivity': host}
            if host:
                totals = host['host_service_totals_ns']
                host_rows.append(f'<tr><th>{escape(name)}</th><td>{"开启" if mode == "on" else "关闭"}</td>'
                    f'<td>{host["invocation_count"]} / {host["event_count"]}</td>'
                    f'<td>{totals["sum_of_event_medians_ns"]/1e6:.3f}</td>'
                    f'<td>{totals["sum_of_event_minima_ns"]/1e6:.3f}</td>'
                    f'<td>{totals["sum_of_event_maxima_ns"]/1e6:.3f}</td>'
                    f'<td><a href="{escape(host["source_file"], quote=True)}">阶段明细</a></td></tr>')
            cells = []
            for key, _ in METRICS:
                measured = native[key]['median_ns'] / 1e6
                prediction = comparison['metrics'][key] if compared else None
                cells.append(f'<td>{measured:.3f}<small>MAD {native[key]["mad_ns"]/1e6:.3f} ms</small>'
                             f'<small>范围 {native[key]["min_ns"]/1e6:.3f}–{native[key]["max_ns"]/1e6:.3f}</small></td>')
                cells.append(f'<td>{prediction["simulation_ns"]/1e6:.3f}'
                             f'<small>绝对误差 {prediction["absolute_error_ns"]/1e6:.3f} ms</small>'
                             f'<small>有符号误差 {prediction["signed_error_percent"]:+.2f}%</small></td>'
                             if prediction else '<td class=pending>待完成有效配对</td>')
            rows.append(f'<tr><th>{escape(name)}</th><td>{"开启" if mode == "on" else "关闭"}</td>{"".join(cells)}</tr>')
        gain = graph_speedup(modes)
        result.append({'case_id': slug, 'name': name, 'modes': modes, 'graph_speedup': gain})
        speedup_data.append({'case_id': slug, 'name': name, **gain})
        predicted = gain['simulation_e2e_reduction_percent']
        error = gain['signed_reduction_error_percentage_points']
        speedups.append(f'<tr><th>{escape(name)}</th><td>{gain["native_e2e_reduction_percent"]:.2f}%</td><td>'
                        + (f'{predicted:.2f}%' if predicted is not None else '待完成') + '</td><td>'
                        + (f'{error:+.2f} 个百分点' if error is not None else '待完成') + '</td></tr>')
    calibration_file = base / 'runtime_calibration_v2.json'
    calibration = read(calibration_file) if calibration_file.is_file() else {}
    phase_rows = ''.join(f'<tr><th>{escape(phase)}</th><td>{error*100:.2f}%</td><td>{calibration.get("validation_absolute_error_ns_by_phase",{}).get(phase,0)/1000:.3f} µs</td></tr>'
                         for phase, error in calibration.get('validation_relative_error_by_phase', {}).items())
    completed = sum(bool(mode['comparison'] and mode['comparison'].get('status') == 'compared')
                    for case in result for mode in case['modes'].values())
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CUDA Graph · 本机模型对照</title><style>
body{font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#f3f6fa;color:#182638;margin:0;padding:30px}main{max-width:1360px;margin:auto}
h1{font-size:28px;margin:8px 0}h2{font-size:20px;margin:30px 0 12px}.sub{color:#56677a}.card{background:white;border:1px solid #dde5ee;border-radius:12px;padding:20px;margin:18px 0;overflow:auto}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:12px 9px;border-bottom:1px solid #e8edf3;white-space:nowrap}thead{background:#edf3fa}small{display:block;color:#667789;font-size:12px}.pending{color:#95651e}.tag{display:inline-block;background:#e2edf9;padding:5px 12px;border-radius:6px;margin-right:8px}a{color:#235cb2}.note{border-left:4px solid #c89a3c;padding-left:14px}code{font-size:12px}
</style><main><div class=sub>本机 RTX 5080 + Ryzen 9 9950X3D · 2026-10-08</div><h1>CUDA Graph 与 native 对照</h1>
<p>测量负载：512 输入 / 128 输出，单序列，context 768，FP16 KV，Flash Attention 关闭，无提示复用、无 MTP。上下文检查点显式关闭，避免额外分段和状态备份干扰 Graph 对照；此前默认检查点结果另行保留。每组 2 次预热，5 次正式实测。下表显示毫秒；native 中位数下方为 MAD（相对中位数的绝对偏差中位数）及实测范围。仿真误差同时显示绝对毫秒和有符号百分比，正值表示仿真高估。</p>
<p>五个模型均属 Qwen 系列，覆盖不同规模和量化精度。前四个为 qwen3，文件名 Qwen3.8 27B 的混合量化模型实际 GGUF 架构为 qwen35；这增加了规模与精度覆盖，仍不构成跨模型家族、跨架构的广泛验证。</p>
'''
    html += f'<div><span class=tag>5 个模型</span><span class=tag>10 组 native 已测完</span><span class=tag>{completed}/10 仿真有效配对</span></div>'
    paired_gains = [item for item in speedup_data
                    if item['simulation_e2e_reduction_percent'] is not None]
    if paired_gains:
        observed = [item['native_e2e_reduction_percent'] for item in paired_gains]
        predicted = [item['simulation_e2e_reduction_percent'] for item in paired_gains]
        largest_gap = max(item['absolute_reduction_error_percentage_points'] for item in paired_gains)
        html += (f'<p class=note>已完成开关配对的 {len(paired_gains)} 个模型：'
                 f'native 的 Graph 耗时降幅为 {min(observed):.2f}%–{max(observed):.2f}%，'
                 f'仿真降幅为 {min(predicted):+.2f}%–{max(predicted):+.2f}%，'
                 f'最大降幅误差为 {largest_gap:.2f} 个百分点。'
                 '流程与生命周期核验通过不等于预测精度合格；本轮独立成本仍属于实验路径。</p>')
    if (base / 'graph_prediction_limitations.md').is_file():
        html += '<p><a href="graph_prediction_limitations.md">0.6B / 1.7B 成本与关键路径审计</a>：已确认 CPU 提交成本参与执行；设备端调度与合成成本的代表性仍有限制，尚不能分别量化归因。</p>'
    html += '<div class=card><table><thead><tr><th rowspan=2>模型</th><th rowspan=2>Graph</th><th colspan=2>首 token</th><th colspan=2>后续每 token</th><th colspan=2>请求总耗时</th></tr><tr>'
    html += '<th>native</th><th>仿真</th>' * 3 + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
    html += '<p class=note>开启环境变量只代表允许使用 Graph。实际重放由独立诊断核验；仿真还须具备完整生命周期、结构匹配的成本与相同请求配置。缺失或失败的配对保留为空，不用旧仿真数字代替。</p>'
    html += '<h2>Graph 对请求总耗时的影响</h2><p>正值表示 Graph 开启后耗时降低，计算为 (关闭耗时 − 开启耗时) / 关闭耗时。两种模式使用相同模型、硬件和请求设置。预测误差为“仿真降幅 − native 降幅”，单位是百分点；正值表示高估 Graph 收益。</p><div class=card><table><thead><tr><th>模型</th><th>native 耗时降低</th><th>仿真耗时降低</th><th>降幅预测误差</th></tr></thead><tbody>' + ''.join(speedups) + '</tbody></table></div>'
    html += '<h2>完成报告中的 host 服务量</h2><p class=note>以下是实际生命周期事件对应的 CPU 服务量之和，包含启动探测、预热和正式请求。各事件中位数相加不等于整次运行的统计中位数；最小值或最大值相加仅作成本敏感性场景。CPU 与 GPU 可以重叠，因此这些总量及其范围均不是请求 E2E 耗时或置信区间，也不能直接加到上表误差。单位：ms。</p>'
    if host_rows:
        html += '<div class=card><table><thead><tr><th>模型</th><th>Graph</th><th>调用 / 阶段事件数</th><th>事件中位数之和</th><th>事件最小值之和</th><th>事件最大值之和</th><th>来源</th></tr></thead><tbody>' + ''.join(host_rows) + '</tbody></table></div>'
    else:
        html += '<p class=pending>尚无完成报告的 host 服务量汇总；不从 native 实测或未完成任务推算。</p>'
    measured_path = base / 'runtime_typed_chain_measurements.json'
    if measured_path.is_file():
        data = read(measured_path)
        measurement_rows = []
        for sample in data['samples']:
            stats = sample['phase_measurements']['replay_submit']
            measurement_rows.append(f'<tr><td>{sample["node_count"]}</td><td>{stats["repeat_count"]}</td>'
                f'<td>{stats["median_ns"]/1000:.3f}</td><td>{stats["repeat_mad_ns"]/1000:.3f}</td>'
                f'<td>{stats["minimum_ns"]/1000:.3f}–{stats["maximum_ns"]/1000:.3f}</td></tr>')
        html += '<h2>本轮采用的独立成本</h2><p class=note>这是显式实验模式：按真实节点类型、边及拷贝尺寸匹配合成 CUDA 程序的独立测点；没有用模型总耗时拟合，也不插值或外推。合成 kernel 的函数、参数和执行体与模型 kernel 有差别，跨结构、跨机器泛化尚未通过验证。重复测量波动不等同于建模误差。</p>'
        html += '<div class=card><table><thead><tr><th>图节点数</th><th>重复次数</th><th>重放 CPU 提交中位数 µs</th><th>MAD µs</th><th>观测范围 µs</th></tr></thead><tbody>' + ''.join(measurement_rows) + '</tbody></table></div>'
    html += '<h2>早期通用插值探索（失败记录）</h2><p>下列是早期通用节点数插值探索的留出点最大误差，未通过准入条件，未用于上述实验预测。该版 capture 计时夹带了节点数量查询，driver 字段记录的是 CUDA 驱动 API 版本而非发行版本；这些问题已在本轮结构测量程序修正。保留该失败记录说明探索过程，其误差不能作为本轮成本质量或泛化性的结论。</p>'
    if phase_rows:
        html += '<div class=card><table><thead><tr><th>阶段</th><th>最大相对误差</th><th>最大绝对误差</th></tr></thead><tbody>' + phase_rows + '</tbody></table></div>'
    html += '<p>CPU 提交成本与 GPU 计算、物理 DRAM 读写分别建模并允许重叠；CUDA event 时间不直接当作额外 GPU 成本相加。GPU 设备调度仍使用原分析参数，Graph 在设备端的调度收益尚未独立校准。结构编译依赖固定版本的 native GGUF 图构建器，只构建和导出结构、不执行模型推理；真实执行决策及耗时仅用于最终核验。新模型若出现未测的类型化拓扑、生命周期阶段或旧→新图更新配对，当前实验成本路径会明确拒绝，不以插值或其他结构代替。仿真显式执行启动探测和预热，只比较最后请求的 engine 边界，不把排队或预热时间混入误差。</p>'
    html += '<p class=note>另有明确的精度限制：保留的每 kernel 1000 ns 启动参数是分析默认值，尚未独立分离 CPU 与设备开销；采样过滤链与完成中断仍有未完整计价项。以上误差属于当前完整预测结果，不能归因于 Graph 单一模块。修复前重复计入的 driver 提交项已移交独立 API 成本；修复前结果另存于 before_runtime_owner_fix，不混入本表。</p>'
    html += '<p><a href="comparison_summary.json">完整比较数据</a> · <a href="native_timing_manifest.json">native 测量配置</a> · <a href="runtime_typed_chain_measurements.json">实验成本原始重复值</a> · <a href="runtime_calibration_v2.json">通用插值验证结果</a></p></main></html>'
    (base / 'report.html').write_text(html, encoding='utf-8')
    (base / 'comparison_summary.json').write_text(json.dumps({'completed_comparisons': completed,
        'expected_comparisons': 10, 'cases': result, 'graph_speedups': speedup_data}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return base / 'report.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/cuda_graph_validation_2026-10-08')
    print(render(parser.parse_args().output))
