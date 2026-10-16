"""Human-readable research results; never equate a check with a research claim."""
from __future__ import annotations

import html
import json

from hbf_hbm_serving_study import POINT_RESULT_VERSION


def uses_current_execution(report):
    """A current envelope cannot turn historical groups or points into new data."""
    return (report.get('point_result_version') == POINT_RESULT_VERSION
        and all(g.get('point_result_version') == POINT_RESULT_VERSION
                for g in report.get('groups', []))
        and all(r.get('point_result_version') == POINT_RESULT_VERSION
                for g in report.get('groups', []) for r in g.get('rows', [])
                if r.get('mode') == 'full_operator_serving'))


def esc(value):
    return html.escape(str(value))


def number(value, digits=3):
    if value is None:
        return '—'
    if 0 < abs(value) < 10 ** -digits:
        return f'{value:.3g}'
    return f'{value:,.{digits}f}'


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+esc('—' if x is None else x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def details(title, value):
    return '<details><summary>'+esc(title)+'</summary><pre>'+esc(json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>'


def supported_scope(group):
    supported=group.get('capability',{}).get('supported',[])
    if isinstance(supported,bool): return group.get('capability',{}).get('scope','')
    return '；'.join(supported)


_BANK_GROUP_TIMING_FIELDS = (
    'column_same_group_ns', 'column_different_group_ns',
    'activate_same_group_ns', 'activate_different_group_ns',
    'four_activate_window_ns', 'write_to_read_same_group_ns',
    'write_to_read_different_group_ns',
)


def bank_group_variants(group):
    """Describe saved execution facts, never re-resolve a mutable profile name."""
    profiles = list(dict.fromkeys(group.get('hardware_profiles', [])))
    for row in group.get('rows', []):
        profile = row.get('hardware_profile')
        if profile is not None and profile not in profiles:
            profiles.append(profile)
    variants = []
    for profile in profiles:
        records = []
        for row in group.get('rows', []):
            if row.get('hardware_profile') != profile:
                continue
            configs = row['executed_hbm_parameters']
            for owner, config in configs.items():
                existing = next((r for r in records if r['owner'] == owner and r['config'] == config), None)
                if existing is None:
                    records.append({'owner': owner, 'config': config, 'row_ids': [row.get('id')]})
                else:
                    existing['row_ids'].append(row.get('id'))
        labels = []
        for record in records:
            config = record['config']; metadata = config.get('metadata', {})
            geometry = str(config.get('bank_groups_per_rank', '?'))+'×'+str(config.get('banks_per_group', '?'))
            generation = config.get('generation') or 'HBM'
            if metadata.get('geometry_control') == 'flat':
                zh, en = '平坦消融 ', 'Flat control '
            elif metadata.get('geometry_status') == 'standard_organization':
                zh, en = generation+'参考 ', generation+' reference '
            else:
                zh, en = '合成/未核实 ', 'Synthetic/unverified '
            timing = [config.get(key) for key in _BANK_GROUP_TIMING_FIELDS]
            if any(value is not None and value > 0 for value in timing):
                timing_zh, timing_en = (('合成组时序', 'synthetic group timing')
                    if metadata.get('geometry_control') == 'timed' else ('已配置组时序', 'configured group timing'))
            elif any(value is None for value in timing):
                timing_zh, timing_en = '组时序未记录', 'group timing unavailable'
            else:
                timing_zh, timing_en = '组时序关闭', 'group timing off'
            label = (zh+geometry+'，'+timing_zh, en+geometry+', '+timing_en)
            if label not in labels:
                labels.append(label)
        variants.append({'profile': profile, 'records': records,
            'label_zh': ' / '.join(x[0] for x in labels) or str(profile)+'（配置未记录）',
            'label_en': ' / '.join(x[1] for x in labels) or str(profile)+' (configuration unavailable)'})
    return variants


def extension_tables(group):
    n=int(group['id'][:2]); rows=group.get('rows',[]); parts=[]
    if n==20:
        variants = bank_group_variants(group)
        parts.append('<p id="bank-group">下列是小型合成Dense模型的完整服务，先执行真实prefill，再测Decode。平坦分组、标准分组和合成时序对照的标签均取自结果中实际执行的HBM配置；不按当前目录重新解析profile名称。总bank数及其余条件是否相同，见本组检查与逐点配置。</p>')
        parts.append('<p>每点一个请求、'+str(group['output_tokens'])+'个输出token；首token来自prefill，Decode TPS只统计之后'+str(group['output_tokens']-1)+'个token。标准CCD约束与合成敏感性时序按保存的来源区分；已知硬下限不得关闭，未知额外门控的零值仅表示未模拟，不表示器件真实延迟为零。具体执行值如下。</p>')
        parts.append(table(('配置ID','实际分组与时序'),[(v['profile'],v['label_zh']) for v in variants]))
        parts.append(table(('上下文','KV放置','平坦分组Decode TPS','标准分组Decode TPS','合成时序Decode TPS','合成/标准'),[
            (r['prompt_tokens'],r['placement'],number(r['baseline_decode_tps'],2),number(r['geometry_control_decode_tps'],2),
             number(r['bank_group_decode_tps'],2),number(r['timing_over_control_tps'],5)) for r in group.get('comparisons',[])]))
        records=[(v['profile'],r['owner'],r['config']) for v in variants for r in v['records']]
        parts.append(table(('配置','owner','代际','stack','通道/stack','伪通道/通道','分组×bank/伪通道','行 B','burst B','容量 bytes'),[
            (profile,owner,config.get('generation') or '未记录',config.get('stacks'),config.get('channels'),config.get('pseudo_channels_per_channel'),
             str(config.get('bank_groups_per_rank'))+'×'+str(config.get('banks_per_group')),
             config.get('row_bytes'),config.get('burst_bytes'),config.get('capacity_bytes'))
            for profile,owner,config in records]))
        parts.append(table(('配置','owner','列命令同/异组 ns','ACT同/异组 ns','四ACT窗口 ns','写结束→读同/异组 ns'),[
            (profile,owner,
             '/'.join(number(config.get(k)) for k in _BANK_GROUP_TIMING_FIELDS[0:2]),
             '/'.join(number(config.get(k)) for k in _BANK_GROUP_TIMING_FIELDS[2:4]),
             number(config.get('four_activate_window_ns')),
             '/'.join(number(config.get(k)) for k in _BANK_GROUP_TIMING_FIELDS[5:7]))
            for profile,owner,config in records]))
        parts.append(table(('配置','owner','控制逻辑范围','已知时序约束检查','厂商速度档验证','未知参数'),[
            (profile,owner,config.get('metadata',{}).get('control_status'),
             config.get('metadata',{}).get('timing_constraint_validation'),
             config.get('metadata',{}).get('timing_validation'),
             '、'.join(config['metadata']['unknown_parameters'])
             if isinstance(config.get('metadata',{}).get('unknown_parameters'),list) else None)
            for profile,owner,config in records]))
        parts.append(details('实际Core配置、来源、未知参数与对应结果行（保留同profile下的不同配置）',variants))
        parts.append('<p>只有保存的metadata声明标准组织时，标签才写明HBM代际参考组织；参考组织不等于已识别的厂商器件，平坦分组也只是合成消融。来源、所选密度/层数、未知的目标器件参数和寻址/时序简化以保存的metadata为准。非零组时序未作厂商校准；小Dense的倍率不能外推为8B/70B性能。</p>')
    elif n==12:
        parts.append('<p>先使完整服务的KV介质等效，再分别加入读延迟、写限制和有限并发。路径实验固定链路带宽总和，权重与冷KV采用固定物理分区；QoS指静态带宽分配和请求优先级。</p>')
        parts.append(table(('实验','TPS','相对理想HBM TPS','中继写/读B','后到请求TTFT µs'),[
            (r['id'],number(r['node_tps'],2),number(r.get('relative_to_ideal_hbm',{}).get('tps_ratio'),5),
             str(r.get('path_observations',{}).get('relay_write_bytes','—'))+'/'+str(r.get('path_observations',{}).get('relay_read_bytes','—')),
             number(r.get('qos_observations',{}).get('later_request_ttft_ns',0)/1000) if 'qos_observations' in r else None) for r in rows]))
    elif n==13:
        parts.append('<p>每轮在前一轮真实完成后等待think time，再带新输入恢复；实际继承KV页。取消会等待当前物理阶段安全结束，已经发生的读写和占用仍计费。</p>')
        parts.append(table(('策略','实际prefill token','独立请求重算token','预取提交/取消B','取消已发生物理读B','GPU decode重叠µs','TPS'),[
            (r['id'],r['computed_prefill_tokens'],r['independent_prompt_tokens_sum'],
             str(r['prefetch']['submitted_bytes'])+'/'+str(r['prefetch'].get('cancelled_bytes',0)),
             r['cancelled_work']['physical_read_bytes'],number(r['prefetch_overlap']['decode_operator_overlap_ns']/1000),number(r['node_tps'],2)) for r in rows]))
    elif n==14:
        parts.append('<p>磨损只由显式erase更新P/E；完整服务的磨损对照采用预老化设备假设，没有暗中生成FTL擦除。</p>')
        parts.append(table(('受控轨迹','时长ms','实际擦除次数','read/program队列峰值','物理plane变异系数'),[
            (r['id'],number(r['duration_ns']/1e6),r['observations']['actual_erase_operations'],
             str(r['observations']['read_queue_depth']['peak'])+'/'+str(r['observations']['program_queue_depth']['peak']),
             number(r['observations']['physical_plane_imbalance']['coefficient_of_variation'])) for r in group.get('trace_rows',[])]))
        parts.append(table(('完整服务对照','TPS反馈/基准','TTFT反馈/基准'),[
            (r['feedback_id'],number(r['feedback_over_baseline_tps'],5),number(r['feedback_over_baseline_ttft_p50'],5)) for r in group.get('serving_comparisons',[])]))
    elif n==15:
        parts.append('<p>显式调度每个GPU分片和GPU间通信，共享端点不按GPU数倍增带宽。机架场景的远端共享池与GPU通信均承担网络带宽/时延。此小模型不一定能摊薄多GPU通信开销。</p>')
        parts.append(table(('配置','GPU数','节点TPS','每GPU TPS','实际执行GPU数','实际GPU间链路资源数'),[
            (r['id'],r['gpu_count'],number(r['node_tps'],2),number(r['per_gpu_tps'],2),
             len(r['physical_execution']['gpu_compute_resources']),len(r['physical_execution']['peer_link_resources'])) for r in rows]))
    elif n==16:
        parts.append('<p>丢弃前25%请求的预热窗口，检查完成速率、积压增长和SLO。候选边界在128/256/512请求轨迹验证；这给出本次假设下的有限窗口边界。</p>')
        parts.append('<p>固定判据：窗口内积压增长≤16请求、末尾积压≤32请求，三个长度的末尾积压差≤16；512请求窗口的完成/到达比例≥95%，且预热后所有请求的TTFT和每次TBT均达标。16是一次批量完成的最大跳变量。</p>')
        parts.append(table(('架构','通过到达率 req/s','失败到达率 req/s','三个窗口均验证'),[
            (r['placement'],number(r.get('passing_rate_rps'),2),number(r.get('failing_rate_rps'),2),r.get('bracket_valid')) for r in group.get('arrival_boundaries',[])]))
        parts.append(table(('架构','通过session/GPU','相邻失败session/GPU','输入/输出'),[
            (r['placement'],r.get('passing_sessions_per_gpu'),r.get('failing_sessions_per_gpu'),str(r.get('prompt_tokens'))+'/'+str(r.get('output_tokens'))) for r in group.get('concurrency_boundaries',[])]))
        boundary_rates={(b['placement'],b['passing_rate_rps']) for b in group.get('arrival_boundaries',[])}|{(b['placement'],b['failing_rate_rps']) for b in group.get('arrival_boundaries',[])}
        chosen=[r for r in rows if 'sustained' in r and r['request_count']==512 and (r['placement'],r['parameters']['rate_rps']) in boundary_rates]
        parts.append(table(('512请求边界点','完成/到达速率','窗口初/末积压','增长请求数','通过'),[
            (r['id'],number(r['sustained']['completed_rps']/r['sustained']['offered_rps'],5),
             str(r['sustained']['backlog_at_start'])+'/'+str(r['sustained']['backlog_at_end']),r['sustained']['backlog_growth_requests'],r['sustained']['finite_horizon_sustainable']) for r in chosen]))
        parts.append('<p>完成/到达可略高于1，因为预热阶段到达的请求也可能在测量窗口内完成；末尾排空期间的完成不计入该速率。</p>')
    elif n==18:
        parts.append('<p>下列为内存顺序读取微基准，不是LLM吞吐。固定384MiB逻辑读量、1.5MiB传输块和有界缓冲；物理page与传输块分别设置。充分并行和内部通道速率是明确假设，不能当作可制造设备规格。</p>')
        parts.append(table(('扫描','端口/链路GB/s','物理页KiB','读延迟µs','实际端到端GB/s','链路忙时窗口GB/s','实际物理读MiB'),[
            (r['section'],str(r['configured']['port_gb_s'])+'/'+str(r['configured']['link_gb_s']),r['configured']['page_bytes']/1024,
             r['configured']['page_read_ns']/1000,number(r['observed']['effective_end_to_end_read_gb_s']),number(r['observed']['link_stream_effective_gb_s']),
             r['observed']['physical_read_bytes']/1024**2) for r in rows]))
    elif n==17:
        parts.append('<p>Qwen3-Coder和合成1T MoE执行现有完整算子服务。DeepSeek V3/V4和Llama4仅运行其结构专属物理工作负载：明确计算矩阵、KV字节和真实HBM/NAND服务，未覆盖的原生算子单列，因此不输出LLM TPS。</p>')
        probes=[r for r in rows if r.get('mode')=='structural_physical_probe']
        parts.append(table(('模型','上下文','放置','几何规格来源','物理探针时长ms','其中内存ms','解析算量时间ms'),[
            (r['model_id'],r['context'],r['placement'],r['support_boundary'].get('geometry_support','见规格'),
             number(r['probe_step_ns']/1e6),number(r['physical_memory_ns']/1e6),number(r['analytical_compute_ns']/1e6)) for r in probes]))
        boundaries={r['model_id']:r['support_boundary'] for r in probes}
        parts.append(details('逐模型事实、假设、未覆盖算子与官方来源',boundaries))
        parts.append('<p>探针采用统一BF16展开对照，不能等同原生量化kernel。Maverick的部分几何沿用公开默认配置作情景假设，未验证为原生checkpoint；不会把这些点计入完整服务数量。</p>')
    return ''.join(parts)


def render_report(report):
    if not uses_current_execution(report):
        return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<title>历史 HBM/HBF 仿真结果</title><main><h1>历史结果：未按当前模型重跑</h1>'
            '<p>载入执行版本 '+esc(report.get('point_result_version', '未记录'))+
            '；当前版本 '+str(POINT_RESULT_VERSION)+'。本页不使用当前默认配置重新解释旧数据。'
            '旧结果和旧图不能证明当前 HBM/HBF 模型的性能或协议行为。</p>'
            '<p>'+esc(report.get('artifact_note', ''))+'</p>'
            '<p><a href="results.json">查看原始历史 JSON</a>。重新运行所需实验后，再生成当前报告和图表。</p></main></html>')
    groups=report['groups']
    output_dir=esc(report.get('output_directory','artifacts/hbf_hbm'))
    service=[r for g in groups for r in g.get('rows',[]) if r.get('mode')=='full_operator_serving']
    rejected=[r for g in groups for r in g.get('rows',[]) if r.get('mode') in {'configuration_rejected','physical_capacity_rejected'} or r.get('status')=='configuration_rejected']
    parts=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HBF/HBM 内部仿真实验审计报告</title>',
           '<style>body{background:#f4f7fa;color:#193446;font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif;margin:0}main{max-width:1420px;margin:auto;padding:32px 24px 80px}h1,h2{color:#123d59}h2{margin-top:36px;border-top:2px solid #d5e0e8;padding-top:22px}.lead{padding:20px;background:#dfedf5;border-left:5px solid #387da9}.scroll{overflow-x:auto}table{border-collapse:collapse;background:white;width:100%;margin:15px 0;font-size:13px}td,th{padding:9px;border:1px solid #cedbe4;text-align:left;vertical-align:top}th{background:#e7eff5;white-space:nowrap}a{color:#146eaf}pre{background:white;padding:15px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}details{margin:12px 0}code{font-size:12px}nav{display:flex;gap:15px;flex-wrap:wrap}.muted{color:#526b7b}.bad{background:#fff0db;padding:15px}</style></head><body><main>',
           '<h1>HBF/HBM：内部仿真实验与审计结果</h1>',
           '<p class="lead">机制检查与研究覆盖分别报告。当前完整服务结果 '+str(len(service))+' 行，配置或物理容量拒绝 '+str(len(rejected))+' 行；其余为微基准或明确假设下的分析。所有数值均为仿真或条件估计，尚无独立硬件预测精度验证。</p>',
           '<p><a href="results.json">完整结果 JSON</a> · <a href="acceptance.json">后端机制检查</a> · <a href="profiles.json">文献参数化探针</a></p>',
           '<nav><a href="#decode">独立Decode结果</a><a href="#bank-group">Bank group对照</a><a href="#core">Core执行口径</a><a href="#coverage">审计完成清单</a><a href="#performance">完整服务性能</a><a href="#memory">内存与迁移指标</a><a href="#micro">各组实验</a><a href="#economics">购置成本与磨损</a><a href="#remaining">范围与限制</a></nav>',
           '<p class="muted">完整复跑：<code>python -B scripts/hbf_hbm_experiments.py --output-dir &quot;'+output_dir+'&quot; --workers 4</code>；生成图表：<code>python -B scripts/plot_hbf_hbm_results.py --input &quot;'+output_dir+'/results.json&quot; --output &quot;'+output_dir+'/figures&quot; --update-report</code>。使用项目虚拟环境。时间原始单位 ns，带宽十进制 GB/s，容量 bytes；表中延迟已转换为 ms/µs。</p>',
           '<h2 id="coverage">修改和覆盖范围</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in report.get('audit_fixes',[]))+'</ul>',
           table(('实验组','检查状态','结果行数','检查支持的范围'),[(g.get('title',g['id']),g['status'],len(g.get('rows',[])),supported_scope(g)) for g in groups])]
    if report.get('artifact_note'):
        parts.insert(4, '<p class="lead">'+esc(report['artifact_note'])+'</p>')
    study=next((g for g in groups if g['id'].startswith('09')),None)
    core=report.get('memory_core_coverage',{})
    parts.append('<h2 id="core">计算、访存与迁移的职责边界</h2>'
        '<p>GPU侧继续使用聚合简化模型估算计算和启动时间。上层算子与KV管理器生成实际读写请求；'
        'HBM统一交给DRAM Core处理地址、行/Bank、命令和数据通道、排队与完成时间；HBF交给NAND Core处理页读取与编程。'
        'KV迁移按源内存读取、链路传输、目标内存写入的依赖顺序执行。调度器组合这些依赖与重叠，不再额外叠加HBM聚合带宽耗时。</p>')
    parts.append('<p>标准参考几何不等于完整JEDEC协议仿真。jedec_hbm 路径执行共享行/列命令接口、PC/SID与激活页状态，以及已配置的刷新约束；已知时序硬下限对手动实验值同样生效。初始化、训练及完整PHY行为仍不在普通读写子集内。具体执行范围、刷新是否启用、约束检查状态和未知参数以每点保存的Core配置为准，不能据此推断已验证某个厂商速度档。</p>')
    parts.append('<p>主实验 HBF 是逻辑页服务分析：4 KiB 页、64 B 主机计费、16 个全双工主机通道、每 bank 两页读缓存；逻辑写可使用 RMW，未模拟主机重映射和 OCP 物理顺序写。独立 hbf_ocp_0_7 预设用于严格物理协议微测，不能把逻辑模型的结果改称 OCP 设备性能。接口预算与冷阵列服务上限分别列出，后者也不包含传输、命令、RMW、擦除和排队；缓存命中不受冷阵列上限约束。</p>')
    hbf_records = {r['id']: r['hbf_service'] for r in service if r.get('hbf_service')}
    parts.append(table(('点ID','抽象层 / 协议','接口读/写 GB/s','冷阵列读/编程上限 GB/s','页/主机粒度 B','有效并行单元'),[
        (key, value['model_abstraction']+' / '+value['protocol_mode'],
         str(value['interface_read_cap_gb_s'])+'/'+str(value['interface_write_cap_gb_s']),
         number(value['ideal_cold_array_read_ceiling_gb_s'])+'/'+number(value['ideal_array_program_ceiling_gb_s']),
         str(value['page_bytes'])+'/'+str(value['host_granularity_bytes']),value['effective_parallel_units'])
        for key,value in hbf_records.items()]))
    if core.get('all_full_serving_hbm_verified'):
        parts.append('<p><strong>本报告 '+str(core['verified_hbm_dram_core_rows'])+' / '+str(core['full_serving_rows'])+
            ' 个完整服务点均记录到实际HBM DRAM Core执行。</strong>证据来自持久在线服务运行时的实际提交，排除规划预演、隔离重放与服务启动前初始化。'
            '其中 '+str(core.get('verified_nand_core_rows',0))+' 点还记录到实际HBF/SSD NAND Core执行。'
            '逐点JSON中的 <code>core_execution_audit</code> 保存物理owner、Core类型、读写字节及prefill/decode分组。主机DDR与HBM分别计数。</p>')
    else:
        parts.append('<p class="bad">当前载入的数据尚未全部具备HBM DRAM Core执行记录，不应据此声称已经完成物理Core重跑。旧版完整服务曾绕过公共HBM Core，详见历史审计。</p>')
    findings=[]
    decode_group=next((g for g in groups if g['id']=='19_decode_physical'),{})
    for pair in decode_group.get('comparisons',[]):
        if pair['prompt_tokens'] not in (2048,1048576): continue
        findings.append((pair['model_id']+'，'+('2K' if pair['prompt_tokens']==2048 else '1M*')+'独立Decode',
            'A '+number(pair['A_decode_tps'])+' → B '+number(pair['B_decode_tps'])+' token/s；B/A='+number(pair['B_over_A_decode_tps'],4),
            '真实prefill后单独测量；固定'+str(pair['output_tokens'])+'输出。*1M为架构压力外推。'))
    for model_id,label in (('llama3_1-8b','Llama 3.1 8B'),('llama3_1-70b','Llama 3.1 70B')):
        prefix='model/'+model_id+'/2048/'
        a=next((r for r in service if r['id']==prefix+'A_hbm_only'),None)
        b=next((r for r in service if r['id']==prefix+'B_all_kv_hbf'),None)
        if a and b:
            findings.append((label+'，2K 输入 / 32 输出',
                'A '+number(a['node_tps'],3)+' → B '+number(b['node_tps'],3)+' token/s；B/A='+number(b['node_tps']/a['node_tps'],4),
                '请求窗口包含初始化；属于单 GPU 假设配置，不能外推普适倍数。'))
    provisioning=next((g for g in groups if g['id']=='11_provisioning'),None)
    if provisioning:
        selected=provisioning.get('selected_candidate_id')
        findings.append(('70B 固定四槽候选搜索',selected or '九个候选中没有同时满足全部预算与 SLO 的配置',
            '按实际候选栈数计条件成本；无可行项时不产生“最优配置”。'))
    ablation=next((g for g in groups if g['id']=='12_ablation_paths'),None)
    if ablation:
        findings.append(('完整服务理想A/B','总时长相对误差 '+number(ablation['ideal_equivalence']['relative_makespan_error'],6),
            '介质服务条件完全等效时性能一致；差异来自加入的具体延迟、写入与并发约束。'))
    agent_group=next((g for g in groups if g['id']=='13_agent_lifecycle'),None)
    if agent_group:
        indexed={r['id']:r for r in agent_group['rows']}
        base=indexed.get('agent-static-cold');heuristic=indexed.get('agent-heuristic')
        if base and heuristic:
            findings.append(('三轮agent的冷热启发式',
                'TPS '+number(base['node_tps'],0)+' → '+number(heuristic['node_tps'],0)+'；冷迁移写 '+str(base['agent_lifecycle']['cold_write_bytes'])+' → '+str(heuristic['agent_lifecycle']['cold_write_bytes'])+' B',
                '两者均复用'+str(heuristic['agent_lifecycle']['reused_tokens'])+'个历史token；含后台请求的实际prefill分别为'+str(base['computed_prefill_tokens'])+'/'+str(heuristic['computed_prefill_tokens'])+'。TPS差异来自冷迁移与预取策略。'))
    wear_group=next((g for g in groups if g['id']=='14_wear'),None)
    if wear_group:
        pair=wear_group['serving_comparisons'][-1]
        findings.append(('磨损压力对照','B的TPS为关闭反馈时的 '+number(pair['feedback_over_baseline_tps'],4)+' 倍',
            '人为压力参数真实改变后续NAND/GPU时间线；不是实际设备寿命预测。'))
    sustained_group=next((g for g in groups if g['id']=='16_sustained'),None)
    if sustained_group:
        for bound in sustained_group.get('arrival_boundaries',[]):
            findings.append((bound['placement']+' 持续到达率',
                number(bound['passing_rate_rps'],1)+' req/s通过；'+number(bound['failing_rate_rps'],1)+' req/s失败',
                '128/256/512请求、去掉25%预热，同时检查SLO、完成率和积压；仅为有限窗口边界。'))
    if findings:
        parts.insert(3,'<h2>先看实际结论</h2>'+table(('场景','观察结果','解释边界'),findings))
    parts.append('<p><strong>本次所选检查：'+('全部通过' if report.get('all_selected_checks_passed') else '存在未通过项，请看各组明细')+'</strong>。真实外部数据校准依用户指示暂不纳入完成条件；检查通过表示实现和所述实验通过，不表示任何硬件预测已经校准。</p>')
    parts.append(table(('原审计项','当前状态','实际完成范围'),[(r['item'],r['status'],r['reason']) for r in report.get('audit_completion',[])]))
    parts.append('<h3>合并后的逐模型执行范围</h3>'+table(('模型ID','实际执行模式','执行/拒绝点数','上下文','完整算子服务'),[
        (r['model_id'],' / '.join(r['modes']),str(r['executed_points'])+'/'+str(r.get('rejected_points',0)),str(r['contexts']),r['native_full_serving']) for r in report.get('model_execution_coverage',[])]))
    if study:
        capacity=[r for r in study.get('rows',[]) if r.get('mode')=='physical_capacity_rejected']
        if capacity:
            parts.append('<h3>512GiB库存的真实容量失败</h3><p>第9组保留原配置的失败结果和可完成的B点；另将70B/1M的A/B/D共同扩为1TiB HBM进行可比较的性能对照。容量控制点不与扩容点混配。</p>'+table(
                ('点ID','HBM GiB','失败的单次申请 GiB','阶段','结果'),[
                    (r['id'],r['capacity_failure']['capacity_bytes']/1024**3,
                     r['capacity_failure']['requested_allocation_bytes']/1024**3,
                     r['capacity_failure']['failure_stage'],r['reason']) for r in capacity])+
                '<p>失败的申请大小不等于整体峰值需求，碎片也可能影响分配；详细占用与空闲区间见原始JSON。不为失败点生成TPS。</p>')
        parts.append('<p>第9组保留其运行时的历史覆盖清单；其中“未运行/缺失preset”仅描述该组当时范围。当前项目模型状态以以上第9+17组合并表为准。</p>')
        parts.append(details('第9组当时的覆盖与边界（非当前项目缺项）', {key:study[key] for key in ('coverage','search','slo_search','slo_boundaries','limitations') if key in study}))
        batch_bounds=[b for b in study.get('slo_boundaries',[]) if b.get('kind')=='batch']
        if batch_bounds:
            parts.append('<p>以下仅为本次有限请求实验中的已测点，不代表稳态最大会话数。</p>'+table(
                ('架构','最高已测合格 batch','最低已测不合格 batch','实际已测 batch'),
                [(b['architecture'],b['maximum_tested_feasible_batch'],b['minimum_tested_infeasible_batch'],str(b['actual_batches'])) for b in batch_bounds]))
    if report.get('checklist_progress'):
        parts.append(table(('审计项','本轮状态','说明'),[(x['item'],x['status'],x['reason']) for x in report['checklist_progress']]))
    decode=next((g for g in groups if g['id']=='19_decode_physical'),None)
    if decode:
        output_tokens=decode.get('output_tokens',64)
        measured_tokens=output_tokens-1
        parts.append('<h2 id="decode">排除Prefill后的Decode影响</h2>'
            f'<p>统一batch=1、每请求输出{output_tokens} token。每点先真实执行完整prefill，首token产生后，单独统计余下{measured_tokens}个token的完成窗口。'
            f'<strong>Decode TPS = {measured_tokens} / 该窗口秒数</strong>；TPOT为首token之后的平均输出间隔。'
            'A/B使用相同硬件配置和工作量。128K加输出略超原生上下文，1M属于架构压力外推。</p>'
            '<p>70B/1M的A/B共同使用1TiB HBM；其余Decode点为512GiB。原512GiB配置在prefill工作空间的连续地址分配时被拒绝，不能据此认定总容量必然不足；失败结果另行保留。扩容时保持命令时钟和方向预算，stack数、并行度与物理接口总速率相应增加。</p>'
            '<p>旧“命名模型随上下文变化的TPS”图按整个请求窗口计算，长prefill会稀释Decode差异。该旧指标继续保留用于衡量端到端输出速率；下表和独立Decode图专门回答生成阶段的影响。</p>')
        parts.append(table(('模型','上下文','A Decode TPS','B Decode TPS','Decode B/A','A TPOT ms','B TPOT ms','TPOT变化','全请求TPS B/A'),[
            (p['model_id'],p['prompt_tokens'],number(p['A_decode_tps'],3),number(p['B_decode_tps'],3),
             number(p['B_over_A_decode_tps'],4),number(p['A_tpot_ns']/1e6),number(p['B_tpot_ns']/1e6),
             f"{(p['B_over_A_tpot']-1)*100:+.2f}%",number(p['B_over_A_total_tps'],4)) for p in decode.get('comparisons',[])]))
        if {p['model_id'] for p in decode.get('comparisons',[]) if p['prompt_tokens']==1048576}>={'llama3_1-8b','llama3_1-70b'}:
            parts.append('<p><strong>为什么70B仍接近？</strong>这是当前GPU简化模型与计算、访存重叠规则下的机制推断：'
                '参考配置的FP16/BF16有效张量吞吐为33.15 TFLOP/s，单token注意力按16行矩阵块扩展计量。'
                '1M时每层attention张量服务估算为8B约8.292ms、70B约16.584ms；相同每层KV在当前NAND参数下的读取扫描下界约16.384ms。'
                '因此8B更容易暴露HBF读取代价，70B较长的计算服务则可覆盖大部分读取；约6.3ms/token的附加时间随共同计算时间增加而被稀释。'
                '这不能推成HBF对大模型影响普遍更小的规律。逐点Core计数确认70B确实读取了HBF；当前结果未保存逐算子关键路径，不能把全部附加时间精确归因给写入。</p>')
        if decode.get('capacity_baseline_observations'):
            parts.append('<h3>原库存的容量负对照</h3>'+table(('模型 / 放置','输入/输出','原HBM GiB','失败的单次申请 GiB','总空闲 GiB','最大连续空闲 GiB','结果'),[
                (r['model_id']+' / '+r['placement'],str(r['prompt_tokens'])+'/'+str(r['output_tokens_per_request']),
                 r['hbm_capacity_bytes']/1024**3,r['failed_allocation_bytes']/1024**3,
                 number(r['capacity_failure']['free_bytes_at_failure']/1024**3) if r.get('capacity_failure',{}).get('free_bytes_at_failure') is not None else None,
                 number(r['capacity_failure']['largest_free_extent_bytes']/1024**3) if r.get('capacity_failure',{}).get('largest_free_extent_bytes') is not None else None,
                 r['reason'])
                for r in decode['capacity_baseline_observations']])+
                '<p>失败发生在规划阶段的物理地址分配，没有完成服务，不产生TPS。单次申请失败不等于整个系统峰值需求；1TiB是本次共同对照配置，不宣称最小可行容量。</p>')
        core_rows=[]
        for row in decode.get('rows',[]):
            audit=row.get('core_execution_audit',{})
            owners=[owner for cohort in audit.get('cohorts',[]) if cohort['phase']=='decode' for owner in cohort['owners']]
            for kind,core_name in (('HBM','DramCore'),('NAND','NandCore')):
                selected=[o for o in owners if o['core']==core_name and (kind=='NAND' or o['memory_kind']==kind)]
                if not selected: continue
                core_rows.append((row['id'],kind+' / '+core_name,sum(o['transactions'] for o in selected),
                    number(sum(o['logical_read_bytes'] for o in selected)/1024**3,5),
                    number(sum(o['logical_write_bytes'] for o in selected)/1024**3,5),
                    row['decode_measurement']['pure_decode_tbt_ns']['count']))
        parts.append('<details><summary>展开：每个Decode点实际Core提交与读写量</summary>'+
            table(('点ID','内存 / Core','事务数','逻辑读 GiB','逻辑写 GiB','TBT样本数'),core_rows)+
            '<p>以上只合计decode cohort；完整请求、迁移及物理传输字节见逐点JSON。HBF物理页放大不等于逻辑流量。</p></details>')
    manifest=report.get('figure_manifest',{})
    if manifest and manifest.get('point_result_version') != POINT_RESULT_VERSION:
        parts.append('<p class="bad">已忽略历史或版本未记录的图表清单；请用本次结果重新生成图表。</p>')
    elif manifest.get('status')=='complete':
        parts.append('<h2 id="figures">主要结果图</h2><p>'+esc(manifest.get('global_caption_zh',''))+'</p>')
        for fig in manifest.get('figures',[]):
            paths=fig.get('report_relative_files',{})
            png=paths.get('png','figures/'+fig['relative_files']['png'])
            svg=paths.get('svg','figures/'+fig['relative_files']['svg'])
            parts.append('<figure style="margin:24px 0"><img style="width:100%;height:auto;background:white" src="'+esc(png)+'" alt="'+esc(fig['title_zh'])+'"><figcaption>'+esc(fig['caption_zh'])+' <a href="'+esc(svg)+'">SVG</a></figcaption></figure>')
    parts.extend(['<h2 id="performance">完整服务性能</h2>',
        '<p>第15组使用显式1/2/4 GPU，其余组按各行GPU数计。每GPU TPS = 节点TPS/GPU数；多GPU节点吞吐来自真实并行任务与通信调度。</p>',
        '<p>TPS 是每秒输出 token 数；goodput 是满足延迟目标的合格 TPS。TTFT 是首 token 等待时间，TPOT 是每请求首 token 后的平均输出间隔，TBT 是相邻 token 的间隔，E2E 是请求总时延；SLO 指本实验设定的延迟目标。每项延迟均保留独立样本数。</p>',
        '<p>主 TPS 和 goodput 使用<strong>最早请求到达至仿真结束</strong>的同一窗口，含请求在初始化期间的等待；另列初始化后 TPS。TTFT/E2E 均从各请求到达起算。下表prefill和decode列均以主请求窗口为分母，表示各阶段token对整段请求的速率贡献，<strong>不是独立Decode TPS</strong>。独立Decode窗口见上方第19组；原有batch忙时速率仍单独保存在JSON。</p>',
        '<p>A=HBM 权重与 KV；B=全部 KV 在 HBF；C=HBM 活跃 KV/HBF 冷 KV；D=HBF 权重/HBM KV。具体目标与控制面实际放置分别保存在结果中。对照约束见每个场景；相同 GPU 或相同槽数不代表相同购置成本和面积。</p>',
        '<p>补充对照 E=主机DRAM中的KV，F=NVMe SSD冷KV，G=CXL连接DRAM中的KV。第9组G是单GPU本地对照，第15组另建多GPU共享池和机架网络变体，路径假设分别保存。</p>',
        '<p>当前未提供真实 token-ID 轨迹；解析 embedding 的远端读取若超过权重表大小，使用明确的循环地址假设保留重复读流量，不扩大权重表驻留容量。不能据此声称复现了真实 token 局部性。</p>',
        '<p>组 02：1 层 hidden=256、4 Q 头/2 KV 头、INT8 小 Dense，每请求 4 输出 token。主实验 HBM 固定0.625ns命令时钟，方向预算1600GB/s；物理接口速率随stack数增加。HBF 的1600/16GB/s是接口读/写预算，4µs/75µs、4KiB页和64个有效单元给出的理想冷阵列读/编程上限仅65.536/3.495253GB/s，实际服务还受传输和队列限制。并行度与内部通道速率均为显式分析假设。named模型、文献探针和研究扩展以每行实际配置为准。</p>'])
    parts.append(table(('ID / 架构','模型','输入/输出','请求数','完成/拒绝','并发上限/实际batch','TPS','初始化后TPS','合格TPS','prefill token/s','decode token/s','初始化µs'),[
        (r['id']+' / '+r['placement'],r['model'],str(r['prompt_tokens'])+'/'+str(r['output_tokens_per_request']),r['request_count'],str(r['completed_requests'])+'/'+str(r['rejected_requests']),str(r['requested_concurrency'])+'/'+str(r['max_observed_batch_requests']),number(r['node_tps'],2),number(r.get('runtime_tps'),2),number(r['goodput_tps'],2),number(r.get('phase_service_rates',{}).get('prefill',{}).get('wall_clock_tokens_per_second'),2),number(r.get('phase_service_rates',{}).get('decode',{}).get('wall_clock_tokens_per_second'),2),number(r.get('measurement_windows',{}).get('initialization_ns',0)/1000)) for r in service]))
    pairs=[]
    for a in service:
        if a['placement']!='A_hbm_only' or not a['id'].startswith(('fixed-','model/','model-capacity/')):
            continue
        key=(a['model'],a['prompt_tokens'],a['output_tokens_per_request'],a['request_count'],a['requested_concurrency'])
        b=next((b for b in service if b['placement']=='B_all_kv_hbf' and b['id'].startswith(('fixed-','model/','model-capacity/')) and
            b.get('fixed_inventory')==a.get('fixed_inventory') and
            (b['model'],b['prompt_tokens'],b['output_tokens_per_request'],b['request_count'],b['requested_concurrency'])==key),None)
        if b: pairs.append((a['model'],a['prompt_tokens'],a['requested_concurrency'],number(a['node_tps'],2),number(b['node_tps'],2),number(b['node_tps']/a['node_tps'],4),number(a['goodput_tps'],2),number(b['goodput_tps'],2)))
    parts.append('<h3>相同模型、请求和并发下的 B/A</h3><p>比值小于 1 表示全部 KV 放入 HBF 后吞吐下降。只能解释表中给定假设和窗口，不能外推为 HBF 的普适加速倍数。</p>')
    parts.append(table(('模型','上下文','并发','A TPS','B TPS','B/A','A合格TPS','B合格TPS'),pairs))
    zero=[r for r in service if r['completed_requests'] and r.get('goodput_tps')==0]
    if zero:
        parts.append('<p class="bad">共有 '+str(len(zero))+' 个已完成请求的服务点 goodput=0，即当前 SLO 下没有合格输出。这是实验结果，不能因为组检查通过而解释为 SLO 达标。</p>')
    parts.append('<h3>所有分位数与样本数</h3><p>下表是合成轨迹的经验分位数，插值不会产生额外样本；即使列出 p999，也不构成真实生产尾延迟保证。</p>')
    parts.append(table(('ID','指标','样本数','p50 ms','p95 ms','p99 ms','p999 ms'),[
        (r['id'],metric,r[metric]['count'],*[number(r[metric][q]/1e6 if r[metric][q] is not None else None,4) for q in ('p50','p95','p99','p999')])
        for r in service for metric in ('ttft_ns','tpot_ns','tbt_ns','e2e_ns')]))
    parts.append('<h2 id="memory">内存与迁移计数</h2><p>分向有效带宽=逻辑读/写字节除以主请求窗口。算子 NAND 流量与 KV 迁移单独列出；尚未记录的队列或物理 plane 指标保持空值。有效并行单元的忙时失衡不等于真实物理 plane 失衡。</p>')
    memory_rows=[]
    for r in service:
        m=r.get('memory_metrics',{}); d=m.get('directional_bandwidth',{}); migration=r.get('migration_ledger',{}).get('bytes_by_kind',{})
        admission=m.get('host_admission_peak_depths',{})
        memory_rows.append((r['id'],number(d.get('read',{}).get('effective_logical_gb_s')),number(d.get('write',{}).get('effective_logical_gb_s')),m.get('page_access_count'),number(m.get('page_access_payload_utilization')),m.get('pages_read'),m.get('pages_programmed'),number(m.get('write_amplification')),admission.get('host_read_outstanding_at_admission_peak'),admission.get('host_write_outstanding_at_admission_peak'),sum(migration.values()),number(r.get('speculative',{}).get('acceptance_rate'))))
    parts.append(table(('ID','逻辑读GB/s','逻辑写GB/s','访问页数','每次访问页载荷率','读取页操作','编程页操作','写放大','host读在途峰值','host写在途峰值','KV迁移B','投机接受率'),memory_rows))
    parts.append('<p>host 在途峰值按每个设备在接纳请求时采样，包含当前请求；缓冲写在主机确认时结束。它不是 NAND 页级读队列或编程队列深度。页载荷率是逻辑 payload/每次访问页跨度，重复访问重复计入，非唯一驻留容量利用率。</p>')
    channel_rows=[]
    for r in service:
        m=r.get('memory_metrics',{}); channels=list(m.get('channel_busy_fraction',{}).values())
        imbalance=m.get('active_service_lane_imbalance',{})
        if channels or imbalance.get('observed_lanes'):
            channel_rows.append((r['id'],len(channels),number(sum(channels)/len(channels) if channels else None,6),
                number(max(channels) if channels else None,6),imbalance.get('observed_lanes'),number(imbalance.get('coefficient_of_variation'))))
    parts.append(table(('ID','观测到的通道数','平均忙时比例','最高忙时比例','观测到的有效服务单元','有效单元忙时变异系数'),channel_rows))
    parts.append('<p>上述有效服务单元只汇总被观察到的资源。物理plane和NAND页队列指标只来自启用详细观测的记录，并包含闲置plane；未启用详细观测的行保留空值，不能混用两类指标。</p>')
    speculative=[r for r in service if r['id'].startswith('speculative/')]
    if speculative:
        parts.append('<h3>投机解码：实际草稿、接受与拒绝</h3>')
        parts.append(table(('ID','继续接受概率参数','草稿token','接受/拒绝token','实际接受比例','TPS','合格TPS'),[
            (r['id'],r.get('parameters',{}).get('speculative_acceptance'),r['speculative']['drafted_tokens'],str(r['speculative']['accepted_draft_tokens'])+'/'+str(r['speculative']['rejected_draft_tokens']),number(r['speculative']['acceptance_rate']),number(r['node_tps'],2),number(r['goodput_tps'],2)) for r in speculative]))
        parts.append('<p>每轮实际提交事件已保存。该实验使用显式合成 MTP 分支和确定性的预期前缀接受规则；输入概率不等于观察接受率，也不是实测模型质量或接受率。</p>')
    prefetched=[r for r in service if r.get('prefetch',{}).get('submitted_bytes') or r['id'].startswith(('prefetch/','prefetch-stress/'))]
    if prefetched:
        parts.append('<h3>异步预取与真正的算子执行交集</h3>')
        parts.append(table(('ID','TPS','提交/完成B','消费B','消费字节比例','其他GPU算子重叠µs','其中decode重叠µs','提前消费交叠µs'),[
            (r['id'],number(r['node_tps'],2),str(r['prefetch']['submitted_bytes'])+'/'+str(r['prefetch']['completed_bytes']),r['prefetch']['useful_bytes'],number(r['prefetch']['useful_byte_fraction']),number(r['prefetch_overlap']['operator_overlap_ns']/1000),number(r['prefetch_overlap']['decode_operator_overlap_ns']/1000),number(r['prefetch_overlap']['premature_consumer_overlap_ns']/1000)) for r in prefetched]))
        parts.append('<p>只与真实 GPU 算子阶段求交集，排除 host 准备和 GPU 前端/控制阶段；算子阶段可含内存等待，不能当作 tensor-core 忙时。零交集表示该场景没有隐藏 GPU 算子时延；固定 lookahead 的消费比例不代表学习型预测器准确率。</p>')
    parts.append('<h2 id="micro">各组机制、微基准及扫描</h2>')
    for g in groups:
        parts.append('<h3>'+esc(g.get('title',g['id']))+'</h3>')
        n=int(g['id'][:2]); rows=g.get('rows',[])
        parts.append(extension_tables(g))
        if n==4:
            parts.append(table(('配置','准入/拒绝','活跃峰值','保留峰值','恢复p50 µs','迁移KiB'),[(r['id'],str(r['admitted_sessions'])+'/'+str(r['rejected_sessions']),r['active_peak'],r['retained_peak'],number((r['resume_latency_p50_ns'] or 0)/1000),number(r['migration_bytes']/1024)) for r in rows]))
            prefixes=[r for r in rows if r['id'].startswith('prefix_')]
            parts.append(table(('前缀实验','目标比例','实际请求比例','复用请求/总请求','成功复用/准入数','准入后实际比例'),[
                (r['id'],number(r['parameters']['prefix_reuse_rate']),number(r['prefix_reuse_rate_offered']),
                 str(r['prefix_offered_sessions'])+'/'+str(r['parameters']['sessions']),
                 str(r['prefix_hits'])+'/'+str(r['admitted_sessions']),number(r['prefix_reuse_rate_admitted'])) for r in prefixes]))
            parts.append('<p>目标与请求比例使用全部提交会话作分母；准入后比例使用成功准入会话作分母。容量拒绝可能改变后者，不能把两个比例混用。</p>')
        elif n==5:
            chosen=[r for r in rows if r['append_bytes']==4096 and r['program_ns']==75000 and r['gc_write_amplification']==1]
            parts.append('<p>以下是固定逻辑写量、4KiB append、75µs program、无额外 GC 放大的代表点；完整组合见 JSON。GC 系数是分析工作量叠加，不是完整 FTL 回收模拟。</p>')
            parts.append(table(('page KiB','缓冲KiB','确认µs','介质排空µs','介质写KiB','缓冲等待µs'),[(r['page_bytes']/1024,r['write_buffer_capacity_bytes']/1024,number(r['host_ack_completion_ns']/1000),number(r['media_drain_completion_ns']/1000),number(r['media_write_bytes']/1024),number(r['write_buffer_blocking_ns']/1000)) for r in chosen]))
        elif n==6:
            parts.append('<p>direct=4GB/s，relay 两跳各8GB/s。这里验证实际传输及共享资源；不能据此宣称同等链路预算的拓扑优势。1.5MiB 扫描指迁移块，物理 NAND page 单独配置。</p>')
            parts.append(table(('路径','完成µs','有效GB/s','链路KiB','relay读写KiB'),[(r['path'],number(r['completion_ns']/1000),number(r['useful_bandwidth_gb_s']),r['link_bytes']/1024,r['relay_bytes']/1024) for r in rows]))
        elif n==7:
            parts.append(table(('HBM/HBF槽','容量页','可运行','服务µs','原因'),[(str(r['hbm_slots'])+'/'+str(r['hbf_slots']),r['aggregate_budget']['capacity_pages'],r['runnable'],number(r['memory_service_duration_ns']/1000 if r['memory_service_duration_ns'] is not None else None),r.get('reason')) for r in rows]))
        elif n==8:
            parts.append(table(('放置/模型','服务µs','HBF读KiB','HBM写B'),[(r['id'],number(r['service_ns']/1000),r['traffic']['hbf']['read']/1024,r['traffic']['hbm']['write']) for r in rows]))
        elif n==11:
            parts.append('<p>下面每个可运行候选都重新执行完整服务；各候选按实际栈数单独计成本。约束是共同上限，不代表各点实际容量、面积或购置成本相等。</p>')
            parts.append(table(('候选','执行/拒绝原因','合格TPS','面积mm²','购置$','条件$/百万合格token','所有预算+SLO可行'),[
                (r['id'],r.get('reason',r['mode']),number(r.get('goodput_tps'),3),number(r['conditional_cost']['memory_package_area_mm2'],1),number(r['conditional_cost']['capex_usd'],0),number(r.get('conditional_slo_cost',{}).get('conditional_usd_per_million_tokens'),3),r['jointly_feasible']) for r in rows]))
        elif n==10:
            parts.append(table(('扫描','HBM比例','目标/实际热读比例','实际读写比','可运行','内存服务µs','迁移KiB'),[
                (r['id'],number(r.get('hbm_fraction')),str(r.get('requested_hot_read_fraction','—'))+'/'+str(r.get('observed_hot_read_fraction','—')),number(r.get('observed_read_write_ratio')),r['runnable'],number(r['memory_service_duration_ns']/1000 if r['memory_service_duration_ns'] is not None else None),number(r.get('migration_bytes',0)/1024)) for r in rows]))
        parts.append('<p>'+esc('；'.join(g.get('limitations',[])))+'</p>')
        parts.append(details('检查明细',g.get('checks',{})))
        for key in ('block_size_scan','slo_search','slo_boundaries','capacity_search','coverage','budgets','selected_candidate_id','feasible_candidate_ids',
                    'scope','ideal_equivalence','serving_comparisons','arrival_boundaries','concurrency_boundaries','measurement_criteria','model_coverage','specification_sources','specifications','bandwidth_targets'):
            if key in g: parts.append(details(key,g[key]))
        parts.append('<p><a href="'+esc(g['id'])+'.json">本组全部原始数据</a></p>')
    parts.append('<h2 id="economics">明确假设下的购置成本、面积及长期磨损敏感性</h2>')
    economics=report.get('economics')
    if economics:
        parts.append('<p>以下结果由显式假设与实测仿真吞吐推导，设备价格、面积及耐久参数均不代表实测或厂商报价。静态预算/长期分析与服务调度结果分别报告。</p><p><a href="economics.json">完整条件估计</a></p>')
        parts.append(details('购置成本与面积输入假设',economics.get('assumptions',{})))
        parts.append(table(('来源实验','条件购置$/百万token','购置成本$'),[
            (r['source_id'],number(r['cost']['conditional_usd_per_million_tokens'],4),number(r['cost']['capex_usd'],2)) for r in economics.get('serving_tps_conditional_projections',[])]))
        parts.append('<p>上述每行把来源 TPS 代入同一假想 2-HBM/2-HBF 节点购置价格公式，未按来源重新计算硬件成本；不能据此宣称某种架构具有真实成本优势。更换费用另列，不含运行费用、人工、停机和税费。</p>')
        parts.append('<h3>3/5 年购置成本摊销</h3><p>以下固定工作时吞吐为 1000 token/s，改变使用年限与工作时间比例，将购置费用按条件输出 token 数摊销。</p>')
        parts.append(table(('年限','工作时间比例','购置成本$','条件购置$/百万token'),[
            (r['years'],number(r['active_fraction'],1),number(r['capex_usd'],2),number(r['conditional_usd_per_million_tokens'],4))
            for r in economics.get('purchase_cost_amortization_rows',[])]))
        parts.append('<p>均匀磨损为独立条件推导，应与直接影响服务时间线的磨损实验分别解读。本节不包含完整 FTL、retention 或坏块机制。</p>')
        wear=[r for r in economics.get('uniform_wear_rows',[]) if r['capacity_bytes']==512_000_000_000 and r['rated_pe_cycles']==3000 and r['additional_gc_factor']==1 and r['study_years']==3]
        parts.append(table(('写率来源','每年P/E','均匀磨损寿命年','3年内更换成本$'),[
            (r['write_rate_source_id'],number(r['pe_cycles_per_year']),number(r['uniform_wear_lifetime_years']),number(r['replacement_cost_usd'])) for r in wear]))
        parts.append(details('条件估计限制',economics.get('limitations',[])))
    else:
        parts.append('<p>本次所选实验未生成经济/长期敏感性结果。</p>')
    parts.append('<h2 id="remaining">本阶段范围与长期研究边界</h2>')
    parts.append(table(('项目','状态','实际限制'),[(r['item'],r['status'],r['reason']) for r in report.get('remaining_research_scope',[])]))
    parts.append('<p>完整服务、结构专属物理探针和条件公式分别标记。文献参数化不等于论文完整复现；未启用的复杂原生算子不会用其他模型更名冒充。所有结论限定于实际运行的模型、参数和窗口。</p></main></body></html>')
    return '\n'.join(parts)
