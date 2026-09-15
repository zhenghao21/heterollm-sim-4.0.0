"""Summarize explicitly pinned optimization evaluations; never run predictions.

Example:
  python tools/render_optimization_loop_report.py --loop-state LOOP/state.json \
    --evaluation v2=DATA/stable_simulation_v2 \
    --evaluation R1=LOOP/round_001/candidate --score-file R1=errors.0002.json \
    --evaluation R5=LOOP/round_005/f32_and_gather_r2 --portable-copy PORTABLE

A directory with no score is pending. A directory with several errors.*.json
files is pending_ambiguous until --score-file LABEL=FILE pins one. Never select
by timestamp, version number, error, state.best_candidate or directory name.
The static portable copy contains reports and optional existing heatmaps only.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
from urllib.parse import quote

SCHEMA = 'optimization-loop-report/v1'
GROUPS = ('qwen25','qwen35','qwen38','qwen38_gpu','smollm2','tinyllama')
LABELS = {'qwen25':'Qwen2.5','qwen35':'Qwen3.5','qwen38':'27B CPU',
          'qwen38_gpu':'27B GPU','smollm2':'SmolLM2','tinyllama':'TinyLlama'}
METRICS = {'ttft':'engine_ttft_ms','tpot':'engine_tpot_ms','e2e':'engine_e2e_ms'}
FIELDS = ('absolute_percentage_error_pct','absolute_error_ms','signed_error_pct','signed_error_ms')
MAX_JSON_BYTES = 128 * 1024 * 1024


class EvidenceError(ValueError):
    pass


def read_json(path):
    path = Path(path).resolve()
    if path.stat().st_size > MAX_JSON_BYTES:
        raise EvidenceError('JSON 超出有界读取限制: ' + str(path))
    raw = path.read_bytes()
    document = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(document, dict):
        raise EvidenceError('证据根节点必须为对象: ' + str(path))
    return document, {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}


def number(value):
    return type(value) in (int,float) and math.isfinite(value)


def rows_by_id(rows):
    result = {}
    for row in rows or []:
        ident = row.get('cell_id', row.get('id'))
        if not isinstance(ident,str) or not ident or ident in result:
            raise EvidenceError('cell_id 缺失或重复')
        result[ident] = row
    return result


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1)*q
    lo,hi = math.floor(index),math.ceil(index)
    return values[lo] + (values[hi]-values[lo])*(index-lo)


def statistics_for(values, *, signed=False):
    values = list(values)
    if not values:
        return {'count':0,'median':None,'p90':None,'worst':None,'min':None,'max':None}
    return {'count':len(values),'median':statistics.median(values),'p90':percentile(values,.9),
            'worst':max(values,key=abs) if signed else max(values),'min':min(values),'max':max(values)}


def raw_identity(refs):
    """Compare declared immutable raw identities, without opening native payloads."""
    paths = {}
    for ref in refs:
        digest = ref.get('sha256')
        name = ref.get('path')
        if not isinstance(name,str) or not name or not isinstance(digest,str) or not re.fullmatch('[0-9a-fA-F]{64}',digest):
            raise EvidenceError('原生 raw 引用缺少 path/SHA256')
        name = name.replace('\\','/').casefold()
        digest = digest.lower()
        if name in paths and paths[name] != digest:
            raise EvidenceError('同一路径记录了冲突的 raw SHA256')
        paths[name] = digest
    if not paths:
        raise EvidenceError('原生 raw 身份缺失')
    return paths


def native_cell_raws(row):
    refs = [item['raw_ref'] for item in row.get('native_actuals',[]) if isinstance(item.get('raw_ref'),dict)]
    if not refs:
        refs = [item['raw_ref'] for item in row.get('static_hardware',{}).get('state_refs',[]) if isinstance(item.get('raw_ref'),dict)]
    return raw_identity(refs)


def compact_rounds(state, state_path):
    result = []
    for item in state.get('rounds',[]):
        text = json.dumps(item,ensure_ascii=False).lower()
        flags = []
        for flag,words in [('zero_effect',('zero_applied','zero_effect','no_effect')),
                           ('deferred',('deferred',)),('failed/regressed',('accuracy_failed','failed_accuracy','regressed'))]:
            if any(word in text for word in words):flags.append(flag)
        if item.get('native_dispatch_proven') is False:flags.append('conditional')
        if 'unpriced' in text or 'unmodeled' in text:flags.append('unpriced')
        refs = {k:v for k,v in item.items() if isinstance(v,str) and (k.endswith('_ref') or k in ('evidence','comparison_ref','candidate','repaired_candidate','full_candidate'))}
        result.append({'round':item.get('round'),'status':item.get('status','pending'),'flags':flags,
            'hypothesis':item.get('hypothesis'), 'decision':item.get('decision',item.get('selection_reason')),
            'original_run_status':item.get('original_run_status'),'evidence':refs,
            'commit':item.get('commit'),'push_status':item.get('push_status')})
    return result


def evidence_path(ref, base):
    raw = ref.get('path') if isinstance(ref,dict) else None
    if not isinstance(raw,str) or not raw:
        raise EvidenceError('证据引用缺少路径')
    path = Path(raw)
    return path if path.is_absolute() else Path(base)/path


def treatments(freeze):
    result = {}
    for key in ('recurrent_batching','cpu_iq_panel_reuse','slot_order','host_offload_source','tensor_storage','gpu_invocation'):
        value = freeze.get(key)
        if value is None:continue
        serialized = json.dumps(value,ensure_ascii=False)
        result[key] = {'enabled_input':True,'conditional':True,
            'unpriced_or_unmodeled_declared':bool(re.search(r'unpriced|unmodeled',serialized,re.I)),
            'native_dispatch_not_inferred_from_errors':True}
    return result


def metric_values(record):
    if not isinstance(record,dict) or record.get('status') != 'scored':
        return None
    required = ('native_median_ms','simulator_median_ms',*FIELDS)
    if not all(number(record.get(key)) for key in required) or record['native_median_ms'] <= 0 or record['simulator_median_ms'] < 0:
        raise EvidenceError('已评分指标含缺失、非有限值或非法时延')
    delta = record['simulator_median_ms']-record['native_median_ms']
    expected = {'signed_error_ms':delta,'absolute_error_ms':abs(delta),
        'signed_error_pct':100*delta/record['native_median_ms'],
        'absolute_percentage_error_pct':100*abs(delta)/record['native_median_ms']}
    for key,value in expected.items():
        if not math.isclose(record[key],value,rel_tol=1e-9,abs_tol=1e-8):
            raise EvidenceError('评分算式不一致: ' + key)
    return {key:record[key] for key in required}


def load_evaluation(label, directory, score_name, selection, selection_ref, selected, raw_map):
    directory = Path(directory).resolve()
    result = {'label':label,'directory':str(directory),'status':'pending','reason':None,
        'sources':{},'groups':[],'cells':[], 'prediction_completed':None,'all3_scored':None,
        'all3_strict_below10':None,'fixed_selected_denominator':len(selected),
        'original_grid_denominator':selection['planned_cells'],'blind_evaluation':False,
        'dataset_role':'development/regression；非独立盲测','accuracy_promotion':False,
        'treatments':{},'existing_visuals':[]}
    freeze_path = directory/'freeze.json'
    if not freeze_path.is_file():
        result['reason']='freeze 尚未生成'
        return result
    freeze,freeze_ref = read_json(freeze_path)
    result['sources']['freeze']=freeze_ref
    if freeze.get('schema')!='stable-native-simulation-freeze/v1':raise EvidenceError('freeze schema 不匹配')
    if freeze.get('selection_sha256') != selection_ref['sha256'] or freeze.get('selection_ref',{}).get('sha256') != selection_ref['sha256']:
        raise EvidenceError('freeze 与固定 selection SHA 不同')
    if freeze.get('selected_denominator') != len(selected) or freeze.get('native_grid_denominator') != selection['planned_cells']:
        raise EvidenceError('freeze 分母与固定选择不一致')
    frozen=rows_by_id(freeze.get('cells'));identities=[]
    for cell,row in frozen.items():
        if cell not in selected:raise EvidenceError('freeze 包含固定选择之外的 cell')
        refs=row.get('static_inputs',{}).get('measurement_state_refs',[])
        actual=raw_identity([r.get('raw_ref',r) for r in refs])
        if actual!=raw_map[cell]:raise EvidenceError('逐格原生 raw SHA 不一致: '+cell)
        identities.append({'cell_id':cell,'raw_sha256':sorted(actual.values())})
    result['native_identity']={'status':'all_frozen_cells_match_fixed_raw_refs','checked_cells':len(identities),
        'cell_raw_identities':identities,'raw_payloads_rehashed':False,
        'scope':'逐格核对冻结/selection 声明的 path+SHA；不读取原生 raw，不重新执行 native'}
    result['treatments']=treatments(freeze)
    if score_name is not None:
        score_path=Path(score_name)
        if not score_path.is_absolute():score_path=directory/score_path
        if not score_path.is_file():result['reason']='明确指定的评分尚未生成';return result
    else:
        choices=sorted(p for p in directory.glob('errors.*.json') if re.fullmatch(r'errors\.\d+\.json',p.name))
        if len(choices)!=1:
            result['status']='pending_ambiguous' if choices else 'pending'
            result['reason']='多个评分版本，请用 --score-file LABEL=FILE 固定，不自动选最新' if choices else '尚无评分文件'
            result['available_score_files']=[p.name for p in choices]
            return result
        score_path=choices[0]
    scores,score_ref=read_json(score_path)
    result['sources']['score']=score_ref
    if scores.get('schema')!='stable-native-simulation-errors/v1':raise EvidenceError('评分 schema 不匹配')
    if scores.get('freeze_ref',{}).get('sha256')!=freeze_ref['sha256']:raise EvidenceError('评分与目录当前 freeze 不同；禁止混用版本')
    if scores.get('selected_denominator')!=len(selected):raise EvidenceError('评分固定选择分母不一致')
    native_ref=scores.get('native_report_ref',{})
    if native_ref.get('sha256')!=selection_ref['sha256']:
        native_path=evidence_path(native_ref,directory)
        if not native_path.is_file():raise EvidenceError('旧原生报告不可读取，不能验证其 selection 绑定')
        native,actual_ref=read_json(native_path)
        if actual_ref['sha256']!=native_ref.get('sha256') or native.get('selection_payload_sha256')!=selection.get('payload_sha256'):
            raise EvidenceError('评分旧原生报告身份或 selection 绑定不一致')
        result['sources']['native_report']=actual_ref
    result['reported_role']={'evaluation_type':scores.get('evaluation_type'),'blind_evaluation':scores.get('blind_evaluation'),
        'formal_prediction_eligible':scores.get('formal_prediction_eligible'),'calibration_applied':scores.get('calibration_applied')}
    if scores.get('blind_evaluation') is True:
        result['role_warning']='评分声称 blind，但本循环固定已披露选择；不作盲测晋级'
    scored=rows_by_id(scores.get('cells'));completed=0;missing_prediction_files=[]
    for cell,row in scored.items():
        if cell not in frozen:raise EvidenceError('评分包含该 freeze 之外的 cell')
        pred_ref=row.get('prediction_ref');prediction_state='pending'
        if isinstance(pred_ref,dict):
            pred_path=evidence_path(pred_ref,directory)
            if pred_path.is_file():
                prediction,actual=read_json(pred_path)
                if actual['sha256']!=pred_ref.get('sha256') or prediction.get('cell_id')!=cell or prediction.get('freeze_ref',{}).get('sha256')!=freeze_ref['sha256']:
                    raise EvidenceError('预测 SHA/cell/freeze 与评分不一致: '+cell)
                prediction_state=prediction.get('status','pending')
                completed+=int(prediction_state=='predicted')
            else:missing_prediction_files.append(cell);prediction_state='snapshot_unavailable'
        metrics={}
        for metric,key in METRICS.items():
            value=metric_values(row.get('metrics',{}).get(key))
            if value is not None:
                expected_native=selected[cell].get('metrics',{}).get(metric,{}).get('native_median_ms')
                if not number(expected_native) or not math.isclose(value['native_median_ms'],expected_native,rel_tol=1e-10,abs_tol=1e-9):
                    raise EvidenceError('评分原生中位数与固定 selection 不一致: '+cell+'/'+metric)
                metrics[metric]=value
        if metrics and (not isinstance(pred_ref,dict) or prediction_state not in ('predicted','snapshot_unavailable')):
            raise EvidenceError('已评分指标没有对应成功预测: '+cell)
        result['cells'].append({'cell_id':cell,'group':selected[cell].get('placement_group',selected[cell].get('model_key')),
            'prediction_state':prediction_state,'prediction_ref':pred_ref,'metrics':metrics,
            'all3_strict_below10':all(metrics[m]['absolute_percentage_error_pct']<10 for m in METRICS) if len(metrics)==3 else None})
    result['prediction_completed']=None if missing_prediction_files else completed
    result['observed_existing_prediction_completed']=completed
    result['prediction_files_unavailable']=missing_prediction_files
    complete=[r for r in result['cells'] if len(r['metrics'])==3]
    result['all3_scored']=len(complete)
    result['all3_strict_below10']=sum(r['all3_strict_below10'] for r in complete) if complete else None
    result['status']='scored' if len(complete)==len(selected) else 'partial'
    if missing_prediction_files:result['status']='scored_snapshot_missing_predictions'
    result['prediction_completion_basis']='实际可读取且匹配评分 SHA/freeze 的 prediction.status；缺文件不补零'
    result['group_completion']=[]
    for group in GROUPS:
        native_cells=[r for r in selected.values() if r.get('placement_group',r.get('model_key'))==group]
        group_cells=[r for r in result['cells'] if r['group']==group]
        fully_scored=[r for r in group_cells if len(r['metrics'])==3]
        result['group_completion'].append({'group':group,'selected_cells':len(native_cells),
            'all3_scored':len(fully_scored),
            'all3_strict_below10':sum(r['all3_strict_below10'] for r in fully_scored) if fully_scored else None})
        for metric in METRICS:
            values=[r['metrics'][metric] for r in group_cells if metric in r['metrics']]
            result['groups'].append({'group':group,'metric':metric,'scored_cells':len(values),'selected_cells':len(native_cells),
                **{field:statistics_for((r[field] for r in values),signed=field.startswith('signed_')) for field in FIELDS}})
    for name in ('report.html','heatmap_ttft.png','heatmap_tpot.png','heatmap_e2e.png',
                 'heatmap_ttft.svg','heatmap_tpot.svg','heatmap_e2e.svg'):
        p=directory/name
        if p.is_file():result['existing_visuals'].append(str(p))
    return result


def make_summary(state_path,evaluations,score_files=None,selection_path=None):
    state_path=Path(state_path).resolve();state,state_ref=read_json(state_path)
    selection_path=Path(selection_path) if selection_path else evidence_path(state.get('native_selection_ref'),state_path.parent)
    selection,selection_ref=read_json(selection_path)
    if selection_ref['sha256']!=state.get('native_selection_ref',{}).get('sha256'):raise EvidenceError('loop state 固定 selection SHA 不一致')
    selected=rows_by_id(selection.get('selected_cells'));excluded=rows_by_id(selection.get('excluded_cells'))
    if set(selected)&set(excluded) or selection.get('selected_count')!=len(selected) or selection.get('excluded_count')!=len(excluded):
        raise EvidenceError('选择/排除记录分母或唯一性不一致')
    raw_map={cell:native_cell_raws(row) for cell,row in selected.items()};owners={}
    for cell,refs in raw_map.items():
        for digest in refs.values():
            if digest in owners and owners[digest]!=cell:raise EvidenceError('同 raw SHA 被分配给多个不同 cell')
            owners[digest]=cell
    if state.get('native_raw_refs') is not None:
        state_raws=raw_identity(state['native_raw_refs']);union={k:v for refs in raw_map.values() for k,v in refs.items()}
        if state_raws!=union:raise EvidenceError('loop state raw 清单与固定选择不一致')
    labels=[label for label,_ in evaluations]
    if len(labels)!=len(set(labels)):raise EvidenceError('evaluation LABEL 重复')
    score_files=score_files or {}
    if set(score_files)-set(labels):raise EvidenceError('--score-file 引用了不存在的 LABEL')
    results=[]
    for label,directory in evaluations:
        try:result=load_evaluation(label,directory,score_files.get(label),selection,selection_ref,selected,raw_map)
        except (OSError,ValueError,TypeError,KeyError) as exc:
            result={'label':label,'directory':str(Path(directory).resolve()),'status':'integrity_failed','reason':str(exc),
                'sources':{},'groups':[],'cells':[],'prediction_completed':None,'all3_scored':None,'all3_strict_below10':None,
                'existing_visuals':[],'treatments':{},'blind_evaluation':False,'accuracy_promotion':False}
        results.append(result)
    attempts=selection.get('failed_attempts');failed_attempts=[]
    if isinstance(attempts,list):
        for row in attempts:failed_attempts.append({k:row.get(k) for k in ('cell_id','source_id','status','error','raw_ref','receipt_ref')})
    reasons=Counter(reason for row in excluded.values() for reason in set(row.get('reasons',[])))
    return {'schema':SCHEMA,'sources':{'state':state_ref,'selection':selection_ref},'loop_status':state.get('status','pending'),
        'dataset_role':'已披露 development/regression；不是独立盲测','blind_evaluation':False,'accuracy_promotion':False,
        'selection_policy':'仅汇总 CLI 明确指定的版本；不按误差/时间选择候选，不合并不同 freeze 的单元',
        'denominators':{'original_grid':selection.get('planned_cells'),'fixed_selected':len(selected),'excluded':len(excluded),
            'failed_native_attempts':len(attempts) if isinstance(attempts,list) else None},
        'native_raw_identity':{'unique_raw_sha256':len(owners),'selected_cells_checked':len(raw_map),'payload_rehash':False},
        'native_coverage':selection.get('coverage'), 'excluded_cells':list(excluded.values()),
        'exclusion_reason_counts':dict(sorted(reasons.items())),'failed_native_attempts':failed_attempts,
        'budget':{**state.get('budget',{}),'full_evaluations_started':state.get('full_131_evaluations_started'),
            'full_evaluations_completed':state.get('full_131_evaluations_completed'),'round_records':len(state.get('rounds',[]))},
        'targets_recorded_only':state.get('targets'),'rounds':compact_rounds(state,state_path),
        'version_control_receipts':state.get('version_control_receipts',[]),
        'git_verification_scope':'仅呈现已有 push receipt，本汇总不提交、不推送、不重新访问远端',
        'operator_microbenchmark':state.get('operator_microbenchmark'),'evaluations':results,
        'statistics_definition':{'unit':'每个已评分 cell 等权；不是按请求数加权','p90':'线性插值',
            'signed_error':'sim - native','signed_worst':'绝对值最大的误差，保留符号','strict_all3':'三个指标 APE 都严格小于 10；缺项不算通过也不补零'},
        'limitations':['表格只来自已保存评分，不重新计算原生测量、不推测 dispatch、不做准确性晋级。',
            'logical 行流量不等于 DRAM 事务；条件性/未计价字段需连同每轮证据阅读。',
            '缺评分与缺源文件显示 pending/未核验；原 162、固定 131 和预测完成数使用不同分母。']}


def fmt(value,signed=False):
    return 'pending' if value is None else format(value,'+.2f' if signed else '.2f')


def triple(stats,signed=False):
    return ' / '.join(fmt(stats.get(k),signed) for k in ('median','p90','worst'))


def relative_link(target,output):
    try:return quote(os.path.relpath(target,output).replace(os.sep,'/'),safe='/._-')
    except ValueError:return None


def md_text(value):
    return html.escape(str(value if value is not None else 'pending')).replace('|','\\|').replace('\n',' ')


def render(summary,output,*,portable=False):
    output=Path(output);den=summary['denominators'];line=lambda x:md_text(x)
    parts=['# 优化循环汇总','',f"原定 **{den['original_grid']}** 格 · 固定选择 **{den['fixed_selected']}** 格 · 排除 **{den['excluded']}** 格 · 保留失败 attempt **{line(den['failed_native_attempts'])}** 次。",'',
        '**已披露开发/回归数据，非独立盲测。报告不选择准确性赢家，不作晋级。**','',
        '## 版本与完成分母','','| 明确指定版本 | 状态 | 预测完成 / 固定选择 / 原定 | 三指标已评分 | 三项均严格 <10% |','|---|---|---|---:|---:|']
    overview_rows=[]
    for e in summary['evaluations']:
        count='pending' if e['prediction_completed'] is None else str(e['prediction_completed'])
        denom=f"{count} / {den['fixed_selected']} / {den['original_grid']}"
        row=[e['label'],e['status'],denom,e['all3_scored'] if e['all3_scored'] is not None else 'pending',e['all3_strict_below10'] if e['all3_strict_below10'] is not None else 'pending']
        overview_rows.append(row);parts.append('| '+' | '.join(line(v) for v in row)+' |')
    esc=html.escape
    def table(headers,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th scope="col">'+esc(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    body=['<header><p class="eyebrow">固定数据 · 明确版本 · 可追溯</p><h1>优化循环汇总</h1><p>'+esc(f"原定 {den['original_grid']} 格 / 固定 {den['fixed_selected']} 格 / 排除 {den['excluded']} 格 / 失败 attempt {den['failed_native_attempts']}")+'</p><p class="notice">开发/回归数据，非独立盲测。没有自动选优，也没有准确性晋级。</p></header>',
        '<section><h2>版本与完成分母</h2>'+table(['版本','状态','预测完成 / 固定 / 原定','三指标已评分','三项均 <10%'],overview_rows)+'</section>']
    for index,e in enumerate(summary['evaluations'],1):
        parts+=['',f"## {line(e['label'])}",'']
        rows=[]
        for g in e['groups']:
            rows.append([LABELS[g['group']],g['metric'].upper(),f"{g['scored_cells']} / {g['selected_cells']}",
                triple(g['absolute_percentage_error_pct']),triple(g['absolute_error_ms']),triple(g['signed_error_pct'],True),triple(g['signed_error_ms'],True)])
        headers=['组','指标','评分 / 固定','APE% 中位/P90/最坏','绝对 ms 中位/P90/最坏','Δ% 中位/P90/最坏带符号','Δms 中位/P90/最坏带符号']
        if rows:
            parts+=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
            parts+=['| '+' | '.join(line(v) for v in row)+' |' for row in rows]
        else:parts.append('pending：'+line(e.get('reason','尚未评分')))
        extras=[]
        if e.get('group_completion'):
            extras.append('分组三项均 <10%（通过 / 固定选择）：'+'；'.join(LABELS[g['group']]+' '+str(g['all3_strict_below10'] if g['all3_strict_below10'] is not None else 'pending')+' / '+str(g['selected_cells']) for g in e['group_completion']))
        if e.get('reason'):extras.append(e['reason'])
        if e.get('role_warning'):extras.append(e['role_warning'])
        flags=[name+('：conditional/unpriced' if t['unpriced_or_unmodeled_declared'] else '：conditional') for name,t in e.get('treatments',{}).items()]
        if flags:extras.append('机制合同：'+'；'.join(flags))
        links=[]
        for target in e.get('existing_visuals',[]):
            link=relative_link(target,output)
            if link:links.append((Path(target).name,link))
        evidence=[]
        for kind,r in e.get('sources',{}).items():
            link=relative_link(r['path'],output) if not portable else None
            text=kind+' · '+Path(r['path']).name+' · SHA '+r['sha256'][:16]
            parts.append('- '+('['+line(text)+']('+link+')' if link else line(text)+'（原始证据保留本地）'))
            evidence.append('<li>'+('<a href="'+esc(link,quote=True)+'">'+esc(text)+'</a>' if link else esc(text)+'（本地证据）')+'</li>')
        parts+=['- '+line(x) for x in extras]
        if links:parts.append('- 既有热图：'+' · '.join('['+line(name)+']('+url+')' for name,url in links))
        body.append('<section><h2>'+esc(e['label'])+'</h2>'+''.join('<p>'+esc(x)+'</p>' for x in extras)+
            ('<details '+('open' if index==1 else '')+'><summary>六组逐指标统计</summary>'+table(headers,rows)+'</details>' if rows else '<p class="pending">pending：'+esc(e.get('reason') or '尚未评分')+'</p>')+
            '<p>'+ ' · '.join('<a href="'+esc(url,quote=True)+'">'+esc(name)+'</a>' for name,url in links)+'</p><ul>'+''.join(evidence)+'</ul></section>')
    parts+=['','## 轮次状态','','| 轮次 | 已记录状态 | 限制标记 |','|---|---|---|']
    round_rows=[]
    for r in summary['rounds']:
        rr=[r['round'],r['status'],', '.join(r['flags']) or '见原证据'];round_rows.append(rr);parts.append('| '+' | '.join(line(x) for x in rr)+' |')
    budget=summary['budget']
    budget_labels={'max_mechanism_rounds':'机制轮次上限','max_full_131_evaluations':'全量评估上限','max_wall_seconds':'总时限秒','deadline_utc':'截止 UTC','prediction_workers':'预测并发数','per_cell_timeout_seconds':'单格超时秒','full_evaluations_started':'全量已开始','full_evaluations_completed':'全量已完成','round_records':'状态轮次记录'}
    budget_text='；'.join(budget_labels.get(k,k)+'='+str(v if v is not None else 'pending') for k,v in budget.items())
    commits=[r for r in summary['version_control_receipts'] if str(r.get('status','')).startswith('pushed')]
    git_rows=[[r.get('kind'),r.get('commit'),r.get('pushed_to'),r.get('status')] for r in commits]
    parts+=['','## 预算与 Git','','预算快照：'+line(budget_text),'']
    parts+=['- '+line(' / '.join(str(x) for x in row)) for row in git_rows] or ['- pending：没有已推送 receipt']
    parts+=['',summary['git_verification_scope'],'','## 原生保留记录','',
        '排除原因：'+line('；'.join(k+'='+str(v) for k,v in summary['exclusion_reason_counts'].items())),
        '完整排除 cell、失败 attempt 原因与证据引用见 summary.json；不重算原生结果。','','## 口径与限制','',*['- '+text for text in summary['limitations']],
        '- 每格等权；P90 线性插值；signed=sim−native；signed 最坏按绝对值选取并保留符号。']
    body+=['<section><h2>轮次状态</h2>'+table(['轮次','已有状态','标记'],round_rows)+'</section>',
        '<section><h2>预算与已推送 Git 记录</h2><p>'+esc(budget_text)+'</p>'+table(['记录','commit','上游','状态'],git_rows)+'<p>'+esc(summary['git_verification_scope'])+'</p></section>',
        '<section><h2>原生排除和失败历史</h2><p>'+esc('；'.join(k+'='+str(v) for k,v in summary['exclusion_reason_counts'].items()))+'</p><p>完整记录见 <a href="summary.json">summary.json</a>；失败 attempt 不从原定分母中消失。</p></section>',
        '<footer><h2>口径与限制</h2><ul>'+''.join('<li>'+esc(x)+'</li>' for x in summary['limitations'])+'</ul><p>每格等权 · P90 线性插值 · signed = sim − native · signed 最坏保留最大绝对值的符号。</p></footer>']
    css='body{font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;background:#f5f7fa;color:#162638;margin:0}main{max-width:1300px;margin:auto;padding:32px 24px}header,section,footer{background:white;padding:24px;margin-bottom:18px;border:1px solid #dce3ec;border-radius:10px}h1{font-size:32px;margin:0 0 12px}h2{font-size:20px}.eyebrow{color:#4a637d}.notice{border-left:4px solid #dd9830;padding-left:14px}.pending{color:#855c12}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #dde3eb;vertical-align:top}th{background:#edf2f8;white-space:nowrap}tr:nth-child(even){background:#fafbfd}td:nth-child(n+3){white-space:nowrap}a{color:#155f96}summary{cursor:pointer;font-weight:600;padding:10px 0}li{overflow-wrap:anywhere}@media(max-width:640px){main{padding:12px}header,section,footer{padding:16px}h1{font-size:26px}}'
    html_doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>优化循环汇总</title><style>'+css+'</style><main>'+''.join(body)+'</main></html>'
    return '\n'.join(parts)+'\n',html_doc


def write_report(summary,output,*,portable=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    protected={Path(r['path']).resolve() for r in summary['sources'].values()}
    for evaluation in summary['evaluations']:
        protected.update(Path(r['path']).resolve() for r in evaluation.get('sources',{}).values())
        if not portable:protected.update(Path(p).resolve() for p in evaluation.get('existing_visuals',[]))
    if any((output/name).resolve() in protected for name in ('summary.json','report.md','report.html')):
        raise EvidenceError('输出路径与已有输入证据/图表冲突')
    md,doc=render(summary,output,portable=portable)
    for name,text in [('summary.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n'),('report.md',md),('report.html',doc)]:
        (output/name).write_text(text,encoding='utf-8')


def portable_copy(summary,destination):
    destination=Path(destination).resolve();copy=json.loads(json.dumps(summary));copy['portable_view']=True
    for index,e in enumerate(copy['evaluations'],1):
        visuals=[]
        for target in e.get('existing_visuals',[]):
            source=Path(target);dest=destination/'evaluations'/str(index)/source.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if source.resolve()!=dest.resolve():shutil.copy2(source,dest)
            visuals.append(str(dest))
        e['existing_visuals']=visuals
    write_report(copy,destination,portable=True)


def assignments(items):
    result=[];seen=set()
    for item in items:
        if '=' not in item:raise EvidenceError('需要 LABEL=PATH: '+item)
        label,value=item.split('=',1)
        if not label.strip() or not value.strip() or label in seen:raise EvidenceError('LABEL/路径为空或标签重复')
        seen.add(label);result.append((label,value))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--loop-state',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--evaluation',action='append',required=True,metavar='LABEL=DIR')
    parser.add_argument('--score-file',action='append',default=[],metavar='LABEL=FILE',help='明确评分快照，相对路径以对应 evaluation 目录为基准')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--portable-copy',type=Path)
    args=parser.parse_args(argv)
    summary=make_summary(args.loop_state,assignments(args.evaluation),dict(assignments(args.score_file)),args.selection)
    output=args.output or args.loop_state.resolve().parent
    write_report(summary,output)
    if args.portable_copy:portable_copy(summary,args.portable_copy)
    failed=[e['label'] for e in summary['evaluations'] if e['status']=='integrity_failed']
    print(json.dumps({'output':str(Path(output).resolve()),'evaluations':{e['label']:e['status'] for e in summary['evaluations']},
        'denominators':summary['denominators'],'integrity_failed':failed,'accuracy_promotion':False},ensure_ascii=False))
    return 2 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
