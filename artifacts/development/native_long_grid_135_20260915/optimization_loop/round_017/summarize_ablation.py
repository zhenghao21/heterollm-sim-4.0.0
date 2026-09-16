from pathlib import Path
import json,datetime,hashlib,html,statistics
P=Path(__file__).resolve().parent
M=('engine_ttft_ms','engine_tpot_ms','engine_e2e_ms')
def ref(p):
 b=p.read_bytes();return {'path':str(p.resolve()),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
scores={v:json.loads((P/v/'errors.0001.json').read_text(encoding='utf-8')) for v in ('pure','current','conversion_cta')}
rows={v:{c['cell_id']:c for c in d['cells'] if all(c['metrics'].get(m,{}).get('status')=='scored' for m in M)} for v,d in scores.items()}
ids=sorted(rows['current']);assert len(ids)==20 and all(set(x)==set(ids) for x in rows.values())
changes=[]
for ident in ids:
 for metric in M:
  a,b=rows['current'][ident]['metrics'][metric],rows['conversion_cta'][ident]['metrics'][metric]
  if a['simulator_median_ms']!=b['simulator_median_ms']:changes.append({'cell_id':ident,'metric':metric,'before':a['simulator_median_ms'],'after':b['simulator_median_ms']})
result={'schema':'source-conversion-cta-ablation/v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'evidence':[ref(P/v/'errors.0001.json') for v in scores],'score_scope':'20 common preregistered development cells; not full131 or independent acceptance',
 'native_remeasured':False,'calibration_used':False,'common_cells':20,'metric_comparisons':60,'changed_metric_count':len(changes),'changes':changes,
 'summary':{v:d['overall'] for v,d in scores.items()},'by_model':{v:d['by_model'] for v,d in scores.items()},
 'decision':'Retain opt-in structural cap and ledger; no accuracy promotion, no full131 rerun justified by zero anchor effect. Proceed to source instructions and host graph submission evidence.',
 'strict_gate':json.loads((P/'strict_gate.json').read_text(encoding='utf-8'))['gate_A']}
with (P/'ablation.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
lines=['# R17 转换CTA上界三路消融','', '20个共同开发场景，三项共60次比较：**新旧机制变化0项，0/20场景逐格三项<10%，A门失败，B门未验证。**', '', '三路采用相同代码、硬件与固定native数据，仅改变预先冻结的MMQ/CTA开关。pure是关闭本轮MMQ/CTA处理的分析对照，保留既有执行语义。没有使用本轮被拒收的算子时间校准。', '', '| 对照 | TTFT 中位/P90/最坏 APE | TPOT 中位/P90/最坏 APE | E2E 中位/P90/最坏 APE |','|---|---|---|---|']
for v,d in scores.items():
 vals=[]
 for m in M:
  e=d['overall'][m]['absolute_percentage_error_pct'];vals.append(' / '.join(f'{e[k]:.2f}%' for k in ('median','p90','max')))
 lines.append('| '+v+' | '+' | '.join(vals)+' |')
lines+=['','仅修正计算资源的必要上界不足以改变这组场景的主要耗时：内存/其他依赖仍主导该阶段。保留默认关闭的结构改进和可审计账本，不宣称降低误差，也不为获得同一答案重跑全131格。','', '下一轮改为独立验证多算子图的CPU提交、driver/API、GPU设备及同步边界，补充转换指令与量化参考证据。全部145/234阶段和0/26算子准入失败历史保留。','', '固定131分母中本次仅20格评分，111格没有该冻结的新结果；不得拼接旧版本成为131格通过。']
(P/'ablation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
w,h,left,cw,rh=1220,140+20*28,340,95,28
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="#fff"/><style>text{{font-family:Segoe UI,sans-serif;font-size:12px}}</style><text x="20" y="25" style="font-size:20px;font-weight:600">R17 — 20 development cells · absolute percentage error</text><text x="20" y="50">Current vs source-CTA: 0 / 60 timing changes. Strict three-metric pass: 0 / 20.</text>']
for group,v in enumerate(scores):
 svg.append(f'<text x="{left+group*3*cw+80}" y="78" text-anchor="middle" font-weight="bold">{v}</text>')
 for col,label in enumerate(('TTFT','TPOT','E2E')):svg.append(f'<text x="{left+(group*3+col)*cw+cw/2}" y="100" text-anchor="middle">{label}</text>')
for ri,ident in enumerate(ids):
 y=113+ri*rh;label=ident.replace('__fixed_runtime','');svg.append(f'<text x="12" y="{y+18}">{html.escape(label)}</text>')
 for gi,v in enumerate(scores):
  for mi,m in enumerate(M):
   val=rows[v][ident]['metrics'][m]['absolute_percentage_error_pct'];t=min(val/85,1);rgb=tuple(round(a+(b-a)*t) for a,b in zip((232,245,233),(230,97,74)));color='#%02x%02x%02x'%rgb;x=left+(gi*3+mi)*cw
   svg.append(f'<rect x="{x}" y="{y}" width="{cw-3}" height="{rh-2}" fill="{color}"/><text x="{x+cw/2}" y="{y+18}" text-anchor="middle">{val:.1f}%</text>')
svg.append('</svg>');(P/'ablation_heatmap.svg').write_text(''.join(svg),encoding='utf-8')
print(json.dumps({'changed':len(changes),'common':len(ids),'summary':str(P/'ablation.md')}))
