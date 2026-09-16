"""R18 three-way development report; imports are inert and no prediction is run.

Root invokes only after all three20-cell predictions/scores are complete.
Native raw files are not parsed. All outputs are exclusive-create. Gate B stays
unvalidated and111 unrun cells remain in the full131 denominator.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

P = Path(__file__).resolve().parent
VARIANTS = ("pure", "current", "sampling")
METRICS = ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms")
LABELS = dict(zip(METRICS, ("TTFT", "TPOT", "E2E")))
SELECTION_SHA = "cab8f3a4baa90f082f2fd83592065aabcb598e3d1b8b2732f21bc5f3e49df9c5"
DENOMINATOR, ANCHORS, THRESHOLD = 131, 20, 10.0
EPSILON = 1e-9
SCORE_NAME = "errors.0001.json"


class EvidenceError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def stable_hash(value):
    # Same canonical serialization as heterollm_sim.serde without code import.
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ": "), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reference(path):
    path = Path(path).resolve(strict=True)
    require(path.is_file(), "not a file: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def normalized_ref(ref):
    require(isinstance(ref, dict) and isinstance(ref.get("path"), str)
            and bool(ref["path"]), "missing reference path")
    sha = ref.get("sha256")
    require(isinstance(sha, str) and len(sha) == 64
            and all(c in "0123456789abcdef" for c in sha), "invalid reference SHA")
    size = ref.get("bytes", ref.get("size_bytes"))
    require(type(size) is int and size >= 0, "missing reference byte size")
    if "bytes" in ref and "size_bytes" in ref:
        require(ref["bytes"] == ref["size_bytes"], "reference byte fields disagree")
    return {"path": str(Path(ref["path"]).resolve()), "sha256": sha, "bytes": size}


def verify_reference(ref):
    expected = normalized_ref(ref)
    require(reference(expected["path"]) == expected,
            "changed evidence: " + expected["path"])
    return expected


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value):
    raise EvidenceError("nonfinite JSON number: " + value)


def read_json(path):
    before = reference(path)
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = json.load(stream, object_pairs_hook=_object_pairs,
                          parse_constant=_reject_constant)
    require(isinstance(value, dict), "JSON root must be an object")
    require(reference(path) == before, "evidence changed while reading: " + str(path))
    return value, before


def finite(value, name, positive=False):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not positive or value > 0),
            "invalid finite metric: " + name)
    return float(value)


def timestamp(value):
    require(isinstance(value, str) and bool(value), "missing timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "timestamp has no timezone")
    return parsed


def unique_cells(document):
    cells = document.get("cells")
    require(isinstance(cells, list) and len(cells) == DENOMINATOR,
            "full131 cell denominator required")
    result = {}
    for row in cells:
        require(isinstance(row, dict), "cell must be an object")
        ident = row.get("cell_id")
        require(isinstance(ident, str) and bool(ident) and ident not in result,
                "missing or duplicate cell identity")
        result[ident] = row
    return result


def anchors(cells):
    chosen = {ident for ident, row in cells.items()
              if row.get("model_key") == "qwen25" or
              (row.get("model_key") in {"qwen35", "smollm2", "tinyllama"}
               and "_p512_o32_c1" in ident)}
    counts = Counter(cells[ident].get("model_key") for ident in chosen)
    require(len(chosen) == ANCHORS and counts == {
        "qwen25": 17, "qwen35": 1, "smollm2": 1, "tinyllama": 1},
        "expected17 Qwen25 plus3 preregistered anchors")
    for ident in chosen:
        row = cells[ident]
        require(row.get("preparation_error") is None
                and isinstance(row.get("static_inputs"), dict),
                "anchor static inputs unavailable: " + ident)
        if row["model_key"] != "qwen25":
            config = row["static_inputs"]["config"]
            prompt = config.get("expected_prompt_tokens", config.get("prompt_tokens"))
            output = config.get("output", config.get("output_tokens"))
            require(prompt == 512 and output == 32 and config.get("parallel") == 1,
                    "anchor ID inconsistent with frozen shape")
    return sorted(chosen)


def source_content(freeze):
    source = freeze.get("source")
    require(isinstance(source, dict) and isinstance(source.get("files"), list)
            and bool(source["files"]) and isinstance(source.get("root"), str)
            and bool(source["root"]), "missing frozen source closure")
    require(source.get("sha256") == stable_hash(source["files"]),
            "frozen source reference-map SHA differs")
    source_root = Path(source["root"]).resolve(strict=True)
    normalized = {}
    for ref in source["files"]:
        actual = verify_reference(ref)
        path = Path(actual["path"])
        require(path.is_relative_to(source_root), "source escapes copied root")
        relative = path.relative_to(source_root).as_posix()
        require(relative not in normalized, "duplicate relative source path")
        normalized[relative] = {"sha256": actual["sha256"], "bytes": actual["bytes"]}
    for required in ("src/heterollm_sim/config.py", "src/heterollm_sim/planner.py",
                     "src/heterollm_sim/cost_models.py", "tools/predict_stable_native_dataset.py",
                     "tools/native_llama_compare.py"):
        require(required in normalized, "required source missing: " + required)
    return normalized


def load_predictions(directory, variant):
    freeze, freeze_ref = read_json(directory / "freeze.json")
    require(freeze.get("schema") == "stable-native-simulation-freeze/v1",
            "wrong frozen prediction schema")
    require(freeze.get("selection_sha256") == SELECTION_SHA
            and freeze.get("selection_ref", {}).get("sha256") == SELECTION_SHA,
            "fixed native selection SHA differs")
    require(freeze.get("selected_denominator") == DENOMINATOR,
            "frozen denominator differs")
    require(freeze.get("blind_evaluation") is False
            and freeze.get("calibration_applied") is False,
            "unexpected blind or calibrated freeze")
    cells = unique_cells(freeze)
    ids = anchors(cells)
    expected_files = {ident + ".prediction.json" for ident in ids}
    actual_files = {path.name for path in (directory / "predictions").glob("*.prediction.json")}
    require(actual_files == expected_files,
            "prediction completion set differs from preregistered20 in " + variant)
    predictions, prediction_refs = {}, {}
    for ident in ids:
        prediction, pred_ref = read_json(directory / "predictions" / (ident + ".prediction.json"))
        require(prediction.get("schema") == "stable-native-cell-prediction/v1"
                and prediction.get("cell_id") == ident
                and prediction.get("status") == "predicted", "prediction incomplete: " + ident)
        require(normalized_ref(prediction.get("freeze_ref")) == freeze_ref,
                "prediction freeze identity differs")
        require(prediction.get("selection_sha256") == SELECTION_SHA
                and prediction.get("source_sha256") == freeze["source"]["sha256"],
                "prediction source or selection SHA differs")
        require(prediction.get("native_answers_used") is False
                and prediction.get("calibration_applied") is False
                and prediction.get("formal_prediction_eligible") is False,
                "prediction provenance flags missing or ineligible")
        require(prediction.get("input_identity", {}).get("static_inputs_sha256")
                == stable_hash(cells[ident]["static_inputs"]), "static input identity differs")
        require(timestamp(freeze["created_utc"]) <= timestamp(prediction["created_utc"])
                <= timestamp(prediction["finished_utc"]), "prediction timestamp order differs")
        for metric in METRICS:
            finite(prediction.get("aggregate", {}).get(metric, {}).get("median_ms"),
                   metric, positive=True)
        predictions[ident], prediction_refs[ident] = prediction, pred_ref
    return {"variant": variant, "directory": directory, "freeze": freeze,
            "freeze_ref": freeze_ref, "cells": cells, "ids": ids,
            "predictions": predictions, "prediction_refs": prediction_refs}


def static_variant_semantics(bundle):
    variant, freeze = bundle["variant"], bundle["freeze"]
    if variant == "sampling":
        require(isinstance(freeze.get("sampling"), dict)
                and isinstance(freeze["sampling"].get("cells"), dict),
                "sampling variant requires frozen contract")
    else:
        require(freeze.get("sampling") is None,
                "pure/current must retain sampling contract off")
    normalized = {}
    for ident, row in bundle["cells"].items():
        inputs = row.get("static_inputs")
        require(isinstance(inputs, dict) and row.get("preparation_error") is None,
                "frozen static cell unavailable: " + ident)
        require(inputs.get("gpu_mmq_source_costs") is (variant != "pure")
                and inputs.get("gpu_conversion_cta_costs") is False,
                "MMQ/CTA switches differ from declared variant")
        proof = inputs.get("gpu_invocation_evidence")
        require(isinstance(proof, dict)
                and proof.get("mmq_source_costs_requested") is (variant != "pure")
                and proof.get("conversion_cta_costs_requested", False) is False,
                "cost switch and evidence binding differ")
        sampling = inputs.get("sampling_binding")
        if variant == "sampling":
            require(isinstance(sampling, dict)
                    and sampling == freeze["sampling"]["cells"].get(ident),
                    "per-cell sampling binding differs")
            policy = sampling.get("typed_policy")
            require(isinstance(policy, dict) and policy.get("mode") == "greedy"
                    and policy.get("implementation") == "llama_cpp_cpu_chain"
                    and type(policy.get("top_k")) is int and policy["top_k"] == 1
                    and type(policy.get("min_keep")) is int and policy["min_keep"] == 0
                    and policy.get("temperature") == 0.0,
                    "not frozen top_k1/min_keep0 CPU chain")
        else:
            require(sampling is None, "unexpected sampling binding")
        common = deepcopy(inputs)
        common.pop("sampling_binding", None)
        common.pop("gpu_mmq_source_costs")
        common["gpu_invocation_evidence"].pop("mmq_source_costs_requested")
        normalized[ident] = stable_hash(common)
    return normalized


def validate_metric(record, prediction, metric):
    fields = ("simulator_median_ms", "native_median_ms", "signed_error_ms",
              "absolute_error_ms", "signed_error_pct", "absolute_percentage_error_pct")
    values = {field: finite(record.get(field), field,
                           positive=field in ("simulator_median_ms", "native_median_ms"))
              for field in fields}
    require(values["simulator_median_ms"] == prediction["aggregate"][metric]["median_ms"],
            "score differs from saved prediction")
    delta = values["simulator_median_ms"] - values["native_median_ms"]
    expected = {"signed_error_ms": delta, "absolute_error_ms": abs(delta),
                "signed_error_pct": 100 * delta / values["native_median_ms"],
                "absolute_percentage_error_pct": 100 * abs(delta) / values["native_median_ms"]}
    for field, value in expected.items():
        require(math.isclose(values[field], value, rel_tol=1e-10, abs_tol=1e-10),
                "score arithmetic differs: " + field)
    repeats = record.get("native_run_medians_ms")
    require(isinstance(repeats, list) and bool(repeats), "missing native repeat medians")
    for value in repeats:
        finite(value, "native repeat", positive=True)
    require(statistics.median(repeats) == values["native_median_ms"],
            "native median does not match run medians")
    return {"status": "scored", **values, "native_run_medians_ms": repeats}


def load_score(bundle):
    """Only called after all current variants'20 predictions are checked."""
    document, score_ref = read_json(bundle["directory"] / SCORE_NAME)
    require(document.get("schema") == "stable-native-simulation-errors/v1"
            and document.get("selected_denominator") == DENOMINATOR,
            "score schema or denominator differs")
    require(normalized_ref(document.get("freeze_ref")) == bundle["freeze_ref"],
            "score freeze identity differs")
    require(document.get("native_report_ref", {}).get("sha256") == SELECTION_SHA,
            "score native source differs")
    require(document.get("blind_evaluation") is False
            and document.get("formal_prediction_eligible") is False
            and document.get("calibration_applied") is False,
            "score lacks development-only status")
    cells = unique_cells(document)
    require(set(cells) == set(bundle["cells"]), "score frozen cell set differs")
    scored, retained = {}, {}
    for ident, row in cells.items():
        statuses = []
        records = row.get("metrics")
        require(isinstance(records, dict), "missing metric status mapping: " + ident)
        for metric in METRICS:
            record = records.get(metric)
            require(isinstance(record, dict) and record.get("status") in {"scored", "unscored"},
                    "missing metric status: " + ident + "/" + metric)
            statuses.append(record["status"])
            if record["status"] == "unscored":
                require(isinstance(record.get("reason"), str) and bool(record["reason"]),
                        "unscored metric lacks retained reason")
        require(len(set(statuses)) == 1, "partial scoring cannot be silently filtered")
        if statuses[0] == "scored":
            require(ident in bundle["predictions"], "score has no complete prediction")
            require(normalized_ref(row.get("prediction_ref")) == bundle["prediction_refs"][ident],
                    "score prediction SHA differs")
            require(timestamp(bundle["predictions"][ident]["finished_utc"])
                    <= timestamp(document["created_utc"]), "score predates prediction")
            scored[ident] = {"cell_id": ident, "model_key": row.get("model_key"),
                            "deployment": row.get("deployment"),
                            "metrics": {metric: validate_metric(records[metric], bundle["predictions"][ident], metric)
                                        for metric in METRICS}}
        else:
            require(ident not in bundle["predictions"], "complete prediction omitted")
            retained[ident] = {"cell_id": ident, "model_key": row.get("model_key"),
                              "deployment": row.get("deployment"), "metrics": records,
                              "prediction_status": "not_run_in_this_freeze"}
        require(row.get("model_key") == bundle["cells"][ident].get("model_key")
                and row.get("deployment") == bundle["cells"][ident].get("deployment"),
                "score grouping identity differs")
    require(sorted(scored) == bundle["ids"] and len(retained) == DENOMINATOR - ANCHORS,
            "score completion set differs from preregistered20")
    bundle.update(score=document, score_ref=score_ref, rows=scored, unrun=retained)



def percentile(values, q):
    x=sorted(values); pos=(len(x)-1)*q; lo=int(pos); hi=min(lo+1,len(x)-1)
    return x[lo]+(x[hi]-x[lo])*(pos-lo)


def distribution(values):
    return {"median":statistics.median(values),"p90":percentile(values,.9),"max":max(values)}


def summarize(rows):
    metrics={}
    for metric in METRICS:
        metrics[metric]={field:distribution([row["metrics"][metric][field] for row in rows.values()])
                        for field in ("absolute_percentage_error_pct","absolute_error_ms","signed_error_pct")}
    return {"cells":len(rows),"strict_pass_cells":sum(all(row["metrics"][m]["absolute_percentage_error_pct"]<THRESHOLD for m in METRICS) for row in rows.values()),"metrics":metrics}


def write_exclusive(path,value):
    with path.open('x',encoding='utf8') as out:
        if isinstance(value,str):out.write(value)
        else:json.dump(value,out,ensure_ascii=False,indent=2,allow_nan=False);out.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.parse_args()
    bundles={v:load_predictions(P/v,v) for v in VARIANTS}
    sources={v:source_content(b['freeze']) for v,b in bundles.items()}
    require(all(x==sources['current'] for x in sources.values()),'variant relative-path source contents differ')
    semantics={v:static_variant_semantics(b) for v,b in bundles.items()}
    require(all(x==semantics['current'] for x in semantics.values()),'static inputs differ beyond declared treatments')
    require(all(b['ids']==bundles['current']['ids'] for b in bundles.values()),'anchor sets differ')
    for b in bundles.values():load_score(b)
    ids=bundles['current']['ids'];changes=[]
    for ident in ids:
        for metric in METRICS:
            a=bundles['current']['rows'][ident]['metrics'][metric];b=bundles['sampling']['rows'][ident]['metrics'][metric]
            changes.append({'cell_id':ident,'model_key':bundles['current']['rows'][ident]['model_key'],'metric':metric,
                'before':a,'after':b,'time_delta_ms':b['simulator_median_ms']-a['simulator_median_ms'],
                'ape_delta_percentage_points':b['absolute_percentage_error_pct']-a['absolute_percentage_error_pct'],
                'outcome':'improved' if b['absolute_percentage_error_pct']<a['absolute_percentage_error_pct'] else 'regressed' if b['absolute_percentage_error_pct']>a['absolute_percentage_error_pct'] else 'unchanged'})
    previous,_=read_json(P.parent/'round_017/current/errors.0001.json');old=unique_cells(previous)
    reproduction=[{'cell_id':ident,'metric':m,'old':old[ident]['metrics'][m]['simulator_median_ms'],'new':bundles['current']['rows'][ident]['metrics'][m]['simulator_median_ms']}
                  for ident in ids for m in METRICS if old[ident]['metrics'][m]['simulator_median_ms']!=bundles['current']['rows'][ident]['metrics'][m]['simulator_median_ms']]
    gate,gate_ref=read_json(P/'strict_gate.0002.json')
    models=sorted({r['model_key'] for r in bundles['current']['rows'].values()})
    summary={v:summarize(b['rows']) for v,b in bundles.items()}
    groups={v:{g:summarize({k:r for k,r in b['rows'].items() if r['model_key']==g}) for g in models} for v,b in bundles.items()}
    result={'schema':'source-sampling-chain-ablation/v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'20 preregistered development anchors; fixed131 denominator,111 not run; not independent acceptance',
        'selection_sha256':SELECTION_SHA,'source_content_equal_across_variants':True,'normalized_source_files':len(sources['current']),
        'declared_treatments':{'pure':'source MMQ cost disabled, sampling disabled','current':'source MMQ cost enabled, sampling disabled','sampling':'source MMQ cost enabled, source-bound CPU sampling enabled'},
        'summary':summary,'by_model':groups,'metric_outcomes':dict(Counter(x['outcome'] for x in changes)),
        'changes':changes,'baseline_reproduction_differences':reproduction,'strict_gate':gate,'strict_gate_ref':gate_ref,
        'native_remeasured':False,'calibration_fitted':False,'policy':'retain source-correct sampling opt-in as partial semantic improvement; target not achieved, continue independent CPU/operator mechanism evidence',
        'unrun_cells':list(bundles['sampling']['unrun'].values())}
    index={'schema':'R18-freeze-prediction-evidence-index/v1','selection_sha256':SELECTION_SHA,'normalized_source_sha256':stable_hash(sources['current']),
        'variants':{v:{'freeze_ref':b['freeze_ref'],'score_ref':b['score_ref'],'source_file_count':len(sources[v]),'predictions':b['prediction_refs']} for v,b in bundles.items()},'gate_ref':gate_ref}
    lines=['# R18 原生采样语义三路消融','',f"20个开发场景、60项指标：改善{result['metric_outcomes'].get('improved',0)}项，退化{result['metric_outcomes'].get('regressed',0)}项。候选仍为0/20格逐项三误差<10%。固定131格中111格本轮未跑；A门失败，B门未验证。",'',
        '原生temperature=0/top-k=1仍执行候选表构造与全词表扫描；本轮按已核验静态策略补入这些工作，不使用目标LLM时延拟合。过滤链余项、bias/suppression、RNG、accept/history仍为部分建模。','',
        '| 对照 | TTFT APE中位/P90/最坏 | TPOT APE中位/P90/最坏 | E2E APE中位/P90/最坏 |','|---|---|---|---|']
    for v,s in summary.items():
        vals=[' / '.join(f"{s['metrics'][m]['absolute_percentage_error_pct'][k]:.2f}%" for k in ('median','p90','max')) for m in METRICS]
        lines.append('| '+v+' | '+' | '.join(vals)+' |')
    lines+=['','| 采样候选模型分组 | 场景 | TTFT绝对毫秒中位/P90/最坏 | TPOT绝对毫秒中位/P90/最坏 | E2E绝对毫秒中位/P90/最坏 |','|---|---:|---|---|---|']
    for g,s in groups['sampling'].items():
        vals=[' / '.join(f"{s['metrics'][m]['absolute_error_ms'][k]:.3f}" for k in ('median','p90','max')) for m in METRICS]
        lines.append('| '+g+' | '+str(s['cells'])+' | '+' | '.join(vals)+' |')
    lines+=['',f'当前机制与R17相同20格的60项预测差异：{len(reproduction)}项。三路冻结的代码内容和静态配置按相对路径及SHA逐项检查，只有已声明开关不同。',
        '', '三路报告保留逐格原生重复中位数、预测中位数、有符号误差和绝对毫秒误差，完整393项门禁没有通过。误差P90是场景误差分位数，不是请求延迟P90。',
        '', '本轮采样结构修复改善有限；下一轮采用独立合成候选表与原DLL top-k微基准检验CPU成本。另有图提交54阶段完成但计时质量0/6准入，不将拒收时间转成校准常数。','']
    w,h,left,cw,rh=1220,146+20*28,340,95,28
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="#fff"/><style>text{{font-family:Segoe UI,sans-serif;font-size:12px}}</style><text x="20" y="25" style="font-size:20px;font-weight:600">R18 — CPU sampling semantics · 20 development cells</text><text x="20" y="50">Absolute percentage error · strict three-metric pass: 0/20 · independent acceptance unvalidated</text>']
    for gi,v in enumerate(VARIANTS):
        svg.append(f'<text x="{left+gi*3*cw+cw*1.5}" y="78" text-anchor="middle" font-weight="bold">{v}</text>')
        for mi,m in enumerate(METRICS):svg.append(f'<text x="{left+(gi*3+mi)*cw+cw/2}" y="100" text-anchor="middle">{LABELS[m]}</text>')
    for ri,ident in enumerate(ids):
        y=113+ri*rh;svg.append(f'<text x="12" y="{y+18}">{html.escape(ident.replace("__fixed_runtime",""))}</text>')
        for gi,v in enumerate(VARIANTS):
            for mi,m in enumerate(METRICS):
                val=bundles[v]['rows'][ident]['metrics'][m]['absolute_percentage_error_pct'];t=min(val/85,1);color='#%02x%02x%02x'%tuple(round(a+(b-a)*t) for a,b in zip((232,245,233),(230,97,74)));x=left+(gi*3+mi)*cw
                svg.append(f'<rect x="{x}" y="{y}" width="{cw-3}" height="{rh-2}" fill="{color}"/><text x="{x+cw/2}" y="{y+18}" text-anchor="middle">{val:.1f}%</text>')
    svg.append('</svg>')
    # Complete validation before any output; never replace an existing result.
    targets=['ablation.json','ablation.md','ablation_heatmap.svg','freeze_prediction_index.json']
    require(not any((P/x).exists() for x in targets),'refusing existing report output')
    write_exclusive(P/targets[0],result);write_exclusive(P/targets[1],'\n'.join(lines));write_exclusive(P/targets[2],''.join(svg));write_exclusive(P/targets[3],index)
    print(json.dumps({'common_cells':20,'metric_outcomes':result['metric_outcomes'],'strict_pass_cells':summary['sampling']['strict_pass_cells'],'baseline_differences':len(reproduction),'report':str(P/'ablation.md')},ensure_ascii=False))


if __name__=='__main__':main()
