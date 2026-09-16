"""R21 reporting-v2 saved-evidence report: anchors three-way, or physical full131.
Only standard-library imports; never imports simulator/evaluator or runs predictions.
Usage: python summarize_ablation.py anchors|full [--directory REPORTING_V2_DIR] [--heatmap]
A development-only posthoc reporting schema correction: source evidence may omit a byte
field only when its path and SHA-256 are present and the observed file SHA-256 matches.
All prediction, freeze, score, protocol, and control references remain byte-size strict.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib,html,json,math,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
ORIGINAL_SUMMARIZER_SHA='d211dfb9ddd3591955abf92cbe676637ba92c5150eb2d0c9af6ba45257d09c91'
VARIANTS=('pure','current','physical')
METRICS=('engine_ttft_ms','engine_tpot_ms','engine_e2e_ms')
LABELS=dict(zip(METRICS,('TTFT','TPOT','E2E')))
SELECTION_SHA='cab8f3a4baa90f082f2fd83592065aabcb598e3d1b8b2732f21bc5f3e49df9c5'
DENOMINATOR,ANCHORS,THRESHOLD,EPSILON=131,20,10.0,1e-9
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


def normalized_source_evidence_ref(ref):
    """Normalize a source-only reference without inventing a missing byte field."""
    require(isinstance(ref, dict) and isinstance(ref.get("path"), str)
            and bool(ref["path"]), "missing source evidence path")
    sha = ref.get("sha256")
    require(isinstance(sha, str) and len(sha) == 64
            and all(c in "0123456789abcdef" for c in sha),
            "invalid source evidence SHA")
    has_bytes = "bytes" in ref or "size_bytes" in ref
    if "bytes" in ref and "size_bytes" in ref:
        require(ref["bytes"] == ref["size_bytes"],
                "source evidence byte fields disagree")
    declared = ref.get("bytes", ref.get("size_bytes"))
    if has_bytes:
        require(type(declared) is int and declared >= 0,
                "invalid source evidence byte size")
    result = {"path": str(Path(ref["path"]).resolve()), "sha256": sha}
    if has_bytes:
        result["declared_bytes"] = declared
    return result


def verify_source_evidence_reference(ref):
    """Verify path+SHA; record observed size and validate declared size if present."""
    expected = normalized_source_evidence_ref(ref)
    observed = reference(expected["path"])
    require(observed["sha256"] == expected["sha256"],
            "changed source evidence: " + expected["path"])
    if "declared_bytes" in expected:
        require(observed["bytes"] == expected["declared_bytes"],
                "changed source evidence size: " + expected["path"])
    return {"path": expected["path"], "sha256": expected["sha256"],
            "observed_bytes": observed["bytes"],
            "declared_bytes": expected.get("declared_bytes")}


def merge_source_evidence_ref(refs, ref):
    """Merge duplicate source refs by path+SHA, tolerating only a missing size."""
    key = (ref["path"], ref["sha256"])
    old = refs.get(key)
    if old is None:
        refs[key] = ref
    elif "declared_bytes" in old and "declared_bytes" in ref:
        require(old["declared_bytes"] == ref["declared_bytes"],
                "conflicting source evidence byte size")
    elif "declared_bytes" in ref:
        refs[key] = ref
    return refs[key]


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



def static_variant_semantics(bundle):
    variant,freeze=bundle['variant'],bundle['freeze'];normalized={}
    for name,enabled in [('sampling',variant!='pure'),('nonflash_kv_view',variant=='physical')]:
        binding=freeze.get(name)
        require(isinstance(binding,dict) if enabled else binding is None,'unexpected '+name+' variant binding')
        if enabled:
            require(isinstance(binding.get('cells'),dict) and set(binding['cells'])==set(bundle['cells']),'contract cell denominator: '+name)
            verify_reference(binding['contract_ref'])
            if name=='nonflash_kv_view':
                contract=binding['contract']
                require(contract.get('schema')=='heterollm.llama-nonflash-kv-view/v1' and contract.get('n_pad')==1 and contract.get('n_kv_padding')==256 and contract.get('context_allocation_alignment')==256 and contract.get('native_latency_used') is False,'physical lower-bound contract identity')
                require(read_json(binding['contract_ref']['path'])[0]==contract,'physical saved contract differs')
                require(contract.get('rules',{}).get('occupied_lower_bound')=='largest_current_sequence_retained_context_only','physical occupied lower bound differs')
    for ident,row in bundle['cells'].items():
        inputs=row.get('static_inputs')
        require(isinstance(inputs,dict) and row.get('preparation_error') is None,'static inputs unavailable: '+ident)
        enabled=variant!='pure';proof=inputs.get('gpu_invocation_evidence')
        require(inputs.get('gpu_mmq_source_costs') is enabled and inputs.get('gpu_conversion_cta_costs') is False,'MMQ/CTA switches differ')
        require(isinstance(proof,dict) and proof.get('mmq_source_costs_requested') is enabled and proof.get('conversion_cta_costs_requested',False) is False,'MMQ evidence differs')
        for name,key,on in [('sampling','sampling_binding',enabled),('nonflash_kv_view','nonflash_kv_view_contract',variant=='physical')]:
            value=inputs.get(key)
            require(value==freeze[name]['cells'][ident] if on else value is None,'per-cell '+key+' differs')
        if enabled:
            policy=inputs['sampling_binding'].get('typed_policy',{})
            require(policy.get('mode')=='greedy' and policy.get('implementation')=='llama_cpp_cpu_chain' and type(policy.get('top_k')) is int and policy['top_k']==1 and type(policy.get('min_keep')) is int and policy['min_keep']==0 and policy.get('temperature')==0.0,'sampling policy differs')
        common=deepcopy(inputs)
        for key in ('sampling_binding','nonflash_kv_view_contract'):common.pop(key,None)
        common.pop('gpu_mmq_source_costs');common['gpu_invocation_evidence'].pop('mmq_source_costs_requested')
        # Only declared static treatment bindings are normalized. Dispatch/phase/query
        # diagnostics in saved predictions remain preserved but are not shared inputs.
        normalized[ident]=stable_hash({'static_inputs':common,'model_key':row.get('model_key'),'deployment':row.get('deployment')})
    return normalized


def load_bundle(directory, variant, input_refs, full=False):
    freeze,fr=read_json(directory/'freeze.json')
    require(fr == verify_reference(input_refs['freeze_ref']),
            'freeze differs from reporting-v2 controls: '+variant)
    require(freeze.get('schema')=='stable-native-simulation-freeze/v1' and freeze.get('selected_denominator')==DENOMINATOR,'freeze schema/full131 denominator')
    require(freeze.get('selection_sha256')==SELECTION_SHA and freeze.get('selection_ref',{}).get('sha256')==SELECTION_SHA,'fixed selection SHA differs')
    require(freeze.get('blind_evaluation') is False and freeze.get('calibration_applied') is False,'freeze development provenance')
    verify_reference(freeze['selection_ref']);cells=unique_cells(freeze);ids=anchors(cells)
    planned=sorted(cells) if full else ids
    expected_refs=input_refs['prediction_refs']
    require(set(expected_refs)==set(planned),
            'reporting-v2 prediction manifest differs from planned '+str(len(planned))+': '+variant)
    actual={q.name for q in (directory/'predictions').glob('*.prediction.json')}
    expected_names={ident+'.prediction.json' for ident in planned}
    if full:
        require(actual==expected_names,'saved terminal prediction set differs from planned '+str(len(planned))+': '+variant)
    else:
        require(expected_names <= actual,'missing saved anchor predictions: '+variant)
        universe={ident+'.prediction.json' for ident in cells}
        require(actual <= universe,'unexpected prediction filename outside freeze plan: '+variant)
    predictions={};refs={}
    for ident in planned:
        pred,pr=read_json(directory/'predictions'/(ident+'.prediction.json'))
        require(pr == verify_reference(expected_refs[ident]),
                'prediction differs from reporting-v2 controls: '+ident)
        require(pred.get('schema')=='stable-native-cell-prediction/v1' and pred.get('cell_id')==ident and pred.get('status') in {'predicted','failed','incomplete'},'prediction identity/terminal status')
        require(pred.get('model_key')==cells[ident].get('model_key') and pred.get('deployment')==cells[ident].get('deployment'),'prediction grouping identity')
        require(normalized_ref(pred.get('freeze_ref'))==fr and pred.get('selection_sha256')==SELECTION_SHA and pred.get('source_sha256')==freeze['source']['sha256'],'prediction freeze/source identity')
        require(all(pred.get(k) is False for k in ('native_answers_used','calibration_applied','formal_prediction_eligible')),'prediction provenance flags')
        require(timestamp(freeze['created_utc'])<=timestamp(pred['created_utc'])<=timestamp(pred['finished_utc']),'prediction timestamp order')
        identity=pred.get('input_identity')
        if pred['status']=='failed':
            require(identity==cells[ident]['static_inputs'] or isinstance(identity,dict) and identity.get('static_inputs_sha256')==stable_hash(cells[ident]['static_inputs']),'failed prediction static identity')
        else:require(isinstance(identity,dict) and identity.get('static_inputs_sha256')==stable_hash(cells[ident]['static_inputs']),'prediction static input hash')
        if pred['status']=='predicted':
            for metric in METRICS:finite(pred.get('aggregate',{}).get(metric,{}).get('median_ms'),metric,positive=True)
        else:require(isinstance(pred.get('reason'),str) and bool(pred['reason']),'failed/incomplete reason missing')
        predictions[ident]=pred;refs[ident]=pr
    return {'variant':variant,'directory':directory,'freeze':freeze,'freeze_ref':fr,'cells':cells,'ids':ids,'planned':planned,'predictions':predictions,'prediction_refs':refs}


def check_controls(directory, phase):
    controls,control_ref=read_json(directory/'reporting_controls.json')
    require(controls.get('schema')=='r21-posthoc-reporting-schema-correction/v1',
            'reporting-v2 control schema')
    require(controls.get('development_only') is True
            and controls.get('posthoc_reporting_schema_correction') is True
            and controls.get('changes_formulas') is False
            and controls.get('changes_thresholds') is False
            and controls.get('changes_scoring_data') is False
            and controls.get('runs_predictions') is False,
            'reporting-v2 provenance')
    require(controls.get('original_summarizer_sha256') == ORIGINAL_SUMMARIZER_SHA,
            'original summarizer SHA differs')
    original_script = verify_reference(controls.get('original_summarizer_ref'))
    require(original_script['sha256'] == ORIGINAL_SUMMARIZER_SHA,
            'original summarizer reference SHA differs')
    v2_script = verify_reference(controls.get('v2_summarizer_ref'))
    require(v2_script == reference(Path(__file__)), 'reporting-v2 script changed')
    round_dir = Path(controls.get('input_round_dir', '')).resolve(strict=True)
    original_controls_ref = verify_reference(controls.get('original_evaluation_controls_ref'))
    require(original_controls_ref == reference(round_dir/'evaluation_controls.json'),
            'original evaluation controls changed')
    original_controls,_ = read_json(round_dir/'evaluation_controls.json')
    require(original_controls.get('schema')=='ablation-controls/v1'
            and original_controls.get('same_source_closure') is True,
            'original control manifest identity')
    protocol,protocol_ref=read_json(round_dir/'evaluation_protocol.json')
    require(protocol_ref == verify_reference(controls.get('original_evaluation_protocol_ref')),
            'original evaluation protocol changed')
    require(protocol.get('schema')=='nonflash-physical-kv-ablation/v1' and protocol.get('variants')==list(VARIANTS) and protocol.get('full_denominator')==DENOMINATOR and protocol.get('threshold_pct_strict')==THRESHOLD and protocol.get('metrics')==list(METRICS),'evaluation protocol domain')
    require(protocol.get('native_lock',{}).get('selection_sha256')==SELECTION_SHA and all(protocol.get(k) is False for k in ('is_blind','formal_acceptance','native_remeasurement','calibration_added')),'evaluation protocol provenance')
    phase_inputs = controls.get('phases', {}).get(phase)
    require(isinstance(phase_inputs, dict), 'reporting-v2 phase inputs not frozen: '+phase)
    require(set(phase_inputs)==set(VARIANTS), 'reporting-v2 variants differ: '+phase)
    return protocol, {'controls':control_ref, 'original_controls':original_controls_ref,
                      'protocol':protocol_ref, 'original_summarizer':original_script,
                      'v2_summarizer':v2_script, 'round_dir':str(round_dir),
                      'phase_inputs':phase_inputs, 'controls_document':controls}


def evidence_closure(freeze):
    refs={}
    for name in ('runtime_build_audit','recurrent_batching','cpu_iq_panel_reuse','slot_order','host_offload_source','tensor_storage','gpu_invocation','sampling','nonflash_kv_view'):
        binding=freeze.get(name)
        if binding:
            for item in binding.get('evidence_refs',[])+([binding['audit_ref']] if 'audit_ref' in binding else []):
                merge_source_evidence_ref(refs, normalized_source_evidence_ref(item))
    return list(refs.values())


def load_score(bundle,name,expected_ids,expected_ref):
    doc,sr=read_json(bundle['directory']/name);require(sr == verify_reference(expected_ref),'score differs from reporting-v2 controls: '+bundle['variant']+'/'+name);require(doc.get('schema')=='stable-native-simulation-errors/v1' and doc.get('selected_denominator')==DENOMINATOR,'score schema/131 denominator')
    require(normalized_ref(doc.get('freeze_ref'))==bundle['freeze_ref'],'score freeze identity differs')
    native=verify_reference(doc['native_report_ref']);require(native['sha256']==SELECTION_SHA,'score fixed native source differs')
    require(all(doc.get(k) is False for k in ('blind_evaluation','formal_prediction_eligible','calibration_applied')),'score development provenance')
    rows=unique_cells(doc);require(set(rows)==set(bundle['cells']),'score cell universe differs');out={}
    for ident,row in rows.items():
        require(row.get('model_key')==bundle['cells'][ident].get('model_key') and row.get('deployment')==bundle['cells'][ident].get('deployment'),'score grouping identity')
        pred=bundle['predictions'].get(ident) if ident in expected_ids else None
        if pred is not None:
            require(normalized_ref(row.get('prediction_ref'))==bundle['prediction_refs'][ident],'score prediction hash differs')
            require(timestamp(pred['finished_utc'])<=timestamp(doc['created_utc']),'score predates saved prediction')
        else:require(row.get('prediction_ref') is None,'historical score claims unplanned prediction')
        records=row.get('metrics');require(isinstance(records,dict) and set(records)==set(METRICS),'score metric denominator')
        validated={}
        for metric in METRICS:
            record=records[metric];require(isinstance(record,dict) and record.get('status') in {'scored','unscored'},'score metric status')
            if record['status']=='scored':
                require(pred is not None and pred['status']=='predicted','scored cell lacks complete saved prediction')
                validated[metric]=validate_metric(record,pred,metric)
            else:
                require(isinstance(record.get('reason'),str) and bool(record['reason']),'unscored reason missing')
                validated[metric]=record
        out[ident]={'cell_id':ident,'model_key':row['model_key'],'deployment':row.get('deployment'),'prediction_status':pred['status'] if pred else 'not_run_at_this_score','prediction_failure_reason':pred.get('reason') if pred else None,'metrics':validated}
    return {'ref':sr,'created_utc':doc['created_utc'],'rows':out,'expected_ids':sorted(expected_ids)}


def distribution(values):
    if not values:return {'count':0,'median':None,'p90':None,'worst':None}
    x=sorted(values);i=(len(x)-1)*.9;lo=int(i);v=x[lo]+(x[min(lo+1,len(x)-1)]-x[lo])*(i-lo)
    return {'count':len(x),'median':statistics.median(x),'p90':v,'worst':max(x)}


def summarize(rows):
    metrics={};failures=[]
    for metric in METRICS:
        valid=[r['metrics'][metric] for r in rows.values() if r['metrics'][metric]['status']=='scored']
        metrics[metric]={field:distribution([r[field] for r in valid]) for field in ('absolute_percentage_error_pct','absolute_error_ms','signed_error_pct')}
        metrics[metric]['unscored_count']=len(rows)-len(valid)
    for ident,row in rows.items():
        reasons={m:x['reason'] for m,x in row['metrics'].items() if x['status']=='unscored'}
        if reasons:failures.append({'cell_id':ident,'prediction_status':row['prediction_status'],'prediction_failure_reason':row.get('prediction_failure_reason'),'unscored_reasons':reasons})
    passed=sum(all(r['metrics'][m]['status']=='scored' and r['metrics'][m]['absolute_percentage_error_pct']<THRESHOLD for m in METRICS) for r in rows.values())
    return {'cells':len(rows),'strict_all3_below10_cells':passed,'full131_denominator':DENOMINATOR,'metrics':metrics,'failed_or_unscored_cells':len(failures),'failures':failures}


def compare(before,after,ids):
    changes=[]
    for ident in ids:
        for metric in METRICS:
            a=before[ident]['metrics'][metric];b=after[ident]['metrics'][metric]
            if a['status']!='scored' or b['status']!='scored':changes.append({'cell_id':ident,'metric':metric,'outcome':'unscored','before':a,'after':b});continue
            require(a['native_median_ms']==b['native_median_ms'] and a['native_run_medians_ms']==b['native_run_medians_ms'],'cross-variant native target differs')
            delta=b['absolute_percentage_error_pct']-a['absolute_percentage_error_pct']
            changes.append({'cell_id':ident,'metric':metric,'ape_delta_percentage_points':delta,'time_delta_ms':b['simulator_median_ms']-a['simulator_median_ms'],'outcome':'unchanged' if abs(delta)<=EPSILON else 'improved' if delta<0 else 'regressed'})
    return {'outcomes':dict(Counter(x['outcome'] for x in changes)),'metric_denominator':len(ids)*3,'changes':changes}


def assess(directory,phase):
    require(phase in ('anchors','full'),'unknown report phase');directory=Path(directory).resolve(strict=True)
    protocol,controls=check_controls(directory, phase)
    round_dir=Path(controls['round_dir']); phase_inputs=controls['phase_inputs']
    bundles={v:load_bundle(round_dir/v,v,phase_inputs[v],full=phase=='full' and v=='physical') for v in VARIANTS}
    sources={v:source_content(b['freeze']) for v,b in bundles.items()}
    require(all(x==sources['current'] for x in sources.values()),'variant source closure differs')
    semantics={v:static_variant_semantics(b) for v,b in bundles.items()}
    require(all(x==semantics['current'] for x in semantics.values()),'static inputs differ beyond declared treatments')
    require(bundles['current']['freeze'].get('sampling')==bundles['physical']['freeze'].get('sampling'),'current/physical sampling binding differs')
    ids=bundles['current']['ids'];require(all(b['ids']==ids for b in bundles.values()) and protocol.get('anchor_ids')==ids,'preregistered anchor IDs differ')
    closure={}
    for b in bundles.values():
        for r in evidence_closure(b['freeze']):
            merge_source_evidence_ref(closure, r)
    for r in closure.values():verify_source_evidence_reference(r)
    # All requested saved predictions were loaded and hashed before reading scores.
    anchor_scores={v:load_score(b,'errors.0001.json',ids,phase_inputs[v]['anchor_score_ref']) for v,b in bundles.items()}
    latest_anchor_prediction=max(timestamp(b['predictions'][ident]['finished_utc']) for b in bundles.values() for ident in ids)
    earliest_score=min(timestamp(x['created_utc']) for x in anchor_scores.values())
    require(latest_anchor_prediction<=earliest_score,'scoring began before all three anchor prediction sets finished')
    anchor_rows={v:{ident:s['rows'][ident] for ident in ids} for v,s in anchor_scores.items()}
    comparisons={a+'_to_'+b:compare(anchor_rows[a],anchor_rows[b],ids) for a,b in [('pure','current'),('current','physical'),('pure','physical')]}
    full=None
    if phase=='full':
        b=bundles['physical'];last_anchor_score=max(timestamp(s['created_utc']) for s in anchor_scores.values())
        require(all(timestamp(b['predictions'][ident]['created_utc'])>=last_anchor_score for ident in b['planned'] if ident not in ids),'full continuation started before anchor scoring ended')
        full=load_score(b,'errors.0002.json',b['planned'],phase_inputs['physical']['full_score_ref'])
        require(timestamp(full['created_utc'])>=max(timestamp(p['finished_utc']) for p in b['predictions'].values()),'full score predates predictions')
        require(all(full['rows'][ident]==anchor_scores['physical']['rows'][ident] for ident in ids),'physical anchors changed between scores')
    # Post-score hashing detects prediction overwrite and all control/source drift.
    for b in bundles.values():
        verify_reference(b['freeze_ref'])
        for r in b['prediction_refs'].values():verify_reference(r)
        source_content(b['freeze'])
    for r in closure.values():verify_source_evidence_reference(r)
    require(check_controls(directory, phase)[1]==controls,'controls changed during analysis')
    for score in [*anchor_scores.values(),*([full] if full else [])]:verify_reference(score['ref'])
    totals=summarize(full['rows'] if full else anchor_scores['physical']['rows'])
    gate_a='passed' if phase=='full' and totals['strict_all3_below10_cells']==DENOMINATOR else 'not_passed'
    return {'schema':'r21-nonflash-three-way-ablation/v1','created_utc':datetime.now(timezone.utc).isoformat(),'phase':phase,'selection_sha256':SELECTION_SHA,'declared_variants':{'pure':'no sampling, no MMQ, no nonflash','current':'R18 sampling + MMQ, no nonflash','physical':'current + source-bound nonflash KV lower bound; retained pool state unknown'},'anchor_denominator':20,'full_denominator':131,'anchor_summary':{v:summarize(rows) for v,rows in anchor_rows.items()},'anchor_comparisons':comparisons,'anchor_rows':anchor_rows,'physical131':totals,'physical131_rows':full['rows'] if full else anchor_scores['physical']['rows'],'coverage':{'physical_saved_predictions':len(bundles['physical']['predictions']),'physical_expected_cells':131,'physical_missing_prediction_cells':131-len(bundles['physical']['predictions']),'physical_anchor_prediction_statuses':dict(Counter(bundles['physical']['predictions'][i]['status'] for i in ids))},'gate_A':gate_a,'gate_B':'unvalidated','native_remeasured':False,'calibration_fitted':False,'coefficients_added':0,'threshold_pct_strict':10,'error_percentile_definition':'Across-cell error percentile; not request-latency P90.','source_content_equal':True,'source_file_count':len(sources['current']),'evidence':{'controls':controls,'source_content_sha256':stable_hash(sources['current']),'variants':{v:{'freeze_ref':b['freeze_ref'],'predictions':b['prediction_refs'],'anchor_score_ref':anchor_scores[v]['ref']} for v,b in bundles.items()},'physical_full_score_ref':full['ref'] if full else None,'external_closure_refs':[verify_source_evidence_reference(r) for r in closure.values()],'predictions_pre_and_post_score_hashes_equal':True}}


def markdown(result):
    lines=['# R21 nonflash KV 三路开发消融','',f"阶段：{result['phase']}。physical 全三项严格小于10%：{result['physical131']['strict_all3_below10_cells']}/131；A={result['gate_A']}，B=unvalidated。新增系数0。",'', 'pure：无 sampling/MMQ/nonflash；current：sampling+MMQ；physical：current 加 source-bound nonflash KV 下界，池状态仍未知。','', '| 20-anchor 对照 | 全三项<10% | TTFT APE中位/P90/最坏 | TPOT APE中位/P90/最坏 | E2E APE中位/P90/最坏 |','|---|---:|---|---|---|']
    def cell(d):return ' / '.join('无有效分数' if d[k] is None else f"{d[k]:.3f}%" for k in ('median','p90','worst'))
    for v,s in result['anchor_summary'].items():lines.append('| '+v+' | '+str(s['strict_all3_below10_cells'])+'/20 | '+' | '.join(cell(s['metrics'][m]['absolute_percentage_error_pct']) for m in METRICS)+' |')
    lines+=['', 'APE为绝对百分比误差；P90是场景误差分位数，不是请求延迟P90。']
    for name,c in result['anchor_comparisons'].items():lines.append(f"- {name}："+', '.join(f'{key}={value}' for key,value in c['outcomes'].items())+f"；指标分母{c['metric_denominator']}。")
    coverage=result['coverage'];lines+=['',f"physical 已保存预测 {coverage['physical_saved_predictions']}/131；未保存 {coverage['physical_missing_prediction_cells']}；失败或未评分 {result['physical131']['failed_or_unscored_cells']}/131。"]
    if result['phase']=='full':
        lines+=['', 'physical 全131 APE：']
        for m in METRICS:lines.append(f"- {LABELS[m]}："+cell(result['physical131']['metrics'][m]['absolute_percentage_error_pct']))
    reasons=Counter(reason for row in result['physical131']['failures'] for reason in row['unscored_reasons'].values())
    if reasons:lines+=['','未评分原因（按指标计数）：']+[f'- {reason}: {count}' for reason,count in reasons.items()]
    lines+=['','逐格完整误差、失败状态、来源引用、预测前后哈希与评分顺序检查保留于同名JSON。缺失/失败不从131分母移除；不得将开发结果当独立验证或校准。','']
    return '\n'.join(lines)


def heatmap(result):
    ids=sorted(result['anchor_rows']['current']);cw,rh,left=85,25,330;w=left+9*cw+20;h=100+len(ids)*rh
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><style>text{{font:12px sans-serif}}</style><text x="12" y="22">R21 anchor APE (%) · B unvalidated · full denominator 131</text>']
    for j,v in enumerate(VARIANTS):
        for k,m in enumerate(METRICS):svg.append(f'<text x="{left+(j*3+k)*cw}" y="52">{v}/{LABELS[m]}</text>')
    for i,ident in enumerate(ids):
        y=65+i*rh;svg.append(f'<text x="10" y="{y+17}">{html.escape(ident)}</text>')
        for j,v in enumerate(VARIANTS):
            for k,m in enumerate(METRICS):
                r=result['anchor_rows'][v][ident]['metrics'][m];value=r.get('absolute_percentage_error_pct');label=f'{value:.1f}' if value is not None else 'unscored';color='#d8efd7' if value is not None and value<10 else '#f6d9d1' if value is not None else '#dddddd';x=left+(j*3+k)*cw
                svg.append(f'<rect x="{x}" y="{y}" width="{cw-2}" height="{rh-2}" fill="{color}"/><text x="{x+8}" y="{y+17}">{label}</text>')
    return ''.join(svg)+ '</svg>'


def write_report(directory,result,with_heatmap=False):
    directory=Path(directory);stem='ablation_'+result['phase'];outputs={stem+'.json':json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',stem+'.md':markdown(result)}
    if with_heatmap:outputs[stem+'_heatmap.svg']=heatmap(result)
    require(not any((directory/name).exists() for name in outputs),'refusing existing report output')
    for name,content in outputs.items():
        with (directory/name).open('x',encoding='utf-8') as stream:stream.write(content)
    return [str(directory/name) for name in outputs]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['anchors','full']);parser.add_argument('--directory',type=Path,default=P);parser.add_argument('--heatmap',action='store_true');args=parser.parse_args(argv)
    result=assess(args.directory,args.phase);paths=write_report(args.directory,result,args.heatmap)
    print(json.dumps({'outputs':paths,'strict_pass_cells':result['physical131']['strict_all3_below10_cells'],'denominator':131,'gate_A':result['gate_A'],'gate_B':'unvalidated'},ensure_ascii=False))
if __name__=='__main__':main()
