"""R22 four-way saved-evidence report; standard library only.
Read frozen controls, terminal predictions and separately authorized scores.
Source-contract evidence references may omit size, including historical source
logs; prediction/freeze/score/protocol/control/phase references remain byte strict. Never executes a simulator or native code.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib,html,json,math,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
VARIANTS=('pure','current','cta_only','cta_issue')
CANDIDATE='cta_issue'
TREATMENTS={
    'pure':{'mmq':False,'sampling':False,'nonflash':False,'cta':False,'issue':False},
    'current':{'mmq':True,'sampling':True,'nonflash':True,'cta':False,'issue':False},
    'cta_only':{'mmq':True,'sampling':True,'nonflash':True,'cta':True,'issue':False},
    'cta_issue':{'mmq':True,'sampling':True,'nonflash':True,'cta':True,'issue':True}}
DECLARED_VARIANTS={
    'pure':'R21 analytical baseline: no source MMQ, sampling or nonflash KV treatment',
    'current':'R21 physical mechanism: source MMQ, sampling and nonflash KV; CTA/issue off',
    'cta_only':'current plus conversion CTA compute-resource cap',
    'cta_issue':'cta_only plus conditional source/PTX MMVQ integer issue bound'}

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
    has_bytes = "bytes" in ref or "size_bytes" in ref or ref.get("declared_bytes") is not None
    if "bytes" in ref and "size_bytes" in ref:
        require(ref["bytes"] == ref["size_bytes"],
                "source evidence byte fields disagree")
    declared = ref.get("bytes", ref.get("size_bytes", ref.get("declared_bytes")))
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
    variant,freeze=bundle['variant'],bundle['freeze'];mode=TREATMENTS[variant];normalized={}
    for name,enabled in [('sampling',mode['sampling']),('nonflash_kv_view',mode['nonflash'])]:
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
    issue=freeze.get('mmvq_issue_bound')
    require(freeze.get('mmvq_vector_issue_bound',False) is mode['issue'],'campaign issue switch differs')
    if mode['issue']:
        require(isinstance(issue,dict) and issue.get('requested') is True and set(issue.get('cells',{}))==set(bundle['cells']),'issue evidence denominator')
    else:require(issue is None,'unexpected issue evidence')
    for ident,row in bundle['cells'].items():
        inputs=row.get('static_inputs')
        require(isinstance(inputs,dict) and row.get('preparation_error') is None,'static inputs unavailable: '+ident)
        proof=inputs.get('gpu_invocation_evidence')
        require(inputs.get('gpu_mmq_source_costs') is mode['mmq'] and inputs.get('gpu_conversion_cta_costs') is mode['cta'],'MMQ/CTA switches differ')
        require(isinstance(proof,dict) and proof.get('mmq_source_costs_requested') is mode['mmq'] and proof.get('conversion_cta_costs_requested',False) is mode['cta'],'MMQ/CTA evidence differs')
        require(inputs.get('gpu_invocation_contract')==proof.get('contract'),'GPU invocation contract differs')
        for name,key,on in [('sampling','sampling_binding',mode['sampling']),('nonflash_kv_view','nonflash_kv_view_contract',mode['nonflash'])]:
            value=inputs.get(key)
            require(value==freeze[name]['cells'][ident] if on else value is None,'per-cell '+key+' differs')
        if mode['sampling']:
            policy=inputs['sampling_binding'].get('typed_policy',{})
            require(policy.get('mode')=='greedy' and policy.get('implementation')=='llama_cpp_cpu_chain' and type(policy.get('top_k')) is int and policy['top_k']==1 and type(policy.get('min_keep')) is int and policy['min_keep']==0 and policy.get('temperature')==0.0,'sampling policy differs')
        require(inputs.get('mmvq_vector_issue_bound',False) is mode['issue'],'cell issue switch differs')
        if mode['issue']:
            cell=issue['cells'][ident]
            require(inputs.get('mmvq_issue_evidence')==cell and inputs.get('mmvq_issue_contract')==cell.get('contract'),'cell issue proof differs')
            require(cell.get('requested') is True and cell.get('status') in ('conditional','uncovered') and cell.get('native_latency_used') is False and cell.get('native_instruction_mapping_proven') is False,'issue qualification provenance')
            require(cell.get('gpu_invocation_sha256')==stable_hash(proof),'issue invocation binding differs')
        else:require(inputs.get('mmvq_issue_evidence') is None and inputs.get('mmvq_issue_contract') is None,'unexpected cell issue proof')
        common=deepcopy(inputs)
        for key in ('sampling_binding','nonflash_kv_view_contract','mmvq_vector_issue_bound','mmvq_issue_contract','mmvq_issue_evidence','gpu_mmq_source_costs','gpu_conversion_cta_costs'):common.pop(key,None)
        common['gpu_invocation_evidence'].pop('mmq_source_costs_requested',None)
        common['gpu_invocation_evidence'].pop('conversion_cta_costs_requested',None)
        normalized[ident]=stable_hash({'static_inputs':common,'model_key':row.get('model_key'),'deployment':row.get('deployment')})
    return normalized


def check_protocol(protocol):
    require(protocol.get('schema')=='mmvq-four-way-ablation/v1' and protocol.get('variants')==list(VARIANTS),'four-way protocol domain')
    require(protocol.get('treatments')==TREATMENTS and protocol.get('candidate')==CANDIDATE,'protocol treatments/candidate differ')
    require(protocol.get('full_denominator')==DENOMINATOR and protocol.get('anchor_denominator')==ANCHORS and protocol.get('threshold_pct_strict')==THRESHOLD and protocol.get('metrics')==list(METRICS),'protocol denominators/threshold differ')
    require(protocol.get('native_lock',{}).get('selection_sha256')==SELECTION_SHA,'protocol selection differs')
    require(all(protocol.get(k) is False for k in ('is_blind','formal_acceptance','native_remeasurement','calibration_added','accuracy_selected_subset')),'protocol provenance differs')
    require(protocol.get('gate_B')=='unvalidated','B has no independent validation')
    ids=protocol.get('anchor_ids')
    require(isinstance(ids,list) and len(ids)==ANCHORS and len(set(ids))==ANCHORS and ids==sorted(ids),'protocol anchors differ')


def check_controls(directory,phase=None):
    directory=Path(directory).resolve(strict=True)
    controls,control_ref=read_json(directory/'evaluation_controls.json')
    require(controls.get('schema')=='mmvq-four-way-controls/v1' and controls.get('same_source_closure') is True and controls.get('variants')==list(VARIANTS),'four-way control manifest identity')
    expected={str((directory/name).resolve()) for name in ('evaluate_candidate.py','summarize_ablation.py','evaluation_protocol.json',*(v+'/freeze.json' for v in VARIANTS))}
    files=controls.get('files')
    require(isinstance(files,list) and len(files)==len(expected),'control file denominator')
    actual=[verify_reference(ref) for ref in files]
    require({ref['path'] for ref in actual}==expected,'control file paths differ')
    require(reference(Path(__file__)) in actual,'running summarizer differs from frozen control')
    protocol,pr=read_json(directory/'evaluation_protocol.json');check_protocol(protocol)
    require(pr in actual,'protocol not in controls')
    return protocol,{'controls':control_ref,'protocol':pr,'files':actual,'source_content_sha256':controls.get('source_content_sha256'),'source_file_count':controls.get('source_file_count')}


def evidence_closure(freeze):
    """R21-compatible source-contract closure, including historical source logs.

    These evidence_refs are source derivation inputs, distinct from prediction,
    freeze, score, protocol/control, native selection, and phase-receipt refs.
    Only this closed source-contract scope permits an omitted declared size.
    """
    refs={}
    for name in ('runtime_build_audit','recurrent_batching','cpu_iq_panel_reuse','slot_order','host_offload_source','tensor_storage','gpu_invocation','sampling','nonflash_kv_view','mmvq_issue_bound'):
        binding=freeze.get(name)
        if binding:
            for ref in binding.get('evidence_refs',[]):
                if name=='runtime_build_audit':verify_reference(ref)
                merge_source_evidence_ref(refs,normalized_source_evidence_ref(ref))
            if 'audit_ref' in binding:
                ref=verify_reference(binding['audit_ref'])
                merge_source_evidence_ref(refs,normalized_source_evidence_ref(ref))
    return list(refs.values())


def frozen_bundles(directory,protocol):
    bundles={};sources={};semantics={};closure={}
    for variant in VARIANTS:
        freeze,fr=read_json(directory/variant/'freeze.json')
        require(freeze.get('schema')=='stable-native-simulation-freeze/v1' and freeze.get('selected_denominator')==DENOMINATOR,'freeze schema/full131 denominator')
        require(freeze.get('selection_sha256')==SELECTION_SHA and freeze.get('selection_ref',{}).get('sha256')==SELECTION_SHA,'fixed selection SHA differs')
        require(freeze.get('blind_evaluation') is False and freeze.get('calibration_applied') is False,'freeze development provenance')
        verify_reference(freeze['selection_ref']);cells=unique_cells(freeze);ids=anchors(cells)
        require(ids==protocol['anchor_ids'],'frozen preregistered anchors differ')
        bundles[variant]={'variant':variant,'directory':directory/variant,'freeze':freeze,'freeze_ref':fr,'cells':cells,'ids':ids}
        sources[variant]=source_content(freeze);semantics[variant]=static_variant_semantics(bundles[variant])
        for ref in evidence_closure(freeze):merge_source_evidence_ref(closure,ref)
    require(all(value==sources['current'] for value in sources.values()),'variant source closure differs')
    require(all(value==semantics['current'] for value in semantics.values()),'static inputs differ beyond declared treatments')
    for variant in ('cta_only','cta_issue'):
        for key in ('sampling','nonflash_kv_view'):
            require(bundles[variant]['freeze'].get(key)==bundles['current']['freeze'].get(key),'current/'+variant+' '+key+' binding differs')
    verified=[verify_source_evidence_reference(ref) for ref in closure.values()]
    return bundles,sources,verified


def load_bundle(directory, variant, input_refs, full=False):
    freeze,fr=read_json(directory/'freeze.json')
    require(fr == verify_reference(input_refs['freeze_ref']),
            'freeze differs from saved receipt: '+variant)
    require(freeze.get('schema')=='stable-native-simulation-freeze/v1' and freeze.get('selected_denominator')==DENOMINATOR,'freeze schema/full131 denominator')
    require(freeze.get('selection_sha256')==SELECTION_SHA and freeze.get('selection_ref',{}).get('sha256')==SELECTION_SHA,'fixed selection SHA differs')
    require(freeze.get('blind_evaluation') is False and freeze.get('calibration_applied') is False,'freeze development provenance')
    verify_reference(freeze['selection_ref']);cells=unique_cells(freeze);ids=anchors(cells)
    planned=sorted(cells) if full else ids
    expected_refs=input_refs['prediction_refs']
    require(set(expected_refs)==set(planned),
            'saved prediction manifest differs from planned '+str(len(planned))+': '+variant)
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
                'prediction differs from saved receipt: '+ident)
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



def load_score(bundle,name,expected_ids,expected_ref):
    doc,sr=read_json(bundle['directory']/name);require(sr == verify_reference(expected_ref),'score differs from saved receipt: '+bundle['variant']+'/'+name);require(doc.get('schema')=='stable-native-simulation-errors/v1' and doc.get('selected_denominator')==DENOMINATOR,'score schema/131 denominator')
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
            if a['status']!='scored' or b['status']!='scored':
                changes.append({'cell_id':ident,'metric':metric,'outcome':'unscored','before':a,'after':b});continue
            require(a['native_median_ms']==b['native_median_ms'] and a['native_run_medians_ms']==b['native_run_medians_ms'],'cross-variant native target differs')
            delta=b['absolute_percentage_error_pct']-a['absolute_percentage_error_pct'];time_delta=b['simulator_median_ms']-a['simulator_median_ms']
            changes.append({'cell_id':ident,'metric':metric,'ape_delta_percentage_points':delta,'time_delta_ms':time_delta,
                'prediction_changed':abs(time_delta)>EPSILON,'outcome':'unchanged' if abs(delta)<=EPSILON else 'improved' if delta<0 else 'regressed'})
    per_metric={}
    for metric in METRICS:
        subset=[row for row in changes if row['metric']==metric];valid=[row for row in subset if row['outcome']!='unscored']
        per_metric[metric]={'denominator':len(ids),'outcomes':dict(Counter(row['outcome'] for row in subset)),
            'prediction_changed_cells':sum(row['prediction_changed'] for row in valid),
            'ape_delta_percentage_points':distribution([row['ape_delta_percentage_points'] for row in valid]),
            'time_delta_ms':distribution([row['time_delta_ms'] for row in valid])}
    return {'outcomes':dict(Counter(row['outcome'] for row in changes)),'metric_denominator':len(ids)*len(METRICS),'by_metric':per_metric,'changes':changes}


def read_phase_receipt(directory,phase,kind,controls):
    name=phase+'_'+kind+'.json';document,ref=read_json(directory/name)
    require(document.get('schema')=='r22-'+('prediction-barrier' if kind=='predictions' else 'score-receipt')+'/v1' and document.get('phase')==phase,'phase receipt schema')
    require(normalized_ref(document.get('controls_ref'))==controls['controls'],'phase receipt control identity')
    timestamp(document.get('created_utc'))
    return document,ref


def load_prediction_barrier(directory,phase,controls):
    document,ref=read_phase_receipt(directory,phase,'predictions',controls)
    variants=VARIANTS if phase=='anchors' else (CANDIDATE,)
    require(set(document.get('variants',{}))==set(variants),'prediction barrier must cover every planned variant')
    require(document.get('native_answers_used') is False and document.get('terminal_failures_preserved') is True,'prediction barrier provenance')
    bundles={v:load_bundle(directory/v,v,document['variants'][v],full=phase=='full') for v in variants}
    latest=max(timestamp(pred['finished_utc']) for bundle in bundles.values() for pred in bundle['predictions'].values())
    require(timestamp(document['created_utc'])>=latest,'prediction barrier predates saved predictions')
    return document,ref,bundles


def load_score_receipt(directory,phase,controls,barrier_ref):
    document,ref=read_phase_receipt(directory,phase,'scores',controls)
    variants=VARIANTS if phase=='anchors' else (CANDIDATE,)
    require(set(document.get('scores',{}))==set(variants),'score receipt must cover every planned variant')
    require(normalized_ref(document.get('prediction_barrier_ref'))==barrier_ref,'score receipt barrier identity')
    for score in document['scores'].values():verify_reference(score)
    return document,ref


def assess(directory,phase):
    require(phase in ('anchors','full'),'unknown report phase');directory=Path(directory).resolve(strict=True)
    protocol,controls=check_controls(directory)
    _,sources,closure=frozen_bundles(directory,protocol)
    require(stable_hash(sources['current'])==controls['source_content_sha256'] and len(sources['current'])==controls['source_file_count'],'control source closure differs')
    barrier,barrier_ref,bundles=load_prediction_barrier(directory,'anchors',controls)
    score_receipt,score_receipt_ref=load_score_receipt(directory,'anchors',controls,barrier_ref)
    ids=protocol['anchor_ids']
    # Every one of the four terminal anchor sets was validated before a score is read.
    scores={v:load_score(bundles[v],'errors.0001.json',ids,score_receipt['scores'][v]) for v in VARIANTS}
    latest=max(timestamp(b['predictions'][i]['finished_utc']) for b in bundles.values() for i in ids)
    earliest=min(timestamp(score['created_utc']) for score in scores.values())
    require(latest<=earliest and timestamp(barrier['created_utc'])<=earliest,'scoring began before all four anchor prediction sets and barrier finished')
    require(timestamp(score_receipt['created_utc'])>=max(timestamp(score['created_utc']) for score in scores.values()),'anchor score receipt predates scores')
    rows={v:{ident:scores[v]['rows'][ident] for ident in ids} for v in VARIANTS}
    comparisons={name:compare(rows[a],rows[b],ids) for name,a,b in (
        ('cta_only_minus_current','current','cta_only'),('cta_issue_minus_cta_only','cta_only','cta_issue'),('current_minus_pure','pure','current'))}
    full=None;full_refs=[]
    if phase=='full':
        full_barrier,full_barrier_ref,full_bundles=load_prediction_barrier(directory,'full',controls)
        require(normalized_ref(full_barrier.get('anchor_score_receipt_ref'))==score_receipt_ref,'full continuation lacks completed four-way anchor scoring receipt')
        full_receipt,full_receipt_ref=load_score_receipt(directory,'full',controls,full_barrier_ref)
        b=full_bundles[CANDIDATE]
        require(all(timestamp(b['predictions'][ident]['created_utc'])>=timestamp(score_receipt['created_utc']) for ident in b['planned'] if ident not in ids),'full continuation started before four-way anchor scoring ended')
        full=load_score(b,'errors.0002.json',b['planned'],full_receipt['scores'][CANDIDATE])
        require(timestamp(full['created_utc'])>=timestamp(full_barrier['created_utc']) and timestamp(full_receipt['created_utc'])>=timestamp(full['created_utc']),'full score/barrier timestamp order')
        require(all(full['rows'][ident]==scores[CANDIDATE]['rows'][ident] for ident in ids),'candidate anchors changed between scores')
        bundles[CANDIDATE]=b;full_refs=[full_barrier_ref,full_receipt_ref,full['ref']]
    totals=summarize(full['rows'] if full else scores[CANDIDATE]['rows'])
    gate_a='passed' if phase=='full' and totals['strict_all3_below10_cells']==DENOMINATOR else 'not_passed'
    for b in bundles.values():
        verify_reference(b['freeze_ref']);source_content(b['freeze'])
        for ref in b['prediction_refs'].values():verify_reference(ref)
    for ref in [barrier_ref,score_receipt_ref,*[score['ref'] for score in scores.values()],*full_refs]:verify_reference(ref)
    for ref in closure:verify_source_evidence_reference(ref)
    require(check_controls(directory)[1]==controls,'controls changed during analysis')
    qualification=bundles[CANDIDATE]['freeze'].get('mmvq_issue_bound',{})
    return {'schema':'r22-mmvq-four-way-ablation/v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'phase':phase,'selection_sha256':SELECTION_SHA,'variants':list(VARIANTS),'candidate':CANDIDATE,'declared_variants':DECLARED_VARIANTS,
        'anchor_denominator':ANCHORS,'full_denominator':DENOMINATOR,'anchor_summary':{v:summarize(rows[v]) for v in VARIANTS},
        'anchor_comparisons':comparisons,'anchor_rows':rows,'candidate131':totals,'candidate131_rows':full['rows'] if full else scores[CANDIDATE]['rows'],
        'coverage':{'saved_predictions_as_of_phase':{v:len(bundles[v]['predictions']) for v in VARIANTS},
            'candidate_missing_prediction_cells':DENOMINATOR-len(bundles[CANDIDATE]['predictions']),
            'terminal_statuses':{v:dict(Counter(pred['status'] for pred in bundles[v]['predictions'].values())) for v in VARIANTS}},
        'issue_static_qualification':{'conditional_cells':qualification.get('conditional_cell_count'),'uncovered_cells':qualification.get('uncovered_cell_count'),
            'actual_MMVQ_application_count':None,'application_inferred_from_cell_switch':False,
            'scope':'Static source/device/document eligibility is not actual GPU MMVQ invocation coverage or a timing change.'},
        'gate_A':gate_a,'gate_B':'unvalidated','formal_success':False,'native_remeasured':False,'calibration_fitted':False,'coefficients_added':0,
        'threshold_pct_strict':THRESHOLD,'error_percentile_definition':'Across-cell error percentile; not request-latency P90.',
        'source_content_equal':True,'source_file_count':len(sources['current']),
        'evidence':{'controls':controls,'source_content_sha256':stable_hash(sources['current']),
            'variants':{v:{'freeze_ref':b['freeze_ref'],'predictions':b['prediction_refs'],'anchor_score_ref':scores[v]['ref']} for v,b in bundles.items()},
            'anchor_prediction_barrier_ref':barrier_ref,'anchor_score_receipt_ref':score_receipt_ref,'full_phase_refs':full_refs,
            'external_closure_refs':closure,'predictions_pre_and_post_score_hashes_equal':True}}


def markdown(result):
    total=result['candidate131']
    lines=['# R22 MMVQ 四路开发消融','',f"阶段：{result['phase']}。cta_issue 全三项严格小于10%：{total['strict_all3_below10_cells']}/131；A={result['gate_A']}，B=unvalidated。新增系数0。",'',
        'current 沿用 R21 physical（MMQ、sampling、nonflash KV），CTA/issue 均关闭。CTA-only 只增加转换 CTA；CTA+issue 再增加条件 MMVQ issue 机制。pure 为整体分析基线。','',
        '| 20-anchor 对照 | 全三项<10% | TTFT APE中位/P90/最坏 | TPOT APE中位/P90/最坏 | E2E APE中位/P90/最坏 |','|---|---:|---|---|---|']
    def cell(d):return ' / '.join('无有效分数' if d[key] is None else f"{d[key]:.3f}%" for key in ('median','p90','worst'))
    for variant in VARIANTS:
        item=result['anchor_summary'][variant]
        lines.append('| '+variant+' | '+str(item['strict_all3_below10_cells'])+'/20 | '+' | '.join(cell(item['metrics'][metric]['absolute_percentage_error_pct']) for metric in METRICS)+' |')
    lines+=['','APE为绝对百分比误差；P90是场景误差分位数，不是请求延迟P90。','',
        '| 独立对照 | 指标 | 改善/恶化/不变/未评分 | 数值改变格数 | APE差中位（百分点） | 模拟时延差中位（ms） |','|---|---|---|---:|---:|---:|']
    for name,comparison in result['anchor_comparisons'].items():
        for metric,item in comparison['by_metric'].items():
            outcomes=' / '.join(str(item['outcomes'].get(key,0)) for key in ('improved','regressed','unchanged','unscored'))
            lines.append(f"| {name} | {LABELS[metric]} | {outcomes} | {item['prediction_changed_cells']} | {item['ape_delta_percentage_points']['median']} | {item['time_delta_ms']['median']} |")
    lines+=['','差值均为后者减前者；APE差为负表示误差下降。未评分保留在每指标20格分母内。',
        f"cta_issue 已保存预测 {result['coverage']['saved_predictions_as_of_phase'][CANDIDATE]}/131；缺失 {result['coverage']['candidate_missing_prediction_cells']}；失败或未评分 {total['failed_or_unscored_cells']}/131。",
        '131格静态资格不等于131格实际MMVQ应用。CPU、非MMVQ和不支持格式不得从cell开关推断数值变化；本报告未取得完整算子应用计数。']
    if result['phase']=='full':
        lines+=['','cta_issue 全131 APE：']+[f'- {LABELS[metric]}：'+cell(total['metrics'][metric]['absolute_percentage_error_pct']) for metric in METRICS]
    reasons=Counter(reason for row in total['failures'] for reason in row['unscored_reasons'].values())
    if reasons:lines+=['','未评分原因（按指标计数）：']+[f'- {reason}: {count}' for reason,count in reasons.items()]
    lines+=['','逐格误差、失败、四路引用和前后哈希保留于JSON。缺失/失败不从131分母移除；A通过也不代表B独立验证通过。','']
    return '\n'.join(lines)


def heatmap(result):
    ids=sorted(result['anchor_rows']['current']);cw,rh,left=85,25,330;w=left+len(VARIANTS)*len(METRICS)*cw+20;h=100+len(ids)*rh
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><rect width="100%" height="100%" fill="white"/><style>text{{font:12px sans-serif}}</style><text x="12" y="22">R22 four-way anchor APE (%) · B unvalidated · full denominator 131</text>']
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['anchors','full']);parser.add_argument('--directory',type=Path,default=P);parser.add_argument('--heatmap',action='store_true')
    args=parser.parse_args(argv);result=assess(args.directory,args.phase);paths=write_report(args.directory,result,args.heatmap)
    print(json.dumps({'outputs':paths,'strict_pass_cells':result['candidate131']['strict_all3_below10_cells'],'denominator':DENOMINATOR,'gate_A':result['gate_A'],'gate_B':'unvalidated'},ensure_ascii=False))

if __name__=='__main__':main()
