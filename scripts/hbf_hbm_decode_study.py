"""Decode-only measurement windows after a real, fully executed prefill.

HBM always executes through DramCore, and HBF through NandCore. This study
does not load a fabricated KV state or divide decode tokens by prefill time.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from hbf_hbm_experiments import group, percentiles
from hbf_hbm_serving_study import run_case, StudyCasesError, POINT_RESULT_VERSION

MODELS = ("llama3_1-8b", "llama3_1-70b")
CONTEXTS = (2048, 32768, 131072, 1048576)
ARCHITECTURES = ("A_hbm_only", "B_all_kv_hbf")
OUTPUT_TOKENS = 64
# Share the serving hardware/execution revision so a geometry change also
# invalidates Decode points and their capacity control.
EXECUTION_CONTRACT = f"physical_core_unified_hardware_catalog_v4/serving-points-v{POINT_RESULT_VERSION}"
CAPACITY_BASELINE_CONTRACT = EXECUTION_CONTRACT + "/capacity-baseline-v1"
CAPACITY_BASELINE_FILENAME = "capacity-baseline-70b-1m-512gib.json"
CAPACITY_BASELINE_HBM_BYTES = 512 * 1024 ** 3


def capacity_baseline_parameters():
    # Omit the capacity override deliberately: this is the original inventory,
    # not point_parameters()'s enlarged A/B comparison.
    return dict(model_id="llama3_1-70b", prompt=1048576, batch=1, output=64)


def capacity_baseline_checks(row):
    failure = row.get("capacity_failure", {})
    return {
        "current_execution_contract": row.get("memory_execution_contract") == CAPACITY_BASELINE_CONTRACT,
        "original_workload_and_inventory": row.get("parameters") == capacity_baseline_parameters()
            and row.get("placement") == "A_hbm_only"
            and row.get("hbm_capacity_bytes") == CAPACITY_BASELINE_HBM_BYTES,
        "actual_physical_capacity_rejection": row.get("mode") == "physical_capacity_rejected"
            and row.get("attempted_serving") is True and row.get("executed_serving") is False
            and failure.get("exception_type") == "AllocationError"
            and failure.get("classification") == "physical_allocator_no_contiguous_extent"
            and failure.get("capacity_bytes") == CAPACITY_BASELINE_HBM_BYTES
            and failure.get("requested_allocation_bytes", 0) > 0
            and row.get("failed_allocation_bytes") == failure.get("requested_allocation_bytes"),
        "rejection_has_no_throughput": row.get("node_tps") is None and row.get("decode_tps") is None,
    }


def run_capacity_baseline():
    """Really attempt the original 70B/1M/64-output A inventory once.

    A normal completion or another rejection remains a failed negative-control
    check; neither is relabelled as the expected physical allocation failure.
    Unclassified exceptions propagate from run_case.
    """
    started = time.perf_counter()
    row = run_case("A_hbm_only", row_id="decode-capacity-llama3_1-70b-1048576-A_hbm_only-64-512GiB",
                   **capacity_baseline_parameters())
    failure = row.get("capacity_failure", {})
    observed_capacity = row.get("fixed_inventory", {}).get("hbm0", {}).get("capacity_bytes")
    row.update(model_id="llama3_1-70b", prompt_tokens=1048576,
        output_tokens_per_request=64, hbm_capacity_bytes=observed_capacity,
        failed_allocation_bytes=failure.get("requested_allocation_bytes"),
        exception_type=failure.get("exception_type"), failure_stage=failure.get("failure_stage"),
        decode_tps=None, wall_seconds=time.perf_counter()-started,
        memory_execution_contract=CAPACITY_BASELINE_CONTRACT,
        evidence="Fresh run_case execution of the original 512 GiB inventory, 70B model, 1M input and 64 requested output tokens; no cached or historical failure is used by this entrypoint.",
        interpretation="A failed contiguous allocation does not establish total peak demand or minimum feasible hardware capacity; occupancy and fragmentation observations are retained in capacity_failure.")
    row.setdefault("reason", "Original inventory completed serving; the expected capacity rejection was not reproduced.")
    row["capacity_baseline_checks"] = capacity_baseline_checks(row)
    row["status"] = "passed" if all(row["capacity_baseline_checks"].values()) else "failed"
    return row


def load_or_run_capacity_baseline(*, point_dir=None, resume=False):
    path = Path(point_dir) / CAPACITY_BASELINE_FILENAME if point_dir else None
    if resume and path and path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if all(capacity_baseline_checks(saved).values()):
            return saved
    row = run_capacity_baseline()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"capacity_baseline": row["id"], "status": row["status"],
                      "mode": row["mode"], "wall_seconds": row["wall_seconds"]}), flush=True)
    return row


def point_parameters(point):
    model,context,_architecture,output=point
    parameters=dict(model_id=model,prompt=context,batch=1,output=output)
    if model=='llama3_1-70b' and context==1048576:
        # The 512 GiB inventory cannot hold KV plus the real prefill workspace.
        # Expand BOTH A and B; do not compare unequal-capacity configurations.
        parameters['hbm_capacity_bytes']=1024**4
    return parameters


def decode_measurement(serving):
    """Count committed tokens in a common window containing no prefill work."""
    by_request = {}
    for event in serving.events:
        if event.event_type != "tokens_committed" or not event.request_id:
            continue
        count = int(event.details.get("visible_tokens", 0))
        by_request.setdefault(event.request_id, []).extend(
            [(float(event.timestamp_ns), event.details.get("source") == "prefill")] * count)
    if not by_request or any(len(tokens) < 2 for tokens in by_request.values()):
        raise ValueError("Decode measurement requires at least two output tokens per request")
    by_request = {key: sorted(tokens) for key, tokens in by_request.items()}
    prefill = [batch for batch in serving.batches if any(item.phase == "prefill" for item in batch.items)]
    decode = [batch for batch in serving.batches if any(item.phase == "decode" for item in batch.items)]
    if not prefill or not decode:
        raise ValueError("Both real prefill and decode batches must execute")
    start = max(max(batch.end_ns for batch in prefill), max(tokens[0][0] for tokens in by_request.values()))
    end = max(tokens[-1][0] for tokens in by_request.values())
    duration = end - start
    if duration <= 0:
        raise ValueError("Decode measurement window must have positive duration")
    counted = sum(not first and start < time <= end for tokens in by_request.values() for time, first in tokens)
    expected = sum(len(tokens) - 1 for tokens in by_request.values())
    pure_gaps = [right[0] - left[0] for tokens in by_request.values()
                 for left, right in zip(tokens, tokens[1:]) if start <= left[0] <= right[0] <= end]
    per_request = [{"request_id": key, "first_token_ns": tokens[0][0], "last_token_ns": tokens[-1][0],
                    "output_tokens": len(tokens), "post_first_tokens": len(tokens)-1,
                    "post_first_window_ns": tokens[-1][0]-tokens[0][0],
                    "tpot_ns": (tokens[-1][0]-tokens[0][0])/(len(tokens)-1),
                    "tbt_ns": percentiles(right[0]-left[0] for left, right in zip(tokens,tokens[1:]))}
                   for key, tokens in by_request.items()]
    checks = {
        "first_token_from_executed_prefill": all(tokens[0][1] and sum(first for _, first in tokens)==1 for tokens in by_request.values()),
        "all_post_first_tokens_counted": counted == expected,
        "no_prefill_overlaps_window": all(batch.end_ns <= start for batch in prefill),
        "decode_batches_start_after_prefill": all(batch.start_ns >= start-1e-6 for batch in decode),
        "event_token_counts_match_requests": all(len(tokens)==serving.request_metrics[key].visible_output_tokens for key,tokens in by_request.items()),
        "event_tpot_matches_runtime": all(math.isclose(row["tpot_ns"],serving.request_metrics[row["request_id"]].tpot_ns,rel_tol=1e-10,abs_tol=1e-4) for row in per_request),
    }
    return {"window":{"start_ns":start,"end_ns":end,"duration_ns":duration},
            "decode_tokens":counted,"expected_decode_tokens":expected,
            "node_decode_tps":counted*1e9/duration,"pure_decode_tbt_ns":percentiles(pure_gaps),
            "maximum_pure_decode_tbt_ns":max(pure_gaps,default=None),
            "excluded_boundary_tbt_count":expected-len(pure_gaps),"per_request":per_request,
            "prefill_batches":len(prefill),"decode_batches":len(decode),"checks":checks,
            "semantics":"Real prefill fully executes. Common decode window starts after the last prefill batch and all first tokens; numerator counts only later committed decode tokens. TBT pairs stay within one request. Standard TPOT retains the complete first-to-last-token interval."}


def core_checks(row):
    audit = row["core_execution_audit"]
    owners = audit["owners"]
    decode_owners = [owner for cohort in audit["cohorts"] if cohort["phase"] == "decode" for owner in cohort["owners"]]
    def real_io(rows, prefix, core, direction):
        return any(owner["physical_owner"].startswith(prefix) and owner["core"] == core
                   and owner["transactions"] > 0 and owner["logical_"+direction+"_bytes"] > 0 for owner in rows)
    checks = {"committed_core_totals_valid":audit["valid_for_committed_byte_totals"],
              "hbm_read_executed_by_dram_core":real_io(owners,"hbm0","DramCore","read"),
              "hbm_write_executed_by_dram_core":real_io(owners,"hbm0","DramCore","write"),
              "decode_hbm_read_executed_by_dram_core":real_io(decode_owners,"hbm0","DramCore","read")}
    if row["placement"] == "B_all_kv_hbf":
        checks.update({"decode_hbf_read_executed_by_nand_core":real_io(decode_owners,"hbf0","NandCore","read"),
                       "decode_hbf_write_executed_by_nand_core":real_io(decode_owners,"hbf0","NandCore","write")})
    else:
        checks["hbm_only_has_no_hbf_kv_io"] = not any(owner["physical_owner"].startswith("hbf0") and owner["transactions"] for owner in owners)
    return checks


def run_decode_point(point):
    model, context, architecture, output = point
    observed = {}
    def capture(result):
        observed.update(decode_measurement(result.serving))
    row = run_case(architecture,row_id=f"decode-{model}-{context}-{architecture}-{output}",
                   result_observer=capture,**point_parameters(point))
    if not row.get("executed_serving"):
        raise RuntimeError("Decode point was rejected: "+str(row))
    row["decode_measurement"] = observed
    row["decode_core_checks"] = core_checks(row)
    row["memory_execution_contract"] = EXECUTION_CONTRACT
    return row


def summarize_decode(rows, *, models=MODELS, contexts=CONTEXTS, output=OUTPUT_TOKENS):
    rows = sorted(rows,key=lambda row:row["id"])
    expected={(model,context,arch) for model in models for context in contexts for arch in ARCHITECTURES}
    actual={(row["parameters"]["model_id"],row["prompt_tokens"],row["placement"]) for row in rows}
    comparisons=[]; paired_inventories=[]
    for model in models:
        for context in contexts:
            pair={row["placement"]:row for row in rows if row["parameters"]["model_id"]==model and row["prompt_tokens"]==context}
            if set(pair)!=set(ARCHITECTURES):
                continue
            a,b=(pair[key] for key in ARCHITECTURES)
            same_inventory=a['fixed_inventory']==b['fixed_inventory']
            paired_inventories.append(same_inventory)
            if not same_inventory:
                continue
            comparisons.append({"model_id":model,"prompt_tokens":context,"output_tokens":output,
                "A_decode_tps":a["decode_measurement"]["node_decode_tps"],"B_decode_tps":b["decode_measurement"]["node_decode_tps"],
                "B_over_A_decode_tps":b["decode_measurement"]["node_decode_tps"]/a["decode_measurement"]["node_decode_tps"],
                "A_tpot_ns":a["tpot_ns"]["p50"],"B_tpot_ns":b["tpot_ns"]["p50"],
                "B_over_A_tpot":b["tpot_ns"]["p50"]/a["tpot_ns"]["p50"],
                "B_over_A_total_tps":b["node_tps"]/a["node_tps"],
                "same_inventory":a["fixed_inventory"]==b["fixed_inventory"],"row_ids":[a["id"],b["id"]]})
    checks={"all_requested_pairs_executed":actual==expected and len(rows)==len(expected),
        "fixed_output_length":bool(rows) and all(row["visible_output_tokens"]==output and row["completed_requests"]==1 for row in rows),
        "pure_decode_windows":bool(rows) and all(all(row["decode_measurement"]["checks"].values()) for row in rows),
        "actual_core_submissions_verified":bool(rows) and all(all(row["decode_core_checks"].values()) for row in rows),
        "placements_verified":all(all(row["placement_verification"].values()) for row in rows),
        "paired_inventory_identical":len(paired_inventories)==len(models)*len(contexts) and all(paired_inventories)}
    return group("19_decode_physical","物理Core下的独立Decode对照",rows,checks,
        ["真实prefill之后的Decode窗口、固定输出长度和实际DRAM/NAND Core调用"],
        ["GPU计算与发射仍使用现有聚合简化模型；HBM/HBF内存完成时间来自各自Core。",
         "参考GPU的单token attention按MMA矩阵块扩展计量，并允许计算与物理访存重叠；模型间的Decode倍率差异受这些假设影响，不是HBF对大模型影响普遍更小的规律。",
         "128K加输出略超原生上下文，1M为架构压力外推，不代表模型语义有效性。",
         "70B/1M的A/B共同使用1TiB HBM容纳真实KV与prefill工作空间；其余点为512GiB。固定16H×32Gbit组织与0.625ns命令时钟，容量增加只增加stack数；方向预算保持1600GB/s，物理接口总速率与并行度随stack数增加。未知厂商时序与刷新值没有补造，实际假设见Core配置。",
         "每点一个请求、固定输出长度；TBT分位数是本次样本统计，不是生产尾延迟保证。"],
        comparisons=comparisons,models=list(models),contexts=list(contexts),output_tokens=output,
        execution_contract="HBM always DramCore; HBF NandCore; GPU compute/launch approximation retained")


def run_decode_study(*,workers=2,models=MODELS,contexts=CONTEXTS,output=OUTPUT_TOKENS,point_dir=None,resume=False):
    points=[(model,context,arch,output) for context in contexts for model in models for arch in ARCHITECTURES]
    rows=[]; pending=[]
    if point_dir:
        point_dir=Path(point_dir);point_dir.mkdir(parents=True,exist_ok=True)
    def path(point):
        return point_dir/('-'.join(map(str,point))+'.json') if point_dir else None
    for point in points:
        saved=path(point)
        if resume and saved and saved.exists():
            row=json.loads(saved.read_text(encoding='utf-8'))
            if (row.get('memory_execution_contract')==EXECUTION_CONTRACT
                    and row.get('core_execution_audit',{}).get('executed_physical_configurations')
                    and row.get('parameters')==point_parameters(point) and all(core_checks(row).values())):
                rows.append(row);continue
        pending.append(point)
    failures=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(run_decode_point,point):point for point in pending}
        for job in as_completed(jobs):
            try:
                row=job.result()
            except Exception as error:
                point=jobs[job]
                failures.append(('-'.join(map(str,point)),error))
                print(json.dumps({'failed_point':point,'error':str(error)},ensure_ascii=False),flush=True)
                continue
            rows.append(row)
            if point_dir:path(jobs[job]).write_text(json.dumps(row,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            print(json.dumps({"completed":row["id"],"decode_tps":row["decode_measurement"]["node_decode_tps"],"tpot_ms":row["tpot_ns"]["p50"]/1e6,"wall_seconds":row["wall_seconds"]}),flush=True)
    if failures:
        raise StudyCasesError(failures,rows) from failures[0][1]
    result=summarize_decode(rows,models=models,contexts=contexts,output=output)
    if "llama3_1-70b" in models and 1048576 in contexts and output == 64:
        baseline = load_or_run_capacity_baseline(point_dir=point_dir, resume=resume)
        result["capacity_baseline_observations"] = [baseline]
        result["checks"]["original_inventory_negative_control_reproduced"] = all(capacity_baseline_checks(baseline).values())
        result["status"] = "passed" if all(result["checks"].values()) else "failed"
        if result["status"] != "passed":
            result["capability"]["supported"] = []
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/hbf_hbm/19_decode_physical.json')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--models',default=','.join(MODELS))
    parser.add_argument('--contexts',default=','.join(map(str,CONTEXTS)))
    parser.add_argument('--output-tokens',type=int,default=OUTPUT_TOKENS)
    parser.add_argument('--resume',action='store_true',help='Reuse only this unchanged physical-Core decode study’s saved points')
    parser.add_argument('--capacity-baseline-only',action='store_true',
        help='Attempt only the original 70B/1M/64-output 512 GiB A inventory; --output is this observation JSON, not the group summary')
    args=parser.parse_args()
    if not 1<=args.workers<=8 or args.output_tokens<2:parser.error('workers must be 1..8 and output-tokens >= 2')
    if args.capacity_baseline_only:
        if args.output == parser.get_default('output'):
            parser.error('--capacity-baseline-only requires a separate --output file; it cannot overwrite the group summary')
        result = run_capacity_baseline()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'status':result['status'],'mode':result['mode'],
                          'wall_seconds':result['wall_seconds'],'output':str(args.output)},ensure_ascii=False))
        return 0 if result['status']=='passed' else 1
    result=run_decode_study(workers=args.workers,models=tuple(args.models.split(',')),contexts=tuple(map(int,args.contexts.split(','))),
        output=args.output_tokens,point_dir=args.output.parent/'decode_points',resume=args.resume)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'status':result['status'],'rows':len(result['rows']),'output':str(args.output)},ensure_ascii=False))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
