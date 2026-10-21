"""Historical bypass-study helpers; its command-line runner is retired.

Existing results keep their original provenance. New comparisons use native
serving/planner/physical execution, as arranged by exploring_hbf_native_plan.
The helpers remain only to interpret the historical experiment contract.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from math import ceil
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from exploring_hbf_paper_config import hardware_by_id, workload_catalog
from exploring_hbf_search import search_candidates, maximum_capacity_batch
from heterollm_sim.paper_inference import PaperExecutionAssumptions, PaperInferenceEngine, PaperParallelism
from heterollm_sim.paper_memory import PaperMemoryCapacityError
from heterollm_sim.paper_models import PaperPrecision, get_paper_model
from heterollm_sim.paper_serving_control import ServingRequest

SEARCH_POLICY = "exact_prepared_capacity_v1"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def case_id(case):
    slo = str(case["slo"]).replace(".", "p")
    variant = case.get("variant", "primary")
    statistic = case.get("slo_statistic", "mean_iteration")
    suffix = "" if statistic == "mean_iteration" else "__"+statistic
    return f'{variant}__{case["model"]}__{case["hardware"]}__{case["workload"]}__G{case["gpus"]}__S{slo}{suffix}'


def run_case(case, output):
    directory = Path(output)
    identifier = case_id(case)
    destination = directory / "points" / (identifier+".json")
    existing = None
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing["case"] != case:
            raise ValueError("existing experiment id has different parameters; select a distinct named variant")
        if existing["status"] == "failed" or (existing.get("search_policy") == SEARCH_POLICY
                and existing["status"] in {"completed", "infeasible", "no_validated_deployment"}):
            return existing
    started = perf_counter()
    prior_wall = existing.get("wall_seconds", 0.) if existing else 0.

    def elapsed():
        return prior_wall + perf_counter()-started
    model, hardware = get_paper_model(case["model"]), hardware_by_id(case["hardware"])
    workload = next(item for item in workload_catalog() if item.workload_id == case["workload"])
    assumptions = replace(PaperExecutionAssumptions(), **case.get("assumptions", {}))
    if existing and existing.get("assumptions") != asdict(assumptions):
        raise ValueError("existing experiment assumptions differ; select a distinct named variant")
    seconds = 86400/workload.output_tokens if case["slo"] == "offline" else float(case["slo"])
    statistic = case.get("slo_statistic", "mean_iteration")
    if statistic not in {"mean_iteration", "max_iteration"}:
        raise ValueError("explicit SLO statistic must be mean_iteration or max_iteration")
    record = dict(case_id=identifier, case=case, status="searching", assumptions=asdict(assumptions),
                  search_policy=SEARCH_POLICY,
                  precision=asdict(PaperPrecision()), paper_slo_seconds=seconds,
                  selection_scope="all discrete layouts screened; best of bounded closed-loop validations, not a proved global optimum")
    if existing:
        record["validations"] = existing.get("validations", [])
        if "screening" in existing:
            record["screening"] = existing["screening"]
    write_json(destination, record)
    candidates = (existing["screening"] if existing and "screening" in existing
                  else search_candidates(model, hardware, case["gpus"], workload, assumptions, slo_seconds=case["slo"]))
    record["screening"] = candidates
    errors = [row for row in candidates if row["status"] == "execution_failed"]
    if errors:
        record.update(status="failed", reason="A deployment screening execution failed; not reclassified as hardware infeasibility.",
                      screening_failures=errors, wall_seconds=elapsed())
        write_json(destination, record)
        return record
    feasible = [row for row in candidates if row["status"] == "screened_candidate"]
    # A search-policy refinement does not invalidate completed physical events.
    # Retain them and execute only genuinely new deployment/batch/lead-time
    # candidates. The same case and full actual assumptions are required.
    validations = list(existing.get("validations", [])) if existing else []
    accepted_prior = [item for item in validations if item.get("accepted")
                      and item.get("result", {}).get("status") == "completed"]
    best = max(accepted_prior, key=lambda item:item["result"]["per_gpu_decode_tokens_per_second"], default=None)
    reused = 0
    # At least two structurally different deployments are validated when
    # available. No observed paper throughput is used to choose or fit them.
    for layout_index, candidate in enumerate(feasible):
        if best is not None and layout_index >= case.get("validate_layouts", 2):
            break
        p = PaperParallelism(**candidate["parallelism"])
        batch = candidate["candidate_batch"]
        multiple = p.ep*p.microbatches
        for attempt in range(4):
            probe = PaperInferenceEngine(model, hardware, p, batch_size=batch,
                input_tokens=workload.input_tokens, output_tokens=workload.output_tokens,
                assumptions=assumptions)
            burst = max(1, ceil(batch/(p.ep*(workload.output_tokens-1))))
            pd_time_ns = max(probe.import_kv(ServingRequest(index*p.ep, workload.input_tokens,
                workload.output_tokens, slot_id=0), 0.).ready_ns for index in range(burst))
            # A derived admission lead time, not an unconstrained free buffer.
            # The corresponding finite successor KV pool is capacity checked.
            step_lower_estimate = candidate["screen_tpot_seconds"]*1e9*batch/candidate["candidate_batch"]*.8
            lookahead = min(workload.output_tokens-1, max(2, ceil(pd_time_ns/step_lower_estimate)+2))
            serving_assumptions = replace(assumptions, prepared_lookahead=lookahead)
            try:
                engine = PaperInferenceEngine(model, hardware, p, batch_size=batch,
                    input_tokens=workload.input_tokens, output_tokens=workload.output_tokens,
                    assumptions=serving_assumptions)
            except PaperMemoryCapacityError as error:
                refined = maximum_capacity_batch(model, hardware, p, upper_batch=batch,
                    input_tokens=workload.input_tokens, output_tokens=workload.output_tokens,
                    assumptions=serving_assumptions)
                validations.append(dict(batch_size=batch, parallelism=asdict(p), accepted=False,
                    stage="derived_prefetch_capacity", reason=str(error), prepared_lookahead=lookahead,
                    capacity_refinement=refined))
                batch = refined["batch_size"]
                if batch < multiple:
                    break
                continue
            tick = perf_counter()
            prior = next((item for item in validations if item.get("batch_size") == batch
                and item.get("parallelism") == asdict(p)
                and item.get("result", {}).get("status") == "completed"
                and item["result"].get("research_assumptions") == asdict(serving_assumptions)), None)
            if prior is not None:
                result = prior["result"]
                reused += 1
            else:
                result = engine.run(warmup_steps=workload.output_tokens,
                                    measurement_steps=2*workload.output_tokens)
            if case["slo"] == "offline":
                deadline = result["max_completed_request_elapsed_ns"]
                accepted = deadline is not None and deadline <= 86400*1e9
            else:
                accepted = result[statistic+"_ns"] <= seconds*1e9*(1+1e-10)
            validation = dict(batch_size=batch, parallelism=asdict(p), accepted=accepted,
                wall_seconds=perf_counter()-tick, result=result,
                slo_statistic=("maximum completed fresh request decode-node residence, not prefill duration" if case["slo"] == "offline"
                               else statistic+": complete full-batch iteration including PD replacement waits"))
            if prior is None:
                validations.append(validation)
            record.update(status="validating", validations=validations, wall_seconds=elapsed(),
                          reused_validation_count=reused)
            write_json(destination, record)
            if accepted:
                if best is None or result["per_gpu_decode_tokens_per_second"] > best["result"]["per_gpu_decode_tokens_per_second"]:
                    best = validation
                break
            next_batch = int(batch*seconds*1e9/result[statistic+"_ns"]*.97)//multiple*multiple
            next_batch = min(batch-multiple, next_batch)
            if next_batch < multiple:
                break
            batch = next_batch
    terminal = "completed" if best else ("infeasible" if not feasible else "no_validated_deployment")
    record.update(status=terminal, selected=best, validations=validations,
                  wall_seconds=elapsed(), reused_validation_count=reused)
    if not best:
        record["reason"] = "No screened deployment passed private capacity and the declared SLO statistic in the tested closed-loop candidates."
    write_json(destination, record)
    return record


def experiments(scope):
    primary = [dict(model=model, hardware=hardware, workload=workload, gpus=gpus, slo=.1)
        for model in ("llama3_405b", "llama4_maverick")
        for hardware in ("HBM4", "CONV", "CONV+", "HBF", "HBF+")
        for workload in ("SHORT", "MID", "LONG")
        for gpus in ((8,) if scope == "eight" else (1,2,4,8,16))]
    if scope == "all":
        primary += [dict(model=model, hardware=hardware, workload="LONG", gpus=gpus, slo=slo)
            for model in ("llama3_405b", "llama4_maverick")
            for hardware in ("HBM4", "CONV", "CONV+", "HBF", "HBF+")
            for gpus in (1,2,4,8,16) for slo in (.05,.2,"offline")]
    # The author's SAFARI talk slide 19 states peak HW performance. Keep this
    # choice in each case, so a research-efficiency run cannot share its id
    # and silently be reused after changing the backend's defaults.
    for case in primary:
        case["assumptions"] = {"matrix_efficiency": 1., "scalar_efficiency": 1.}
    return sorted(primary,key=lambda case:(case["gpus"] != 8, case["slo"] != .1,
        case["model"], {"HBM4":0,"HBF":1,"HBF+":2,"CONV":3,"CONV+":4}[case["hardware"]],
        ("SHORT","MID","LONG").index(case["workload"]), case["gpus"], str(case["slo"])))


def main():
    raise SystemExit(
        "独立 paper_* 实验入口已停用；历史结果保留，不再启动旁路实验。"
        "新原生实验安排：scripts/exploring_hbf_native_plan.py；"
        "短请求功能预检：scripts/exploring_hbf_native_pilot.py。"
    )


if __name__ == "__main__":
    main()
