"""Prepare a small, unfitted sensitivity plan; default invocation runs no jobs.

Core: eight complete workload executions plus two zero-simulation SLO rechecks.
Extended: eight optional complete workload executions. Fixed deployments keep
the selected layout, batch, precision and admission lookahead unchanged unless
the named control explicitly changes that lookahead. Reoptimization is a
different experiment, delegates to the maintained runner, and is never mixed
with fixed-deployment causality. No paper output is a fitting target.

Examples (preparation is safe while the primary queue is running)::

    python scripts/exploring_hbf_sensitivity_plan.py
    python scripts/exploring_hbf_sensitivity_plan.py --write-plan

After the primary queue has finished, explicitly execute ONE independent case::

    python scripts/exploring_hbf_sensitivity_plan.py --execute --case-id fixed_compute_l3_hbm

This helper has no background scheduler, automatic resume or old-scan entry.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
DEFAULT_BASELINE = ROOT / "artifacts/exploring_hbf_reproduction_20261019"
DEFAULT_OUTPUT = ROOT / "artifacts/exploring_hbf_sensitivity_20261019"


def source_case(model, hardware, workload="SHORT", gpus=8):
    return {"model": model, "hardware": hardware, "workload": workload, "gpus": gpus, "slo": .1}


def source_filename(case):
    return (f'primary__{case["model"]}__{case["hardware"]}__{case["workload"]}'
            f'__G{case["gpus"]}__S0p1.json')


def specifications():
    """Controls are selected by mechanism, not proximity to any published bar."""
    l3, l4 = "llama3_405b", "llama4_maverick"
    rows = []
    for label, model in (("l3", l3), ("l4", l4)):
        for arch, hardware in (("hbm", "HBM4"), ("hbf_plus", "HBF+")):
            rows.append(dict(case_id=f"fixed_compute_{label}_{arch}", priority="core",
                mode="fixed", baseline=source_case(model, hardware),
                assumption_overrides={"matrix_efficiency": .6},
                question="作者slides规定峰值算力的基线上，若矩阵有效效率只有60%，同部署代价如何变化？",
                limit="只改变matrix_efficiency；scalar效率保留源结果峰值设置，不把未知效率反调到论文吞吐。"))
    for label, model in (("l3", l3), ("l4", l4)):
        for arch, hardware in (("hbm", "HBM4"), ("hbf_plus", "HBF+")):
            rows.append(dict(case_id=f"reopt_capacity_gib_{label}_{arch}", priority="core",
                mode="reoptimized", baseline=source_case(model, hardware),
                assumption_overrides={"capacity_unit_bytes": 2**30},
                question="论文GB单位歧义能增加多少可达批量和吞吐？",
                limit="作者slides的405B约756GB与官方756GiB、Maverick约746GB与746GiB构成歧义线索，非所有单位的证明。只把设备每GB容量解释为GiB；模型张量数、精度、介质速率、3.13MB读暂存、40MB SRAM与HOST容量均不缩放。"))
    for label, model in (("l3", l3), ("l4", l4)):
        rows.append(dict(case_id=f"recheck_slo_{label}_hbf_long", priority="core",
            mode="postprocess", baseline=source_case(model, "HBF+", "LONG"),
            assumption_overrides={},
            question="同一个真实episode在平均100ms和最大100ms下是否得到不同验收？",
            limit="复用已有完整episode；不重跑、不改变物理时序，不把最大值规则偷偷替换主实验平均规则。"))
    rows.extend([
        dict(case_id="fixed_node_nic_l3_hbf_short", priority="extended", mode="fixed",
            baseline=source_case(l3, "HBF+"), assumption_overrides={"nic_scope": "node"},
            question="共置DP副本和EP/TP组共享100GB/s节点NIC时，原部署吞吐损失多大？",
            limit="使用当前固定份额节点策略，不把空闲份额借用或公平调度当成已实现。"),
        dict(case_id="fixed_host50_l4_hbf_long", priority="extended", mode="fixed",
            baseline=source_case(l4, "HBF+", "LONG"), assumption_overrides={"pd_host_gb_s": 50.},
            question="HOST DRAM端口从200降到50GB/s，PD驻留和均值/峰值是否敏感？",
            limit="只改变HOST读写带宽；保留64MiB缓冲和原导入预备窗口，不免费扩大缓冲或自动改batch。"),
        dict(case_id="fixed_sram2500_l4_hbf_long", priority="extended", mode="fixed",
            baseline=source_case(l4, "HBF+", "LONG"),
            assumption_overrides={"sram_gb_s_per_stack": 2500.},
            question="每堆SRAM端口从10降到2.5TB/s后，读暂存和中间张量服务是否成为瓶颈？",
            limit="这是未披露片上端口的研究区间，不是对某颗SRAM产品的实测描述；容量不变。"),
        dict(case_id="fixed_prefetch2x_l4_hbf_long", priority="extended", mode="fixed",
            baseline=source_case(l4, "HBF+", "LONG"), assumption_overrides={},
            lookahead_multiplier=2,
            question="保持FIFO但更早准备PD，减少补位停顿还是加剧未来预约的队头阻塞？",
            limit="预备KV池仍计入私有容量；若原batch因此放不下就真实拒绝，不自动缩批。此项不是FIFO与公平调度的直接对比。"),
    ])
    for arch, hardware in (("hbm", "HBM4"), ("hbf_plus", "HBF+")):
        rows.append(dict(case_id=f"reopt_compute_l3_{arch}", priority="extended", mode="reoptimized",
            baseline=source_case(l3, hardware), assumption_overrides={"matrix_efficiency": .6},
            question="若矩阵有效效率只有60%，重新优化会否改变并行方案和批量？",
            limit="须与同参数固定回放分开展示；部署切换的收益不能归为纯算力弹性。"))
    for label, model in (("l3", l3), ("l4", l4)):
        rows.append(dict(case_id=f"reopt_max_slo_{label}_hbf_long", priority="extended", mode="reoptimized",
            baseline=source_case(model, "HBF+", "LONG"), assumption_overrides={},
            slo_statistic="max_iteration",
            question="若选择逐步最大100ms门槛，可验证部署会怎样变化？",
            limit="论文未公开TPOT聚合口径；此项是独立目标，不能与平均100ms主曲线合并。"))
    return rows


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_selected(directory, spec):
    path = Path(directory) / "points" / source_filename(spec["baseline"])
    if not path.is_file():
        raise ValueError(f"主实验尚无该点：{path.name}")
    record = _read(path)
    selected = record.get("selected")
    if (record.get("status") != "completed" or not isinstance(selected, dict)
            or selected.get("accepted") is not True or not isinstance(selected.get("result"), dict)
            or selected["result"].get("status") != "completed"):
        raise ValueError(f"主实验不是已验收完整episode：{path.name}")
    actual = record["case"]
    if (any(actual.get(key) != value for key, value in spec["baseline"].items())
            or actual.get("variant", "primary") != "primary"
            or actual.get("slo_statistic", "mean_iteration") != "mean_iteration"):
        # Additional, fully recorded source assumptions are retained below.
        # A control variant or a different workload cannot silently become a baseline.
        raise ValueError("主实验参数与声明的基线不一致")
    return path, record, selected


def plan(directory):
    rows = []
    for spec in specifications():
        row = dict(spec)
        try:
            path, record, selected = load_selected(directory, spec)
            result = selected["result"]
            row.update(baseline_status="ready", source=str(path),
                selected_batch=selected["batch_size"], selected_parallelism=selected["parallelism"],
                selected_lookahead=result["research_assumptions"]["prepared_lookahead"],
                baseline_mean_iteration_ms=result["mean_iteration_ns"]/1e6,
                baseline_max_iteration_ms=result["max_iteration_ns"]/1e6)
        except (OSError, ValueError, KeyError) as error:
            row.update(baseline_status="not_ready", reason=str(error))
        rows.append(row)
    return {"status": "planned_not_started", "baseline_directory": str(Path(directory)),
        "cases": rows,
        "new_complete_workloads": {priority: sum(row["priority"] == priority
            and row["mode"] != "postprocess" for row in rows) for priority in ("core", "extended")},
        "existing_episode_rechecks": sum(row["mode"] == "postprocess" for row in rows),
        "fifo_boundary": "只有FIFO实现；提前窗口/链路/缓冲敏感性不能证明公平调度的结果。要直接比较须另建保持服务费用与信用一致的ready-chunk调度器。",
        "staging_unit_prerequisite": "每堆读暂存3.13MiB须先暴露独立的字节输入，单独更改该容量；不得借capacity_unit_bytes连同HBM/HBF、40MB SRAM或HOST容量一起改。修复版结果之前不执行。",
        "no_fitting": "参数预声明为物理上界、容量单位解释和未披露端口的研究控制；不读取论文吞吐作为求解或调参目标。",
        "execution": "默认只计划。--execute --case-id仅串行一个独立控制点；主实验运行时拒绝启动。"}


def _slo_recheck(result, case):
    if case.get("slo") == "offline":
        elapsed = result.get("max_completed_request_elapsed_ns")
        return {"offline_decode_residence_accepted": elapsed is not None and elapsed <= 86400*1e9}
    limit = float(case.get("slo", .1))*1e9
    return {"mean_iteration_accepted": result["mean_iteration_ns"] <= limit*(1+1e-10),
        "max_iteration_accepted": result["max_iteration_ns"] <= limit*(1+1e-10),
        "mean_iteration_ms": result["mean_iteration_ns"]/1e6,
        "max_iteration_ms": result["max_iteration_ns"]/1e6}


def execute_one(spec, baseline_directory, output_directory):
    """One explicit serial action; no automatic queue or old-scan resumption."""
    status_path = Path(baseline_directory)/"status.json"
    if (not status_path.exists()
            or _read(status_path).get("status") not in {"completed", "completed_with_failures"}):
        raise ValueError("仅使用已结束的修复版主实验；运行、暂停或失效的队列不能启动敏感性回放。")
    source, record, selected = load_selected(baseline_directory, spec)
    result = selected["result"]
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    target = output_directory/(spec["case_id"]+".json")
    if target.exists():
        existing = _read(target)
        if existing.get("specification") != spec:
            raise ValueError("同名控制点已有不同参数，请使用新的具名case")
        return existing
    if spec["mode"] == "postprocess":
        answer = {"status": "existing_episode_rechecked", "specification": spec,
            "source": str(source), "new_simulation": False,
            "slo": _slo_recheck(result, spec["baseline"])}
    elif spec["mode"] == "reoptimized":
        from reproduce_exploring_hbf import run_case
        source_assumptions = dict(record["assumptions"])
        source_assumptions.update(spec["assumption_overrides"])
        case = {**record["case"], "variant": spec["case_id"], "assumptions": source_assumptions}
        if "slo_statistic" in spec:
            case["slo_statistic"] = spec["slo_statistic"]
        reopt = output_directory/"reoptimized"
        (reopt/"points").mkdir(parents=True, exist_ok=True)
        completed = run_case(case, str(reopt))
        answer = {"status": completed["status"], "specification": spec,
            "source": str(source), "new_simulation": True, "reoptimized_record": completed}
    else:
        from exploring_hbf_paper_config import hardware_by_id
        from heterollm_sim.paper_inference import PaperExecutionAssumptions, PaperInferenceEngine, PaperParallelism
        from heterollm_sim.paper_memory import PaperMemoryCapacityError
        from heterollm_sim.paper_models import PaperPrecision, get_paper_model
        assumptions = dict(result["research_assumptions"])
        assumptions.update(spec["assumption_overrides"])
        if "lookahead_multiplier" in spec:
            assumptions["prepared_lookahead"] = min(result["config"]["output_tokens"]-1,
                assumptions["prepared_lookahead"]*spec["lookahead_multiplier"])
        assumptions["basis"] += (" Fixed-deployment sensitivity " + spec["case_id"]
            + ": the explicit numeric fields contain the declared overrides; baseline prose above "
              "describes the original settings. No layout/batch or unpublished efficiency is fitted.")
        known = {field.name for field in fields(PaperExecutionAssumptions)}
        if set(assumptions)-known:
            raise ValueError("基线包含当前引擎无法解释的研究参数")
        tick = perf_counter()
        try:
            engine = PaperInferenceEngine(get_paper_model(spec["baseline"]["model"]),
                hardware_by_id(spec["baseline"]["hardware"]),
                PaperParallelism(**selected["parallelism"]), batch_size=selected["batch_size"],
                input_tokens=result["config"]["input_tokens"], output_tokens=result["config"]["output_tokens"],
                assumptions=PaperExecutionAssumptions(**assumptions),
                precision=PaperPrecision(**record["precision"]))
            replay = engine.run(warmup_steps=result["config"]["warmup_steps"],
                                measurement_steps=result["config"]["measurement_steps"])
            answer = {"status": "fixed_deployment_completed", "specification": spec,
                "source": str(source), "new_simulation": True, "wall_seconds": perf_counter()-tick,
                "fixed_batch": selected["batch_size"], "fixed_parallelism": selected["parallelism"],
                "result": replay, "slo": _slo_recheck(replay, spec["baseline"]),
                "throughput_ratio_to_fixed_baseline": replay["decode_tokens_per_second"]/result["decode_tokens_per_second"]}
        except PaperMemoryCapacityError as error:
            answer = {"status": "fixed_deployment_capacity_rejected", "specification": spec,
                "source": str(source), "new_simulation": False, "reason": str(error),
                "fixed_batch": selected["batch_size"], "fixed_parallelism": selected["parallelism"]}
    target.write_text(json.dumps(answer, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--case-id", choices=[row["case_id"] for row in specifications()])
    args = parser.parse_args()
    if args.execute:
        if not args.case_id:
            parser.error("--execute必须指定单个--case-id")
        spec = next(row for row in specifications() if row["case_id"] == args.case_id)
        answer = execute_one(spec, args.baseline_dir, args.output)
        print(json.dumps({"case_id": spec["case_id"], "status": answer["status"]}, ensure_ascii=False))
        return
    planned = plan(args.baseline_dir)
    if args.write_plan:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"plan.json").write_text(json.dumps(planned, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(planned, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
