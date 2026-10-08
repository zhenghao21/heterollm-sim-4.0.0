"""Read-only first-layer/output-tail ownership audit, without GPU execution."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from heterollm_sim import planner as p
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.cuda_graph_dispatch import iter_source_dispatches
from heterollm_sim.cuda_graph_task_binding import match_cuda_dispatch_tasks


def render_html(result):
    structure = result["source_structure_validation"]
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>CUDA 调用来源覆盖审计</title>',
        '<style>body{font:16px/1.65 system-ui,sans-serif;color:#162238;background:#f4f6fa;margin:0}main{max-width:1200px;margin:auto;padding:36px}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{padding:10px;border:1px solid #d5dce6;text-align:left}th{background:#eaf0f7}.note{background:#fff1d6;padding:16px;border-left:4px solid #ca8418}code{overflow-wrap:anywhere}a{color:#155ca5}</style><main>',
        '<h1>CUDA 调用来源覆盖审计</h1>',
        '<p>真实 CUDA 调用与模型成本任务之间的对应关系已检查。这里只检查来源和覆盖，不改变设备调度或成本，也没有运行 native 推理或前端仿真。</p>',
        '<p class="note">设备启动成本尚未部署。当前独立测量仍不能排除驱动批量提交和测量工具的影响，不能据此填写一个固定启动延迟，也不能宣称 Graph 预测精度已修复。</p>',
        f'<p>{structure["model_count"]} 个模型的来源结构共完成 {structure["call_count"]:,} 次调用导出；与旧结构的生命周期判断条件和 CUDA 节点拓扑没有差异。下面的语义对应检查覆盖每个稠密模型的首层及最后一层到输出头，分别使用 512-token prefill 和已有 512-token KV 的单 token decode。</p>',
        '<table><thead><tr><th>模型</th><th>状态</th><th>区域检查</th><th>含多个 CUDA 节点的组</th><th>任务序列逆序跳转（最大值）</th></tr></thead><tbody>']
    for case in result["cases"]:
        rows = case["regions"]
        matched = [row for row in rows if row["status"] == "matched"]
        status = {"matched": "来源对应通过", "unmatched": "存在未对应项", "unsupported": "混合架构未适配"}[case["status"]]
        parts.append('<tr><td>' + html.escape(case["case_id"]) + '</td><td>' + status + '</td><td>'
            + (f'{len(matched)}/{len(rows)}' if rows else '未套用稠密模型规则') + '</td><td>'
            + str(sum(row.get("source_multi_node_dispatches", 0) for row in matched)) + '</td><td>'
            + (str(max(row["modeled_order_backward_transitions"] for row in matched)) if matched else '—') + '</td></tr>')
    parts.extend(['</tbody></table>',
        '<p>“通过”表示每个检查区域的原生操作和模拟事件能按真实权重、张量关系及源码融合边界找到唯一对应。任务序列中的逆序跳转仍存在；当前匹配器没有重排依赖，该数值也不等同于执行时间误差。</p>',
        '<h2>部署设备成本之前仍需完成</h2><ol>',
        '<li>取得可独立验证的普通启动和 Graph 启动设备成本，区分 GPU 本体、CPU 供给、驱动排队及测量扰动。</li>',
        '<li>按原生调用边界建模提交与执行顺序、CPU/GPU 重叠及程序化依赖启动（PDL）的完成点。</li>',
        '<li>补齐一个原生调用组内部的多个 CUDA 节点阶段。不得按节点数比例切分已有计算时长或 DRAM 流量。</li>',
        '<li>核实融合内核的临时缓冲和资源生命周期；把已有任务分到一组，不代表中间内存访问已与 native 完全一致。</li>',
        '<li>适配 Qwen3.8-27B 的循环状态、线性注意力与状态复制来源。其结构导出成功不代表语义匹配已完成。</li>',
        '<li>接入完整服务流程后，再从前端运行并用留出的 native 数据检查误差。此次没有用模型总耗时拟合参数。</li>',
        '</ol><p><a href="source_dispatch_coverage.json">逐区域对应明细 JSON</a></p></main></html>'])
    return ''.join(parts)


def source_structure_validation(structures, cases):
    path = structures.parent / "ownership_status.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    models = value.get("models", [])
    if (value.get("status") != "complete" or value.get("capture_only") is not True
            or value.get("target_llm_latency_used") is not False
            or len(models) != len(cases)
            or {row["case_id"] for row in models} != {row["case_id"] for row in cases}
            or any(row.get("status") != "complete" or type(row.get("calls")) is not int
                   or row["calls"] < 1 or row.get("source_lifecycle_differences") != 0
                   or row.get("capture_topology_differences") != 0 for row in models)):
        raise ValueError("source structure matrix does not confirm complete unchanged capture-only ownership")
    return {"status": "complete", "manifest": str(path.resolve()), "model_count": len(models),
            "call_count": sum(row["calls"] for row in models), "source_lifecycle_differences": 0,
            "capture_topology_differences": 0, "target_llm_latency_used": False}


def compile_region(scenario, phase, region):
    tokens, prior = (512, 0) if phase == "prefill" else (1, 512)
    layers = p._execution_layers(scenario)
    layer = layers[0 if region == "first_layer" else -1]
    with p._compilation_scope(scenario):
        plan, router = p._parallel_plan(scenario), p._topology_router(scenario)
        request = scenario.workload.requests[-1]
        builder = p._TaskBuilder(request)
        builder._linear_state_owner_request_ids = (request.request_id,)
        selection, indices, prepared = None, None, ()
        if region == "output_tail":
            selection = p._final_output_selection(scenario, plan, tokens, (tokens - 1,))
            prepared, indices = p._prepare_output_selection_inputs(
                builder, scenario, router, plan, selection, phase, ())
        end = p._compile_parallel_layer_body(builder, scenario, plan, router, layer,
            token_batch=tokens, context_tokens=tokens + prior, kv_read_tokens=prior,
            kv_append_tokens=tokens, kv_materialized_tokens=tokens, linear_state_runtime=None,
            phase=phase, dependencies=prepared, output_selection=selection,
            output_indices_dependency=indices)
        if region == "output_tail":
            p._compile_parallel_lm_head(builder, scenario, plan, router, phase, (end,),
                token_batch=1, output_selection=selection, output_indices_dependency=indices)
    return layer, p._promote_physical_allocation_extents(builder.tasks)


def layer_start(groups, index):
    weight = f"blk.{index}.attn_norm.weight"
    matches = [i for i, group in enumerate(groups) if any(
        node["output"]["op"] == "MUL" and any(
            source["slot"] == 1 and source["tensor"]["name"] == weight
            for source in node["sources"]) for node in group["source_nodes"])]
    if len(matches) != 1:
        raise ValueError("source layer has no unique physical attention norm boundary: " + weight)
    return matches[0]


def audit_case(case, inputs, structures):
    slug = case["case_id"]
    path = inputs / f"scenario_{slug}_graph_off.json"
    scenario = scenario_from_dict(json.loads(path.read_text(encoding="utf-8")))
    capture = structures / slug / "capture.jsonl"
    layers = p._execution_layers(scenario)
    record = {"case_id": slug, "preset_id": case["preset_id"], "scenario": str(path.resolve()),
              "capture": str(capture.resolve()), "regions": []}
    if any(layer.sequence_mixer != "full_attention" for layer in layers):
        record.update(status="unsupported", reason="Hybrid recurrent/linear-attention source operators and state-copy ownership are not adapted; no dense surrogate was used.")
        return record
    wanted = {"measured:0:prefill": "prefill", "measured:0:decode:1": "decode"}
    selected = {}
    for label, groups in iter_source_dispatches(capture):
        if label in wanted:
            selected[wanted[label]] = groups
        if len(selected) == len(wanted):
            break
    if len(selected) != len(wanted):
        raise ValueError("source capture lacks both measured prefill and decode calls")
    for phase, groups in selected.items():
        for region in ("first_layer", "output_tail"):
            row = {"phase": phase, "region": region}
            try:
                layer, tasks = compile_region(scenario, phase, region)
                index = int(layer.layer_id.removeprefix("layer-"))
                start = layer_start(groups, index)
                end = layer_start(groups, index + 1) if region == "first_layer" else len(groups)
                source = groups[start:end]
                mapping = match_cuda_dispatch_tasks(tasks, source)
                owner = {task_id: group["dispatch_index"] for group in mapping for task_id in group["launch_task_ids"]}
                modeled_order = [owner[task.task_id] for task in tasks if task.task_id in owner]
                row.update(status="matched", source_dispatch_count=len(source),
                    source_cuda_node_count=sum(group["node_count"] for group in source),
                    source_multi_node_dispatches=sum(group["node_count"] > 1 for group in source),
                    modeled_owned_event_count=len(owner),
                    modeled_order_backward_transitions=sum(b < a for a, b in zip(modeled_order, modeled_order[1:])),
                    mapping=mapping, scheduling_rewritten=False, physical_costs_modified=False)
            except ValueError as error:
                row.update(status="unmatched", reason=str(error), scheduling_rewritten=False, physical_costs_modified=False)
            record["regions"].append(row)
    record["status"] = "matched" if all(row["status"] == "matched" for row in record["regions"]) else "unmatched"
    return record


def audit(inputs, structures, output):
    cases = json.loads((inputs / "cases.json").read_text(encoding="utf-8"))["cases"]
    result = {"schema": "heterollm.cuda-source-dispatch-coverage/v1",
        "scope": "Isolated physical lowering of first dense layer and final layer/output head, prefill 512 and decode after 512 KV tokens; not a frontend simulation or native accuracy result.",
        "target_llm_latency_used": False, "gpu_execution_performed": False,
        "device_dispatch_cost_deployed": False,
        "source_structure_validation": source_structure_validation(structures, cases), "cases": []}
    for case in cases:
        row = audit_case(case, inputs, structures)
        result["cases"].append(row)
        print(case["case_id"], row["status"], flush=True)
    rows = [row for case in result["cases"] for row in case["regions"]]
    result["summary"] = {"model_count": len(cases),
        "matched_models": sum(case["status"] == "matched" for case in result["cases"]),
        "unsupported_models": sum(case["status"] == "unsupported" for case in result["cases"]),
        "audited_regions": len(rows), "matched_regions": sum(row["status"] == "matched" for row in rows)}
    result["remaining_requirements"] = [
        "Independently qualified ordinary/Graph device-dispatch measurements that separate GPU body, host queue supply, driver batching and instrumentation overhead; current synthetic/CUPTI observations do not qualify a scalar dispatch cost.",
        "Source-correct event scheduling: ownership alone does not change the current modeled operation order, host submission overlap, CUDA stream order or programmatic dependency launch milestones.",
        "Internal body stages for source dispatches emitting multiple CUDA nodes, including conversion/cuBLAS/copy work; do not divide a modeled body or DRAM traffic proportionally by node count.",
        "Fusion internal buffers and resource lifetimes: grouping existing physical tasks does not prove native register/shared-memory reuse or remove intermediate DRAM accesses.",
        "Hybrid Qwen3.8-27B recurrent/linear-attention and state-copy semantic ownership adaptation.",
        "Full serving/cohort integration and frontend plus held-out native latency validation after scheduling/cost changes; these ownership checks do not establish prediction accuracy.",
    ]
    output.mkdir(parents=True, exist_ok=True)
    target = output / "source_dispatch_coverage.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "source_dispatch_coverage.html").write_text(render_html(result), encoding="utf-8")
    return result, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("docs/gguf_preset_native_validation_2026-10-08"))
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/cuda_graph_dispatch_validation_2026-10-08"))
    args = parser.parse_args()
    _, target = audit(args.inputs, args.structures, args.output)
    print(target)
