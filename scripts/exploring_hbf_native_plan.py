"""Author the native-backend comparison inputs; never execute a simulation.

The exporter never runs experiments or invents metrics. The separate native
runner executes the phase barriers after integration validation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from exploring_hbf_native_models import build_native_paper_model, native_paper_model_catalog
from exploring_hbf_paper_config import (
    build_paper_hbm_config, build_paper_nand_config, default_research_assumptions,
    hardware_by_id, workload_catalog,
)

DEFAULT_DIRECTORY = ROOT / "artifacts/exploring_hbf_native_20261019"
HARDWARE_ORDER = ("HBM4", "CONV", "HBF", "CONV+", "HBF+")
NATIVE_CHAIN = (
    "reporting.run_scenario", "serving.simulate_online",
    "serving.TopologyAwareBatchCostProvider", "planner native task DAG",
    "UnifiedEventKernel", "PhysicalRuntimeContext / DramCore / NandCore",
)
RETIRED_EXECUTORS = (
    "paper_inference.PaperInferenceEngine", "paper_memory.PaperMemorySystem",
    "paper_pd.PaperPDImporter", "paper_serving_control.run_serving",
)

# These extensions belong to the existing backend, not alternate services.
NATIVE_EXTENSIONS = {
    "external_prefill_admission": {
        "targets": ["src/heterollm_sim/serving.py", "src/heterollm_sim/kv_transfer.py",
                    "src/heterollm_sim/kv_pool.py"],
        "reuse": "PhysicalKVTransferExecutor.prepare_async + DynamicKVPool prepare/complete + OnlineRuntime admission",
        "implementation": "原生外部KV入站、有限HOST源缓冲、物理导入完成后准入；最后一个输入Token实际生成首输出。",
        "acceptance": "同一物理上下文付源读、HOST 写/读、链路和目标写费用；传输完成才提交 KV；取消回滚；不得直接设置 KV 长度。",
    },
    "decode_measurement_boundary": {
        "targets": ["src/heterollm_sim/serving.py", "src/heterollm_sim/reporting.py"],
        "reuse": "原生 serving events、统计窗口和 OnlineRuntime cohort 完成边界",
        "implementation": "OnlineRuntime按cohort边界预热、测量和停止；统计窗为(start,end]，完整fresh请求单独计算TPOT。",
        "acceptance": "预热、测量和最后 drain 分离；等待队列存在时不得用 run_until_decode_steps 的全员 Prefill 完成条件冒充稳态窗口。",
    },
    "logic_die_sram": {
        "targets": ["src/heterollm_sim/memory_types.py", "src/heterollm_sim/physical_contract.py",
                    "src/heterollm_sim/data_motion.py", "src/heterollm_sim/workspace_memory.py"],
        "reuse": "现有物理服务派发、ResourceTimeline、allocator 与 workspace lifetime/capacity",
        "implementation": "原生SramConfig/SramCore，有限端口、带宽、时延及队列，复用allocator和workspace。",
        "acceptance": "每 GPU 8×40MB 的有限激活/临时区，显式端口/时延假设；不得伪装成 HBM、DRAM 或 SRAM-CIM。",
    },
    "maverick_architecture": {
        "targets": ["src/heterollm_sim/ir.py", "src/heterollm_sim/planner.py",
                    "src/heterollm_sim/serving.py", "src/heterollm_sim/kv_pool.py"],
        "reuse": "原生 GQA、MoE、专家通信、逐层图、KV 页所有权与回收",
        "implementation": "原生固定块环形KV与全注意力交错、跨块访问、sigmoid及专家输入加权、NoPE和可选温度缩放。",
        "acceptance": "原生图到任务和物理 KV 访问范围一致；不能拿全注意力、滑窗、softmax 输出加权替代。",
    },
    "paper_gpu_profile": {
        "targets": ["src/heterollm_sim/component_presets.py", "src/heterollm_sim/cost_models.py"],
        "reuse": "现有结构化 GPUProfile、TensorCoreProfile、cache/kernel/occupancy 模型",
        "implementation": "原生结构化GPUProfile，公开计算峰值与SM/缓存/kernel研究结构分别标注。",
        "acceptance": "公开 BF16 4PFLOPS、FP32 130TFLOPS 与本地补充结构分别标注；不能把 B200 改名 Rubin，不能用论文柱高调参。",
    },
}


def _case(model: str, hardware: str, workload) -> dict:
    required = ["paper_gpu_profile", "external_prefill_admission", "decode_measurement_boundary"]
    if hardware.endswith("+"):
        required.append("logic_die_sram")
    if model == "llama4_maverick":
        required.append("maverick_architecture")
    return {
        "case_id": f"native__{model}__{hardware.replace('+', '_plus')}__{workload.workload_id}__G8",
        "model": model, "hardware": hardware, "workload": workload.workload_id,
        "input_tokens": workload.input_tokens, "output_tokens": workload.output_tokens,
        "gpu_count": 8, "weight_dtype": "bf16", "kv_dtype": "bf16",
        "status": "prepared", "required_native_extensions": required,
        "execution_enabled": True,
        "metrics": None,
    }


def build_native_plan() -> dict:
    """A separate thirty-case plan, not entries in the cancelled B200 queue."""
    models = native_paper_model_catalog()
    return {
        "schema": "exploring-hbf-native-plan-v1", "backend": "native_serving_physical",
        "native_execution_chain": list(NATIVE_CHAIN),
        "retired_executors": list(RETIRED_EXECUTORS),
        "status": "authored_inputs", "formal_case_count": 30,
        "runtime_status_file": "status.json", "automatic_execution": False,
        "models": models, "workloads": [asdict(item) for item in workload_catalog()],
        "required_native_extensions": NATIVE_EXTENSIONS,
        "cases": [_case(model, hardware, workload)
                  for model in models for hardware in HARDWARE_ORDER for workload in workload_catalog()],
        "phases": [
            {"id": "native_pilot", "purpose": "短请求原生执行贯通；不作论文吞吐或稳态结果",
             "entry_point": "scripts/exploring_hbf_native_pilot.py",
             "cases": ["HBM continuous multi-wave", "HBF KV continuous multi-wave", "HBM TP2",
                       "HBF weights and KV / HBM intermediates continuous multi-wave"]},
            {"id": "dense_native", "purpose": "先跑 405B 的 HBM4/CONV/HBF 短中长 9 点",
             "case_count": 9, "start_when": ["paper_gpu_profile", "external_prefill_admission", "decode_measurement_boundary"]},
            {"id": "dense_plus", "purpose": "原生普通 SRAM 完成后追加 405B 的两个 plus 方案 6 点",
             "case_count": 6, "start_when": ["paper_gpu_profile", "external_prefill_admission", "decode_measurement_boundary", "logic_die_sram"]},
            {"id": "maverick_native", "purpose": "原生模型语义补齐后追加 Maverick 五方案 15 点",
             "case_count": 15, "start_when": ["paper_gpu_profile", "external_prefill_admission", "decode_measurement_boundary", "maverick_architecture", "logic_die_sram"]},
        ],
        "deployment_search": {
            "implementation": "脚本枚举原生 ParallelSpec(TP, PP, EP) 与独立 DP 副本数的布局，每个候选（布局＋每副本批量）调用同一 run_scenario，在独立子进程中运行；不计算替代算子时延。",
            "layout": "DP×TP×PP×EP 恰好占满每配置的 GPU 数（8）。只仿真一个副本（TP×PP×EP 张 GPU），批量是每副本批量，由 EP 个注意力数据并行组平分（每组 B/EP 个请求）。",
            "first_control": {"tp_degree": 8, "pp_degree": 1, "ep_degree": 1, "dp_replicas": 1},
            "dense_followups": [{"tp_degree": 4, "pp_degree": 2, "ep_degree": 1, "dp_replicas": 1},
                                {"tp_degree": 2, "pp_degree": 4, "ep_degree": 1, "dp_replicas": 1}],
            "ep_scope": "仅 Maverick（MoE）可取 EP>1：每个 EP 组独占 B/EP 个请求及其 KV，注意力、稠密 MLP、路由和共享专家在本组 TP 卡上执行（稠密权重每组一份）；专家由各组共享，经原生 all_to_all 分发/合并，每层在跨组屏障处同步。要求 B 与每个 MoE 层的专家数都能被 EP 整除。",
            "dp_scope": "DP 副本是互相独立的拷贝，不共享 KV 或批次；只仿真一个副本，单GPU吞吐＝副本吞吐÷(TP×PP×EP)，集群吞吐＝单GPU吞吐×GPU总数。每个副本有自己的 PD 入站，未建模副本之间共享网卡或 PD 源的争用，也不添加不存在的 ParallelSpec 字段。",
            "pp_scope": "原生没有GPipe微批；稳态快速模型不组合流水线各级之间的发送，PP>1布局不进入原生搜索。",
            "selection": "每配置在已实际执行的候选中选择通过 100ms 请求均值 TPOT 约束的最高单GPU吞吐；容量不足的布局或批量记为容量拒绝。不是论文最大批量或全局最优搜索。",
            "batch_search": ("搜索驱动（run_exploring_hbf_native.py --search，总是fast）：旁路模型（exploring_hbf_search）对每个配置筛选全部合法布局并给出批量区间；"
                             "取旁路排名前2且原生可运行的布局（无微批、PP=1、通过原生布局校验），首点取旁路批量的一半且不超过512，"
                             "第二点按原生/旁路TPOT比例缩放旁路TPOT直线推到100ms，未夹住100ms时再定第三点；"
                             "批量点不超过旁路KV容量（直到原生点达到它，之后以原生容量拒绝为界；KV容量拒绝报告原生放得下的最大批量，下一点直接测它）、壁钟（已测壁钟不除以并行数，"
                             "线性投影到每worker 1800s的90%）和峰值内存（线性投影到物理内存减保留量）；过大的点减半或二分。"
                             "k个候选并行时单候选壁钟上限为k×1800s，按预测峰值内存自动增开并行。夹住100ms的相邻两点线性插值TPOT上界，"
                             "在预测单GPU吞吐最高的布局上做一次原生快速验收；验收未过退回已测通过点；够不到100ms的布局停在已测最大点并标注受限，不外推。"
                             "旁路数值只用于放置批量点，最终数字只取原生候选。固定批量管理器仅用于对照点。"),
            "control_batch_candidates": [1, 8],
        },
        "measurement": {
            "decode_nodes_only": True,
            "slo_tpot_ns": 100_000_000, "slo_interpretation": "mean_request_tpot",
            "request_tpot_definition": "测量窗内完整可见输出请求的 (最后Token-首Token)/(输出Token数-1)，再按请求等权算术平均；少于2个输出未定义。",
            "non_slo_diagnostics": "最大请求TPOT、TBT分布、系统迭代仅单独报告，不当作同一个100ms均值门槛。",
            "slo_note": "论文未公开精确聚合口径；同时报告平均/最大 TBT、平均/最大请求 TPOT 和系统迭代，不能混为同一个值。",
            "decode_ingress_ttft": "请求实付KV导入开始至本节点首个可见输出；包含PD导入与准入等待，不含Prefill计算。",
            "ttft": "TTFT = Prefill通路 + 入口TTFT。Prefill副本为HBM4、与Decode副本相同的TP/PP/EP布局，请求到达即单独一次不分块前向计算前P-1个提示Token，至最后一层完成（开始导出KV）；快速合约为分层采样区间，详细合约为一次覆盖全部层的原生运行。",
            "output_count_basis": "外部只提供输入prompt的KV；本节点真实生成声明的全部输出，首输出仍付实际Decode成本；TPOT分母为D-1。",
            "window": "原生闭环预热一个输出生命周期、测量两个生命周期，排除最终 drain；实际事件定义边界。",
            "report": ["tokens/s/GPU", "cluster tokens/s", "TTFT with Prefill, its Prefill pass and decode-ingress parts",
                       "request TPOT", "TBT distribution",
                       "system iteration duration", "maximum validated batch", "weight/KV/workspace capacity",
                       "PD bytes/time and waits", "DRAM/NAND read/program/erase traffic", "wall time"],
            "normalization": "同模型、负载、GPU 数、精度及 SLO 口径，除以各自原生 HBM4 基线。",
            "cross_paper_difference": "原生结果与论文图标注的差异是跨实现对照，不是快速模式的预测误差。",
        },
        "execution_policy": {
            "max_concurrent_configurations": 2,
            "existing_scans_restarted": False, "cancelled_5k_10k_restored": False,
            "historical_results_reused_as_native": False,
            "full_detailed_references_rerun": False,
            "pilot_fidelity": "bounded native physical execution; no forecast",
            "formal_fidelity": "原生物理预热和测量窗，无预测尾段；现有单GPU/dense fast guard不扩为多GPU/MoE，10%约束不变。无预测不代表对真实硬件或论文误差小于10%。",
            "unsupported_behavior": "标记 pending/unsupported；不转入 paper_*，不免费写 KV、不给 plus 增配 HBM。",
        },
        "sources": ["https://arxiv.org/abs/2608.13868",
                    "https://people.inf.ethz.ch/omutlu/pub/HBF-for-LLM-cal26-talk.pdf"],
    }


def export_native_plan(directory: Path) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    plan = build_native_plan()
    from exploring_hbf_native_config import sram_reference
    inputs = {"schema": "native-paper-inputs-v1", "scope": "原生模型和物理端点输入；完整场景由exploring_hbf_native_config构造",
              "models": {identity: asdict(build_native_paper_model(identity))
                         for identity in native_paper_model_catalog()}, "hardware": {}}
    for identity in HARDWARE_ORDER:
        hardware = hardware_by_id(identity)
        nand = (build_paper_nand_config(identity, assumptions=default_research_assumptions(identity))
                if hardware.hbf_stacks else None)
        hbm = build_paper_hbm_config(identity)
        inputs["hardware"][identity] = {
            "per_gpu_table": hardware.as_dict(),
            "native_hbm_config": asdict(hbm) if hbm else None,
            "native_nand_config": asdict(nand) if nand else None,
            "native_intermediate_sram_config": asdict(sram_reference()) if identity.endswith("+") else None,
            "logic_sram_status": "native_pooled_sram" if identity.endswith("+") else "not_required",
        }
    for name, payload in (("plan.json", plan), ("native_inputs.json", inputs)):
        (directory / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"directory": str(directory.resolve()), "formal_cases_prepared": len(plan["cases"]),
            "experiments_started_by_export": 0, "backend": plan["backend"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    print(json.dumps(export_native_plan(args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
