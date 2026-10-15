"""Plot recorded full-serving observations without rerunning the simulator.

Example (project environment, matplotlib 3.10.x installed):
  .venv\\Scripts\\python.exe scripts/plot_hbf_hbm_results.py \
      --input artifacts/hbf_hbm/results.json --output artifacts/hbf_hbm/figures \
      --update-report

Missing experiment groups produce labelled placeholders and a nonzero exit
unless --allow-partial is supplied for a smoke run. No microbenchmark row is
accepted as LLM throughput, and no SLO observations are joined into a boundary.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
A, B = "A_hbm_only", "B_all_kv_hbf"
COLORS = {A: "#2364AA", B: "#E07632"}
MARKERS = {A: "o", B: "s"}
LABELS = {A: "A · HBM KV", B: "B · HBF KV"}
ZH = False


def tr(chinese, english):
    return chinese if ZH else english


def setup_font():
    global ZH
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            family = font_manager.FontProperties(fname=str(candidate)).get_name()
            plt.rcParams["font.family"] = [family, "DejaVu Sans"]
            ZH = True
            break
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#ADB7C4", "axes.labelcolor": "#233348",
        "text.color": "#233348", "xtick.color": "#44546A", "ytick.color": "#44546A",
        "axes.unicode_minus": False, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "path", "lines.linewidth": 1.8})
    return {"language": "zh-CN" if ZH else "en", "font_family": plt.rcParams["font.family"]}


def valid_number(value, *, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def full_row(row):
    return row.get("mode") == "full_operator_serving" and valid_number(row.get("node_tps"))


def metric(row, name, quantile=None):
    value = row.get(name)
    if quantile is not None:
        value = value.get(quantile) if isinstance(value, dict) else None
    return float(value) if valid_number(value) else math.nan


def context_extrapolation(row):
    return row["model_structure"]["context_extrapolation"]


def nullable_numbers(value):
    """Keep absent numeric observations JSON-null rather than nonstandard NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: nullable_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [nullable_numbers(item) for item in value]
    return value


def style(ax, *, xgrid=False):
    ax.set_axisbelow(True)
    ax.grid(axis="both" if xgrid else "y", color="#E1E7EE", linewidth=.7)


def missing(ax, message=None):
    ax.set_axis_off()
    ax.text(.5, .5, message or tr("输入结果尚无此组完整服务数据", "This full-serving dataset is absent"),
            ha="center", va="center", wrap=True, transform=ax.transAxes, color="#6C7785")


def save(fig, output, stem, *, title, caption, rows, status="complete", observations=None):
    files = {}
    for extension in ("png", "svg"):
        path = output / f"{stem}.{extension}"
        fig.savefig(path, dpi=190, bbox_inches="tight", pad_inches=.2)
        files[extension] = str(path.resolve())
    plt.close(fig)
    return {"id": stem, "title_zh": title, "caption_zh": caption, "status": status,
            "files": files, "relative_files": {k: Path(v).name for k, v in files.items()},
            "source_row_ids": [row["id"] for row in rows], "source_row_count": len(rows),
            "observations": observations or [], "row_scope": "full_operator_serving only"}


def small_batch_figure(groups, output):
    rows = [r for r in groups.get("02_fixed_workload", {}).get("rows", ())
            if full_row(r) and str(r.get("id", "")).startswith("fixed-")
            and r.get("prompt_tokens") == 2048 and r.get("placement") in (A, B)
            and r.get("requested_concurrency") in (1, 4, 16, 64)
            and valid_number(r.get("max_observed_batch_requests"), positive=True)]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.7))
    fig.subplots_adjust(top=.79, bottom=.23, left=.085, right=.98, wspace=.25)
    fig.suptitle(tr("小模型：实际 batch 下的输出吞吐与 SLO goodput",
                    "Small model: output throughput and SLO goodput by actual batch"), y=.98, fontsize=15)
    outputs = sorted({r.get("output_tokens_per_request") for r in rows})
    output_label = "/".join(str(value) for value in outputs) or "?"
    fig.text(.5, .895, tr(f"完整 LLM 服务 · 2K 提示词 · 每请求输出 {output_label} tokens · 单 GPU",
                          f"Full LLM serving · 2K prompt · output tokens/request: {output_label} · one GPU"),
             ha="center", fontsize=10)
    for ax, key, heading in zip(axes, ("node_tps", "goodput_tps"),
            (tr("全部可见输出 tokens/s", "All visible output tokens/s"),
             tr("满足 SLO 的输出 tokens/s", "SLO-qualified output tokens/s"))):
        if not rows:
            missing(ax); continue
        for architecture in (A, B):
            series = sorted((r for r in rows if r["placement"] == architecture),
                            key=lambda r: r["max_observed_batch_requests"])
            ax.plot([r["max_observed_batch_requests"] for r in series],
                    [metric(r, key) for r in series], marker=MARKERS[architecture],
                    color=COLORS[architecture], label=LABELS[architecture], markersize=7)
        ax.set_xscale("log", base=4)
        ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
        ax.set_xlabel(tr("实际最大 batch（请求数）", "Observed maximum batch (requests)"))
        ax.set_title(heading, pad=13)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:,.0f}"))
        style(ax); ax.legend(frameon=False, fontsize=9)
    caption = "只取固定硬件的小模型 2K 完整服务点，排除带宽扫描及微基准。横轴取实际最大 batch；TPS/goodput 均按包含初始化等待的请求观测窗口计算。短输出与有限轨迹不代表稳态服务能力。"
    fig.text(.06, .075, tr("TPS 与 goodput 使用同一请求窗口，包含初始化等待；goodput 仅计入全部满足请求 SLO 的输出。\n这些是有限短输出服务观测；连线仅帮助读数，不表示未测试 batch 的性能。",
        "TPS and goodput use the same request window including initialization; only SLO-qualified requests enter goodput.\nFinite short-output observations; lines are visual guides, not predictions for untested batch sizes."), fontsize=9)
    complete = all(any(r["placement"] == arch and r["requested_concurrency"] == n
                       and r["max_observed_batch_requests"] == n for r in rows)
                   for arch in (A, B) for n in (1, 4, 16, 64))
    observations = [{"id": r["id"], "requested_batch": r["requested_concurrency"],
        "actual_batch": r["max_observed_batch_requests"], "node_tps": r["node_tps"],
        "goodput_tps": r.get("goodput_tps")} for r in rows]
    return save(fig, output, "01_small_actual_batch", title="小模型实际 batch：TPS 与 goodput",
                caption=caption, rows=rows, status="complete" if complete else "partial_data", observations=observations)


def named_context_figure(groups, output):
    rows = [r for r in groups.get("09_serving_study", {}).get("rows", ())
            if full_row(r) and str(r.get("id", "")).startswith(("model/","model-capacity/"))
            and r.get("inventory_role")!='capacity_control_512gib' and r.get("placement") in (A, B)]
    model_order = {"llama3_1-8b": 0, "llama3_1-70b": 1}
    model_ids = sorted({r["parameters"]["model_id"] for r in rows},
                       key=lambda name: (model_order.get(name, 2), name))
    nrows = max(1, len(model_ids))
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 3.65 * nrows + 2.0), squeeze=False)
    fig.subplots_adjust(top=.86, bottom=.18, left=.09, right=.98, hspace=.48, wspace=.27)
    fig.suptitle(tr("命名模型：上下文长度与完整服务输出吞吐", "Named models: context length and full-serving output throughput"),
                 y=.98, fontsize=15)
    observations = []
    if not model_ids:
        for ax in axes.flat: missing(ax)
    for index, model_id in enumerate(model_ids):
        selected = [r for r in rows if r["parameters"]["model_id"] == model_id]
        title = selected[0].get("model", model_id)
        left, right = axes[index]
        for architecture in (A, B):
            series = sorted((r for r in selected if r["placement"] == architecture), key=lambda r: r["prompt_tokens"])
            left.plot([r["prompt_tokens"] for r in series], [r["node_tps"] for r in series],
                      color=COLORS[architecture], linestyle="--", alpha=.75, linewidth=1.)
            for row in series:
                extrapolated = bool(context_extrapolation(row))
                left.scatter(row["prompt_tokens"], row["node_tps"], s=57,
                    marker="D" if extrapolated else MARKERS[architecture],
                    facecolors="white" if extrapolated else COLORS[architecture],
                    edgecolors=COLORS[architecture], linewidths=1.7, zorder=3)
            left.plot([], [], marker=MARKERS[architecture], color=COLORS[architecture], label=LABELS[architecture])
        contexts = sorted({r["prompt_tokens"] for r in selected})
        extrapolated_contexts = {r["prompt_tokens"] for r in selected if context_extrapolation(r)}
        ratios = []
        for context in contexts:
            pair = {r["placement"]: r for r in selected if r["prompt_tokens"] == context}
            if set(pair) != {A, B} or not valid_number(pair[A]["node_tps"], positive=True):
                continue
            if pair[A].get("output_tokens_per_request") != pair[B].get("output_tokens_per_request"):
                continue
            if pair[A].get("fixed_inventory") != pair[B].get("fixed_inventory"):
                continue
            ratio = pair[B]["node_tps"] / pair[A]["node_tps"]
            ratios.append((context, ratio))
            observations.append({"model_id": model_id, "context_tokens": context, "B_over_A_tps": ratio,
                "output_tokens_per_request": pair[A].get("output_tokens_per_request"),
                "context_extrapolation": context_extrapolation(pair[A]),
                "paired_row_ids": [pair[A]["id"], pair[B]["id"]]})
        if ratios:
            right.plot([x for x, _y in ratios], [y for _x, y in ratios], "o--", color="#6D52A4", linewidth=1.)
            for x, y in ratios:
                right.annotate(f"{y:.5f}×", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
        else:
            missing(right, tr("缺少同输出长度、同硬件的 A/B 配对", "No matched A/B pairs with equal output length and inventory"))
        for ax in (left, right):
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(contexts, [(f"{x // 1024}K" if x < 1048576 else "1M")
                                    + ("*" if x in extrapolated_contexts else "") for x in contexts])
            ax.set_xlabel(tr("提示词 / 上下文 tokens（对数坐标）", "Prompt/context tokens (log scale)"))
            ax.margins(x=.18, y=.32); style(ax, xgrid=True)
        left.set_title(title + " · TPS", pad=10); left.set_ylabel("tokens/s")
        left.legend(frameon=False, fontsize=9)
        right.axhline(1., color="#667281", linestyle=":", linewidth=1.)
        right.set_title(title + " · B/A", pad=10); right.set_ylabel(tr("输出 TPS 比值（对数坐标）", "Output TPS ratio (log scale)"))
    caption = "仅使用完整命名模型 A/B 配对；不是内存微基准。2K 点为 32 输出，长上下文点仅 4 输出，不能把折线斜率解释为纯上下文效应。70B/1M共同扩为1TiB HBM，原512GiB容量对照不参与配对。128K加输出与1M超出原生上下文，采用星号及空心菱形标记。比值仅匹配相同输出长度和硬件清单。"
    fig.text(.055, .045, tr("2K 点：32 输出；128K / 1M 点：仅 4 输出，吞吐受 prefill 主导。折线仅连接已测点。\n* 空心菱形：提示词加输出超出原生上限；128K+4 已略超上限，1M 属大幅架构外推。\n70B/1M的A/B共同扩为1TiB HBM；原512GiB容量对照不参与配对。",
        "2K: 32 output tokens; 128K/1M: only 4 output tokens, dominated by prefill. Lines join observations only.\n* Hollow diamonds: prompt plus output exceeds native context; 128K+4 slightly exceeds it, while 1M is a large extrapolation.\n70B/1M uses 1 TiB HBM for BOTH A/B; original 512 GiB capacity controls excluded."), fontsize=9)
    complete = all(any(o["model_id"] == model and o["context_tokens"] == context
                      for o in observations) for model in ("llama3_1-8b", "llama3_1-70b")
                   for context in (2048, 131072, 1048576))
    return save(fig, output, "02_named_context", title="命名模型上下文：TPS 与 B/A 比率", caption=caption,
                rows=rows, status="complete" if complete else "partial_data", observations=observations)


def slo_figure(groups, output):
    rows = [r for r in groups.get("09_serving_study", {}).get("rows", ())
            if full_row(r) and str(r.get("id", "")).startswith(("slo/", "slo-refine/", "slo-batch/"))
            and r.get("placement") in (A, B)]
    fig, axes = plt.subplots(3, 2, figsize=(13.4, 12.5))
    fig.subplots_adjust(top=.87, bottom=.15, left=.09, right=.98, hspace=.44, wspace=.26)
    fig.suptitle(tr("SLO：有限实验的可行性与延迟原始观测点", "SLO feasibility and latency: finite observed points"), y=.975, fontsize=15)
    fig.text(.5, .932, tr("散点之间不连线；未将有限搜索结果解释为全局稳态边界", "Unconnected observations; no global or steady-state boundary is inferred"), ha="center")
    observations = []
    for column, is_batch in enumerate((False, True)):
        selected = [r for r in rows if str(r["id"]).startswith("slo-batch/") == is_batch]
        if not selected:
            for ax in axes[:, column]: missing(ax)
            continue
        for architecture in (A, B):
            series = [r for r in selected if r["placement"] == architecture]
            xs = [(r.get("max_observed_batch_requests") if is_batch else r["parameters"]["interval_ns"] / 1000.) for r in series]
            feasible = [r.get("qualified_requests", 0) / r["request_count"] * 100. if r.get("request_count") else math.nan for r in series]
            for ax, values in zip(axes[:, column], (feasible,
                    [metric(r, "ttft_ns", "p99") / 1e6 for r in series],
                    [metric(r, "tbt_ns", "p99") / 1e6 for r in series])):
                ax.scatter(xs, values, marker=MARKERS[architecture], s=70 if architecture == B else 45,
                           facecolors="none" if architecture == B else COLORS[architecture],
                           edgecolors=COLORS[architecture] if architecture == B else "white",
                           linewidths=1.4 if architecture == B else .7, label=LABELS[architecture], zorder=3)
            for x, row in zip(xs, series):
                observations.append({"id": row["id"], "axis": "actual_batch" if is_batch else "arrival_interval_us",
                    "x": x, "requested_batch": row.get("requested_concurrency"),
                    "actual_batch": row.get("max_observed_batch_requests"),
                    "qualified_requests": row.get("qualified_requests"), "request_count": row.get("request_count"),
                    "ttft_p99_ns": metric(row, "ttft_ns", "p99"), "tbt_p99_ns": metric(row, "tbt_ns", "p99")})
        for i, ax in enumerate(axes[:, column]):
            style(ax)
            if is_batch:
                ax.set_xscale("log", base=4); ax.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
            else:
                ax.set_xscale("symlog", linthresh=10.)
                # Refinement observations remain in the scatter; sparse major
                # labels avoid suggesting a separate category for every step.
                ax.set_xticks([0., 10., 100., 1000.], ["0", "10", "100", "1000"])
            ax.set_xlabel(tr("实际最大 batch（请求数）", "Observed maximum batch (requests)") if is_batch
                          else tr("请求到达间隔（µs；0=同时到达）", "Arrival interval (µs; 0 = simultaneous)"))
            if i:
                metric_name = "ttft_ns" if i == 1 else "tbt_ns"
                limits = {r.get("slo", {}).get(metric_name) for r in selected}
                limits = sorted(x for x in limits if valid_number(x, positive=True))
                for limit in limits:
                    ax.axhline(limit / 1e6, color="#8B4A54", linestyle=":", linewidth=1.3,
                               label=f"SLO = {limit / 1e6:g} ms")
                ax.set_yscale("log")
                ax.set_ylabel("TTFT p99 (ms)" if i == 1 else "TBT p99 (ms)")
            else:
                ax.axhline(100., color="#8391A1", linestyle=":", linewidth=1.)
                ax.set_ylim(-5, 110); ax.set_ylabel(tr("满足全部请求 SLO 的比例（%）", "SLO-qualified requests (%)"))
                ax.set_title(tr("同时到达 batch 扫描", "Simultaneous-batch observations") if is_batch
                             else tr("32 请求有限到达轨迹", "Finite 32-request arrival traces"), pad=13)
            ax.legend(frameon=False, fontsize=8, loc="best")
    caption = "到达间隔和实际 batch 均保留原始可行/不可行点，不拟合、不插值稳态边界。到达间隔轴只标注 0/10/100/1000 微秒主刻度，所有细化散点仍保留，细化值见报告和 manifest 原始记录。p99 是各有限请求/TBT 样本的经验分位数；请求样本量随 batch 改变。水平 SLO 线仅为配置阈值，不能解释为估计服务容量。"
    fig.text(.055, .059, tr("所有点来自完整算子服务；水平虚线仅表示配置的 SLO 阈值，不是搜索所得边界。\n细化散点全部保留，详细到达间隔见报告；p99 为有限样本经验分位数。\n到达率可持续性、全局单调性和未测试 batch 均未得到证明。",
        "All points are full-operator observations. Horizontal dotted lines are configured SLO thresholds only.\nAll refinement points remain visible; detailed intervals are in the report. p99 is an empirical finite-sample quantile.\nSustainable arrival rate, global monotonicity and untested batches remain unproven."), fontsize=9)
    arrival_complete = all(any(str(r["id"]).startswith("slo/") and r["placement"] == arch
        and r.get("parameters", {}).get("interval_ns") == interval for r in rows)
        for arch in (A, B) for interval in (0., 10_000., 100_000., 1_000_000.))
    batch_complete = all(any(str(r["id"]).startswith("slo-batch/") and r["placement"] == arch
        and r.get("parameters", {}).get("batch") == batch for r in rows)
        for arch in (A, B) for batch in (1, 4, 16, 64))
    complete = arrival_complete and batch_complete
    return save(fig, output, "03_slo_observations", title="SLO 原始点：到达间隔、实际 batch 与延迟", caption=caption,
                rows=rows, status="complete" if complete else "partial_data", observations=observations)


def prefetch_figure(groups, output):
    rows = [r for r in groups.get("09_serving_study", {}).get("rows", ())
            if full_row(r) and str(r.get("id", "")).startswith(("prefetch/", "prefetch-stress/"))]
    def read_latency(row):
        return row.get("parameters", {}).get("hbf_page_read_ns") or 4000.

    # Never compare stress points to another read-latency case's baseline.
    cases = [(False, 4000.)] + [(True, latency) for latency in sorted(
        {25_000., 75_000.} | {read_latency(r) for r in rows if r["id"].startswith("prefetch-stress/")})]
    fig, axes = plt.subplots(2, len(cases), figsize=(6.1 * len(cases), 8.7), squeeze=False)
    fig.subplots_adjust(top=.84, bottom=.2, left=.06, right=.98, hspace=.42, wspace=.3)
    fig.suptitle(tr("异步 KV 预取：实际 GPU 算子重叠与完整服务吞吐", "Async KV prefetch: realized GPU-operator overlap and serving throughput"),
                 y=.975, fontsize=15)
    observations = []
    for column, (stress, read_ns) in enumerate(cases):
        selected = sorted((r for r in rows if str(r["id"]).startswith("prefetch-stress/") == stress
                           and read_latency(r) == read_ns),
                          key=lambda r: r["parameters"]["prefetch_distance"])
        if not selected:
            for ax in axes[:, column]: missing(ax)
            continue
        distances = [r["parameters"]["prefetch_distance"] for r in selected]
        xs = list(range(len(selected)))
        axes[0, column].set_title(tr(f"{'延迟压力' if stress else '基准'} · HBF read = {read_ns / 1000:g} µs",
                                     f"{'Latency stress' if stress else 'Baseline'} · HBF read = {read_ns / 1000:g} µs"), pad=13)
        for key, label, color, marker in (("node_tps", "TPS", "#2364AA", "o"),
                                          ("goodput_tps", "Goodput", "#21887B", "s")):
            axes[0, column].plot(xs, [metric(r, key) for r in selected], color=color, marker=marker,
                                label=label, linestyle="--" if key == "goodput_tps" else "-",
                                markerfacecolor="none" if key == "goodput_tps" else color,
                                markersize=8 if key == "goodput_tps" else 5)
        baseline = next((r for r in selected if r["parameters"]["prefetch_distance"] == 0), None)
        for i, row in enumerate(selected):
            relative = row["node_tps"] / baseline["node_tps"] if baseline and baseline["node_tps"] else None
            if relative is not None:
                axes[0, column].annotate(f"{relative:.5f}×", (i, row["node_tps"]),
                                        xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
            overlap = row.get("prefetch_overlap", {})
            operator = overlap.get("operator_overlap_ns")
            decode = overlap.get("decode_operator_overlap_ns")
            observations.append({"id": row["id"], "prefetch_distance_tokens": distances[i],
                "hbf_page_read_ns": read_ns,
                "node_tps": row["node_tps"], "tps_relative_to_same_case_zero_distance": relative,
                "operator_overlap_ns": operator, "decode_operator_overlap_ns": decode,
                "premature_consumer_overlap_ns": overlap.get("premature_consumer_overlap_ns")})
        for shift, key, label, color in ((-.17, "operator_overlap_ns", tr("全部 GPU 算子阶段", "All GPU operator stages"), "#6D52A4"),
                                         (.17, "decode_operator_overlap_ns", tr("其中 decode 阶段", "Decode stages (subset)"), "#E0A143")):
            values = [metric(r.get("prefetch_overlap", {}), key) / 1000. for r in selected]
            bars = axes[1, column].bar([x + shift for x in xs], values, width=.31, color=color, label=label)
            for bar, value in zip(bars, values):
                if math.isfinite(value):
                    axes[1, column].annotate(f"{value:.2f}", (bar.get_x() + bar.get_width()/2, value),
                                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        for index, ax in enumerate(axes[:, column]):
            ax.set_xticks(xs, [str(x) for x in distances]); ax.set_xlim(-.6, len(xs)-.4)
            ax.set_xlabel(tr("prefetch distance（tokens；0=关闭）", "Prefetch distance (tokens; 0 = disabled)"))
            values = ([metric(r, key) for r in selected for key in ("node_tps", "goodput_tps")]
                      if index == 0 else [metric(r.get("prefetch_overlap", {}), key) / 1000.
                          for r in selected for key in ("operator_overlap_ns", "decode_operator_overlap_ns")])
            maximum = max((value for value in values if math.isfinite(value)), default=0.)
            ax.set_ylim(0., max(1., maximum) * 1.27)
            style(ax); ax.legend(frameon=False, fontsize=8)
        axes[0, column].set_ylabel("tokens/s")
        axes[0, column].yaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:,.0f}"))
        axes[1, column].set_ylabel(tr("与其他请求 GPU 算子阶段重叠（µs）", "Overlap with other-request GPU stages (µs)"))
    caption = "C 热冷 KV 策略，按同一场景 distance=0 对照比较 TPS。重叠取真实 GPU invocation 算子阶段与预取复制区间的交集，排除主机设置和批次空白包络；算子阶段仍可能包含内存等待，不等于 tensor-core 忙碌时间。decode 为全部算子重叠的子集，不能相加。"
    fig.text(.055, .071, tr("倍率以同列 distance=0 的实际 TPS 为基准；有重叠不自动意味着吞吐提升。\n重叠来自真实 GPU 算子阶段，可能包含内存等待；decode 是其子集，不相加，也不等同于 tensor-core 忙碌时间。",
        "Multipliers use the same-column distance=0 TPS. Overlap alone does not establish a throughput gain.\nGPU operator envelopes may include memory waits; decode is a subset, not additive or tensor-core busy time."), fontsize=9)
    complete = all(any(str(r["id"]).startswith("prefetch-stress/") == stress
                       and read_latency(r) == read_ns and r["parameters"]["prefetch_distance"] == distance
                       and valid_number(r.get("prefetch_overlap", {}).get("operator_overlap_ns")) for r in rows)
                   for stress, read_ns in cases for distance in ((0, 16) if stress else (0, 16, 32)))
    return save(fig, output, "04_prefetch_overlap", title="预取对照：GPU 算子重叠与 TPS", caption=caption,
                rows=rows, status="complete" if complete else "partial_data", observations=observations)


def system_figure(groups, output):
    all_rows=[r for r in groups.get('15_system',{}).get('rows',[]) if full_row(r)]
    rows=[r for r in all_rows if
          r['system_contract']['parameters'].get('model_id','small')=='small']
    named=[r for r in all_rows if r not in rows]
    fig,axes=plt.subplots(1,3,figsize=(18.5,5.7))
    fig.subplots_adjust(top=.79,bottom=.22,wspace=.3)
    fig.suptitle(tr('多GPU共享池：节点吞吐与每GPU效率','Shared pool: node throughput and per-GPU efficiency'),y=.97,fontsize=15)
    for architecture,color in zip(('A_hbm_only','B_all_kv_hbf','D_hbf_weights','G_cxl_dram'),('#2364AA','#E07632','#6D52A4','#21887B')):
        selected=sorted([r for r in rows if r['placement']==architecture and not r['system_contract']['rack']],key=lambda r:r['gpu_count'])
        for ax,key in zip(axes,('node_tps','per_gpu_tps')):
            if selected: ax.plot([r['gpu_count'] for r in selected],[r[key] for r in selected],marker='o',label=architecture[:1],color=color)
    for ax,key in zip(axes,('node_tps','per_gpu_tps')):
        for row in rows:
            if row['system_contract']['rack']:
                ax.scatter(row['gpu_count'],row[key],marker='x',s=65,color='#b22d43')
                ax.annotate(row['placement'][0]+' rack',(row['gpu_count'],row[key]),xytext=(-8,9),textcoords='offset points',ha='right',fontsize=8,color='#b22d43')
        ax.set_xticks([1,2,4]);ax.set_xlabel('GPU');ax.set_ylabel(key+' (token/s)');style(ax);ax.legend(frameon=False)
    for architecture,color in ((A,'#2364AA'),(B,'#E07632')):
        selected=sorted([r for r in named if r['placement']==architecture],key=lambda r:r['gpu_count'])
        if selected: axes[2].plot([r['gpu_count'] for r in selected],[r['node_tps'] for r in selected],marker='o',label=architecture[:1],color=color)
    axes[2].set_xticks([1,2,4]);axes[2].set_xlabel('GPU');axes[2].set_ylabel('node_tps (token/s)');style(axes[2])
    if named: axes[2].legend(frameon=False)
    for ax,title in zip(axes,[tr('小模型：节点TPS','Small: node TPS'),tr('小模型：每GPU TPS','Small: per-GPU TPS'),'Llama 3.1 8B · 2K / 4']): ax.set_title(title)
    fig.text(.075,.06,tr('实际张量并行任务与通信；共享内存带宽不随GPU数倍增。红叉为50GB/s机架网络。\n左/中为小模型4请求、128输入/8输出；右为8B单请求2K输入/4输出。全部含初始化等待，不能跨工作负载比较绝对TPS。',
        'Explicit TP and communication; fixed shared bandwidth. Red crosses: 50 GB/s rack fabric.\nLeft/middle: small model, 4 requests, 128/8 tokens. Right: 8B, one 2K/4-token request. All include initialization wait.'),fontsize=9)
    return save(fig,output,'05_multi_gpu',title='多GPU共享池：实际节点与每GPU吞吐',
        caption='小模型A/B/D/G与8B命名模型A/B固定共享池，显式1/2/4 GPU执行；红叉为机架网络变体。多GPU没有直接按单GPU吞吐倍乘。',rows=all_rows,status='complete' if len(rows)==14 and len(named)==6 else 'partial_data')


def internal_feedback_figure(groups, output):
    ablation=[r for r in groups.get('12_ablation_paths',{}).get('rows',[]) if full_row(r) and r['id'].startswith('ablation/')]
    agents=[r for r in groups.get('13_agent_lifecycle',{}).get('rows',[]) if r['id'] in ('agent-static-cold','agent-heuristic')]
    wear=[r for r in groups.get('14_wear',{}).get('rows',[]) if full_row(r) and r['placement']==B]
    fig,axes=plt.subplots(1,3,figsize=(17,5.6));fig.subplots_adjust(top=.79,bottom=.28,wspace=.32)
    fig.suptitle(tr('内部机制进入完整服务：消融、会话与磨损反馈','Full serving: ablation, agent sessions and wear feedback'),y=.97,fontsize=15)
    if ablation:
        xs=list(range(len(ablation)));vs=[r['relative_to_ideal_hbm']['tps_ratio'] for r in ablation]
        axes[0].bar(xs,vs,color='#2364AA');axes[0].set_xticks(xs,['HBM','HBF','Read','Write','Units','R+W','R+W+U'],rotation=35)
        axes[0].set_ylabel('TPS / ideal HBM TPS');axes[0].axhline(1,color='#666',linestyle=':',lw=1)
    if agents:
        axes[1].bar([0,1],[r['agent_lifecycle']['cold_write_bytes']/1024 for r in agents],color=['#6D52A4','#21887B'])
        axes[1].set_xticks([0,1],['Static cold','Heuristic']);axes[1].set_ylabel(tr('实际冷迁移写入（KiB）','Cold migration writes (KiB)'))
        for i,r in enumerate(agents): axes[1].text(i,r['agent_lifecycle']['cold_write_bytes']/1024+1,f"{r['node_tps']:,.0f} TPS",ha='center',fontsize=9)
    if wear:
        base=wear[0]['node_tps'];vals=[r['node_tps']/base for r in wear]
        bars=axes[2].bar(range(len(vals)),vals,color=['#8C9BAB','#6D52A4'])
        axes[2].set_xticks(range(len(vals)),['Off','Wear']);axes[2].set_ylabel('B TPS / feedback-off TPS')
        for b,v in zip(bars,vals): axes[2].text(b.get_x()+b.get_width()/2,v+.025,f'{v:.3f}',ha='center')
    for ax,title in zip(axes,[tr('完整服务逐项消融','Full-serving ablation'),tr('真实三轮KV继承','Three-turn physical KV reuse'),tr('动态NAND时延反馈','Dynamic NAND latency feedback')]):
        ax.set_title(title);style(ax);ax.set_ylim(0,max(1,ax.get_ylim()[1])*1.15)
    if ablation:
        axes[0].set_yscale('log');axes[0].set_ylim(min(vs)*.35,2.)
        axes[0].set_ylabel(tr('TPS / 理想HBM（对数轴）','TPS / ideal HBM (log scale)'))
        for i,v in enumerate(vs): axes[0].text(i,v*1.18,f'{v:.2g}',ha='center',fontsize=8)
    fig.text(.06,.065,tr('三列是各自严格配对的独立实验，不能跨列比较绝对TPS。磨损曲线是压力假设；wear为预老化P/E敏感性。\nRead=读延迟，Write=写限制，Units=有限并行；理想零延迟下Units单独加入可不改变吞吐。',
        'Each panel is a separate paired experiment. Wear curves are stress assumptions; wear is pre-aged P/E sensitivity.\nRead=latency, Write=write limit, Units=finite parallelism; units alone may have no effect at ideal zero latency.'),fontsize=9)
    return save(fig,output,'06_internal_feedback',title='完整服务中的消融、真实多轮会话与动态磨损反馈',
        caption='分别对照理想介质、静态/启发式冷热策略和开关磨损反馈。所有倍率只使用同一列对应基准；压力参数不代表真实设备。',rows=ablation+agents+wear,
        status='complete' if len(ablation)==7 and len(agents)==2 and len(wear)==2 else 'partial_data')


def sustained_figure(groups,output):
    group=groups.get('16_sustained',{});bounds=group.get('arrival_boundaries',[])
    rows=[r for r in group.get('rows',[]) if full_row(r)]
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.5));fig.subplots_adjust(top=.79,bottom=.23,wspace=.3)
    fig.suptitle(tr('持续负载SLO：有限窗口边界与并发会话','Sustained SLO: finite-window and concurrency brackets'),y=.97,fontsize=15)
    for i,b in enumerate(bounds):
        axes[0].bar(i-.17,b['passing_rate_rps'],width=.3,color='#21887B',label='Pass' if i==0 else None)
        axes[0].bar(i+.17,b['failing_rate_rps'],width=.3,color='#B22D43',label='Fail' if i==0 else None)
    axes[0].set_xticks(range(len(bounds)),[b['placement'][0] for b in bounds]);axes[0].set_ylabel('requests/s')
    if bounds: axes[0].legend(frameon=False)
    concurrent=group.get('concurrency_boundaries',[])
    for i,b in enumerate(concurrent):
        axes[1].bar(i-.17,b['passing_sessions_per_gpu'],width=.3,color='#21887B')
        axes[1].bar(i+.17,b['failing_sessions_per_gpu'],width=.3,color='#B22D43')
        axes[1].text(i,b['failing_sessions_per_gpu']+1,f"{b['passing_sessions_per_gpu']} / {b['failing_sessions_per_gpu']}",ha='center')
    axes[1].set_xticks(range(len(concurrent)),[b['placement'][0] for b in concurrent]);axes[1].set_ylabel('sessions/GPU')
    for ax in axes: style(ax); ax.set_ylim(0,ax.get_ylim()[1]*1.12)
    fig.text(.075,.06,tr('左：去掉前25%预热，128/256/512请求均验证到达率边界，同时检查SLO和积压增长。\n右：2K输入/8输出的同时到达会话，TTFT≤5ms、TBT≤1ms；是局部相邻整数边界，不是无限期生产容量。',
        'Left: 25% warmup excluded; rate boundaries validated at 128/256/512 requests, with SLO and backlog checks.\nRight: simultaneous 2K/8-token requests, TTFT <= 5ms and TBT <= 1ms; local adjacent bracket, not infinite-horizon capacity.'),fontsize=9)
    return save(fig,output,'07_sustained_slo',title='持续负载与session/GPU边界',caption='负载搜索使用三个轨迹长度核验。并发边界与到达率边界采用不同的明确工作负载，不混为统一生产容量。',rows=rows,
        status='complete' if len(bounds)==len(concurrent)==2 and group.get('status')=='passed' else 'partial_data')


def bandwidth_figure(groups,output):
    rows=groups.get('18_bandwidth',{}).get('rows',[])
    # Identify the bandwidth target and its negative control by physical inputs,
    # independent of labels authored by the experiment runner.
    selected=[r for r in rows if (r.get('target_effective_gb_s') is not None or
        r['configured']['port_gb_s']==3000. and r['configured']['link_gb_s']==1600.)]
    fig,ax=plt.subplots(figsize=(10,5.7));fig.subplots_adjust(top=.78,bottom=.25)
    xs=list(range(len(selected)));actual=[r['observed']['effective_end_to_end_read_gb_s'] for r in selected]
    ceiling=[r['configured']['path_bandwidth_upper_bound_gb_s'] for r in selected]
    ax.bar([x-.17 for x in xs],ceiling,width=.3,color='#B5C4D3',label=tr('路径带宽上限','Path ceiling'))
    bars=ax.bar([x+.17 for x in xs],actual,width=.3,color='#2364AA',label=tr('实际端到端有效值','Actual end-to-end'))
    for b,v in zip(bars,actual): ax.text(b.get_x()+b.get_width()/2,v+45,f'{v:,.1f}',ha='center')
    ax.set_xticks(xs,[f"{r['configured']['port_gb_s']:g} / {r['configured']['link_gb_s']:g}" for r in selected]);ax.set_ylabel('GB/s');ax.set_xlabel(tr('端口 / 链路配置（GB/s）','Port / link configuration (GB/s)'));style(ax);ax.legend(frameon=False)
    fig.suptitle(tr('有效读带宽：实际内存与链路完成窗口','Effective reads: actual media and link completion window'),y=.95,fontsize=15)
    fig.text(.08,.055,tr('384MiB顺序读、16KiB物理页、4µs读延迟、有限64块FIFO。\n2048并行单元与高内部通道速率是内部敏感性假设；不是设备规格，也不是LLM TPS。',
        '384 MiB sequential read, 16 KiB pages, 4 us read latency, bounded 64-block FIFO.\n2048 units and high internal bandwidth are sensitivity assumptions, not device specifications or LLM TPS.'),fontsize=9)
    result=save(fig,output,'08_effective_bandwidth',title='端口与有效带宽：含链路瓶颈负对照',caption='物理内存读取与真实链路占用后计算端到端有效带宽；计入启动、延迟与有限缓冲反压。',rows=selected,status='complete' if len(selected)==4 else 'partial_data')
    result['row_scope']='physical sequential-read and link microbenchmark; no LLM TPS';return result


def decode_figure(groups,output):
    from hbf_hbm_experiments import decode_matrix_complete
    group=groups.get('19_decode_physical',{})
    rows=[r for r in group.get('rows',[]) if full_row(r) and 'decode_measurement' in r]
    output_tokens=group.get('output_tokens',64)
    measured_tokens=output_tokens-1
    contexts=[2048,32768,131072,1048576]
    labels=['2K','32K','128K*','1M*']
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    fig.subplots_adjust(top=.85,bottom=.19,left=.075,right=.98,hspace=.42,wspace=.3)
    fig.suptitle(tr('生成阶段的真实影响：独立Decode窗口','Decode impact measured after prefill'),y=.98,fontsize=17)
    fig.text(.5,.905,tr(f'HBM → DRAM Core · HBF → NAND Core · GPU计算/启动保留简化模型\n相同配置与工作量 · batch=1 · {output_tokens}输出 · 每点{measured_tokens}个Decode token',
        f'HBM: DRAM Core · HBF: NAND Core · GPU compute/launch approximation retained\nIdentical paired inventory/workload · batch=1 · {output_tokens} outputs · {measured_tokens} measured decode tokens'),ha='center',fontsize=10)
    for i,(model,label) in enumerate((('llama3_1-8b','Llama 3.1 8B'),('llama3_1-70b','Llama 3.1 70B'))):
        selected=[r for r in rows if r.get('parameters',{}).get('model_id')==model]
        pairs=sorted([p for p in group.get('comparisons',[]) if p['model_id']==model],key=lambda p:p['prompt_tokens'])
        for j,(key,heading) in enumerate((('node_decode_tps','Decode TPS'),('tpot_ns','TPOT (ms)'))):
            ax=axes[i,j]
            if not selected: missing(ax);continue
            for arch in (A,B):
                series=sorted([r for r in selected if r['placement']==arch],key=lambda r:r['prompt_tokens'])
                values=[r['decode_measurement']['node_decode_tps'] if key=='node_decode_tps' else r['tpot_ns']['p50']/1e6 for r in series]
                ax.plot([contexts.index(r['prompt_tokens']) for r in series],values,marker=MARKERS[arch],
                    color=COLORS[arch],label=LABELS[arch],markersize=6)
            ax.set_yscale('log');ax.set_ylabel(heading+tr('（对数轴）',' (log)'));ax.set_title(label+' · '+heading)
            ax.legend(frameon=False,fontsize=8)
        ax=axes[i,2]
        if pairs:
            xs=[contexts.index(p['prompt_tokens']) for p in pairs]
            ratios=[p['B_over_A_decode_tps'] for p in pairs]
            ax.plot(xs,ratios,'o-',color='#6D52A4',label=tr('独立Decode TPS','Decode TPS'))
            ax.plot(xs,[p['B_over_A_total_tps'] for p in pairs],'s--',color='#8C9BAB',label=tr('全请求TPS','Whole-request TPS'))
            ax.axhline(1,color='#64748B',linestyle=':',lw=1)
            for x,value in zip(xs,ratios):ax.annotate(f'{value:.3f}',(x,value),xytext=(0,-16),textcoords='offset points',ha='center',fontsize=9,color='#6D52A4')
            ax.set_ylim(0,max(1.12,max(ratios)*1.15));ax.set_ylabel('B / A');ax.legend(frameon=False,fontsize=8,loc='lower left')
        else:missing(ax)
        ax.set_title(label+tr(' · 两种窗口的倍率',' · ratios by measurement window'))
        for ax in axes[i]:
            if ax.axison:
                ax.set_xticks(range(len(contexts)),labels);ax.set_xlabel(tr('输入上下文','Prompt context'));style(ax)
    fig.text(.07,.045,tr(f'先真实执行全部prefill，再统计首token之后的{measured_tokens}个token。Decode TPS={measured_tokens}/生成窗口秒数；TPOT=该窗口/{measured_tokens}。\n右列低于1表示HBF KV的吞吐低于HBM KV；全请求窗口含prefill，可能掩盖生成阶段差异。\n70B/1M的A/B共同使用1TiB HBM，其余512GiB；带宽/时序不变，原配置连续地址分配失败另存为负对照。\n* 128K加输出略超原生上下文；1M为架构压力外推。数值来自仿真假设，不是实测设备性能。',
        f'Real prefill executes first. Decode TPS={measured_tokens} / post-first-token window; TPOT=window / {measured_tokens}.\nRight: below 1 means lower throughput with HBF KV; prefill in the whole-request window can obscure decode impact.\n70B/1M uses 1 TiB HBM for BOTH A/B; other points use 512 GiB. Bandwidth/timing unchanged; original capacity failure retained.\n* 128K plus outputs exceeds native context slightly; 1M is an architecture stress case. Simulation, not hardware measurements.'),fontsize=9)
    return save(fig,output,'09_decode_physical',title='固定输出下的独立Decode TPS、TPOT及A/B倍率',
        caption=f'统一{output_tokens}输出，真实prefill后测量{measured_tokens}个Decode token。HBM与HBF均执行对应物理Core；GPU计算和启动继续采用原有简化模型。',
        rows=rows,status='complete' if decode_matrix_complete([group] if group else []) else 'partial_data',observations=group.get('comparisons',[]))


def bank_group_figure(groups, output):
    from hbf_hbm_report import bank_group_variants
    study = groups['20_bank_group_decode']
    rows = [r for r in study.get('rows', []) if full_row(r)]
    variants = bank_group_variants(study)
    profiles = [v['profile'] for v in variants]
    contexts = study['contexts']
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.subplots_adjust(top=.77, bottom=.25, left=.065, right=.98, wspace=.3)
    fig.suptitle(tr('Bank group 对 Decode 的实际影响：小Dense完整服务',
                   'Bank-group effect on Decode: full small-Dense serving'), fontsize=17, y=.97)
    fig.text(.5, .86, tr('真实prefill后统计 · 分组与时序图例读取本次实际Core配置\n参考组织不代表已识别厂商器件；非零组时序为合成假设，不是8B/70B性能预测。',
                        'Real prefill; geometry and timing labels come from saved Core configurations\nReference organization is not an identified vendor part; synthetic timing is not an 8B/70B prediction.'), ha='center', fontsize=10)
    names = [tr(v['label_zh'].replace('，', '\n', 1), v['label_en'].replace(', ', '\n', 1)) for v in variants]
    for axis, arch in zip(axes[:2], (A, B)):
        for profile, name, color, marker in zip(profiles, names, ('#8C9BAB', '#2364AA', '#E07632'), ('s', 'o', '^')):
            selected = sorted((r for r in rows if r['hardware_profile'] == profile and r['placement'] == arch), key=lambda r: r['prompt_tokens'])
            axis.plot([contexts.index(r['prompt_tokens']) for r in selected],
                      [r['decode_measurement']['node_decode_tps'] for r in selected],
                      label=name, color=color, marker=marker, linestyle='--' if profile == profiles[0] else '-',
                      markersize=8 if profile == profiles[0] else 5, markerfacecolor='none' if profile == profiles[0] else color)
        axis.set_title(LABELS[arch]); axis.set_ylabel('Decode token/s'); axis.legend(frameon=False, fontsize=8)
    for arch in (A, B):
        selected = sorted((c for c in study.get('comparisons', []) if c['placement'] == arch), key=lambda c: c['prompt_tokens'])
        axes[2].plot([contexts.index(c['prompt_tokens']) for c in selected], [c['timing_over_control_tps'] for c in selected],
                     label=LABELS[arch], color=COLORS[arch], marker=MARKERS[arch])
    axes[2].axhline(1., color='#8C9BAB', linestyle=':', linewidth=1)
    axes[2].set_title(tr('非零组时序 / 分组控制', 'Nonzero timing / geometry control'))
    axes[2].set_ylabel(tr('Decode TPS倍率', 'Decode TPS ratio')); axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xticks(range(len(contexts)), [str(c) for c in contexts]); axis.set_xlabel(tr('输入上下文token', 'Prompt tokens')); style(axis)
    fig.text(.065, .075, tr(f'1层合成Dense，hidden=256、FFN=512；每点{study["output_tokens"]}输出，完整prefill后测量{study["output_tokens"]-1}个Decode token。\n分组标为group×bank/group；“组时序关闭”不代表器件真实时序为零。HBM/HBF仍执行DRAM/NAND Core。\n完整几何、时序、参数来源和未知项见结果JSON；不按当前目录重新解释旧profile名称。',
                           f'One synthetic Dense layer, hidden=256, FFN=512; {study["output_tokens"]} outputs, measuring {study["output_tokens"]-1} Decode tokens after real prefill.\nLabels show groups x banks/group. Disabled gates do not imply zero device timing. HBM/HBF execute DRAM/NAND Core.\nGeometry, timing, provenance and unknowns are saved in JSON; current catalog defaults do not reinterpret old profile names.'), fontsize=9)
    caption = '小Dense完整服务；实际配置：'+'；'.join(v['label_zh'] for v in variants)+'。配置与来源读取保存的Core记录；参考组织和合成时序不能外推为目标器件或8B/70B性能。'
    result = save(fig, output, '10_bank_group_decode', title='统一参数入口下的Bank group Decode敏感性',
                caption=caption, rows=rows,
                status='complete' if study.get('status') == 'passed' and all(v['records'] for v in variants) else 'partial_data',
                observations=study.get('comparisons', []))
    result['hardware_variant_labels'] = [{key: v[key] for key in ('profile', 'label_zh', 'label_en')} for v in variants]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/hbf_hbm/results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/hbf_hbm/figures")
    parser.add_argument("--allow-partial", action="store_true", help="Allow an explicitly labelled partial-data smoke run")
    parser.add_argument("--update-report", action="store_true",
                        help="When all figures are complete, attach their manifest to results and refresh report.html")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    groups = {group["id"]: group for group in payload.get("groups", ())}
    args.output.mkdir(parents=True, exist_ok=True)
    font = setup_font()
    figures = [make(groups, args.output) for make in
               (small_batch_figure, named_context_figure, slo_figure, prefetch_figure,
                system_figure,internal_feedback_figure,sustained_figure,bandwidth_figure,decode_figure)]
    if '20_bank_group_decode' in groups:
        figures.append(bank_group_figure(groups, args.output))
    def report_path(path):
        try:
            return Path(os.path.relpath(Path(path).resolve(), args.input.resolve().parent)).as_posix()
        except ValueError:  # Windows paths on separate drives have no relative path.
            return Path(path).resolve().as_uri()

    for figure in figures:
        figure["report_relative_files"] = {extension: report_path(path) for extension, path in figure["files"].items()}
    complete = all(figure["status"] == "complete" for figure in figures)
    manifest = {"schema_version": 1, "status": "complete" if complete else "partial_data",
        "input_results": str(args.input.resolve()), "input_selected_checks_passed": payload.get("all_selected_checks_passed"),
        "output_directory_relative_to_results": report_path(args.output),
        "plotting_python": sys.executable, "matplotlib_version": matplotlib.__version__, **font,
        "global_caption_zh": "图表来自实际运行的仿真：完整算子服务与物理读带宽微基准分别标记，不是实测硬件性能。微基准不作为LLM TPS；有限SLO实验不解释为无限期稳定性证明。",
        "figures": figures}
    manifest = nullable_numbers(manifest)
    manifest_path = args.output / "figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report_updated = False
    if args.update_report and complete:
        from hbf_hbm_report import render_report
        payload["figure_manifest"] = manifest
        report = render_report(payload)
        args.input.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        args.input.with_name("report.html").write_text(report, encoding="utf-8")
        report_updated = True
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path.resolve()),
                      "figures": len(figures), "report_updated": report_updated}, ensure_ascii=False))
    return 0 if complete or args.allow_partial else 2


if __name__ == "__main__":
    raise SystemExit(main())
