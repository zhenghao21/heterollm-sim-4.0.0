"""Render the bounded comparison without running a simulation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ORDER = ("scan_hbm", "scan_hbf", "scan_hbf_hbm_intermediates", "paper_hbm4", "paper_hbf", "paper_conv")
LABELS = ("原 HBM3E", "原全 HBF", "原 HBF\n中间数据放 HBM", "论文 HBM4\n参数控制", "论文 HBF\n参数控制", "论文 CONV\n参数控制")


def render(directory: Path):
    data = json.loads((directory / "controls.json").read_text(encoding="utf-8"))
    rows = {(row["case"], row["batch_size"]): row for row in data["rows"]}
    expected = {(case, batch) for case in ORDER for batch in (1, 4)}
    if set(rows) != expected or any(row["status"] != "completed" for row in rows.values()):
        raise ValueError("the six-case, two-batch comparison is not fully completed")
    with (directory / "controls.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("方案", "批量", "TTFT毫秒", "TPOT毫秒", "有限窗口tokens每秒",
                         "TPOT不超过100毫秒", "已执行NAND写入MB", "实际运行秒"))
        for case in ORDER:
            for batch in (1, 4):
                row = rows[case, batch]
                writer.writerow((row["label"], batch, row["ttft_ns_mean"] / 1e6,
                    row["tpot_ns_mean"] / 1e6, row["decode_window_tps"], row["tpot_slo_met"],
                    row["storage_traffic"].get("physical_write_bytes", 0) / 1e6, row["wall_seconds"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import numpy as np

    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax, writes) = plt.subplots(1, 2, figsize=(15.5, 6.3), gridspec_kw={"width_ratios": (1.8, 1)})
    x = np.arange(len(ORDER))
    for batch, offset, color in ((1, -.18, "#4276ee"), (4, .18, "#e8a031")):
        values = [rows[case, batch]["tpot_ns_mean"] / 1e6 for case in ORDER]
        bars = ax.bar(x + offset, values, .34, label=f"批量 {batch}", color=color)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=5, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(.3, 210)
    ax.axhline(100, color="#bf4054", linestyle="--", linewidth=1.1)
    ax.text(5.4, 105, "100ms 阈值", ha="right", va="bottom", color="#bf4054", fontsize=10)
    ax.set_xticks(x, LABELS, fontsize=10)
    ax.set_ylabel("每请求 TPOT（毫秒，对数坐标）")
    ax.set_title("短物理请求：不同存储控制的生成延迟", fontsize=14, pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1, .82), frameon=False)
    for case, offset, label, color in (("scan_hbf", -.18, "全部中间数据放 HBF", "#e8a031"),
                                      ("scan_hbf_hbm_intermediates", .18, "中间数据放 HBM", "#38a58a")):
        values = [rows[case, batch]["storage_traffic"]["physical_write_bytes"] / 1e6 for batch in (1, 4)]
        bars = writes.bar(np.arange(2) + offset, values, .34, label=label, color=color)
        writes.bar_label(bars, labels=[f"{value:,.1f}" for value in values], padding=6, fontsize=10)
    writes.set_yscale("log")
    writes.set_ylim(10, 25000)
    writes.set_xticks((0, 1), ("批量 1", "批量 4"))
    writes.set_ylabel("请求内已执行 NAND 写入（十进制 MB）")
    writes.set_title("同一 HBF 介质：中间数据落点对照", fontsize=14, pad=15)
    writes.legend(loc="upper left", frameon=False, fontsize=10)
    for axis in (ax, writes):
        axis.grid(axis="y", alpha=.2, which="major")
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("固定 B200 / Qwen2.5-0.5B / BF16，输入 1,660、输出 17 词元", fontsize=17, y=.99)
    fig.text(.5, .025, "论文参数组采用通用页服务与补充假设；没有稳态补位、PD 导入或读双缓冲，不能视为论文吞吐复现。",
             ha="center", fontsize=11, color="#475569")
    fig.tight_layout(rect=(0, .08, 1, .94))
    for extension in ("png", "svg"):
        fig.savefig(directory / ("comparison." + extension), dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    render(parser.parse_args().results_dir)
