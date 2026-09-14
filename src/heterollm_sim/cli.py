"""Command-line interface for validation and reference simulation."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__
from .compiler_ir import compile_canonical_scenario
from .config import load_scenario
from .planner import validate_scenario
from .reference import build_reference_scenario
from .reporting import (
    compare_with_gpu_baseline,
    format_comparison,
    format_report,
    report_dict,
    run_scenario,
)
from .runtime_diagnostics import format_doctor_report, health_payload
from .serde import canonical_json, write_json
from .topology import validate_topology


def _argparse_message_zh(message: str) -> str:
    patterns = (
        (r"the following arguments are required: (.+)", r"缺少必填参数：\1"),
        (r"unrecognized arguments: (.+)", r"无法识别的参数：\1"),
        (r"argument (.+): invalid choice: (.+)", r"参数 \1 的选项无效：\2"),
        (r"argument (.+): invalid int value: (.+)", r"参数 \1 的整数值无效：\2"),
        (r"argument (.+): expected one argument", r"参数 \1 需要提供一个值"),
    )
    for pattern, replacement in patterns:
        if re.fullmatch(pattern, message):
            return re.sub(pattern, replacement, message)
    return "参数不合法，请检查命令格式并使用 --help 查看说明。"


class ChineseArgumentParser(argparse.ArgumentParser):
    """Use Chinese headings and diagnostics without changing command names."""

    def __init__(self, *args, **kwargs):
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self.add_argument("-h", "--help", action="help", help="显示帮助并退出")

    def format_help(self) -> str:
        return (
            super()
            .format_help()
            .replace("usage:", "用法：")
            .replace("positional arguments:", "位置参数：")
            .replace("optional arguments:", "选项：")
            .replace("options:", "选项：")
        )

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "用法：")

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, "{}：参数错误：{}\n".format(self.prog, _argparse_message_zh(message)))


def _exception_message_zh(exc: BaseException) -> str:
    message = str(exc).strip()
    if re.search(r"[\u3400-\u9fff]", message):
        return message
    return "操作失败。请检查输入文件、场景配置和命令参数；可先运行 validate 命令定位问题。"


def build_parser() -> argparse.ArgumentParser:
    parser = ChineseArgumentParser(
        prog="heterollm-sim",
        description="面向异构集成芯片的大语言模型推理性能仿真器",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="校验场景配置")
    validate_parser.add_argument("scenario", type=Path)

    run_parser = subparsers.add_parser("run", help="运行场景仿真")
    run_parser.add_argument("scenario", type=Path)
    run_parser.add_argument(
        "--retention-policy",
        choices=("exact", "streaming", "aggregate"),
        default=None,
        help="结果保留策略（默认：static=exact，continuous=aggregate）",
    )
    run_parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    run_parser.add_argument("--output", type=Path, help="将 JSON 报告写入文件")

    canonical_parser = subparsers.add_parser(
        "export-ir",
        help="将场景编译并导出为 Canonical Schema 1.1",
    )
    canonical_parser.add_argument("scenario", type=Path)
    canonical_parser.add_argument(
        "--output",
        type=Path,
        help="写入 Canonical IR JSON；未指定时输出到标准输出",
    )

    demo_parser = subparsers.add_parser("demo", help="运行内置参考场景")
    demo_parser.add_argument(
        "--retention-policy",
        choices=("exact", "streaming", "aggregate"),
        default=None,
        help="结果保留策略（默认：static=exact，continuous=aggregate）",
    )
    demo_parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    demo_parser.add_argument("--output", type=Path, help="将 JSON 报告写入文件")

    compare_parser = subparsers.add_parser(
        "compare-gpu", help="将当前场景与纯 GPU 映射进行比较"
    )
    compare_parser.add_argument(
        "scenario", nargs="?", type=Path, default=None
    )
    compare_parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    compare_parser.add_argument("--output", type=Path, help="将 JSON 报告写入文件")

    doctor_parser = subparsers.add_parser(
        "doctor", help="诊断 Python 与 OR-Tools CP-SAT 运行环境"
    )
    doctor_parser.add_argument(
        "--json", action="store_true", help="以 JSON 输出完整诊断"
    )

    ui_parser = subparsers.add_parser("ui", help="启动本地 Web 界面")
    ui_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址（默认：127.0.0.1）",
    )
    ui_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="监听 TCP 端口（默认：8765）",
    )
    ui_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="启动后不自动打开浏览器",
    )

    catalog_parser = subparsers.add_parser(
        "catalog", help="列出、搜索或显式导入模型配置"
    )
    catalog_subparsers = catalog_parser.add_subparsers(
        dest="catalog_command", required=True
    )
    catalog_list = catalog_subparsers.add_parser("list", help="列出模型元数据")
    _add_catalog_cache_argument(catalog_list)
    catalog_list.add_argument("--offset", type=int, default=0)
    catalog_list.add_argument("--limit", type=int, default=50)
    catalog_list.add_argument("--query")
    catalog_list.add_argument("--family")
    catalog_list.add_argument("--architecture")
    catalog_list.add_argument("--support-level")
    catalog_list.add_argument("--openness")
    catalog_list.add_argument("--source")

    catalog_import = catalog_subparsers.add_parser(
        "import", help="固定一个 Hugging Face 配置，但不下载权重"
    )
    _add_catalog_cache_argument(catalog_import)
    catalog_import.add_argument("repo_id")
    catalog_import.add_argument("--revision", default="main")

    catalog_search = catalog_subparsers.add_parser(
        "search", help="显式搜索 Hugging Face 上公开的文本生成仓库"
    )
    catalog_search.add_argument("query")
    catalog_search.add_argument("--limit", type=int, default=20)

    architecture_parser = subparsers.add_parser(
        "architecture-presets", help="列出、查看或导出架构拓扑预设"
    )
    architecture_subparsers = architecture_parser.add_subparsers(
        dest="architecture_presets_command", required=True
    )
    architecture_list = architecture_subparsers.add_parser(
        "list", help="列出架构拓扑预设元数据"
    )
    architecture_list.add_argument("--query")
    architecture_list.add_argument("--vendor")
    architecture_list.add_argument("--family")
    architecture_list.add_argument("--topology-class")
    architecture_list.add_argument("--scale")
    architecture_list.add_argument("--support-level")
    architecture_list.add_argument("--protocol")
    architecture_list.add_argument("--tag")
    architecture_list.add_argument(
        "--loadable",
        choices=("true", "false", "1", "0", "yes", "no"),
        help="按是否可载入筛选",
    )
    architecture_detail = architecture_subparsers.add_parser(
        "detail", help="查看一个架构拓扑预设详情"
    )
    architecture_detail.add_argument("preset_id")
    architecture_export = architecture_subparsers.add_parser(
        "export-hardware", help="导出架构预设的 hardware JSON"
    )
    architecture_export.add_argument("preset_id")
    architecture_export.add_argument(
        "--output",
        type=Path,
        help="写入 hardware JSON；未指定时输出到标准输出",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            payload = health_payload(__version__)
            print(canonical_json(payload) if args.json else format_doctor_report(payload))
            return 0 if payload["ortools"]["available"] else 1

        if args.command == "ui":
            from .web import serve

            serve(args.host, args.port, open_browser=not args.no_browser)
            return 0

        if args.command == "catalog":
            return _run_catalog_command(args)

        if args.command == "architecture-presets":
            return _run_architecture_presets_command(args)

        if args.command == "demo" or (
            args.command == "compare-gpu" and args.scenario is None
        ):
            scenario = build_reference_scenario()
        else:
            scenario = load_scenario(args.scenario)
        if args.command == "validate":
            topology = validate_topology(scenario.hardware)
            scenario_report = validate_scenario(scenario)
            if not topology.is_valid:
                print(topology.format(), file=sys.stderr)
            for warning in scenario_report.warnings:
                print("警告：{}".format(warning))
            for information in scenario_report.information:
                print("信息：{}".format(information))
            if not scenario_report.is_valid:
                print("场景校验未通过：", file=sys.stderr)
                for error in scenario_report.errors:
                    print("- {}".format(error), file=sys.stderr)
                return 2
            print("场景校验通过：{}".format(scenario.name))
            return 0

        if args.command == "export-ir":
            topology = validate_topology(scenario.hardware)
            scenario_report = validate_scenario(scenario)
            if not topology.is_valid or not scenario_report.is_valid:
                print("错误：场景校验未通过，无法导出 Canonical IR。", file=sys.stderr)
                return 2
            canonical = compile_canonical_scenario(scenario)
            payload = canonical.to_dict()
            if args.output:
                write_json(args.output, payload)
                print("Canonical IR 已写入：{}".format(args.output), file=sys.stderr)
            else:
                print(canonical_json(payload))
            return 0

        if args.command == "compare-gpu":
            payload = compare_with_gpu_baseline(scenario)
            print(canonical_json(payload) if args.json else format_comparison(scenario))
            if args.output:
                write_json(args.output, payload)
                print("报告已写入：{}".format(args.output), file=sys.stderr)
            return 0

        result = run_scenario(
            scenario, retention_policy=args.retention_policy
        )
        payload = report_dict(result)
        print(canonical_json(payload) if args.json else format_report(result))
        if args.output:
            write_json(args.output, payload)
            print("报告已写入：{}".format(args.output), file=sys.stderr)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("错误：{}".format(_exception_message_zh(exc)), file=sys.stderr)
        return 2


def _add_catalog_cache_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="覆盖当前用户的模型目录缓存路径",
    )


def _run_catalog_command(args: argparse.Namespace) -> int:
    from .model_catalog import ModelCatalog

    catalog = ModelCatalog(getattr(args, "cache_dir", None))
    if args.catalog_command == "list":
        payload = catalog.page(
            offset=args.offset,
            limit=args.limit,
            query=args.query,
            family=args.family,
            architecture=args.architecture,
            support_level=args.support_level,
            openness=args.openness,
            source=args.source,
        )
    elif args.catalog_command == "import":
        payload = catalog.import_repository(args.repo_id, args.revision)
    elif args.catalog_command == "search":
        items = catalog.remote_search(args.query, args.limit)
        payload = {
            "items": items,
            "total": len(items),
            "query": args.query,
            "limit": args.limit,
        }
    else:  # argparse enforces the subcommand; retain a fail-closed guard.
        raise ValueError("未知的模型目录命令。")
    print(canonical_json(payload))
    return 0


def _run_architecture_presets_command(args: argparse.Namespace) -> int:
    from .architecture_presets import (
        architecture_preset_detail,
        architecture_preset_page,
        materialize_architecture_payload,
    )

    if args.architecture_presets_command == "list":
        payload = architecture_preset_page(
            query=args.query or "",
            vendor=args.vendor or "",
            family=args.family or "",
            topology_class=args.topology_class or "",
            scale=args.scale or "",
            support_level=args.support_level or "",
            protocol=args.protocol or "",
            tag=args.tag or "",
            loadable=_optional_bool_text(args.loadable),
        )
        print(canonical_json(payload))
        return 0
    if args.architecture_presets_command == "detail":
        try:
            payload = architecture_preset_detail(args.preset_id)
        except KeyError as exc:
            raise ValueError(
                "未找到指定的架构拓扑预设：{}".format(args.preset_id)
            ) from exc
        print(canonical_json(payload))
        return 0
    if args.architecture_presets_command == "export-hardware":
        try:
            payload = materialize_architecture_payload(args.preset_id)
        except KeyError as exc:
            raise ValueError(
                "未找到指定的架构拓扑预设：{}".format(args.preset_id)
            ) from exc
        if args.output:
            write_json(args.output, payload)
            print("架构硬件 JSON 已写入：{}".format(args.output), file=sys.stderr)
        else:
            print(canonical_json(payload))
        return 0
    raise ValueError("未知的架构拓扑预设命令。")


def _optional_bool_text(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError("布尔值必须为 true、false、1、0、yes 或 no。")


if __name__ == "__main__":
    raise SystemExit(main())
