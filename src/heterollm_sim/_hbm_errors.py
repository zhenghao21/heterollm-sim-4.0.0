"""Actionable HBM configuration diagnostics shared by Python and the web UI."""
from __future__ import annotations

import math
from collections.abc import Mapping


_LABELS = {
    "command_clock_ns": "命令时钟周期",
    "column_same_group_ns": "同 bank group 列命令间隔（tCCD_L）",
    "column_different_group_ns": "不同 bank group 列命令间隔（tCCD_S）",
    "column_different_sid_ns": "跨 SID 读命令间隔（tCCD_R）",
    "precharge_to_precharge_ns": "连续预充电命令间隔",
    "read_latency_ns": "读延迟（RL）", "write_latency_ns": "写延迟（WL）",
    "refresh_cycle_ns": "全 bank 刷新耗时（tRFCab）",
    "refresh_interval_ns": "平均刷新间隔（tREFI）",
    "maximum_active_ns": "最大开行时间（tRASmax）",
    "minimum_active_ns": "最小开行时间（tRASmin）",
    "activate_cycle_ns": "同 bank 重复激活间隔（tRC）",
    "precharge_to_activate_ns": "预充电到激活间隔（tRP）",
    "activate_to_read_ns": "激活到读命令间隔（tRCDRD）",
    "activate_to_write_ns": "激活到写命令间隔（tRCDWR）",
    "activate_command_cycles": "激活命令占用周期数",
    "precharge_command_cycles": "预充电命令占用周期数",
    "column_command_cycles": "列命令占用周期数",
    "interface_bandwidth_gb_s": "物理接口带宽",
    "data_rate_mt_s": "每引脚数据速率",
    "effective_pin_data_rate_gbps": "每引脚有效数据速率",
    "lane_bandwidth_gb_s": "单数据通道带宽",
    "read_bandwidth_gb_s": "读带宽", "write_bandwidth_gb_s": "写带宽",
    "channels": "物理通道数", "subchannels_per_channel": "每通道子通道数",
    "pseudo_channels_per_channel": "每通道伪通道数",
    "stacks": "堆叠数量", "dies_per_stack": "每堆叠裸片数",
    "ranks_per_channel": "每通道 rank 数", "bank_groups_per_rank": "每 rank 的 bank group 数",
    "banks_per_group": "每组 bank 数", "rows_per_bank": "每 bank 行数",
    "row_bytes": "行大小", "burst_bytes": "突发传输大小", "data_width_bits": "数据宽度",
    "data_lanes": "数据通道数", "capacity_bytes": "可见容量",
    "activated_row_msb": "有效行地址最高位",
    "bank_geometry_scope": "bank 几何统计范围", "stack_interface_mode": "堆叠接口模式",
}


def _json_value(value):
    # Invalid numeric input must still produce valid JSON diagnostics.
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return repr(value)


class HBMConfigError(ValueError):
    """A validation error carrying bilingual HBM diagnostics."""

    def __init__(self, *, field, value, category, generation, clock_ns,
                 rule, requirement_zh, requirement_en, remedy_zh, remedy_en,
                 source=None, related_values=None, label=None, unit=None):
        self.code = "hbm_" + rule
        label = label or _LABELS.get(field, field)
        if unit is None:
            unit = "ns" if field.endswith("_ns") else "CK" if field.endswith("_cycles") else "GB/s" if field.endswith("_gb_s") else "MT/s" if field.endswith("_mt_s") else "Gb/s" if field.endswith("_gbps") else "B" if field.endswith("_bytes") else "bit" if field.endswith("_bits") else ""
        self.details = _json_value(dict(
            code=self.code, field=field, label=label, value=value, unit=unit,
            category=category, generation=generation, clock_ns=clock_ns,
            rule=rule, requirement_zh=requirement_zh, requirement_en=requirement_en,
            remedy_zh=remedy_zh, remedy_en=remedy_en, source=source,
            related_values=related_values or {},
        ))
        actual = f"{value:g}" if isinstance(value, (float, int)) and not isinstance(value, bool) else repr(value)
        actual += (" " + unit) if unit else ""
        category_zh = {"standard_limit": "JEDEC 已知约束", "model_limit": "当前模型限制", "invalid_input": "输入错误"}[category]
        self.message_zh = f"{generation} 配置被拒绝：{label}（{field}）= {actual}。\n{category_zh}：{requirement_zh}"
        self.message_en = f"{generation}: {field}={actual}; {requirement_en}"
        if isinstance(clock_ns, (int, float)) and not isinstance(clock_ns, bool) and math.isfinite(clock_ns) and clock_ns > 0:
            self.message_zh += f"\n当前命令时钟周期：{clock_ns:g} ns。"
            self.message_en += f"; command_clock_ns={clock_ns:g} ns"
        self.message_zh += f"\n处理建议：{remedy_zh}"
        self.message_en += f". Suggested correction: {remedy_en}"
        if source:
            self.message_zh += f"\n依据：{source}。"
            self.message_en += f". Source: {source}"
        super().__init__(self.message_en)

    def add_context(self, context):
        """Retain the component identity without replacing the typed error."""
        if context not in self.details.setdefault("context", []):
            self.details["context"].append(context)
            self.message_zh = context + "\n" + self.message_zh
            self.message_en = context + " " + self.message_en
            self.args = (self.message_en,)
        return self
