"""Pure HBM organization/timing dependencies shared by all config entrances.

This module deliberately imports no configuration dataclasses. Provenance
identifies derived/reference values; explicit experiments retain legal values.
Geometry is insufficient to infer vendor array delays or a valid speed bin.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Iterable, Mapping

from ._hbm_errors import HBMConfigError


SOURCES = {
    "HBM2": "https://gitcode.com/Open-source-documentation-tutorial/a1451",
    "HBM3": "https://gitcode.com/Open-source-documentation-tutorial/69c72",
    "HBM4": "https://github.com/Adilohit/HBM4_verification/blob/main/resources/JEDECJESD270-4A-2025.pdf",
}
DOCUMENTS = {
    "HBM2": ("JESD235B", ".refs/jedec/JESD235B-2018.pdf",
             "JESD235B (November 2018), sections 3.2.1.2/3.2.3/3.2.4, Tables 4/5, printed pp. 6-8",
             "JESD235B pp.30-41,107-113"),
    "HBM3": ("JESD238", ".refs/jedec/JESD238-2022.pdf",
             "JESD238 (January 2022), sections 3.1.2/3.2, Tables 4/5, printed pp. 6-7",
             "JESD238 pp.26,34-46,161-165"),
    "HBM4": ("JESD270-4A", ".refs/jedec/JESD270-4A-2025.pdf",
             "JESD270-4A (2025), sections 3.1.2/3.2, Tables 4/5, printed pp. 6-7",
             "JESD270-4A pp.33,41-55,194-199"),
}
AUTOMATIC = {"standard_organization", "derived_standard_organization", "standard_reference", "standard_minimum",
             "standard_command_semantics", "derived_from_input", "jedec_idd_example", "selected_reference"}
GEOMETRY_FIELDS = {"channels", "subchannels_per_channel", "pseudo_channels_per_channel",
    "ranks_per_channel", "bank_groups_per_rank", "banks_per_group", "rows_per_bank",
    "row_bytes", "burst_bytes", "data_width_bits", "bank_geometry_scope", "stack_interface_mode"}
ARRAY_UNKNOWN = {"activate_to_read_ns", "activate_to_write_ns", "write_to_precharge_ns",
    "activate_same_group_ns", "activate_different_group_ns", "four_activate_window_ns",
    "read_to_precharge_ns", "write_to_read_same_group_ns", "write_to_read_different_group_ns",
    "read_to_write_command_ns"}
TIMING_FIELDS = ARRAY_UNKNOWN | {"command_clock_ns", "activate_command_cycles",
    "precharge_command_cycles", "column_command_cycles", "precharge_to_activate_ns",
    "minimum_active_ns", "maximum_active_ns", "activate_cycle_ns", "precharge_to_precharge_ns",
    "column_same_group_ns", "column_different_group_ns", "column_different_sid_ns",
    "refresh_interval_ns", "refresh_cycle_ns", "read_latency_ns", "write_latency_ns",
    "refresh_policy", "activated_row_msb", "close_ns", "burst_interval_ns"}


def family(generation):
    return "HBM2" if generation in {"HBM2", "HBM2E"} else "HBM3" if generation in {"HBM3", "HBM3E"} else generation


def organization_for(generation: str, density: int, height: int) -> dict:
    """Return the imported JEDEC organization, or raise for unsupported inputs."""
    gen = family(generation)
    densities = (16,) if gen == "HBM2" else (8, 16, 24, 32) if gen == "HBM3" else (24, 32)
    heights = (4, 8) if gen == "HBM2" else (4, 8, 12, 16)
    if gen not in SOURCES or density not in densities or height not in heights:
        raise ValueError("unsupported HBM standard density/height organization")
    return dict(channels=8 if gen == "HBM2" else 16 if gen == "HBM3" else 32,
        subchannels_per_channel=1, pseudo_channels_per_channel=2, ranks_per_channel=1,
        bank_groups_per_rank=height // 2 if gen == "HBM4" else height,
        banks_per_group=8 if gen == "HBM4" else 4,
        rows_per_bank=density * (2048 if gen == "HBM2" else 1024 if gen == "HBM3" else 512),
        row_bytes=1024, burst_bytes=32, data_width_bits=64 if gen == "HBM2" else 32,
        bank_geometry_scope="per_stack", stack_interface_mode="independent")


def refresh_cycle_ns_for(generation: str, density: int, height: int):
    """JEDEC tRFCab table lookup; None means unavailable, never a zero delay."""
    gen = family(generation)
    if gen == "HBM4":
        return {24: {4: 360., 8: 410., 12: 450., 16: 490.},
                32: {4: 400., 8: 450., 12: 490., 16: 530.}}.get(density, {}).get(height)
    if gen == "HBM3":
        return {(8, 8): 260., (8, 12): 310., (8, 16): 350.,
                (16, 4): 260., (16, 8): 350., (16, 12): 410., (16, 16): 450.}.get((density, height))
    if gen == "HBM2" and density == 16:
        return {4: 350., 8: 450.}.get(height)
    return None


def clock_ns_for(generation: str, width: int, lanes: int, interface_gb_s: float) -> float:
    return (2 if family(generation) == "HBM2" else 4) * width * lanes / (8.0 * interface_gb_s)


def reference_timings(generation: str, clock: float) -> dict:
    """Protocol minima and the separately labelled IDD example condition."""
    hbm2 = family(generation) == "HBM2"
    ras, rp = math.ceil(33.0 / clock) * clock, math.ceil(15.0 / clock) * clock
    return dict(activate_command_cycles=2. if hbm2 else 1.5,
        precharge_command_cycles=1. if hbm2 else .5, column_command_cycles=1.,
        column_same_group_ns=max(4 * clock, 2.8 if hbm2 else 2.5),
        column_different_group_ns=2 * clock, column_different_sid_ns=4 * clock,
        precharge_to_precharge_ns=(1 if hbm2 else 2) * clock, minimum_active_ns=ras,
        precharge_to_activate_ns=rp, close_ns=rp, activate_cycle_ns=ras + rp)


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


VENDOR_TIMING_FIELDS = ARRAY_UNKNOWN | {
    "minimum_active_ns", "precharge_to_activate_ns", "activate_cycle_ns",
    "read_latency_ns", "write_latency_ns", "column_different_sid_ns",
}


def _organization_refresh(payload):
    """Recognize actual geometry, never trust a caller's standard/source label."""
    gen = family(str(payload.get("generation", "")).upper())
    factor = 2048 if gen == "HBM2" else 1024 if gen == "HBM3" else 512
    rows = payload.get("rows_per_bank")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0 or rows % factor:
        return None
    density, height = rows // factor, payload.get("dies_per_stack")
    try:
        org = organization_for(gen, density, height)
    except (TypeError, ValueError):
        return None
    if any(payload.get(name) != value for name, value in org.items()):
        return None
    if payload.get("data_lanes") not in (None, org["channels"] * 2):
        return None
    return refresh_cycle_ns_for(gen, density, height)


def _access_precharge_windows(payload, clock, gen):
    """Earliest explicit PRE after each isolated operation, from ACT anchor."""
    pre_tick = clock if gen == "HBM2" else clock / 2
    for op, rcd, recovery in (
        ("READ", payload.get("activate_to_read_ns", 0), payload.get("read_to_precharge_ns", 0)),
        ("WRITE", payload.get("activate_to_write_ns", 0), payload.get("write_latency_ns", 0) + 2 * clock + payload.get("write_to_precharge_ns", 0)),
    ):
        earliest = max(payload.get("minimum_active_ns", 0), math.ceil(rcd / clock - 1e-12) * clock + recovery)
        yield op, math.ceil(earliest / pre_tick - 1e-12) * pre_tick


def _input_error(payload, field, requirement_en, requirement_zh, remedy_zh,
                 remedy_en="Correct the named input and validate the configuration again.", *, value=None, related_values=None):
    raise HBMConfigError(field=field, value=payload.get(field) if value is None else value,
        category="invalid_input", generation=payload.get("generation", "HBM"),
        clock_ns=payload.get("command_clock_ns"), rule="invalid_input",
        requirement_en=requirement_en, requirement_zh=requirement_zh,
        remedy_en=remedy_en, remedy_zh=remedy_zh, related_values=related_values)


def validate_hbm_timing_constraints(payload: Mapping):
    """Check imported JEDEC limits independently of provenance/assumption tags.

    The represented values are controller waiting gates, not certified device
    speed-bin minima. Longer minimum waits are safe; maximum deadlines may not
    be increased. Blank AC cells, IDD examples and optional auto-PRE MR encodings
    do not establish missing analog minima. The periodic refresh implementation
    supports intervals up to 3900 ns. This model limit does not describe every
    refresh mode allowed by the standards or establish vendor applicability.
    """
    if payload.get("control_mode") != "jedec_hbm":
        return
    gen = family(str(payload.get("generation", "")).upper())
    if gen not in DOCUMENTS:
        raise ValueError("jedec_hbm requires a supported HBM generation")
    selected = set(payload)
    document = DOCUMENTS[gen][0]

    ac_source = document + {"HBM2": " Table 68, pp.109–112", "HBM3": " Table 93, pp.161–165", "HBM4": " Table 108, pp.194–199"}[gen]

    def fail(name, requirement, requirement_zh, remedy_zh, remedy_en,
             *, category="standard_limit", rule="timing_constraint", source=None, related_values=None):
        raise HBMConfigError(field=name, value=payload.get(name),
            category=category, generation=payload.get("generation"), clock_ns=payload.get("command_clock_ns"),
            rule=rule, requirement_en=requirement, requirement_zh=requirement_zh,
            remedy_zh=remedy_zh, remedy_en=remedy_en,
            source=source or (ac_source if category == "standard_limit" else None),
            related_values=related_values)

    for name in selected & (TIMING_FIELDS - {"refresh_policy", "activated_row_msb"}):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            _input_error(payload, name, "timing must be finite and non-negative", "时序必须是有限的非负数。", "填写有效数值；不要使用负数、NaN 或无穷大。")
    clock = payload.get("command_clock_ns")
    if not _positive(clock):
        _input_error(payload, "command_clock_ns", "command clock must be finite and positive", "命令时钟周期必须是有限的正数。", "填写大于 0 的时钟周期，并使其与物理接口速率一致。")

    def minimum(name, limit, basis, basis_zh, source=None):
        if name in selected and payload.get(name, 0) < limit - max(1e-12, abs(limit) * 1e-12):
            fail(name, f"must be >= {limit:g} ns ({basis})", f"至少需要 {limit:g} ns；{basis_zh}。",
                 f"将 {name} 设置为不小于 {limit:g} ns 的值；手动值也必须满足该下限。",
                 f"Set {name} to at least {limit:g} ns; manual assumptions cannot bypass this minimum.",
                 rule="timing_below_minimum", source=source,
                 related_values={"minimum_ns": limit, "dies_per_stack": payload.get("dies_per_stack"),
                                 "rows_per_bank": payload.get("rows_per_bank")})

    minimum("column_same_group_ns", max(4 * clock, 2.8 if gen == "HBM2" else 2.5), "tCCD_L AC minimum: max(4*tCK, absolute ns floor)",
            f"tCCD_L 下限取 4 个时钟周期与 {2.8 if gen == 'HBM2' else 2.5:g} ns 中较大的一个")
    minimum("column_different_group_ns", 2 * clock, "tCCD_S AC minimum: 2*tCK", "tCCD_S 下限是 2 个时钟周期")
    minimum("column_different_sid_ns", (2 if gen == "HBM2" else 3) * clock,
            "tCCD_R vendor-minimum range lower bound; actual vendor value remains unknown", "这是 tCCD_R 厂商取值范围的公共下界，实际器件要求仍未知")
    minimum("precharge_to_precharge_ns", (1 if gen == "HBM2" else 2) * clock,
            "Table 33 PRE command spacing: 1*tCK" if gen == "HBM2" else "tPPD AC minimum: 2*tCK",
            "HBM2 的 PRE 命令间隔至少为 1 个时钟周期" if gen == "HBM2" else "tPPD 至少为 2 个时钟周期",
            "JESD235B Table 33, p.38" if gen == "HBM2" else None)
    refresh_min = _organization_refresh(payload)
    if refresh_min is not None:
        minimum("refresh_cycle_ns", refresh_min, "tRFCab AC table for the actual density/height", "按当前密度和堆叠层数查询 tRFCab 表得到该下限")
    expected_commands = (2., 1., 1.) if gen == "HBM2" else (1.5, .5, 1.)
    for name, expected in zip(("activate_command_cycles", "precharge_command_cycles", "column_command_cycles"), expected_commands):
        if name in selected and payload.get(name) != expected:
            fail(name, f"command occupancy must be {expected:g} CK for this generation", f"本代命令占用必须为 {expected:g} 个时钟周期。",
                 f"将 {name} 恢复为 {expected:g}；如需切换技术代，请同时更新整套配置。",
                 f"Set {name} to {expected:g}; change the full generation configuration together if needed.",
                 rule="command_occupancy", related_values={"required_cycles": expected},
                 source=document + {"HBM2": " Tables 30/31, pp.30–31", "HBM3": " Tables 30/31, pp.35–36", "HBM4": " Tables 33/34, pp.42–43"}[gen])
    limits = ((3, 48), (1, 16)) if gen == "HBM2" else ((4, 63), (4, 16)) if gen == "HBM3" else ((17, 90), (4, 19))
    for name, (low, high) in zip(("read_latency_ns", "write_latency_ns"), limits):
        if name in selected:
            cycles = payload.get(name, 0) / clock
            if not math.isclose(cycles, round(cycles), rel_tol=0, abs_tol=1e-9) or not low <= round(cycles) <= high:
                fail(name, f"must encode {low}..{high} whole command clocks; actual vendor support remains unknown",
                     f"必须对应 {low}–{high} 个整数时钟周期；当前为 {cycles:g} 个周期。合法编码不代表厂商已验证。",
                     f"选择 {low}–{high} 范围内的整数周期数，再乘以当前 {clock:g} ns 时钟周期；不要直接取整掩盖输入错误。",
                     f"Select an integer cycle count in {low}..{high} and multiply by {clock:g} ns; vendor support still needs verification.",
                     rule="latency_encoding", related_values={"actual_cycles": cycles, "minimum_cycles": low, "maximum_cycles": high},
                     source=document + {"HBM2": " Table 11, p.17", "HBM3": " Tables 12/13, pp.18–19", "HBM4": " Tables 11/12, pp.21–22"}[gen])
    interval = payload.get("refresh_interval_ns", 0)
    if "refresh_interval_ns" in selected and interval > 3900. * (1 + 1e-12):
        fail("refresh_interval_ns", "must be <= 3900 ns within the supported periodic refresh interval range; this is a model limit, not a universal JEDEC limit or vendor guarantee",
             "当前周期刷新实现支持的间隔不超过 3900 ns；这是模型限制，不代表 JEDEC 所有刷新模式的统一上限或厂商保证。",
             "在当前模型中使用不超过 3900 ns 的间隔，并保留适用的器件时序要求；更长间隔尚不受支持。",
             "Use an interval no greater than 3900 ns while retaining applicable device timing requirements; longer intervals are unsupported.",
             category="model_limit", rule="refresh_interval_scope", related_values={"maximum_ns": 3900.})
    if "maximum_active_ns" in selected:
        maximum = payload.get("maximum_active_ns", 0)
        limit = 9 * (interval or 3900.)
        if not _positive(maximum) or maximum > limit * (1 + 1e-12):
            fail("maximum_active_ns", f"tRASmax must be > 0 and <= {limit:g} ns (9*tREFI; 3900 ns reference when refresh timing is unknown)",
                 f"必须大于 0 且不超过 {limit:g} ns，上限为 9×tREFI；刷新间隔未知时采用 3900 ns 参考条件。",
                 f"将最大开行时间设在 (0, {limit:g}] ns 内，同时保证足以完成读写恢复；不能用 0 停用这项限制。",
                 f"Choose 0 < maximum_active_ns <= {limit:g} ns while preserving access recovery time; zero cannot disable the limit.",
                 rule="maximum_active_limit", related_values={"maximum_ns": limit, "refresh_interval_ns": interval or 3900.})
    if payload.get("refresh_policy") == "all_bank":
        if not _positive(interval) or not _positive(payload.get("refresh_cycle_ns")):
            fail("refresh_cycle_ns", "all_bank refresh requires positive tRFC and tREFI", "启用全 bank 刷新时，刷新耗时和刷新间隔都必须大于 0。",
                 "提供适用的 tRFC/tREFI；若参数未知，只能明确关闭刷新并标记数据保持未建模，不能把 0 当真实刷新耗时。",
                 "Supply applicable positive tRFC/tREFI, or explicitly disable unknown refresh and mark data retention unmodeled.",
                 category="model_limit", rule="refresh_inputs", related_values={"refresh_interval_ns": interval})
        if payload["refresh_cycle_ns"] > interval:
            fail("refresh_cycle_ns", "periodic all_bank policy cannot sustain tRFC > tREFI",
                 f"一次刷新耗时 {payload['refresh_cycle_ns']:g} ns，超过刷新间隔 {interval:g} ns，当前周期刷新策略无法持续执行。",
                 "重新选择可排程的刷新配置，使 tRFC≤tREFI；仍须满足 tRFC 表中下限和支持的刷新间隔范围，不能任意缩短刷新耗时。",
                 "Choose a feasible refresh configuration with tRFC <= tREFI, preserving the device-table minimum and supported refresh interval range.",
                 category="model_limit", rule="refresh_schedule", related_values={"refresh_interval_ns": interval, "maximum_ns": interval})
        ca_occupancy = payload.get("pseudo_channels_per_channel", 2) * max(
            payload.get("precharge_command_cycles", 0), payload.get("column_command_cycles", 0)) * clock
        if ca_occupancy > interval:
            fail("refresh_interval_ns", f"periodic all_bank policy needs at least {ca_occupancy:g} ns of shared command-bus time per refresh period",
                 f"各伪通道的刷新命令每周期至少占用共享命令总线 {ca_occupancy:g} ns，当前间隔只有 {interval:g} ns。",
                 "调整合法的时钟与刷新工作点，使刷新间隔容纳全部命令占用；同时保留刷新间隔上限。",
                 "Select a legal clock/refresh operating point that provides sufficient shared command-bus time without exceeding the refresh-interval limit.",
                 category="model_limit", rule="refresh_bus_capacity", related_values={"minimum_ns": ca_occupancy})
    maximum = payload.get("maximum_active_ns", 0)
    if maximum < payload.get("minimum_active_ns", 0):
        fail("maximum_active_ns", "tRASmax cannot be shorter than minimum_active_ns (tRASmin)",
             f"最大开行时间 {maximum:g} ns 小于最小开行时间 {payload.get('minimum_active_ns', 0):g} ns，两项要求无法同时满足。",
             "核对最小和最大开行时间，使最小值≤最大值；最大值仍不得超过 9×tREFI。",
             "Reconcile tRASmin <= tRASmax while retaining tRASmax <= 9*tREFI.",
             category="model_limit", rule="active_window_conflict", related_values={"minimum_active_ns": payload.get("minimum_active_ns", 0)})
    # Feasibility of one isolated access followed by explicit PRE. These are
    # scheduling relationships from the command tables, not guessed array
    # minima. Account for column rising edges and PRE half/full CK edges.
    for op, earliest in _access_precharge_windows(payload, clock, gen):
        if earliest > maximum + max(1e-12, abs(maximum) * 1e-12):
            fail("maximum_active_ns", f"{op} recovery cannot fit before tRASmax; earliest legal PRE is {earliest:g} ns after ACT array anchor",
                 f"按当前{'读' if op == 'READ' else '写'}延迟和恢复时间，最早只能在激活计时起点后 {earliest:g} ns 预充电，超过 {maximum:g} ns 的开行期限。",
                 f"核对激活到读写、写延迟和恢复时间；或在不超过 9×tREFI 的前提下，将开行期限设为至少 {earliest:g} ns。不要为通过校验而随意缩短未知器件时序。",
                 f"Check array/latency/recovery inputs, or set maximum_active_ns to at least {earliest:g} ns without exceeding 9*tREFI. Do not arbitrarily shorten unknown device timings.",
                 category="model_limit", rule="access_recovery_window", related_values={"operation": op, "earliest_precharge_ns": earliest,
                     "minimum_active_ns": payload.get("minimum_active_ns"), "activate_to_read_ns": payload.get("activate_to_read_ns"),
                     "activate_to_write_ns": payload.get("activate_to_write_ns"), "write_latency_ns": payload.get("write_latency_ns"),
                     "read_to_precharge_ns": payload.get("read_to_precharge_ns"), "write_to_precharge_ns": payload.get("write_to_precharge_ns")})


def _authored_timing_fields(payload, previous, explicit, provenance):
    """Identify raw edits by re-evaluating formulas at their prior inputs.

    A different scalar is authored even if an old source tag remains. Final
    checks use the new operating point, allowing simultaneous legal edits.
    No saved value snapshot is needed.
    """
    if previous:
        return explicit
    for name in TIMING_FIELDS - {"refresh_policy", "activated_row_msb"}:
        if name in payload:
            value = payload[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                _input_error(payload, name, "must be finite and non-negative", "时序必须是有限的非负数。", "填写有效数值；不要使用负数、NaN 或无穷大。")
    clock = payload.get("command_clock_ns")
    if not _positive(clock):
        _input_error(payload, "command_clock_ns", "must be finite and positive", "命令时钟周期必须是有限的正数。", "填写大于 0 的时钟周期，并使其与物理接口速率一致。")
    context = dict(payload)
    meta = payload.get("metadata") or {}
    managed = meta.get("geometry_status") in {"standard_organization", "explicit_assumption"} and (
        bool(provenance) or meta.get("organization_source") or meta.get("jedec_standard"))
    old_gen = {v[0]: k for k, v in DOCUMENTS.items()}.get(meta.get("jedec_standard"))
    if old_gen:
        context["generation"] = old_gen
        height = meta.get("sid_count", 0) * 4
        try:
            context.update(organization_for(old_gen, meta.get("die_density_gbits"), height))
            context["dies_per_stack"] = height
        except (TypeError, ValueError):
            pass
    expected = reference_timings(context["generation"], clock)
    # Recognizing an unchanged sum output is distinct from enforcing that
    # equality on authored tRC. The independent gate may be larger or smaller.
    cycle_entry = provenance.get("activate_cycle_ns", {})
    if "input_minimum_active_ns" in cycle_entry and "input_precharge_to_activate_ns" in cycle_entry:
        ras_input, rp_input = cycle_entry["input_minimum_active_ns"], cycle_entry["input_precharge_to_activate_ns"]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in (ras_input, rp_input)):
            _input_error(payload, "activate_cycle_ns", "activate_cycle_ns formula inputs must be finite and non-negative",
                         "来源记录中的 tRAS/tRP 公式输入必须是有限的非负数。", "修正 parameter_provenance.activate_cycle_ns 中的公式输入，并重新校验；不能靠修改来源标签绕过检查。",
                         related_values={"input_minimum_active_ns": ras_input, "input_precharge_to_activate_ns": rp_input})
        expected["activate_cycle_ns"] = ras_input + rp_input
    elif not math.isclose(payload.get("activate_cycle_ns", 0), expected["activate_cycle_ns"], rel_tol=1e-12):
        expected["activate_cycle_ns"] = payload.get("minimum_active_ns", 0) + payload.get("precharge_to_activate_ns", 0)
    interval_input = provenance.get("maximum_active_ns", {}).get("input_refresh_interval_ns")
    if interval_input is None:
        interval_input = payload.get("refresh_interval_ns") or 3900.
    if not _positive(interval_input) or interval_input > 3900. * (1 + 1e-12):
        _input_error(payload, "maximum_active_ns", "maximum_active_ns source input_refresh_interval_ns must be finite, positive and within the modeled 3900 ns envelope",
                     f"来源记录中的刷新间隔必须大于 0 且不超过当前模型支持的 3900 ns，当前为 {interval_input!r}。",
                     "修正 parameter_provenance.maximum_active_ns.input_refresh_interval_ns，使来源公式与合法的刷新配置一致。",
                     related_values={"input_refresh_interval_ns": interval_input, "maximum_ns": 3900.})
    expected["maximum_active_ns"] = 9 * interval_input
    # Formula metadata is editable too. An impossible old reference window
    # cannot authorize overwriting an authored scalar during normalization.
    minimum_window = max(value for _, value in _access_precharge_windows(context, clock, family(context["generation"])))
    if expected["maximum_active_ns"] < minimum_window:
        explicit.add("maximum_active_ns")
    expected["refresh_cycle_ns"] = _organization_refresh(context) or 0.
    for name, default in (("read_latency_ns", 32), ("write_latency_ns", 8)):
        cycles = provenance.get(name, {}).get("cycles", default)
        if not _positive(cycles):
            _input_error(payload, name, f"{name} source cycles must be finite and positive",
                         f"来源记录中的周期数必须是有限的正数，当前为 {cycles!r}。", f"修正 parameter_provenance.{name}.cycles，并确保对应本代支持的整数周期编码。",
                         related_values={"cycles": cycles})
        expected[name] = cycles * clock
    for name, reference in expected.items():
        entry = provenance.get(name, {})
        if (entry.get("status") not in AUTOMATIC and (name in provenance or not managed)) or name not in payload:
            continue
        if entry.get("rule") == "max_standard_minimum_and_analytical_floor":
            floor = entry.get("analytical_floor_ns")
            if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor < 0:
                _input_error(payload, name, "analytical_floor_ns must be finite and non-negative",
                             f"来源记录中的分析下限必须是有限的非负数，当前为 {floor!r}。", f"修正 parameter_provenance.{name}.analytical_floor_ns；分析假设不能替代标准下限。",
                             related_values={"analytical_floor_ns": floor})
            reference = max(reference, floor)
        if not math.isclose(payload[name], reference, rel_tol=1e-12, abs_tol=1e-12):
            explicit.add(name)
    return explicit


def _finish_timing_validation(result, meta, provenance, unknown, profile_context=None):
    if profile_context is not None:
        from .hbm_timing_profiles import finish_timing_profile
        finish_timing_profile(result, meta, provenance, profile_context)
    validate_hbm_timing_constraints(result)
    unknown.update(VENDOR_TIMING_FIELDS | {"vendor_speed_bin", "phy_turnaround_parameters"})
    vendor_unknown = VENDOR_TIMING_FIELDS | ({"refresh_cycle_ns"} if _organization_refresh(result) is None else set())
    for name in vendor_unknown:
        entry = provenance.setdefault(name, {"status": "unknown_with_assumption" if result.get(name) else "unknown_disabled"})
        if entry.get("status") not in {"unknown_with_assumption", "unknown_disabled", "unknown", "explicit_assumption", "selected_reference", "jedec_idd_example", "derived_from_input", "caller_supplied"}:
            entry["status"] = "explicit_assumption" if result.get(name) else "unknown_disabled"
        entry["vendor_value_status"] = "unknown"
    if _organization_refresh(result) is None:
        unknown.add("refresh_cycle_ns")
        provenance.setdefault("refresh_cycle_ns", {})["vendor_value_status"] = "unknown"
    meta.update(parameter_provenance=provenance, unknown_parameters=sorted(unknown),
                vendor_verified=False, timing_validation="not_a_vendor_speed_bin",
                timing_constraint_validation="known_jedec_constraints_checked",
                timing_constraint_basis=DOCUMENTS[family(result["generation"])][3],
                refresh_interval_scope="supported periodic interval <= 3900 ns; not a universal JEDEC limit or vendor guarantee",
                data_retention_validation="not_vendor_validated" if result.get("refresh_policy") == "all_bank" else "not_modeled_refresh_disabled")
    result["metadata"] = meta
    return result


def normalize_hbm_dependencies(payload: Mapping, *, previous: Mapping | None = None,
                               explicit_fields: Iterable[str] = ()) -> dict:
    """Normalize managed HBM values; preserve explicitly authored experiments.

    Bandwidth mode uses the aggregate interface as its primary operating
    point. Data-rate mode uses per-pin rate. A clock-only reconfiguration
    changes that operating point. Unchanged formula outputs follow dependencies;
    edited scalars remain authored inputs even when old source tags are retained.
    Every resulting JEDEC timing is checked independently of its provenance.
    """
    result = deepcopy(dict(payload))
    if result.get("control_mode") != "jedec_hbm":
        return result
    generation = str(result.get("generation", "")).upper()
    gen = family(generation)
    if gen not in SOURCES:
        return result
    explicit = set(explicit_fields)
    prior = dict(previous or {})
    if not isinstance(result.get("metadata", {}), Mapping):
        _input_error(result, "metadata", "HBM metadata must be a mapping", "元数据必须是键值对象。", "使用对象形式填写 metadata，例如 {}。")
    meta = deepcopy(dict(result.get("metadata") or {}))
    prior_meta = dict(prior.get("metadata") or {})
    if not isinstance(meta.get("parameter_provenance", {}), Mapping):
        _input_error(result, "metadata.parameter_provenance", "HBM parameter_provenance must be a mapping", "参数来源记录必须是键值对象。", "使用对象形式填写 parameter_provenance；没有依据的参数应标为未知。", value=meta.get("parameter_provenance"))
    provenance = deepcopy(dict(meta.get("parameter_provenance") or {}))
    if any(not isinstance(entry, Mapping) for entry in provenance.values()):
        name = next(name for name, entry in provenance.items() if not isinstance(entry, Mapping))
        _input_error(result, f"metadata.parameter_provenance.{name}", "HBM parameter provenance entries must be mappings", "每项参数来源记录都必须是键值对象。", "把该参数的来源记录改为对象，并如实填写来源；未知值不能标为已验证。", value=provenance[name])
    from .hbm_timing_profiles import prepare_timing_profile
    profile_context = prepare_timing_profile(result, prior, explicit)
    explicit = _authored_timing_fields(result, previous, explicit, provenance)
    managed = meta.get("geometry_status") in {"standard_organization", "explicit_assumption"} and (
        bool(provenance) or meta.get("organization_source") or meta.get("jedec_standard"))
    source = SOURCES[gen]
    unknown = set(meta.get("unknown_parameters") or ())
    meta["vendor_verified"] = False

    def record(name, status, basis, **extra):
        provenance[name] = dict(status=status, source=source if status in AUTOMATIC else "physical_config_override",
                                basis=basis, **extra)

    def automatic(name):
        return name not in explicit and (provenance.get(name, {}).get("status") in AUTOMATIC
                                        or (managed and name not in provenance))

    for name in explicit & TIMING_FIELDS:
        record(name, "explicit_assumption", "explicit timing/control input; not a certified vendor value")
    if not managed:
        # Synthetic configurations remain executable with their authored
        # numbers, while missing manufacturer evidence remains visible.
        for name in TIMING_FIELDS & result.keys():
            if name not in provenance:
                record(name, "explicit_assumption" if result[name] else "unknown_disabled",
                       "authored synthetic control configuration; vendor timing is unknown")
        for name in ARRAY_UNKNOWN:
            unknown.add(name)
            if result.get(name):
                record(name, "explicit_assumption" if name in explicit else "unknown_with_assumption",
                       "device minimum unknown; explicit analytical value is executed", vendor_value_status="unknown")
        if automatic("activate_cycle_ns") and explicit & {"minimum_active_ns", "precharge_to_activate_ns"}:
            result["activate_cycle_ns"] = result["minimum_active_ns"] + result["precharge_to_activate_ns"]
        if "maximum_active_ns" not in explicit and "refresh_interval_ns" in explicit and _positive(result.get("refresh_interval_ns")):
            result["maximum_active_ns"] = 9 * result["refresh_interval_ns"]
        return _finish_timing_validation(result, meta, provenance, unknown, profile_context)

    for name in ("channels", "subchannels_per_channel", "pseudo_channels_per_channel",
                 "stacks", "dies_per_stack", "ranks_per_channel", "bank_groups_per_rank",
                 "banks_per_group", "rows_per_bank", "row_bytes", "burst_bytes", "data_width_bits"):
        if isinstance(result.get(name), bool) or not isinstance(result.get(name), int) or result[name] <= 0:
            _input_error(result, name, "must be a positive integer", "几何数量必须是正整数。", "填写大于 0 的整数，并使容量、行大小及通道组织互相一致。")

    old_gen = family(str(prior.get("generation", generation)))
    if not prior:
        old_gen = {"JESD235B": "HBM2", "JESD238": "HBM3", "JESD270-4A": "HBM4"}.get(meta.get("jedec_standard"), gen)
    old_density = prior_meta.get("die_density_gbits", meta.get("die_density_gbits"))
    old_height = prior.get("dies_per_stack", int(meta.get("sid_count", 0)) * 4 or result.get("dies_per_stack"))
    if not prior and _positive(meta.get("raw_stack_capacity_bytes")) and _positive(old_height):
        capacity_density = meta["raw_stack_capacity_bytes"] * 8 / (old_height * (1 << 30))
        if float(capacity_density).is_integer():
            old_density = int(capacity_density)
    try:
        old_org = organization_for(old_gen, old_density, old_height)
    except (TypeError, ValueError):
        old_org = {}
    density = meta.get("die_density_gbits")
    height = result.get("dies_per_stack")
    rows = result.get("rows_per_bank")
    rows_authored = "rows_per_bank" in explicit or density is None or (not prior and old_org and rows != old_org["rows_per_bank"])
    density_authored = (("metadata" in explicit and density != prior_meta.get("die_density_gbits"))
                        or (not prior and density is not None and density != old_density))
    factor = 2048 if gen == "HBM2" else 1024 if gen == "HBM3" else 512
    if rows_authored:
        inferred = rows // factor if isinstance(rows, int) and rows > 0 and rows % factor == 0 else None
        if density_authored and inferred != density:
            _input_error(result, "rows_per_bank", "explicit HBM die density and row organization conflict",
                         f"行数与声明的每裸片密度 {density!r} Gbit 冲突；当前技术代的标准组织要求 rows_per_bank = 密度×{factor}。",
                         "同时更新行数与密度，或只修改其中一个并让联动规则更新另一个；自定义组织不能保留标准组织声明。",
                         related_values={"die_density_gbits": density, "rows_per_gbit": factor, "inferred_die_density_gbits": inferred})
        density = inferred
    try:
        expected = organization_for(generation, density, height)
    except (TypeError, ValueError):
        expected = None
    if expected:
        for name, value in expected.items():
            authored = name in explicit or (not prior and name in old_org and result.get(name) != old_org[name])
            if name == "rows_per_bank":
                authored = rows_authored
            if not authored and (automatic(name) or name in {"bank_geometry_scope", "stack_interface_mode"}):
                result[name] = value
        standard = all(result.get(name) == value for name, value in expected.items())
        standard = standard and result.get("data_lanes") in (None, expected["channels"] * 2)
    else:
        standard = False
    if standard:
        standard_name, source_file, organization_basis, control_basis = DOCUMENTS[gen]
        limitations = ["linear mixed-radix simulation mapping is not a vendor address mapping",
                      "standard organization does not establish the target vendor part or timing"]
        if gen == "HBM2":
            limitations.append("catalog imports only 16Gb base/8H HBM2 organizations; other legal JEDEC organizations are not yet imported")
        if height == 4 and gen != "HBM4":
            limitations.append(f"4H instantiates the no-SID base organization; {standard_name} Table 4 leaves base-organization physical stack height vendor-specific")
        meta.update(geometry_status="standard_organization", die_density_gbits=density,
            sid_count=height // 4, physical_channels_per_stack=expected["channels"],
            physical_channel_width_bits=expected["data_width_bits"] * 2,
            pseudo_channel_width_bits=expected["data_width_bits"],
            stack_data_width_bits=expected["channels"] * expected["data_width_bits"] * 2,
            banks_per_pseudo_channel=expected["bank_groups_per_rank"] * expected["banks_per_group"],
            bank_groups_per_pseudo_channel=expected["bank_groups_per_rank"],
            effective_addressed_page_bytes=1024, organization_source=source,
            jedec_standard=standard_name, jedec_reference_file=source_file,
            organization_basis=organization_basis, control_basis=control_basis,
            geometry_basis=organization_basis, geometry_limitations=limitations,
            channel_operation_mode="pseudo_channel",
            channel_operation_mode_basis=(f"{standard_name} section 3.1.2 defines two pseudo-channels per channel; no alternative Legacy mode is defined for this generation"
                if gen != "HBM2" else "JESD235B section 3.2.1 defines Legacy (HBM1) and Pseudo Channel (HBM2); this catalog imports only PC-mode organizations. Mode support is fixed by device design: DEVICE_ID[17:16] is 01 for PC-only or 10 for Legacy-only; 00 and 11 are illegal. This is not a runtime Legacy/PC switch."),
            capacity_formula="stacks * die_density_gbits * dies_per_stack * 2**30 / 8",
            source_documents=[dict(source=source, basis=organization_basis, local_file=source_file,
                                   document_author="JEDEC", access_method="public mirror of original protocol PDF")],
            parameter_basis="sourced_organization_with_explicit_timing_inputs")
        meta["row_bytes_basis"] = (f"{standard_name} Table 4 note 3: row_bytes is the 1KiB effective addressed PC page; the MSB of RA selects half of an open 2KiB page"
                                  if gen != "HBM2" else "JESD235B Table 4: effective PC page is 1KiB; physical activation organization is not asserted")
        if gen != "HBM2":
            meta["activated_page_bytes"] = 2048
        else:
            meta.pop("activated_page_bytes", None)
            provenance.pop("activated_page_bytes", None)
        for name in GEOMETRY_FIELDS:
            record(name, "standard_organization", "JEDEC channel addressing table for selected density/height")
        for name in ("effective_addressed_page_bytes", "activated_page_bytes"):
            if name in meta:
                record(name, "standard_organization", meta["row_bytes_basis"])
        for name, entry in provenance.items():
            if entry.get("status") in AUTOMATIC:
                entry["source"] = source
    else:
        meta.update(geometry_status="explicit_assumption", control_status="jedec_command_model_on_custom_geometry",
                    parameter_basis="explicit_organization_with_analytical_timing")
        for name in ("die_density_gbits", "sid_count", "organization_source", "organization_basis",
                     "organization_selection", "jedec_standard", "jedec_reference_file", "geometry_standard",
                     "geometry_source", "geometry_verified", "source_provenance", "capacity_formula",
                     "physical_channels_per_stack", "physical_channel_width_bits", "pseudo_channel_width_bits",
                     "stack_data_width_bits", "banks_per_pseudo_channel", "bank_groups_per_pseudo_channel"):
            meta.pop(name, None)
        for name in GEOMETRY_FIELDS:
            if name in explicit or not expected or result.get(name) != expected[name]:
                record(name, "explicit_assumption", "custom organization; no standard device organization is asserted")
        unknown.add("die_density_gbits")
        # The executed command protocol still follows this generation, while
        # its source table no longer certifies the custom organization.
        _, source_file, _, control_basis = DOCUMENTS[gen]
        meta["control_basis"] = control_basis
        meta["source_documents"] = [dict(source=source, basis=control_basis,
            local_file=source_file, document_author="JEDEC",
            access_method="public mirror of original protocol PDF")]
        if gen != old_gen:
            for name in ("row_bytes_basis", "geometry_limitations", "channel_operation_mode_basis",
                         "effective_addressed_page_bytes", "activated_page_bytes"):
                meta.pop(name, None)
                provenance.pop(name, None)

    lanes = (result.get("data_lanes") or result["channels"] * result["subchannels_per_channel"] * result["pseudo_channels_per_channel"])
    if result.get("stack_interface_mode") == "independent":
        lanes *= result["stacks"]
    width = result["data_width_bits"]
    prior_clock = prior.get("command_clock_ns", result.get("command_clock_ns"))
    old_interface = prior.get("interface_bandwidth_gb_s", result.get("interface_bandwidth_gb_s"))
    frequency_fields = {"interface_bandwidth_gb_s", "lane_bandwidth_gb_s", "effective_pin_data_rate_gbps", "data_rate_mt_s"}
    clock_authored = "command_clock_ns" in explicit
    mode = result.get("bandwidth_input_mode", "bandwidth")
    for name in explicit & (frequency_fields | {"command_clock_ns"}):
        if result.get(name) is not None and not _positive(result[name]):
            _input_error(result, name, "must be finite and positive", "频率、带宽或时钟周期必须是有限的正数。", "填写大于 0 的有效数值；时钟、每引脚速率与接口带宽须互相一致。")
    if clock_authored and not _positive(result.get("command_clock_ns")):
        _input_error(result, "command_clock_ns", "must be finite and positive", "命令时钟周期必须是有限的正数。", "填写大于 0 的时钟周期，并使其与物理接口速率一致。")
    if "bandwidth_input_mode" not in explicit:
        if explicit & {"effective_pin_data_rate_gbps", "data_rate_mt_s"}:
            mode = "data_rate"
        elif explicit & {"interface_bandwidth_gb_s", "lane_bandwidth_gb_s"}:
            mode = "bandwidth"
    result["bandwidth_input_mode"] = mode
    def frequency_conflict(field, requirement_en, requirement_zh, **related):
        _input_error(result, field, requirement_en, requirement_zh,
                     "只指定一个主输入，让其他值自动联动；若同时指定时钟、速率或带宽，必须先按相同通道宽度与通道数换算为一致值。",
                     "Specify one primary clock/rate/bandwidth input, or supply mutually consistent values for the same physical width and lane count.",
                     related_values={"data_width_bits": width, "data_lanes": lanes,
                                     "inputs": {name: result.get(name) for name in sorted(explicit & (frequency_fields | {"command_clock_ns"}))}, **related})

    declared_interfaces = []
    for name in sorted(explicit & frequency_fields):
        value = result.get(name)
        if value is None:
            continue
        supplied = value * lanes if name == "lane_bandwidth_gb_s" else value * width * lanes / (8.0 * (1000 if name == "data_rate_mt_s" else 1)) if name in {"data_rate_mt_s", "effective_pin_data_rate_gbps"} else value
        declared_interfaces.append(supplied)
    if any(not math.isclose(value, declared_interfaces[0], rel_tol=1e-10) for value in declared_interfaces[1:]):
        frequency_conflict(next(name for name in sorted(explicit & frequency_fields) if result.get(name) is not None),
                           "explicit HBM interface and data-rate inputs conflict",
                           f"同时填写的带宽和数据速率换算后不一致，分别得到接口带宽 {declared_interfaces} GB/s。",
                           derived_interfaces_gb_s=declared_interfaces)
    if clock_authored and _positive(result.get("command_clock_ns")):
        interface = (2 if gen == "HBM2" else 4) * width * lanes / (8.0 * result["command_clock_ns"])
        for name in explicit & frequency_fields:
            value = result.get(name)
            if value is None:
                continue
            supplied = value * lanes if name == "lane_bandwidth_gb_s" else value * width * lanes / (8.0 * (1000 if name == "data_rate_mt_s" else 1)) if name in {"data_rate_mt_s", "effective_pin_data_rate_gbps"} else value
            if not math.isclose(interface, supplied, rel_tol=1e-10):
                frequency_conflict("command_clock_ns", "explicit HBM command clock and interface/data-rate inputs conflict",
                                   f"时钟对应接口带宽 {interface:g} GB/s，但 {name} 对应 {supplied:g} GB/s。",
                                   interface_from_clock_gb_s=interface, interface_from_rate_gb_s=supplied, conflicting_field=name)
        result["interface_bandwidth_gb_s"] = interface
        result["lane_bandwidth_gb_s"] = None
        operating_basis = "explicit command_clock_ns; interface and per-pin rate follow JEDEC CK/DQ ratio"
    elif mode == "data_rate":
        rate = result["data_rate_mt_s"] / 1000.0 if "data_rate_mt_s" in explicit and "effective_pin_data_rate_gbps" not in explicit else result.get("effective_pin_data_rate_gbps")
        rate = rate if rate is not None else result["data_rate_mt_s"] / 1000.0
        interface = rate * width * lanes / 8.0
        if "interface_bandwidth_gb_s" in explicit and result.get("interface_bandwidth_gb_s") is not None and not math.isclose(interface, result["interface_bandwidth_gb_s"], rel_tol=1e-10):
            frequency_conflict("interface_bandwidth_gb_s", "explicit HBM data-rate and interface bandwidth inputs conflict",
                               f"数据速率对应接口带宽 {interface:g} GB/s，与填写的接口带宽不一致。", expected_interface_gb_s=interface)
        result["interface_bandwidth_gb_s"] = interface
        result["lane_bandwidth_gb_s"] = None
        operating_basis = "data_rate mode: per-pin rate is primary; aggregate interface and CK are derived"
    else:
        if "lane_bandwidth_gb_s" in explicit and "interface_bandwidth_gb_s" not in explicit:
            result["interface_bandwidth_gb_s"] = None
        if "interface_bandwidth_gb_s" in explicit and "lane_bandwidth_gb_s" not in explicit:
            result["lane_bandwidth_gb_s"] = None
        interface = result.get("interface_bandwidth_gb_s")
        if interface is None:
            interface = result.get("lane_bandwidth_gb_s")
            interface = interface * lanes if interface is not None else result["data_rate_mt_s"] * width * lanes / 8000.0
        operating_basis = "bandwidth mode: aggregate interface bandwidth is primary; per-PC rate and CK are derived"
    if not _positive(interface) or not _positive(lanes):
        _input_error(result, "interface_bandwidth_gb_s", "HBM physical interface bandwidth and lane count must be positive",
                     "换算后的物理接口带宽及数据通道数都必须是有限的正数。", "检查接口带宽、每引脚数据速率和几何中的数据通道数；不能使用零、负数或无效数值。",
                     value=interface, related_values={"data_lanes": lanes})
    clock = clock_ns_for(generation, width, lanes, interface)
    result["command_clock_ns"] = clock
    result["data_rate_mt_s"] = interface * 8000.0 / (width * lanes)
    if mode == "data_rate" or result.get("effective_pin_data_rate_gbps") is not None:
        result["effective_pin_data_rate_gbps"] = result["data_rate_mt_s"] / 1000.0
    if previous and old_interface is not None and not math.isclose(interface, old_interface, rel_tol=1e-10):
        for name in ("read_bandwidth_gb_s", "write_bandwidth_gb_s"):
            if name not in explicit and result.get(name) is not None:
                if math.isclose(result[name], old_interface, rel_tol=1e-10):
                    result[name] = interface
                else:
                    result[name] = min(result[name], interface)
    for name in ("command_clock_ns", "data_rate_mt_s"):
        record(name, "derived_from_input", operating_basis)
    meta["operating_point_basis"] = operating_basis
    result["burst_interval_ns"] = result["burst_bytes"] * lanes / interface
    record("burst_interval_ns", "derived_from_input", "full DQ burst duration from physical interface; direction caps only pace starts")
    if automatic("activated_row_msb"):
        result["activated_row_msb"] = None if gen == "HBM2" else (result["rows_per_bank"] - 1).bit_length() - 1
        record("activated_row_msb", "standard_command_semantics", "effective RA highest bit chooses the activated page half")

    for name, value in reference_timings(generation, clock).items():
        if automatic(name):
            entry = provenance.get(name, {})
            floor = entry.get("analytical_floor_ns")
            floor_rule = entry.get("rule") == "max_standard_minimum_and_analytical_floor"
            if floor_rule:
                if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(floor) or floor < 0:
                    _input_error(result, name, "analytical_floor_ns must be finite and non-negative",
                                 f"来源记录中的分析下限必须是有限的非负数，当前为 {floor!r}。", f"修正 parameter_provenance.{name}.analytical_floor_ns；分析假设不能替代标准下限。",
                                 related_values={"analytical_floor_ns": floor})
                value = max(value, floor)
            result[name] = value
            status = "jedec_idd_example" if name in {"minimum_active_ns", "precharge_to_activate_ns", "close_ns", "activate_cycle_ns"} else "selected_reference" if name == "column_different_sid_ns" else "standard_command_semantics" if name.endswith("command_cycles") else "standard_minimum"
            record(name, "derived_from_input" if floor_rule else status,
                   "maximum of the current standard minimum and an explicit analytical floor" if floor_rule else
                   "IDD example 33ns tRAS/15ns tRP quantized in CK" if status == "jedec_idd_example" else "JEDEC command clocks; tCCDR selects the 4CK reference within vendor range",
                   **({"rule": "max_standard_minimum_and_analytical_floor", "analytical_floor_ns": floor} if floor_rule else {}))
    for name, default in (("read_latency_ns", 32), ("write_latency_ns", 8)):
        entry = provenance.get(name, {})
        cycles = entry.get("cycles")
        if automatic(name) or (name not in explicit and entry.get("input_unit") == "cycles"):
            if cycles is None:
                cycles = result.get(name, 0) / prior_clock if _positive(prior_clock) else default
            result[name] = cycles * clock
            record(name, "caller_supplied" if entry.get("input_unit") == "cycles" and entry.get("status") == "caller_supplied" else "selected_reference",
                   "selected whole-CK MR latency; vendor speed-bin support unknown", cycles=cycles, input_unit="cycles")
        else:
            record(name, "explicit_assumption", "caller-authored ns latency; legal MR encoding validated separately",
                   cycles=result.get(name, 0) / clock, input_unit="ns")
        unknown.add(name)
    if automatic("activate_cycle_ns"):
        if "activate_cycle_ns" not in explicit:
            result["activate_cycle_ns"] = result["minimum_active_ns"] + result["precharge_to_activate_ns"]
            record("activate_cycle_ns", "derived_from_input", "tRC = configured tRAS + tRP, preserving their clock quantization",
                   input_minimum_active_ns=result["minimum_active_ns"], input_precharge_to_activate_ns=result["precharge_to_activate_ns"])
    refresh = refresh_cycle_ns_for(generation, density, height) if standard else None
    if automatic("refresh_cycle_ns") or provenance.get("refresh_cycle_ns", {}).get("status") in {"unknown", "unknown_disabled"}:
        result["refresh_cycle_ns"] = refresh or 0.0
        record("refresh_cycle_ns", "standard_reference" if refresh is not None else "unknown",
               "JEDEC tRFCab table selected by generation, density and height" if refresh is not None else "no applicable known tRFCab entry; refresh duration is unknown")
    elif refresh is None:
        record("refresh_cycle_ns", "explicit_assumption", "no known device tRFCab; explicit analytical duration retained",
               vendor_value_status="unknown")
    if refresh is None:
        unknown.add("refresh_cycle_ns")
    else:
        unknown.discard("refresh_cycle_ns")
    if automatic("refresh_interval_ns") or provenance.get("refresh_interval_ns", {}).get("status") in {"unknown", "unknown_disabled"}:
        interval = result.get("refresh_interval_ns")
        if _positive(interval) and interval != 3900.0:
            record("refresh_interval_ns", "explicit_assumption", "authored periodic refresh interval within the supported range; vendor applicability unverified")
        else:
            result["refresh_interval_ns"] = 3900.0 if result.get("refresh_cycle_ns", 0) > 0 else 0.0
            record("refresh_interval_ns", "standard_reference" if result["refresh_interval_ns"] else "unknown",
                   "selected 3900ns periodic refresh interval; vendor applicability unverified")
    if "refresh_policy" not in explicit and provenance.get("refresh_policy", {}).get("status") in AUTOMATIC | {"controller_policy", "unknown_disabled"}:
        result["refresh_policy"] = "all_bank" if result.get("refresh_cycle_ns", 0) > 0 and result.get("refresh_interval_ns", 0) > 0 else "disabled"
    if automatic("maximum_active_ns"):
        result["maximum_active_ns"] = 9 * (result.get("refresh_interval_ns") or 3900.0)
        record("maximum_active_ns", "derived_from_input", "tRASmax = 9*tREFI; 3900ns reference remains when automatic refresh is unavailable",
               input_refresh_interval_ns=result.get("refresh_interval_ns") or 3900.0)
    for name in ARRAY_UNKNOWN:
        unknown.add(name)
        status = provenance.get(name, {}).get("status")
        record(name, "explicit_assumption" if status == "explicit_assumption" else "unknown_with_assumption" if result.get(name) else "unknown_disabled",
               "device timing cannot be inferred from geometry; explicit analytical value retained" if result.get(name) else "device timing not supplied; zero disables this gate",
               vendor_value_status="unknown")
    unknown.update({"minimum_active_ns", "precharge_to_activate_ns", "activate_cycle_ns", "column_different_sid_ns"})
    if standard:
        raw_stack = density * height * (1 << 30) // 8
        raw = raw_stack * result["stacks"]
        meta.update(raw_stack_capacity_bytes=raw_stack, raw_capacity_bytes=raw,
                    visible_capacity_bytes=result.get("capacity_bytes"),
                    unexposed_capacity_bytes=raw - (result.get("capacity_bytes") or raw),
                    control_status="jedec_normal_operation_subset")
    meta.update(parameter_provenance=provenance, unknown_parameters=sorted(unknown),
                unconfigured_timing_parameters=sorted(name for name in ARRAY_UNKNOWN | {"refresh_cycle_ns"} if not result.get(name)),
                timing_validation="not_a_vendor_speed_bin")
    return _finish_timing_validation(result, meta, provenance, unknown, profile_context)
