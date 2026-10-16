"""Traceable HBM3/E engineering scenarios, never vendor-calibrated speed bins.

The packs target the project's explicit 32 Gbit/die, 16-high organization.
They change timing only. Public manufacturer material checked on 2026-10-15
did not provide an applicable complete AC table. Fast/slow are sensitivity
endpoints, not guaranteed physical bounds or statistical confidence intervals.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from .memory_types import DramConfig


_STANDARD = ".refs/jedec/JESD238-2022.pdf"
_VENDOR_SEARCH = (
    {"url": "https://www.micron.com/products/memory/hbm/hbm3e/part-catalog",
     "finding": "Public HBM3E catalog includes 24GB/36GB classes; no applicable 32Gb 16-high AC table obtained."},
    {"url": "https://semiconductor.samsung.com/dram/hbm/hbm3e/",
     "finding": "Public 12-layer product overview; not an applicable 32Gb 16-high AC timing table."},
    {"url": "https://news.skhynix.com/en/sk-hynix-announces-16-layer-hbm3e-at-sk-ai-summit-2024/",
     "finding": "48GB 16-layer announcement; does not establish this 64GiB stack's AC timings."},
)

# Tuple order: fast / reference / slow. Every chosen value is an assumption.
_INPUTS = {
    "activate_to_read_ns": ("ns", (12., 14., 18.), "tRCDRD", "Table 93 p.161; AC minimum unspecified"),
    "activate_to_write_ns": ("ns", (12., 14., 18.), "tRCDWR", "Table 93 p.161; AC minimum unspecified"),
    "minimum_active_ns": ("ns", (28., 33., 40.), "tRAS", "Table 84 p.152 is an IDD example, not an AC guarantee"),
    "precharge_to_activate_ns": ("ns", (12., 15., 18.), "tRP", "Table 84 p.152 is an IDD example, not an AC guarantee"),
    "write_to_precharge_ns": ("ns", (12., 15., 20.), "tWR", "Table 93 p.161; analog value is vendor-specific"),
    "activate_different_group_ns": ("ns", (2.5, 4., 6.), "tRRD_S", "Table 93 p.161, notes 8/9 p.164"),
    "activate_same_group_ns": ("ns", (4., 6., 8.), "tRRD_L", "Table 93 p.161, notes 8/9 p.164"),
    "four_activate_window_ns": ("ns", (12., 20., 32.), "tFAW", "Section 6.3.2.2.1 p.41; numeric window unspecified"),
    "read_to_precharge_ns": ("nCK", (4, 6, 8), "tRTP", "Table 93 p.161 and section 6.3.3.2.4 p.65"),
    "write_to_read_different_group_ns": ("nCK", (4, 6, 8), "tWTR_S", "Table 93 p.161; section 6.3.3.3.3 p.72"),
    "write_to_read_same_group_ns": ("nCK", (6, 8, 12), "tWTR_L", "Table 93 p.161; section 6.3.3.3.3 p.72"),
    "read_latency_ns": ("nCK", (28, 32, 40), "RL", "MR2, Table 13 p.19; actual supported range is vendor-specific"),
    "write_latency_ns": ("nCK", (6, 8, 10), "WL", "MR1, Table 12 p.18; actual supported range is vendor-specific"),
    "column_different_sid_ns": ("nCK", (3, 4, 4), "tCCDR", "Table 93 note 17 p.164; 3..4 CK vendor-minimum range"),
    "refresh_cycle_ns": ("ns", (350., 500., 650.), "tRFCab", "Table 93 p.162: 32Gb/16-high is TBD; independent assumptions"),
    "refresh_interval_ns": ("ns", (3900., 3900., 3900.), "tREFI", "Table 93 p.163: 3.9us maximum average interval; selected periodic reference, no temperature scenario inferred"),
}
_SENSITIVITY_NAMES = ("fast", "reference", "slow")
_NAMES = (*_SENSITIVITY_NAMES, "ramulator2")
_DERIVED = {"activate_cycle_ns", "read_to_write_command_ns", "maximum_active_ns", "refresh_policy"}
_RTW_FORMULA = "ceil((RL_ns-WL_ns+2.5*tCK+max(-0.2ns,-0.2*tCK)+tWDQS2DQ_O_max_ns+tDQSQtra_max_ns+system_turnaround_margin_ns)/tCK)*tCK"
_RAMULATOR_COMMIT = "54e90367191062067e9be74840439b0daca40343"
_RAMULATOR_FILE = "python/ramulator/dram/hbm3.py"
_RAMULATOR_SOURCE = f"https://github.com/CMU-SAFARI/ramulator2/blob/{_RAMULATOR_COMMIT}/{_RAMULATOR_FILE}"
_RAMULATOR_RTW_FORMULA = "(RL_CK+BL_CK-WL_CK+ceil((max(-2000,-2*tCK_ps)+5*tCK_ps+10*(2500+20))/(10*tCK_ps)))*tCK_ns"
# Source cycles at HBM3_6400Mbps / HBM3_32Gb_16hi. The quoted source
# labels the array/latency inputs, nCCDR and this organization's nRFC as
# Ramulator Guesstimate; importing them does not establish vendor calibration.
_RAMULATOR_INPUTS = {
    "read_latency_ns": ("nCL", 20, "upstream_estimate"),
    "activate_to_read_ns": ("nRCDRD", 31, "upstream_estimate"),
    "activate_to_write_ns": ("nRCDWR", 15, "upstream_estimate"),
    "precharge_to_activate_ns": ("nRP", 26, "upstream_estimate"),
    "minimum_active_ns": ("nRAS", 45, "upstream_estimate"),
    "write_to_precharge_ns": ("nWR", 33, "upstream_estimate"),
    "read_to_precharge_ns": ("nRTP", 9, "upstream_estimate"),
    "write_latency_ns": ("nCWL", 10, "upstream_estimate"),
    "activate_different_group_ns": ("nRRDS", 4, "upstream_estimate"),
    "activate_same_group_ns": ("nRRDL", 5, "upstream_estimate"),
    "four_activate_window_ns": ("nFAW", 24, "upstream_estimate"),
    "write_to_read_different_group_ns": ("nWTRS", 7, "upstream_estimate"),
    "write_to_read_same_group_ns": ("nWTRL", 10, "upstream_estimate"),
    "column_different_group_ns": ("nCCDS", 2, "upstream_standard_reference"),
    "column_same_group_ns": ("nCCDL", 4, "upstream_formula"),
    "column_different_sid_ns": ("nCCDR", 3, "upstream_estimate"),
    "precharge_to_precharge_ns": ("nPPD", 2, "upstream_standard_reference"),
    "refresh_cycle_ns": ("nRFC", 880, "upstream_estimate"),
    "refresh_interval_ns": ("nREFI", 6240, "upstream_standard_reference"),
}


def _ramulator_pack():
    parameters = {field: dict(symbol=symbol, upstream_parameter=symbol,
        input_unit="nCK", input_value=cycles, status="explicit_assumption",
        vendor_value_status="unknown", source=_RAMULATOR_SOURCE, source_kind=kind,
        basis=("Upstream Ramulator Guesstimate, not a manufacturer value" if kind == "upstream_estimate"
               else "Upstream HBM3 standard/formula reference at the selected operating point; not vendor validation"))
        for field, (symbol, cycles, kind) in _RAMULATOR_INPUTS.items()}
    parameters["column_same_group_ns"]["upstream_formula"] = "max(4, ceil(2500/tCK_ps))"
    parameters["refresh_cycle_ns"].update(upstream_input_ns=550., upstream_formula="ceil(550000/tCK_ps)")
    parameters["refresh_interval_ns"]["upstream_formula"] = "3900000 // tCK_ps"
    parameters.update({
        "activate_cycle_ns": dict(symbol="nRC", upstream_parameter="nRC", input_unit="formula",
            formula="executed tRAS + executed tRP", source_kind="upstream_formula",
            basis="Upstream nRC=nRAS+nRP; derived from the executed profile inputs"),
        "read_to_write_command_ns": dict(symbol="nRTW", upstream_parameter="nRTW", input_unit="formula",
            formula=_RAMULATOR_RTW_FORMULA, source_kind="upstream_formula",
            basis="Upstream _resolve_nRTW; source adopts 2500 ps and 20 ps PHY extrema, no additional local margin"),
        "maximum_active_ns": dict(symbol="tRASmax", input_unit="formula", source=_STANDARD,
            formula="9 * (positive configured tREFI, otherwise 3900 ns reference interval)",
            source_kind="local_standard_supplement",
            basis="Local engine retains JESD238 Table 93 tRASmax; upstream HBM3 timing table has no nRASmax field"),
        "refresh_policy": dict(symbol="REFab", input_unit="policy", input_value="all_bank",
            source="heterollm_sim.hbm_timing_profiles", source_kind="local_controller_policy",
            basis="Local periodic all-bank controller policy, not an imported Ramulator controller configuration"),
    })
    for entry in parameters.values():
        entry.setdefault("source", _RAMULATOR_SOURCE)
        entry.update(status="explicit_assumption", vendor_value_status="unknown")
    raw = {symbol: cycles for symbol, cycles, _ in _RAMULATOR_INPUTS.values()}
    raw.update(rate=6400, tCK_ps=625, nBL=2, nRC=71, nRTW=17,
               nRFCpb=320, nRFMab=880, nRFMpb=320, nRREFD=13, nREFIpb=97)
    return dict(name="ramulator2", version=1, vendor_calibrated=False,
        scope="Ramulator2 HBM3_6400Mbps / HBM3_32Gb_16hi timing inputs mapped into this HBM3/E engine; not a complete Ramulator execution replica",
        range_scope="One upstream engineering reference at 6.4 Gb/s per pin; no extrapolation to other operating points",
        required_command_clock_ns=.625, parameters=parameters,
        upstream_source=dict(repository="CMU-SAFARI/ramulator2", commit=_RAMULATOR_COMMIT,
            file=_RAMULATOR_FILE, url=_RAMULATOR_SOURCE, timing_preset="HBM3_6400Mbps",
            organization_preset="HBM3_32Gb_16hi", checked_date="2026-10-15"),
        upstream_raw_timings=raw,
        unmapped_upstream_parameters={name: "Per-bank refresh or refresh-management command is outside the current engine scope"
            for name in ("nRFCpb", "nRFMab", "nRFMpb", "nRREFD", "nREFIpb")},
        mapping_notes={
            "nBL": "2 CK burst duration follows the existing 32-byte, 32-bit, 6.4 Gb/s interface; directional bandwidth caps can extend spacing between transfers",
            "command_cycles": "Existing HBM3 ACT=1.5 CK, PRE=0.5 CK and column=1 CK are retained",
            "nWTRS_nWTRL_nWR": "Recovery begins after write data ends; CWL and burst duration are already accounted for by this engine",
            "execution": "Existing address mapping, queue, scheduler, command anchors and periodic REFab remain this engine's implementation; upstream spec.py adjusts constraints to the last command-reception tick, so imported scalar values do not reproduce its command timeline"},
        phy_assumptions=dict(twdqs2dq_o_max_ns=2.5, tdqsqtra_max_ns=.020,
            system_turnaround_margin_ns=0., tdqss_min_rule="max(-0.2 ns, -0.2*tCK)",
            status="explicit_assumption", vendor_value_status="unknown", source=_RAMULATOR_SOURCE,
            basis="Values used by upstream _resolve_nRTW, not measurements of a vendor part"),
        standard_source=_STANDARD,
        missing_vendor_evidence="Applicable complete AC table and supported MR settings not obtained")


def _pack(name):
    if name == "ramulator2":
        return _ramulator_pack()
    index = _SENSITIVITY_NAMES.index(name)
    parameters = {field: dict(input_unit=unit, input_value=values[index],
        sensitivity_range=[min(values), max(values)], symbol=symbol,
        status="explicit_assumption", vendor_value_status="unknown", source=_STANDARD,
        basis="Engineering sensitivity choice; " + basis)
        for field, (unit, values, symbol, basis) in _INPUTS.items()}
    parameters.update({
        "activate_cycle_ns": dict(symbol="tRC", input_unit="formula", output_unit="ns",
            formula="executed tRAS + executed tRP", basis="Section 6.3.2.4 p.45; sum the rounded array minima"),
        "read_to_write_command_ns": dict(symbol="tRTW", input_unit="formula", output_unit="ns",
            formula=_RTW_FORMULA, basis="Table 93 note 18 p.164; bus turnaround with explicitly assumed PHY extrema"),
        "maximum_active_ns": dict(symbol="tRASmax", input_unit="formula", output_unit="ns",
            formula="9 * (positive configured tREFI, otherwise 3900 ns reference interval)",
            basis="Table 93 p.161: 9*tREFI; disabling refresh does not disable the open-row limit"),
        "refresh_policy": dict(symbol="REFab", input_unit="policy", input_value="all_bank",
            sensitivity_range=["all_bank"], basis="Section 6.3.2.5 pp.46-47; selected controller policy, REFpb/RFM not modeled"),
    })
    for entry in parameters.values():
        entry.update(status="explicit_assumption", vendor_value_status="unknown", source=_STANDARD)
    # PHY extrema are scenario inputs in ns. tDQSS(min) uses the public
    # envelope max(-0.2 ns, -0.2 CK), converted to CK inside the formula.
    phy = {"tdqss_min_rule": "max(-0.2 ns, -0.2*tCK)",
           "twdqs2dq_o_max_ns": (1., 1.5, 2.5)[index],
           "tdqsqtra_max_ns": .020,
           "system_turnaround_margin_ns": (.030, .080, .180)[index],
           "status": "explicit_assumption", "vendor_value_status": "unknown",
           "basis": "JESD238 Table 93 note 18 p.164 formula plus an explicit system margin; not a characterized package",
           "source": _STANDARD,
           "sensitivity_range": {"twdqs2dq_o_max_ns": [1., 2.5], "tdqsqtra_max_ns": [.020, .020],
                                 "system_turnaround_margin_ns": [.030, .180]},
           "parameter_basis": {
               "tdqss_min_rule": "Table 93 p.163: public -200 ps / -0.2 CK envelope; converted to ns",
               "twdqs2dq_o_max_ns": "Engineering maximum assumption inside Table 93 p.160 example 0.6..2.5 ns range; device maximum not obtained",
               "tdqsqtra_max_ns": "Table 93 p.160 example maximum 20 ps = 0.020 ns; selected public reference, not target validation",
               "system_turnaround_margin_ns": "Additional engineering system guard band; not a JEDEC device parameter or limit"}}
    return dict(name=name, version=1, vendor_calibrated=False,
        scope="HBM3/E 32Gbit-per-die 16-high analytical timing sensitivity; not a vendor speed bin",
        range_scope="Sensitivity scenarios, not guaranteed device bounds or confidence intervals",
        parameters=parameters, phy_assumptions=phy, vendor_sources_checked=deepcopy(list(_VENDOR_SEARCH)),
        vendor_search_date="2026-10-15", standard_source=_STANDARD,
        missing_vendor_evidence="Applicable complete AC table and supported MR settings not obtained")


def timing_profile_catalog() -> dict:
    """Return independent, JSON-compatible descriptions of the exact names."""
    return {"none": {"name": "none", "scope": "Leave the supplied configuration unchanged; no timing pack applied"},
            **{name: _pack(name) for name in _NAMES}}


def apply_hbm_timing_profile(config: "DramConfig", name: str = "reference") -> "DramConfig":
    """Apply a pack through the public reconfiguration contract.

    'none' leaves the supplied baseline unchanged; it does not undo already
    authored timings. Reapplying a named pack resets that pack's manual edits.
    """
    if not isinstance(name, str) or name not in (*_NAMES, "none"):
        raise ValueError("unknown HBM timing profile; choose none, fast, reference, slow, or ramulator2")
    from .memory_types import DramConfig
    if not isinstance(config, DramConfig):
        raise TypeError("HBM timing profiles require a DramConfig")
    if name == "none":
        return config
    from .physical_contract import reconfigure_physical_config
    return reconfigure_physical_config(config, overrides={"metadata": {
        "hbm_parameter_profile": {"name": name, "version": 1}}})


def _values(pack, clock, authored=None):
    authored = authored or {}
    values = {name: entry["input_value"] * (clock if entry["input_unit"] == "nCK" else 1.)
              for name, entry in pack["parameters"].items() if entry["input_unit"] in {"nCK", "ns"}}
    # Explicit PRE can occur on half CK; array minima are rounded upwards.
    for name in ("minimum_active_ns", "precharge_to_activate_ns"):
        values[name] = math.ceil(values[name] / (clock / 2)) * (clock / 2)
    values.update(authored)
    values["activate_cycle_ns"] = authored.get("activate_cycle_ns",
        values["minimum_active_ns"] + values["precharge_to_activate_ns"])
    interval_entry = pack["parameters"]["refresh_interval_ns"]
    reference_interval = interval_entry["input_value"] * (clock if interval_entry["input_unit"] == "nCK" else 1.)
    active_limit_interval = values["refresh_interval_ns"] or reference_interval
    values["maximum_active_ns"] = authored.get("maximum_active_ns", 9 * active_limit_interval)
    values["refresh_policy"] = authored.get("refresh_policy", "all_bank")
    phy = pack["phy_assumptions"]
    tdqss_ns = max(-.2, -.2 * clock)
    # Note 18 puts tDQSS inside its nCK term; BL=8 gives BL/4+0.5=2.5 CK.
    # Normalize the ns envelope, add the declared system guard band, then
    # round to a legal rising command edge, never to integer ns.
    raw_rtw = (values["read_latency_ns"] - values["write_latency_ns"]
               + 2.5 * clock + tdqss_ns + phy["twdqs2dq_o_max_ns"] + phy["tdqsqtra_max_ns"]
               + phy["system_turnaround_margin_ns"])
    if pack["name"] == "ramulator2":
        # Preserve upstream's placement of ceil: only the analog adjustment
        # is rounded; CL/BL/CWL are already whole cycles at the fixed point.
        tck_ps = clock * 1000
        correction = math.ceil((max(-2000., -2 * tck_ps) + 5 * tck_ps + 10 * (2500 + 20)) / (10 * tck_ps))
        raw_rtw = values["read_latency_ns"] + 2 * clock - values["write_latency_ns"] + correction * clock
    values["read_to_write_command_ns"] = authored.get("read_to_write_command_ns",
        math.ceil(max(0., raw_rtw) / clock) * clock)
    return values


def prepare_timing_profile(payload: Mapping, previous: Mapping, explicit_fields: set):
    """Recognize authored scalar edits before generic dependency normalization."""
    descriptor = (payload.get("metadata") or {}).get("hbm_parameter_profile")
    if descriptor is None:
        return None
    if (not isinstance(descriptor, Mapping) or descriptor.get("name") not in _NAMES
            or descriptor.get("version") != 1):
        raise ValueError("invalid hbm_parameter_profile; apply a named HBM timing profile")
    if (payload.get("generation") not in {"HBM3", "HBM3E"}
            or payload.get("control_mode") != "jedec_hbm"
            or payload.get("dies_per_stack") != 16 or payload.get("rows_per_bank") != 32768):
        raise ValueError("HBM timing packs require the explicit HBM3/E 32Gb 16-high organization")
    pack = _pack(descriptor["name"])
    profile_fields = set(pack["parameters"])
    old_clock = descriptor.get("input_command_clock_ns")
    if old_clock is None:
        return pack, {}
    if not isinstance(old_clock, (int, float)) or isinstance(old_clock, bool) or not math.isfinite(old_clock) or old_clock <= 0:
        raise ValueError("HBM timing profile input_command_clock_ns must be finite and positive")
    overridden = descriptor.get("overridden_fields", [])
    if not isinstance(overridden, (list, tuple)) or any(name not in profile_fields for name in overridden):
        raise ValueError("HBM timing profile has invalid overridden_fields")
    authored = {name: payload[name] for name in overridden}
    if previous:
        authored.update({name: payload[name] for name in explicit_fields & profile_fields})
    else:
        # Reconstruct former formula outputs from declared inputs, without a
        # state hash or saved scalar snapshot. Raw scalar edits stay authored.
        expected = _values(pack, old_clock, authored)
        provenance = (payload.get("metadata") or {}).get("parameter_provenance", {})
        for name in _DERIVED - {"refresh_policy"}:
            dependencies = provenance.get(name, {}).get("profile_dependencies")
            if isinstance(dependencies, Mapping):
                expected[name] = _values(pack, old_clock, {**authored, **dependencies})[name]
        for name, value in expected.items():
            actual = payload.get(name)
            same = actual == value if isinstance(value, str) else (
                isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-12))
            if not same:
                authored[name] = actual
    return pack, authored


def finish_timing_profile(result, meta, provenance, context):
    """Re-evaluate a selected pack after CK/geometry normalization."""
    if context is None:
        return
    pack, authored = context
    clock = result["command_clock_ns"]
    required_clock = pack.get("required_command_clock_ns")
    if required_clock is not None and not math.isclose(clock, required_clock, rel_tol=0, abs_tol=1e-12):
        raise ValueError("ramulator2 timing profile requires command_clock_ns=0.625 ns (6.4 Gb/s per pin); other operating points require a separately sourced timing profile")
    values = _values(pack, clock, authored)
    dependencies = {
        "activate_cycle_ns": {name: values[name] for name in ("minimum_active_ns", "precharge_to_activate_ns")},
        "maximum_active_ns": {"refresh_interval_ns": values["refresh_interval_ns"]},
        "read_to_write_command_ns": {name: values[name] for name in ("read_latency_ns", "write_latency_ns")},
    }
    for name, value in values.items():
        result[name] = value
        entry = deepcopy(pack["parameters"].get(name, {}))
        entry.update(status="explicit_assumption", vendor_value_status="unknown",
                     profile_name=pack["name"], output_value=value,
                     output_unit="policy" if name == "refresh_policy" else "ns")
        if name in authored:
            entry.update(source="physical_config_override", basis="Caller override of an engineering pack; not vendor calibration",
                         input_unit="policy" if name == "refresh_policy" else "ns", input_value=value,
                         rule="explicit_profile_override")
            if "source_kind" in entry:
                entry["source_kind"] = "caller_override"
        else:
            entry["rule"] = ("half_clock_ceil_of_analog_minimum" if name in {"minimum_active_ns", "precharge_to_activate_ns"}
                             else "cycles_times_command_clock" if entry.get("input_unit") == "nCK" else "constant_ns")
        if name in dependencies and name not in authored:
            entry.update(rule={"activate_cycle_ns": "tRAS_plus_tRP", "maximum_active_ns": "9_times_effective_or_reference_tREFI",
                               "read_to_write_command_ns": "jedec_rtw_with_assumed_phy"}[name],
                         profile_dependencies=dependencies[name])
        if name == "read_to_write_command_ns" and name not in authored:
            entry.update(phy_assumptions=deepcopy(pack["phy_assumptions"]), standard_location="Table 93 note 18 p.164")
            if pack["name"] == "ramulator2":
                entry["rule"] = "ramulator2_rtw_with_upstream_phy"
        if name in {"read_latency_ns", "write_latency_ns"}:
            entry["cycles"] = value / clock
            entry["input_unit"] = "ns" if name in authored else "cycles"
        provenance[name] = entry
    pack.update(input_command_clock_ns=clock, overridden_fields=sorted(authored))
    meta.update(hbm_parameter_profile=pack, target_hardware_calibrated=False,
                timing_basis=("Mapped Ramulator2 HBM3 engineering timing reference in this engine; no vendor AC calibration"
                              if pack["name"] == "ramulator2" else "JEDEC command subset with explicit engineering timing sensitivity; no vendor AC calibration"),
                unconfigured_timing_parameters=[name for name in meta.get("unconfigured_timing_parameters", []) if not result.get(name)])
