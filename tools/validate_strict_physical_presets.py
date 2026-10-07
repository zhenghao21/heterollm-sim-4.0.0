"""Quantify physical preset execution drift against burst/page reference loops.

This measures software equivalence, not prediction accuracy against hardware.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from heterollm_sim.component_presets import _PRESETS
from heterollm_sim.architecture_presets import _LEGACY_ARCHITECTURE_PRESETS
from heterollm_sim.memory_types import AccessRequest, DramConfig, Operation
from heterollm_sim.physical_contract import PHYSICAL_MEMORY_COMPONENT_KINDS, require_physical_memory_config
from measure_memory_precision import compare_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    rows = []
    components = [(preset.preset_id, preset.component) for preset in _PRESETS]
    lpddr = next(component for preset in _LEGACY_ARCHITECTURE_PRESETS
                 for component in preset.hardware.components
                 if component.metadata.get("physical_memory_config", {}).get("kind") == "LPDDR")
    components.append(("legacy-lpddr/" + lpddr.component_id, lpddr))
    for name, component in components:
        if component.normalized_kind not in PHYSICAL_MEMORY_COMPONENT_KINDS:
            continue
        config = require_physical_memory_config(component)
        is_dram = isinstance(config, DramConfig)
        unit = config.burst_bytes if is_dram else config.page_bytes
        count = 10001 if is_dram else 5001
        size = count * unit - 7
        requests = [
            AccessRequest("read", Operation.READ, 1, size, 0.0),
            AccessRequest("write", Operation.WRITE, 1, size, 0.0),
            AccessRequest("read-after-write", Operation.READ, 1, size, 0.0),
            AccessRequest("later-read", Operation.READ, size - unit, unit, 1e9),
        ]
        row = compare_case(name, "dram" if is_dram else "nand", config, requests)
        rows.append(row)
        print(json.dumps({"preset": name, "max_duration_error": row["max_duration_error"],
                          "integer_counters_pass": row["integer_counters_pass"]}), flush=True)
    report = {
        "scope": "All public physical memory presets plus one legacy LPDDR; accelerated execution versus per-burst/per-page reference execution",
        "limitation": "Not hardware calibration or whole-model prediction error",
        "elapsed_seconds": time.monotonic() - started,
        "cases": rows,
        "max_duration_absolute_error_ns": max(row["max_duration_error"]["absolute_error"] for row in rows),
        "max_duration_relative_error": max(request["duration_relative_error"] for row in rows
                                           for request in row["requests"]),
        "discrete_states_and_counters_match": all(row["integer_counters_pass"] and row["discrete_state_pass"]
                                                    and row["serial_numeric_vs_python_integer_counters_pass"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["discrete_states_and_counters_match"]:
        raise SystemExit("physical state or counter mismatch")


if __name__ == "__main__":
    main()
