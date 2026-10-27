"""Micro-probes behind the memory-modeling deck.

Each case submits explicit AccessRequests to the simulator's own HBM4 / HBF /
SRAM cores (paper Table-I configurations, cold core per case unless noted) and
prints JSON. These are controlled mechanism probes, not serving results.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from exploring_hbf_paper_config import (build_paper_hbm_config, build_paper_nand_config,
                                        default_research_assumptions)
from heterollm_sim.dram_core import DramCore
from heterollm_sim.memory_types import AccessRequest, SramConfig
from heterollm_sim.nand_core import NandCore
from heterollm_sim.sram_core import SramCore

HBM = build_paper_hbm_config("HBM4")
HBF = build_paper_nand_config("HBF", assumptions=default_research_assumptions("HBF"))
SRAM = SramConfig(capacity_bytes=320_000_000, burst_bytes=64, interface_bandwidth_gb_s=80000.,
                  read_bandwidth_gb_s=80000., write_bandwidth_gb_s=80000., read_latency_ns=1.,
                  write_latency_ns=1., ports=8, max_outstanding_requests=256)
ROW = 256  # 405B, TP8: 1 KV head x 128 dim x BF16 per token, per layer, per rank (K or V)
SIZES = {"1 token": ROW, "16 token (4KiB)": 16 * ROW, "SHORT 1660": 1660 * ROW,
         "MID 5900": 5900 * ROW, "LONG 103500": 103500 * ROW}


def core(kind):
    return {"hbm": lambda: DramCore(HBM, capture_details=False),
            "hbf": lambda: NandCore(HBF, capture_details=False),
            "sram": lambda: SramCore(SRAM, capture_details=False)}[kind]()


def run(kind, requests, c=None):
    c = c or core(kind)
    started = time.perf_counter()
    results = c.submit_many(requests) if kind == "hbm" else tuple(c.submit(r) for r in requests)
    span = max(r.completion_ns for r in results) - min(r.arrival_ns for r in results)
    logical = sum(r.logical_bytes for r in results)
    sums = {}
    for r in results:
        for key in ("row_hits", "row_misses", "row_conflicts", "burst_count", "pages_read",
                    "pages_programmed", "host_transfer_bytes", "internal_transfer_bytes",
                    "read_cache_hits", "physical_read_bytes", "physical_write_bytes"):
            value = r.counters.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                sums[key] = sums.get(key, 0) + value
    return {"requests": len(results), "logical_bytes": logical,
            "transfer_bytes": sum(r.transfer_bytes for r in results),
            "span_ns": span, "gb_s": logical / span if span else None,
            "wall_s": round(time.perf_counter() - started, 3), **sums}


def single(kind, size, op="read"):
    return run(kind, [AccessRequest("r", op, 0, size)])


def scattered(kind, count, size, align, capacity, seed=7):
    rng = random.Random(seed)
    slots = rng.sample(range(capacity // align), count)
    return run(kind, [AccessRequest(f"r{i}", "read", slot * align, size) for i, slot in enumerate(slots)])


def adjacent(kind, count, size, stride=None):
    stride = stride or size
    return run(kind, [AccessRequest(f"r{i}", "read", i * stride, size) for i in range(count)])


out = {"hbm_config": {"lanes": HBM.lane_count, "banks_per_pc": HBM.bank_count, "burst_bytes": HBM.burst_bytes,
                      "row_bytes": HBM.row_bytes, "interface_gb_s": HBM.interface_bandwidth_gb_s},
       "hbf_config": {"channels": HBF.channels, "planes_per_die": HBF.planes_per_lun, "page_bytes": HBF.page_bytes,
                      "host_granularity": HBF.host_granularity_bytes, "page_read_ns": HBF.page_read_ns,
                      "page_program_ns": HBF.page_program_ns, "internal_gb_s_per_channel": HBF.internal_bandwidth_gb_s,
                      "host_read_gb_s": HBF.host_read_bandwidth_gb_s}}

# 1. One request of growing KV size, cold core.
out["size_sweep"] = {kind: {name: single(kind, size) for name, size in SIZES.items()}
                     for kind in ("hbm", "hbf", "sram")}

# 2. Same 1 MiB, different placement / useful bytes per burst.
MiB = 1 << 20
cap = HBM.effective_capacity_bytes
out["hbm_placement"] = {
    "contiguous 1MiB": single("hbm", MiB),
    "256 x 4KiB adjacent": adjacent("hbm", 256, 4096),
    "256 x 4KiB scattered": scattered("hbm", 256, 4096, 4096, cap),
    "32768 x 32B adjacent": adjacent("hbm", 32768, 32),
    "32768 x 32B scattered": scattered("hbm", 32768, 32, 32, cap),
    "32768 x 8B scattered": scattered("hbm", 32768, 8, 32, cap),
}
# 64 decode requests of a SHORT context, each its own page-aligned K buffer.
short = SIZES["SHORT 1660"]
stride = -(-short // 4096) * 4096
out["batch64_short"] = {kind: adjacent(kind, 64, short, stride) for kind in ("hbm", "hbf")}
out["sram_placement"] = {
    "contiguous 1MiB": single("sram", MiB),
    "32768 x 32B scattered": scattered("sram", 32768, 32, 64, SRAM.capacity_bytes),
}
out["hbf_placement"] = {
    "contiguous 1MiB": single("hbf", MiB),
    "256 x 4KiB scattered": scattered("hbf", 256, 4096, 4096, HBF.effective_capacity_bytes),
    "4096 x 256B scattered": scattered("hbf", 4096, 256, 4096, HBF.effective_capacity_bytes),
}

# 3. HBF page buffer: the next token's row lies in the page just read.
c = core("hbf")
first = c.submit(AccessRequest("a", "read", 0, ROW))
second = c.submit(AccessRequest("b", "read", ROW, ROW, arrival_ns=first.completion_ns))
out["hbf_cache"] = {"first_ns": first.latency_ns, "second_same_page_ns": second.latency_ns,
                    "second_hits": second.counters.get("read_cache_hits")}

# 4. HBF writes: one decode token append vs a full page.
out["hbf_write"] = {"1 token 256B": single("hbf", ROW, "write"),
                    "full 4KiB page": single("hbf", 4096, "write"),
                    "1 MiB": single("hbf", MiB, "write")}
out["hbm_write"] = {"1 token 256B": single("hbm", ROW, "write")}

print(json.dumps(out, indent=1, ensure_ascii=False))
