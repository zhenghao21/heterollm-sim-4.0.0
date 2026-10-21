"""Capacity and representative-decode screening for the Exploring HBF study.

Every legal DP/TP/PP/EP strategy is kept, including rejected/failed strategies.
The event at the middle output age is only a screening observation. It has no
PD import, admission stall, warmup, throughput convergence or measured router
trace, and is NOT a steady-state optimum or final SLO acceptance. The caller
must run complete closed-loop serving on ranked candidates, shrinking batch
when that serving run fails the SLO. This script never starts the old scan.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import ceil, isfinite
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from heterollm_sim.paper_inference import (
    PaperExecutionAssumptions, PaperInferenceEngine, enumerate_parallelism, prepared_slot_count,
)
from heterollm_sim.paper_models import PaperPrecision, get_paper_model
from heterollm_sim.paper_serving_control import ServingRequest


def _slo_seconds(value, output_tokens: int) -> tuple[float, str]:
    if value == "offline":
        # The paper's offline requirement is whole-response completion within
        # 24 hours. This conversion divides by L_OUT, not decode-only L_OUT-1.
        return 24 * 60 * 60 / output_tokens, "offline_24_hours_divided_by_L_OUT"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError("slo_seconds must be positive and finite or 'offline'")
    return float(value), "explicit_per_token_seconds"


def _capacity_probe(model, hardware, parallel, batch, input_tokens, output_tokens, assumptions, precision):
    """Run only the maintained capacity method; allocate no memory actors."""
    engine = object.__new__(PaperInferenceEngine)
    engine.model, engine.hardware, engine.parallel = model, hardware, parallel
    engine.assumptions, engine.precision = assumptions, precision
    engine.batch_size, engine.input_tokens, engine.output_tokens = batch, input_tokens, output_tokens
    engine.prepared_slots = prepared_slot_count(batch, output_tokens, assumptions.prepared_lookahead)
    return engine.capacity_requirements()


def maximum_capacity_batch(model, hardware, parallel, *, upper_batch, input_tokens,
                           output_tokens, assumptions, precision=None):
    """Largest capacity-accepted legal batch at a fixed admission lookahead.

    Only the maintained private/paged capacity method runs; no GPU/memory
    actors, decode operators, PD transfers or serving requests are executed.
    The same monotone capacity predicate used by the main layout search is
    bisected on EP*microbatch quanta, with no percentage safety haircut.
    ``upper_batch`` is a caller-provided ceiling, not an assertion of capacity.
    A zero result means even the smallest legal batch fails this lookahead.
    The result records the exact endpoint and adjacent rejected quantum.
    """
    if isinstance(upper_batch, bool) or not isinstance(upper_batch, int) or upper_batch < 1:
        raise ValueError("upper_batch must be a positive integer")
    if (isinstance(input_tokens, bool) or not isinstance(input_tokens, int) or input_tokens < 1
            or isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 2):
        raise ValueError("capacity search requires positive integer input and at least two output tokens")
    if input_tokens + output_tokens > model.max_context_tokens:
        raise ValueError("capacity search workload exceeds the model context")
    if not isinstance(assumptions, PaperExecutionAssumptions):
        raise TypeError("fixed admission assumptions must be PaperExecutionAssumptions")
    precision = precision or PaperPrecision()
    quantum = parallel.ep * parallel.microbatches
    maximum_units = upper_batch // quantum
    cache = {}

    def probe(units):
        batch = units*quantum
        if batch not in cache:
            cache[batch] = _capacity_probe(model, hardware, parallel, batch,
                input_tokens, output_tokens, assumptions, precision)
        return cache[batch]

    chosen = 0
    capacity_at_minimum = probe(1) if maximum_units else None
    if maximum_units and capacity_at_minimum["accepted"]:
        low, high = 1, maximum_units
        while low < high:
            middle = (low+high+1)//2
            if probe(middle)["accepted"]:
                low = middle
            else:
                high = middle-1
        chosen = low*quantum
    next_units = chosen//quantum + 1
    next_rejected = probe(next_units) if maximum_units and next_units <= maximum_units else None
    if next_rejected is not None and next_rejected["accepted"]:
        raise RuntimeError("capacity predicate did not yield an adjacent rejected quantum")
    return {"batch_size": chosen, "batch_quantum": quantum,
        "upper_batch": upper_batch, "rounded_upper_batch": maximum_units*quantum,
        "prepared_lookahead": assumptions.prepared_lookahead,
        "capacity_at_minimum": capacity_at_minimum,
        "capacity_at_batch": cache.get(chosen),
        "next_rejected_batch": next_units*quantum if next_rejected is not None else None,
        "next_rejected_capacity": next_rejected,
        "checked_batches": list(cache),
        "execution_scope": "capacity_only_no_actors_or_timing_execution"}


def _capacity_upper_bound(model, parallel, input_tokens, output_tokens, precision, minimum_capacity):
    """Finite upper bound from active KV and linear workspace, without actors.

    Prepared imports can only lower it, and are handled by actual capacity
    probes. Resident weights are private; dense/shared EP replication and GQA
    duplication therefore remain exactly those of capacity_requirements().
    """
    maximum = input_tokens + output_tokens - 2
    if model.attention_chunk_size:
        chunk = model.attention_chunk_size
        chunk_minimum = (1 if input_tokens // chunk != maximum // chunk else input_tokens % chunk+1)
    else:
        chunk_minimum = 0
    element = 2 * model.local_kv_head_count(parallel.tp) * model.head_dim * precision.kv_bytes
    quantum = parallel.ep * parallel.microbatches
    upper = None
    for row in minimum_capacity["stages"]:
        stage = row["stage"]
        # A loose upper bound must not use the old full-sequence reservation:
        # active paged KV ages are staggered. Their minimum possible tokens
        # bound capacity above; exact cyclic occupancy is checked below.
        kv_per_request = sum(element * (input_tokens+1 if model.is_full_attention(layer) else chunk_minimum)
                             for layer in model.layer_indices(parallel.pp, stage))
        available = row["device_capacity_bytes"] - row["resident_weight_bytes_per_gpu"]
        bound = max(0, available) * parallel.ep // max(1, kv_per_request)
        workspace = row["workspace_peak_bytes_per_gpu"]
        if workspace:
            bound = min(bound, row["workspace_capacity_bytes"] * quantum // workspace)
        upper = bound if upper is None else min(upper, bound)
    return int(upper or 0) // quantum * quantum


def search_candidates(model, hardware, gpu_count, workload, assumptions=None, slo_seconds=.1, *, precision=None):
    """Return one row per legal strategy, sorted by screened per-GPU throughput.

    Batch is per independent DP replica and always divisible by EP and GPipe
    microbatches. Binary search assumes nondecreasing representative-event
    TPOT; it retains every tested point and does not claim a proof of global
    monotonicity or a globally best full-serving batch.
    """
    assumptions, precision = assumptions or PaperExecutionAssumptions(), precision or PaperPrecision()
    input_tokens, output_tokens = workload.input_tokens, workload.output_tokens
    if not isinstance(input_tokens, int) or input_tokens < 1 or not isinstance(output_tokens, int) or output_tokens < 3:
        raise ValueError("Screening requires positive integer input and at least three output tokens")
    if input_tokens + output_tokens > model.max_context_tokens:
        raise ValueError("workload exceeds the model context")
    limit, slo_basis = _slo_seconds(slo_seconds, output_tokens)
    age = ceil((output_tokens - 1) / 2)
    representative_context = input_tokens + age
    rows = []
    for parallel in enumerate_parallelism(model, gpu_count):
        quantum = parallel.ep * parallel.microbatches
        row = {
            "parallelism": asdict(parallel), "gpu_count": parallel.gpu_count,
            "batch_quantum": quantum, "capacity_upper_bound_batch": 0,
            "capacity_max_batch": 0, "candidate_batch": 0,
            "representative_context_tokens": representative_context,
            "representative_generated_decode_tokens": age,
            "slo_seconds": limit, "slo_basis": slo_basis,
            "screen_tpot_seconds": None, "screen_per_gpu_tokens_s": None,
            "screen_cluster_tokens_s": None, "trials": [], "capacity_checked_batches": [],
            "status": "capacity_rejected", "reason": "",
            "screening_scope": "one_representative_decode_event_without_PD_import_or_admission_stalls",
            "final_closed_loop_validated": False,
            "search_basis": "Every legal strategy retained; batch binary search assumes monotone representative-event TPOT; not a global steady-state optimum.",
        }
        capacity_cache = {}

        def capacity(batch):
            if batch not in capacity_cache:
                capacity_cache[batch] = _capacity_probe(model, hardware, parallel, batch,
                    input_tokens, output_tokens, assumptions, precision)
                row["capacity_checked_batches"].append(batch)
            return capacity_cache[batch]

        try:
            minimum = capacity(quantum)
            row["capacity_at_minimum"] = minimum
            upper = _capacity_upper_bound(model, parallel, input_tokens, output_tokens, precision, minimum)
            row["capacity_upper_bound_batch"] = upper
            if not minimum["accepted"] or upper < quantum:
                row["reason"] = minimum.get("reason") or "minimum batch exceeds private GPU capacity"
                rows.append(row)
                continue
            low, high = 1, upper // quantum
            while low < high:
                middle = (low + high + 1) // 2
                if capacity(middle * quantum)["accepted"]:
                    low = middle
                else:
                    high = middle - 1
            maximum = low * quantum
            row["capacity_max_batch"] = maximum
            row["capacity_at_maximum"] = capacity(maximum)
            if maximum < upper:
                row["capacity_next_rejected"] = capacity(maximum + quantum)

            trial_cache = {}

            def trial(batch):
                if batch in trial_cache:
                    return trial_cache[batch]
                engine = PaperInferenceEngine(model, hardware, parallel, batch_size=batch,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    assumptions=assumptions, precision=precision)
                requests = tuple(ServingRequest(slot, input_tokens, output_tokens,
                    generated_decode_tokens=age, slot_id=slot) for slot in range(batch))
                execution = engine.execute_decode(requests, (), 0.)
                seconds = execution.end_ns / 1e9
                if not isfinite(seconds) or seconds <= 0:
                    raise ValueError("representative decode returned invalid elapsed time")
                record = {"batch_size": batch, "tpot_seconds": seconds,
                          "per_gpu_tokens_s": batch / (seconds * parallel.gpus_per_replica),
                          "cluster_tokens_s": batch * parallel.dp / seconds,
                          "slo_accepted": seconds <= limit}
                row["trials"].append(record)
                trial_cache[batch] = record
                return record

            # Capacity and screening failures remain distinct. A real execution
            # exception aborts this strategy; it is never silently reclassified
            # as SLO rejection and searched around.
            if not trial(quantum)["slo_accepted"]:
                row["status"] = "slo_rejected"
                row["reason"] = "minimum legal batch exceeds representative-event SLO"
                rows.append(row)
                continue
            low, high = 1, maximum // quantum
            while low < high:
                middle = (low + high + 1) // 2
                if trial(middle * quantum)["slo_accepted"]:
                    low = middle
                else:
                    high = middle - 1
            accepted = trial(low * quantum)
            row.update(status="screened_candidate", reason="Complete closed-loop serving validation is still required.",
                       candidate_batch=accepted["batch_size"], screen_tpot_seconds=accepted["tpot_seconds"],
                       screen_per_gpu_tokens_s=accepted["per_gpu_tokens_s"],
                       screen_cluster_tokens_s=accepted["cluster_tokens_s"])
        except Exception as error:
            row["status"] = "execution_failed"
            row["reason"] = f"{type(error).__name__}: {error}"
            row["failure_phase"] = "decode_screening" if row["capacity_max_batch"] else "capacity_probe"
        rows.append(row)
    return sorted(rows, key=lambda row: (
        row["status"] != "screened_candidate", -(row["screen_per_gpu_tokens_s"] or 0),
        tuple(row["parallelism"].values())))


def main():
    from exploring_hbf_paper_config import hardware_by_id, workload_catalog
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llama3_405b", "llama4_maverick"), required=True)
    parser.add_argument("--hardware", choices=("HBM4", "HBF", "HBF+", "CONV", "CONV+"), required=True)
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument("--workload", choices=("SHORT", "MID", "LONG"), required=True)
    parser.add_argument("--slo", default="0.1", help="seconds per token, or offline for 24 hours / L_OUT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    slo = "offline" if args.slo == "offline" else float(args.slo)
    workload = next(item for item in workload_catalog() if item.workload_id == args.workload)
    rows = search_candidates(get_paper_model(args.model), hardware_by_id(args.hardware), args.gpus, workload,
                             slo_seconds=slo)
    output = {"model": args.model, "hardware": args.hardware, "gpu_count": args.gpus,
              "workload": asdict(workload), "candidates": rows,
              "scope": "screening_only_not_final_closed_loop_validation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Screened {len(rows)} strategies; final closed-loop validation required. Saved {args.output}")


if __name__ == "__main__":
    main()
