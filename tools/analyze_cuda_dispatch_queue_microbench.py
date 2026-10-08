"""Fit declared queue training only, then report untouched holdout predictions.

This candidate intentionally retains failed or unsupported predictions. It does
not update runtime profiles and does not use any model/native latency.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics

from run_cuda_dispatch_queue_microbench import summarize

QUANTUM_NS = 2048.0


def read_case(directory, case):
    raw = json.loads(gzip.decompress((directory / (case["case_id"] + ".json.gz")).read_bytes()))
    summarize(raw, case)
    return raw


def regular_gap(body, tail):
    return math.ceil((body + tail) / QUANTUM_NS) * QUANTUM_NS - body


def fit_training(training):
    if not training or any(case.get("split") not in ("train_threshold", "train_payload") for case, _ in training):
        raise ValueError("queue fitting accepts declared training cases only")
    lower, upper = 0.0, float("inf")
    threshold_rows = []
    for case, raw in training:
        if case["split"] != "train_threshold":
            continue
        body = statistics.median(end - begin for sample in raw["samples"]
                                 for begin, end in zip(sample["node_begin_ns"], sample["node_end_ns"]))
        # The quantized period is an existing hypothesis. Modal periods avoid
        # fitting the sparse queue-stall pulses as a hardware launch quantum.
        bins = Counter(round((sample["node_begin_ns"][i] - sample["node_begin_ns"][i - 1]) / QUANTUM_NS)
                       for sample in raw["samples"] for i in range(2, case["node_count"]))
        multiple = bins.most_common(1)[0][0]
        if multiple < 1:
            raise ValueError("training does not support a positive launch cadence")
        low, high = (multiple - 1) * QUANTUM_NS - body, multiple * QUANTUM_NS - body
        lower, upper = max(lower, low), min(upper, high)
        threshold_rows.append({"case_id": case["case_id"], "actual_body_median_ns": body,
                               "modal_start_period_ns": multiple * QUANTUM_NS,
                               "tail_lower_exclusive_ns": low, "tail_upper_inclusive_ns": high})
    if not threshold_rows or not math.isfinite(upper) or lower >= upper:
        raise ValueError("no shared bounded tail interval fits the declared training")
    tail = (lower + upper) / 2
    families = defaultdict(list)
    for case, raw in training:
        families[case["extra_parameter_bytes"]].append((case, raw))
    packet_models = []
    for payload, entries in sorted(families.items()):
        zero_body = [(case, raw) for case, raw in entries if set(case["requested_bodies_ns"]) == {0}]
        if not zero_body:
            raise ValueError("packet family requires a declared zero-body training control")
        # Infer the pulse period using zero-body train data only, weighted to
        # prefer the fundamental period over a sparse harmonic of the same peak.
        edge_residuals = defaultdict(list)
        for case, raw in zero_body:
            for sample in raw["samples"]:
                for i in range(2, case["node_count"]):
                    body = sample["node_end_ns"][i - 1] - sample["node_begin_ns"][i - 1]
                    gap = sample["node_begin_ns"][i] - sample["node_end_ns"][i - 1]
                    edge_residuals[i].append(gap - regular_gap(body, tail))
        medians = {i: statistics.median(values) for i, values in edge_residuals.items()}
        scores = {}
        for period in range(2, 25):
            inside = [value for i, value in medians.items() if i % period == 0]
            outside = [value for i, value in medians.items() if i % period != 0]
            if not inside or not outside:
                continue
            scores[period] = (statistics.median(inside) - statistics.median(outside)) * math.sqrt(len(inside))
        period = max(scores, key=scores.get)
        pulses = {0: [], 1: []}
        per_case = []
        for case, raw in entries:
            local = {0: [], 1: []}
            for sample in raw["samples"]:
                for i in range(period, case["node_count"], period):
                    body = sample["node_end_ns"][i - 1] - sample["node_begin_ns"][i - 1]
                    gap = sample["node_begin_ns"][i] - sample["node_end_ns"][i - 1]
                    local[(i // period) % 2].append(gap - regular_gap(body, tail))
            for parity in (0, 1):
                pulses[parity].extend(local[parity])
            per_case.append({"case_id": case["case_id"], "pulse_even_ns": statistics.median(local[0]),
                             "pulse_odd_ns": statistics.median(local[1])})
        packet_models.append({"extra_parameter_bytes": payload, "actual_parameter_extent_bytes": 32 + payload,
                              "packet_period_nodes": period, "pulse_even_ns": statistics.median(pulses[0]),
                              "pulse_odd_ns": statistics.median(pulses[1]), "training_case_pulses": per_case})
    return {"quantum_ns": QUANTUM_NS, "tail_lower_exclusive_ns": lower, "tail_upper_inclusive_ns": upper,
            "tail_midpoint_ns": tail, "threshold_training": threshold_rows, "packet_models": packet_models}


def predict_case(raw, case, fitted):
    models = [entry for entry in fitted["packet_models"] if entry["extra_parameter_bytes"] == case["extra_parameter_bytes"]]
    if not models:
        return {"status": "unsupported_parameter_extent", "reason": "Training does not identify a packet-period rule for this new argument footprint; no nearest-family fallback."}
    packet = models[0]; period = packet["packet_period_nodes"]
    actual_span, predicted_span, actual_gap, predicted_gap, cadence_hits, cadence_total = [], [], [], [], 0, 0
    for sample in raw["samples"]:
        body = [end - begin for begin, end in zip(sample["node_begin_ns"], sample["node_end_ns"])]
        gaps = []
        for i in range(1, case["node_count"]):
            regular = regular_gap(body[i - 1], fitted["tail_midpoint_ns"])
            pulse = 0 if i % period else packet["pulse_odd_ns" if (i // period) % 2 else "pulse_even_ns"]
            gaps.append(regular + pulse)
            if gaps[-1] < 0:
                return {"status": "invalid_negative_candidate_gap", "reason": "Queue candidate cannot emit negative serial dispatch duration."}
            if i > 1 and i % period:
                observed = sample["node_begin_ns"][i] - sample["node_begin_ns"][i - 1]
                estimated = regular + body[i - 1]
                cadence_hits += abs(observed - estimated) <= 64
                cadence_total += 1
        actual_span.append(max(sample["node_end_ns"]))
        actual_gap.append(sum(sample["node_begin_ns"][i] - sample["node_end_ns"][i - 1] for i in range(1, case["node_count"])))
        predicted_gap.append(sum(gaps)); predicted_span.append(sum(body) + sum(gaps))
    observed, predicted = statistics.median(actual_span), statistics.median(predicted_span)
    observed_gap, predicted_gaps = statistics.median(actual_gap), statistics.median(predicted_gap)
    return {"status": "candidate_evaluated_not_qualified", "observed_span_ns": observed, "predicted_span_ns": predicted,
            "span_error_percent": 100 * (predicted / observed - 1), "observed_gap_ns": observed_gap,
            "predicted_gap_ns": predicted_gaps, "gap_error_percent": 100 * (predicted_gaps / observed_gap - 1),
            "regular_edge_cadence_agreement_within_64ns_percent": 100 * cadence_hits / cadence_total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--fixed-model", type=Path,
                        help="Evaluate a replication with this previously fitted model; do not fit any new parameters")
    args = parser.parse_args()
    plan = json.loads((args.directory / "experiment_plan.json").read_text(encoding="utf-8"))
    if args.fixed_model:
        prior = json.loads(args.fixed_model.read_text(encoding="utf-8"))
        if prior.get("schema") != "heterollm.cuda-queue-model-identification/v1" or prior.get("target_llm_latency_used") is not False:
            raise ValueError("fixed model must be prior independent queue identification")
        fitted = prior["fitted_from_train_only"]
    else:
        training = [(case, read_case(args.directory, case)) for case in plan["cases"] if case["split"].startswith("train_")]
        fitted = fit_training(training)
    # No holdout data is loaded until all fitted parameters are final.
    results = []
    for case in plan["cases"]:
        if not case["split"].startswith("holdout_"):
            continue
        raw = read_case(args.directory, case)
        results.append({"case_id": case["case_id"], "split": case["split"], "result": predict_case(raw, case, fitted)})
    controls = []
    for payload in (0, 256):
        off = next(case for case in plan["cases"] if case["case_id"] == f"observer_off_payload_{payload}")
        on = next(case for case in plan["cases"] if case["extra_parameter_bytes"] == payload and case["split"].startswith("train_") and set(case["requested_bodies_ns"]) == {0})
        off_raw, on_raw = read_case(args.directory, off), read_case(args.directory, on)
        off_time = statistics.median(sample["device_event_ns"] for sample in off_raw["samples"])
        on_time = statistics.median(sample["device_event_ns"] for sample in on_raw["samples"])
        controls.append({"extra_parameter_bytes": payload, "observer_off_event_ns": off_time, "observer_on_event_ns": on_time,
                         "difference_percent": 100 * (on_time / off_time - 1), "matched_environment_qualified": False})
    output = {"schema": "heterollm.cuda-queue-model-identification/v1", "target_llm_latency_used": False,
              "prediction_qualified": False, "fitted_from_train_only": fitted, "holdout_results": results,
              "parameter_estimation": "fixed_prior_model_no_refit" if args.fixed_model else "declared_training_only",
              "fixed_model_source": str(args.fixed_model.resolve()) if args.fixed_model else None,
              "observer_controls": controls, "limitations": [
                  ("The original model was fitted under concurrent simulator CPU work; this fixed-model replication records its own CPU environment separately."
                   if args.fixed_model else "Heavy concurrent simulator CPU work was active; driver supply pulse amplitudes need quiet replication."),
                  "Known synthetic parameter footprints have independently observed packet periods; a general size-to-packet mapping is not identified.",
                  "The quantized effective-body tail is an empirical probe-domain parameter, not a pure hardware dispatch latency.",
                  "CUPTI, native model latency, D2D costs, and Graph replay timings do not enter this fit."]}
    (args.directory / "queue_model_identification.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tail_interval_ns": [fitted["tail_lower_exclusive_ns"], fitted["tail_upper_inclusive_ns"]],
                      "packet_models": [{key: p[key] for key in ("extra_parameter_bytes", "packet_period_nodes", "pulse_even_ns", "pulse_odd_ns")} for p in fitted["packet_models"]],
                      "holdout_results": results, "observer_controls": controls}, indent=2))


if __name__ == "__main__":
    main()
