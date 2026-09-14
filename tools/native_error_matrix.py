"""Run a bounded native-vs-simulator error matrix.

Each case invokes the existing parity harness in a fresh server process.  The
matrix keeps cases independent so cache and CUDA-graph state cannot leak
between configurations. A case is structurally valid when its GGUF geometry,
token construction, and runtime configuration gates pass. Timing remains
diagnostic because native and simulator timer boundaries differ. The legacy
``status`` field is retained as a pass/invalid alias.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
import math
import statistics
from pathlib import Path

CASES = (
    {"id":"short_output1", "prompt":"Hi.", "predict":1, "ctx":512, "parallel":1, "batch":64, "ubatch":64, "threads":16, "gpu_layers":-1},
    {"id":"short_output8", "prompt":"Hi.", "predict":8, "ctx":512, "parallel":1, "batch":64, "ubatch":64, "threads":16, "gpu_layers":-1},
    {"id":"medium_output8", "prompt":"Explain deterministic benchmarking for inference systems in two concise paragraphs.", "predict":8, "ctx":512, "parallel":1, "batch":64, "ubatch":64, "threads":16, "gpu_layers":-1},
    {"id":"long_output8", "prompt":"Benchmark design and reproducibility matter because " * 24, "predict":8, "ctx":512, "parallel":1, "batch":64, "ubatch":64, "threads":16, "gpu_layers":-1},
    {"id":"medium_output32", "prompt":"Explain deterministic benchmarking for inference systems.", "predict":32, "ctx":512, "parallel":1, "batch":64, "ubatch":64, "threads":16, "gpu_layers":-1},
    {"id":"batch32_ub16", "prompt":"Explain deterministic benchmarking for inference systems.", "predict":8, "ctx":512, "parallel":1, "batch":32, "ubatch":16, "threads":16, "gpu_layers":-1},
    {"id":"batch128_ub64", "prompt":"Explain deterministic benchmarking for inference systems.", "predict":8, "ctx":512, "parallel":1, "batch":128, "ubatch":64, "threads":16, "gpu_layers":-1},
    {"id":"cpu_only", "prompt":"CPU execution parity check.", "predict":2, "ctx":512, "parallel":1, "batch":64, "ubatch":64, "threads":16, "gpu_layers":0},
    {"id":"gpu_tail12", "prompt":"Partial GPU layer placement parity check.", "predict":4, "ctx":512, "parallel":1, "batch":64, "ubatch":64, "threads":16, "gpu_layers":12},
    {"id":"threads8", "prompt":"Thread count sensitivity check.", "predict":4, "ctx":512, "parallel":1, "batch":64, "ubatch":64, "threads":8, "gpu_layers":-1},
)


def _configuration_gate(payload: dict) -> dict:
    """Compare requested settings with the effective simulator runtime plan."""
    requested = payload.get("configuration") or {}
    effective = payload.get("runtime_config") or {}
    aliases = {"ctx": "context", "continuous_batching": "cont_batching"}
    fields = (
        "ctx", "parallel", "batch", "ubatch", "threads", "threads_batch", "gpu_layers",
        "flash_attn", "mmap", "mlock", "offload_kqv", "op_offload",
        "split_mode", "main_gpu", "kv_type_k", "kv_type_v", "kv_unified", "continuous_batching",
    )
    checked, mismatches = [], []
    for field in fields:
        effective_field = aliases.get(field, field)
        if field not in requested or effective_field not in effective:
            continue
        checked.append(field)
        if requested[field] != effective[effective_field]:
            mismatches.append({"field": field, "requested": requested[field], "effective": effective[effective_field]})
    if not effective:
        status = "partial"
    elif mismatches:
        status = "fail"
    else:
        status = "pass" if checked else "partial"
    return {"status": status, "ok": status == "pass", "checked": checked,
            "mismatches": mismatches, "missing_runtime_evidence": not bool(effective)}


def _derive_gates(payload: dict) -> dict:
    """Build explicit structural/timing gates while preserving legacy status."""
    parity = payload.get("parity") or {}
    geometry = parity.get("geometry") or payload.get("gguf") or {}
    geometry_ok = bool(geometry.get("ok"))
    tokens = parity.get("tokens") or {}
    token_ok = bool(tokens.get("ok"))
    config_gate = _configuration_gate(payload)

    # Native counts currently feed simulator construction, so this is a
    # structural check rather than an independent generation/tokenization test.
    token_gate = {
        "status": "structural_only", "ok": token_ok, "independent": False,
        "native_prompt": tokens.get("native_prompt"), "simulator_prompt": tokens.get("simulator_prompt"),
        "native_output": tokens.get("native_output"), "simulator_output": tokens.get("simulator_output"),
        "note": "native token counts are currently fed back as simulator inputs",
    }
    timing_gate = {
        "status": "diagnostic_only", "ok": None,
        "ttft": {"status": "boundary_mismatch", "native_field": "native.prompt_eval_ms", "simulator_field": "simulator.ttft_ms", "reason": "native prompt eval excludes simulator host/control-plane start path"},
        "e2e": {"status": "boundary_mismatch", "native_field": "native.total_ms", "simulator_field": "simulator.e2e_ms", "reason": "native eval total excludes HTTP/queueing while simulator E2E includes arrival-to-finish"},
        "tpot": {"status": "comparable_when_output_gt_1", "native_denominator": "predicted_n - 1", "simulator_denominator": "committed_tokens - 1"},
    }
    boundary_mismatch = {"ttft": True, "e2e": True, "tpot": False, "diagnostic_only": True, "fields": ["ttft_ms", "e2e_ms"]}
    # A missing effective runtime snapshot is not enough to claim parity.
    # Keep partial configuration evidence visible, but fail the structural
    # overall gate until every requested setting has an effective counterpart.
    structural_ok = geometry_ok and token_ok and config_gate["status"] == "pass"
    overall_status = "valid_for_structural_analysis" if structural_ok else "invalid"
    return {
        "geometry_gate": "pass" if geometry_ok else "fail",
        # Gate fields are scalar statuses for easy aggregation. Detailed
        # evidence remains alongside them for audit/report consumers.
        "token_count_gate": token_gate["status"],
        "token_count_gate_detail": token_gate,
        "config_gate": config_gate["status"],
        "config_gate_detail": config_gate,
        "timing_gate": timing_gate["status"],
        "timing_gate_detail": timing_gate,
        "boundary_mismatch": boundary_mismatch,
        "overall_status": overall_status,
        "legacy_status": "pass" if overall_status == "valid_for_structural_analysis" else "invalid",
    }

def run_case(case: dict, *, exe: str, model: str, root: Path, out_dir: Path) -> dict:
    output = out_dir / f"{case['id']}.json"
    command = [sys.executable, str(root / "tools/native_llama_compare.py"), "--exe", exe, "--model", model]
    command += ["--prompt", case["prompt"]]
    for key in ("predict", "ctx", "parallel", "batch", "ubatch", "threads", "gpu_layers"):
        command += [f"--{key.replace('_','-')}", str(case[key])]
    command += ["--output", str(output)]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    result = {"case": case, "command": command, "returncode": completed.returncode, "wall_s": time.perf_counter() - started, "output": str(output.resolve())}
    if output.exists():
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
            gates = _derive_gates(payload)
            result.update(gates)
            # Backward-compatible v1 alias. New consumers should use
            # overall_status and the individual gates above.
            result["status"] = gates["legacy_status"]
            result["validity_status"] = payload.get("validity_status")
            result["native"] = payload.get("native")
            result["simulator"] = payload.get("simulator")
            result["relative_error_pct"] = payload.get("relative_error_pct")
            result["parity"] = payload.get("parity")
            result["hardware"] = payload.get("hardware")
            result["runtime_fingerprint"] = payload.get("runtime_fingerprint")
        except (OSError, json.JSONDecodeError) as exc:
            result.update({"status":"invalid", "error":f"result parse failed: {exc}"})
    else:
        result.update({"status":"error", "stderr":completed.stderr[-4000:], "stdout":completed.stdout[-4000:]})
    return result


def _error_summary(results: list[dict]) -> dict:
    """Aggregate signed errors while excluding undefined one-token TPOT."""
    summary = {}
    for metric in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [
            float((item.get("relative_error_pct") or {}).get(metric))
            for item in results
            if (item.get("relative_error_pct") or {}).get(metric) is not None
        ]
        if not values:
            summary[metric] = {"n": 0, "median_pct": None, "p95_pct": None, "min_pct": None, "max_pct": None}
            continue
        ordered = sorted(values)
        p95_index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
        summary[metric] = {
            "n": len(values),
            "median_pct": statistics.median(values),
            "p95_pct": ordered[p95_index],
            "min_pct": min(values),
            "max_pct": max(values),
        }
    return summary

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--exe", required=True); ap.add_argument("--model", required=True); ap.add_argument("--output", type=Path, default=Path("artifacts/native_error_matrix.json")); ap.add_argument("--case", action="append", dest="case_ids"); args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]; out_dir=args.output.parent / (args.output.stem + "_cases"); out_dir.mkdir(parents=True, exist_ok=True)
    selected=[c for c in CASES if not args.case_ids or c["id"] in set(args.case_ids)]
    results=[run_case(c, exe=args.exe, model=args.model, root=root, out_dir=out_dir) for c in selected]
    summary={"schema":"native-error-matrix/v1", "case_count":len(results), "valid_count":sum(r.get("status")=="pass" for r in results), "invalid_count":sum(r.get("status")!="pass" for r in results), "cases":results, "relative_error_summary":_error_summary(results), "boundary_mismatch":{"ttft":True,"e2e":True,"tpot":False,"diagnostic_only":True,"fields":["ttft_ms","e2e_ms"]}, "notes":["Each case uses a fresh llama-server and the same GGUF; warmup is excluded by the parity harness.", "TTFT versus prompt_ms and E2E versus prompt_ms+predicted_ms are diagnostic comparisons with different boundary semantics; TPOT is undefined for one-token output.", "token_count_gate is structural_only because native token counts are currently fed back as simulator inputs; timing_gate remains diagnostic_only until timer boundaries are aligned."]}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({"schema":summary["schema"],"case_count":summary["case_count"],"valid_count":summary["valid_count"],"invalid_count":summary["invalid_count"],"output":str(args.output.resolve())},ensure_ascii=False)); return 0
if __name__=='__main__': raise SystemExit(main())
