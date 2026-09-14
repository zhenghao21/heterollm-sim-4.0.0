"""Derive stage-scoped coefficients from a native_llama_profile artifact."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from heterollm_sim.calibration import NativeCalibrationProfile
from heterollm_sim.kernel_mapping import load_kernel_csv, map_kernel_profile
from heterollm_sim.serde import stable_hash
from native_llama_compare import _hardware_fingerprint

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("profile", type=Path); ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cuda-api-phase", type=Path, default=None,
                    help="可选 extract_cuda_api_phase 输出；仅用于一次性 phase boundary 证据")
    ap.add_argument("--runtime-fingerprint", default=None,
                    help="可选 llama.cpp-runtime-v1 fingerprint；缺省时沿用旧 server-command 摘要")
    ap.add_argument("--hardware-fingerprint", default=None,
                    help="可选 native_llama_compare 硬件 fingerprint；缺省时沿用 profile 摘要")
    a = ap.parse_args(); raw = json.loads(a.profile.read_text(encoding="utf-8")); t = raw.get("formal", {}).get("timings", {})
    def api_avg(names):
        rows = raw.get("stats", {}).get("api", {}).get("rows", []); total = calls = 0.0
        for row in rows:
            if str(row.get("Name", "")) in names:
                total += float(str(row.get("Total Time (ns)", "0")).replace(",", "")); calls += float(str(row.get("Num Calls", "0")).replace(",", ""))
        return total / calls if calls else None
    # ``d2h_ns_per_byte`` is a time-per-byte coefficient.  Nsight's summary
    # table exposes aggregate time and count but not a trustworthy byte total;
    # keep the value unknown unless the detailed trace below provides bytes.
    d2h = None
    trace_path = a.profile.with_suffix('.trace.json')
    if trace_path.exists():
        trace = json.loads(trace_path.read_text(encoding='utf-8'))
        d2h_events = [e for e in trace.get('events', []) if e.get('kind') == 'memcpy' and e.get('copy_kind') == 2]
        total_bytes = sum(int(e.get('bytes') or 0) for e in d2h_events)
        total_ns = sum(float(e.get('duration_ns') or 0.0) for e in d2h_events)
        d2h = total_ns / total_bytes if total_bytes > 0 else None
    kernel_csv = a.profile.with_name(a.profile.stem + ".kernel.csv")
    kernel_mapping = map_kernel_profile(load_kernel_csv(kernel_csv), source=str(kernel_csv.resolve())) if kernel_csv.exists() else None
    phase_boundary = None
    phase_boundary_policy = None
    if a.cuda_api_phase is not None:
        api_path = a.cuda_api_phase.resolve()
        api = json.loads(api_path.read_text(encoding="utf-8"))
        ranges = api.get("phase_ranges", {})
        rows = api.get("api", {})
        candidate = {}
        for phase in ("prefill", "decode"):
            invocation_count = int(ranges.get(phase, 0) or 0)
            phase_rows = rows.get(phase, {}) if isinstance(rows, dict) else {}
            total_ns = sum(float(item.get("total_ns", 0.0) or 0.0)
                           for item in phase_rows.values()
                           if isinstance(item, dict))
            if invocation_count <= 0 or total_ns <= 0.0:
                candidate = {}
                break
            candidate[phase] = total_ns / invocation_count
        if candidate:
            phase_boundary = candidate
            phase_boundary_policy = "one_task_per_phase_invocation"
    gguf_info = raw.get("gguf", {}).get("gguf", {})
    output = NativeCalibrationProfile(
        prompt_ns_per_token=(float(t["prompt_ms"]) * 1e6 / int(t["prompt_n"])) if t.get("prompt_n") else None,
        decode_ns_per_token=(float(t["predicted_ms"]) * 1e6 / (int(t["predicted_n"]) - 1)) if int(t.get("predicted_n", 0)) > 1 else None,
        launch_ns_per_call=api_avg({"cudaLaunchKernel", "cudaLaunchKernelExC_v11060"}),
        synchronize_ns_per_call=api_avg({"cudaStreamSynchronize"}),
        phase_boundary_ns_per_invocation=phase_boundary,
        phase_boundary_policy=phase_boundary_policy,
        d2h_ns_per_byte=d2h,
        source=str(a.profile.resolve()),
        source_sha256=hashlib.sha256(a.profile.read_bytes()).hexdigest(),
        kernel_stage_mapping=kernel_mapping,
        model_sha256=gguf_info.get("sha256"),
        hardware_fingerprint=a.hardware_fingerprint or (_hardware_fingerprint(raw.get("hardware", {})) if raw.get("hardware") else None),
        runtime_fingerprint=a.runtime_fingerprint or (stable_hash(raw.get("server_command", [])) if raw.get("server_command") else None),
        backend_version=raw.get("hardware", {}).get("llama_cpp") if isinstance(raw.get("hardware"), dict) else None,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(output.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(output.to_dict(), ensure_ascii=False, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
