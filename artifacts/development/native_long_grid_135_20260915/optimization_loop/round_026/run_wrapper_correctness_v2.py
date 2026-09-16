"""One frozen synthetic GPU correctness run; no target LLM and no timing acceptance.

--check reads only files and never queries/launches a GPU. --execute requires the
reviewed campaign closure, no concurrent project test/simulation process, immutable target-capture prerequisite,
immutable wrapper build and hardware before/after evidence. The wrapper's numeric
result does not establish runtime equivalence, cache behavior or performance.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import psutil

P = Path(__file__).resolve().parent
LOOP = P.parent
CAP = P / "mmvq_wrapper_correctness"
INPUTS = CAP / "build_inputs.json"
BUILD = CAP / "build_receipt.json"
STATIC = CAP / "static_device_identity.json"
HOST = CAP / "cpu_self_test_result.json"
EXE = CAP / "mmvq_gpu_correctness.exe"
PREDECESSOR_DRIVER = P / "run_wrapper_correctness.py"
PREDECESSOR_DRIVER_SHA = "09c1d2e8f93218a6d867402d4d4a0f517acb59ed542d71851ac1f5b6a856096e"
PREDECESSOR_FINISH = P / "wrapper_correctness_run.0001/finish.json"
PREDECESSOR_FINISH_SHA = "3d2f2ce0f9a7739bec8cda05af9580f669a4d78df8b4141fbb01e717cbae0ff4"
PINS = {
    "build_inputs.json": "66f3923e9280c9d552aa5df76e6bc9b75ab81bf904558b8eabed5a9fdef1420d",
    "build_receipt.json": "56950f99829dc964cc98480a7bd6b8ca633c459e9a4a4248bab7e6d325511602",
    "static_device_identity.json": "5eff9211532604ebc099cb0eeb8105ae3276e76d0e373c6dd7cfbf5989eb5e59",
    "cpu_self_test_result.json": "f86a2f806227848c08541bb978e978c85b87a2ae4a4777c0644300557e73eef0",
    "mmvq_gpu_correctness.exe": "e85281dcd62f2c60dd9137f4ffa4f0563b9b907e26b9b68425486c1fb09a5ccd",
}
PRIOR_GATE = P / "run_target_capture_ex.py"
PRIOR_GATE_SHA = "a24f9614f5e98ffff0c9f4b8acf880edbb8dc1174721695e093d7d4ed4bf668e"
PRIOR_BUILDER = P / "mmvq_target_capture_ex/build.py"
PRIOR_BUILDER_SHA = "5ae315369fc825ea4a50c25c9f559a1b694eb3435fffd8ea87193123652e48b2"
CLOSED_CAMPAIGN = LOOP / "round_025/execution_closed.json"
CLOSED_CAMPAIGN_SHA = "c42fd307580b4473c111010d7282e31a3c6130018bac3c43d187402730f3fa20"
TARGET_FINISH = P / "target_capture_ex_run.0001/finish.json"
TARGET_FINISH_SHA = "f991bcf6d7df48bbf2e48baee4135297251bd2e3abb7ac766c9a90c32460bfd8"
HARDWARE = LOOP / "operator_microbench_v2/driver_device_properties.json"
HARDWARE_SHA = "351f81b15fd1706021fb449da7330000559b628a6cae5914a0cfb9bcd3783eaa"
SMI = Path(r"C:\Windows\System32\nvidia-smi.exe")
SMI_SHA = "7f98ec4f7563624b9817d3c83a3b42e74c847581a6aed7887604c44344977791"
GPU_FIELDS = ("index", "uuid", "name", "pci.bus_id", "driver_version", "memory.total",
              "pstate", "temperature.gpu", "clocks.sm", "clocks.mem", "utilization.gpu", "memory.used")
FIXTURE = dict(weight_type="Q5_0", ggml_type_id=6, K=4096, N=3072, M=1,
               padded_K=4096, q8_1_stride_blocks=128, seed=0x51354D31)


def now():
    return datetime.now(timezone.utc).isoformat()


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def ref(path):
    path = Path(path).resolve()
    require(path.is_file(), "evidence file missing: " + str(path))
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def check_ref(item):
    require(isinstance(item, dict) and set(item) == {"path", "sha256", "bytes"}, "exact nonempty reference required")
    require(isinstance(item["path"], str) and Path(item["path"]).is_absolute(), "absolute evidence path required")
    require(isinstance(item["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "complete lowercase SHA required")
    require(type(item["bytes"]) is int and item["bytes"] >= 0, "exact byte length required")
    actual = ref(item["path"])
    # The frozen C++ receipt uses forward slashes for its embedded manifest path.
    # Normalize only the absolute filesystem path; digest and exact byte count
    # remain mandatory and are never substituted from another evidence item.
    require(actual["sha256"] == item["sha256"] and actual["bytes"] == item["bytes"],
            "evidence identity changed: " + item["path"])
    return actual


def pinned(path, digest):
    actual = ref(path)
    require(actual["sha256"] == digest, "unreviewed frozen file: " + str(path))
    return actual


def write_new(path, payload):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def process_conflicts(records):
    conflicts = []
    for item in records:
        name = (item.get("name") or "").lower()
        args = item.get("cmdline")
        text = " ".join(args or []).replace("\\", "/").lower()
        native = (name.startswith(("llama-", "mmvq-", "mmvq_", "launch-recorder"))
                  or name in {"test-backend-ops.exe", "nsys.exe", "ncu.exe"}
                  or "microbench" in name or ("probe" in name and name.endswith(".exe")))
        simulation = any(v in text for v in (
            "predict_stable_native_dataset.py", "run_candidate.py", "native_grid_predict.py",
            "evaluate_identity_repair.py", "evaluate_candidate.py", "native_llama_compare.py",
            " -m heterollm_sim", "--run-recorder-only", "--run-correctness"))
        unknown = name.startswith("python") and args is None
        if native or simulation or unknown:
            # Record the reason and PID, not unrelated command-line content/secrets.
            conflicts.append({"pid": item["pid"], "name": item.get("name"),
                              "native_or_probe": native, "simulation_or_correctness": simulation,
                              "unknown_python_commandline": unknown})
    return conflicts


def assert_idle():
    conflicts = process_conflicts([p.info for p in psutil.process_iter(["pid", "name", "cmdline"])])
    require(not conflicts, "native/simulation process live or uninspectable: " + json.dumps(conflicts))


def campaign_finished():
    require(CLOSED_CAMPAIGN.is_file(), "reviewed campaign closure missing")
    pinned(CLOSED_CAMPAIGN, CLOSED_CAMPAIGN_SHA)
    # Reuse the existing complete closure check only after checking its code and
    # the builder imported at module scope. Never infer stopped from a lock file.
    pinned(PRIOR_GATE, PRIOR_GATE_SHA)
    pinned(PRIOR_BUILDER, PRIOR_BUILDER_SHA)
    spec = importlib.util.spec_from_file_location("r26_wrapper_prior_gate", PRIOR_GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.campaign_finished()


def verify_prerequisite_capture():
    pinned(TARGET_FINISH, TARGET_FINISH_SHA)
    finished = read_json(TARGET_FINISH)
    require(finished.get("status") == "synthetic_runtime_pair_qualified" and
            finished.get("returncode") == 0 and finished.get("runtime_pair_qualified") is True and
            finished.get("identity_unchanged") is True and finished.get("performance_parameter_admitted") is False,
            "reviewed target ExC capture is not qualified")
    for key in ("start_ref", "stdout_ref", "stderr_ref", "launches_ref"):
        check_ref(finished.get(key))
    raw = read_json(finished["launches_ref"]["path"])
    require(raw.get("status") == "qualified_runtime_pair" and raw.get("timed") is False and
            raw.get("performance_parameter_admitted") is False and raw.get("target_LLM_latency_used") is False,
            "target raw capture scope changed")
    pinned(HARDWARE, HARDWARE_SHA)
    hw = read_json(HARDWARE)
    require(raw["hardware"]["uuid"] == hw["gpu_uuid"], "target/static hardware UUID differs")
    return {"finish_ref": ref(TARGET_FINISH), "launches_ref": finished["launches_ref"],
            "hardware_ref": ref(HARDWARE), "target_hardware": raw["hardware"]}


def verify_predecessor():
    # The first attempt was refused before the executable started. Preserve its
    # exact driver and every retained failure artifact instead of rewriting it.
    pinned(PREDECESSOR_DRIVER, PREDECESSOR_DRIVER_SHA)
    pinned(PREDECESSOR_FINISH, PREDECESSOR_FINISH_SHA)
    failure = read_json(PREDECESSOR_FINISH)
    require(failure.get("status") == "rejected" and failure.get("returncode") is None and
            failure.get("numerical_qualification") is False, "predecessor failure scope changed")
    require(isinstance(failure.get("retained_files"), dict) and failure["retained_files"], "predecessor raw evidence missing")
    for item in failure["retained_files"].values():
        check_ref(item)
    require(not (PREDECESSOR_FINISH.parent/"correctness.json").exists() and
            not (PREDECESSOR_FINISH.parent/"child_started.json").exists(), "rejected predecessor acquired GPU execution evidence")
    return {"driver_ref": ref(PREDECESSOR_DRIVER), "failed_finish_ref": ref(PREDECESSOR_FINISH)}


def verify_build():
    predecessor = verify_predecessor()
    for name, digest in PINS.items():
        pinned(CAP/name, digest)
    inputs, build, static, host = map(read_json, (INPUTS, BUILD, STATIC, HOST))
    require(build.get("status") == "compiled_and_linked_runtime_unverified", "unexpected wrapper build status")
    require(build.get("gpu_driver_executed") is False and build.get("gpu_execution_performed") is False and
            type(build.get("timed_runs")) is int and build["timed_runs"] == 0 and
            build.get("r6_files_unchanged") is True, "build scope or R6 preservation not evidenced")
    require(check_ref(build["build_inputs"]) == ref(INPUTS), "build input manifest binding differs")
    check_ref(build["build_identity_header"])
    require(isinstance(inputs.get("input_refs"), list) and inputs["input_refs"], "empty build source closure")
    for item in inputs["input_refs"]:
        check_ref(item)
    for item in inputs["expected_runtime_dlls"].values():
        check_ref(item)
    check_ref(inputs["locked_target_cuda"])
    for label in ("gpu_correctness", "cpu_self_test"):
        variant = build["variants"][label]
        require(variant["compile"].get("returncode") == 0 and variant["link"].get("returncode") == 0,
                "compile/link failed: " + label)
        for key in ("executable", "object", "compiler_dependency_file"):
            check_ref(variant[key])
        for key in ("compile", "link", "import_inspection"):
            check_ref(variant[key]["log"])
        require(variant.get("compiler_discovered_includes"), "empty include dependency closure")
        for item in variant["compiler_discovered_includes"]:
            check_ref(item)
    require(build["variants"]["gpu_correctness"]["executable"] == ref(EXE), "GPU executable binding differs")
    cpu = build["variants"]["cpu_self_test"]
    require(cpu.get("cuda_imports_absent") is True, "CPU import qualification absent")
    imports = Path(cpu["import_inspection"]["log"]["path"]).read_text(encoding="utf-8").lower()
    require(not any(t in imports for t in ("cudart", "nvcuda", "ggml-cuda")), "CPU test imported CUDA")
    require(static.get("runtime_equivalence_verified") is False and static.get("gpu_execution_performed") is False and
            static.get("timed_runs") == 0 and static.get("performance_parameters_admitted") == 0,
            "static cubin evidence claims runtime/performance qualification")
    require(check_ref(static["executable"]) == ref(EXE) and
            check_ref(static["target_dll"]) == inputs["locked_target_cuda"], "static device-code identity mismatch")
    require(set(static.get("pairs", {})) == {"mmvq_main", "q8_1_conversion"}, "missing device-code pair")
    for pair in static["pairs"].values():
        a, b = check_ref(pair["target_cubin"]), check_ref(pair["wrapper_cubin"])
        require(pair.get("bytes_equal") is True and a["sha256"] == b["sha256"] and a["bytes"] == b["bytes"], "device code differs")
    require(host.get("status") == "cpu_reference_self_test_passed_gpu_unverified" and host.get("cpu_only") is True and
            host.get("cuda_api_calls_started") is False and host.get("gpu_execution_performed") is False and
            host.get("cpu_self_test_cuda_modules_absent") is True and host.get("runtime_equivalence_verified") is False,
            "host-only evidence missing or contradictory")
    require(check_ref(host["identity"]["build_inputs"]) == ref(INPUTS) and
            check_ref(host["identity"]["executable"]) == cpu["executable"], "host fixture covers another build")
    require(host.get("fixture") == FIXTURE and host["tolerance"].get("selected_before_observing_gpu_output") is True,
            "host fixture or preset tolerance differs")
    require(host["cpu_self_test"].get("status") == "passed" and host["cpu_self_test"]["output_bound_positive_fixture"].get("failed") == 0,
            "CPU reference self-test not passed")
    prereq = verify_prerequisite_capture()
    pinned(SMI, SMI_SHA)
    return {"inputs": inputs, "build": build, "static": static, "host": host,
            "prerequisite": prereq, "predecessor": predecessor, "hardware": read_json(HARDWARE),
            "frozen_refs": {"inputs": ref(INPUTS), "build": ref(BUILD), "static": ref(STATIC), "host": ref(HOST), "executable": ref(EXE)},
            "host_evidence_scope": {"recorded_CPU_status": host["status"],
                "recorded_cuda_api_calls_started": host["cuda_api_calls_started"],
                "recorded_cuda_modules_absent": host["cpu_self_test_cuda_modules_absent"],
                "external_host_process_returncode_verified": False,
                "note": "No separate host-process exit receipt exists; do not invent one or assert callback/runtime equivalence."}}


def environment_for(inputs):
    keep = {v.lower() for v in ("SystemRoot", "WINDIR", "COMSPEC", "TEMP", "TMP", "LOCALAPPDATA", "APPDATA", "USERPROFILE",
                                  "HOMEDRIVE", "HOMEPATH", "ProgramData", "ProgramFiles", "ProgramFiles(x86)")}
    env = {k: v for k, v in os.environ.items() if k.lower() in keep}
    system = os.environ.get("SystemRoot", r"C:\Windows")
    env["SystemRoot"] = system
    dirs = sorted({str(Path(v["path"]).parent) for v in inputs["expected_runtime_dlls"].values()})
    env["PATH"] = os.pathsep.join(dirs + [str(Path(system)/"System32"), system])
    env["GGML_CUDA_DISABLE_GRAPHS"] = "1"  # Requested policy, not an observed/runtime-qualified claim.
    return env, dirs


def parse_gpu_csv(text):
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if r and any(v.strip() for v in r)]
    require(len(rows) == 1 and len(rows[0]) == len(GPU_FIELDS), "expected one complete GPU property row")
    result = dict(zip(GPU_FIELDS, [v.strip() for v in rows[0]]))
    require(result["index"] == "0" and re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", result["uuid"]), "missing/wrong GPU index or UUID")
    require(result["driver_version"] and result["driver_version"] not in {"N/A", "[N/A]"}, "driver version unavailable")
    require(result["name"] and result["pci.bus_id"] and result["memory.total"].isdigit(), "static GPU identity incomplete")
    return result


def parse_compute_csv(text):
    rows = [r for r in csv.reader(io.StringIO(text)) if r and any(v.strip() for v in r)]
    result = []
    for row in rows:
        require(len(row) == 2 and row[0].strip().isdigit(), "compute-process inventory unavailable or malformed")
        result.append({"pid": int(row[0].strip()), "name": row[1].strip()})
    return result


def capture_hardware(label, output, env, expected):
    pinned(SMI, SMI_SHA)
    commands = {
        "gpu": [str(SMI), "--id=0", "--query-gpu=" + ",".join(GPU_FIELDS), "--format=csv,noheader,nounits"],
        "compute": [str(SMI), "--id=0", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
    }
    raw = {}
    for key, argv in commands.items():
        with (output/(label+"."+key+".stdout.log")).open("xb") as stdout, (output/(label+"."+key+".stderr.log")).open("xb") as stderr:
            p = subprocess.run(argv, env=env, stdout=stdout, stderr=stderr, timeout=30,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        require(p.returncode == 0, "hardware query failed: " + label + "/" + key)
        raw[key] = (output/(label+"."+key+".stdout.log")).read_text(encoding="utf-8", errors="strict")
    gpu, compute = parse_gpu_csv(raw["gpu"]), parse_compute_csv(raw["compute"])
    snapshot = {"schema": "r26-wrapper-hardware-snapshot/v2", "created_utc": now(), "tool_ref": ref(SMI),
                "commands": commands, "gpu": gpu, "reported_gpu_processes": compute,
                "process_inventory_scope": "raw nvidia-smi pid/name observations; graphics/CUDA activity category is not established",
                "background_GPU_isolation_verified": False, "timing_not_qualified": True,
                "raw_refs": {name: ref(output/(label+"."+name+".log"))
                             for name in ("gpu.stdout", "gpu.stderr", "compute.stdout", "compute.stderr")},
                "dynamic_clock_temperature_utilization_policy": "recorded only; no performance stability qualification"}
    write_new(output/(label+".json"), snapshot)
    require(gpu["uuid"].lower() == expected["gpu_uuid"].lower(), "actual GPU differs from frozen hardware")
    # This is numerical qualification. A nonempty NVML process inventory does
    # not falsify byte/output equality and does not identify an activity class.
    # The separate live-project-process guard still runs immediately around it.
    return snapshot


def compare_hardware(before, after, device, expected):
    require(before["tool_ref"] == after["tool_ref"], "hardware query tool changed")
    for key in ("index", "uuid", "name", "pci.bus_id", "driver_version", "memory.total"):
        require(before["gpu"][key] == after["gpu"][key], "hardware/driver changed during correctness run: " + key)
    for snapshot in (before, after):
        require(isinstance(snapshot.get("reported_gpu_processes"), list), "raw GPU process inventory missing")
        require(snapshot.get("background_GPU_isolation_verified") is False and snapshot.get("timing_not_qualified") is True,
                "unknown background activity must not claim isolation or timing qualification")
    require(isinstance(device, dict), "raw CUDA device identity missing")
    require(device.get("index") == 0 and device.get("uuid_hex", "").lower() == expected["gpu_uuid"].lower().removeprefix("gpu-").replace("-", ""), "raw CUDA device UUID mismatch")
    attrs = expected["attributes"]
    for key, attr in (("compute_major", "COMPUTE_CAPABILITY_MAJOR"), ("compute_minor", "COMPUTE_CAPABILITY_MINOR"),
                      ("sm_count", "MULTIPROCESSOR_COUNT"), ("warp_size", "WARP_SIZE"), ("l2_cache_bytes", "L2_CACHE_SIZE")):
        require(type(device.get(key)) is int and device[key] == attrs["CU_DEVICE_ATTRIBUTE_"+attr], "raw CUDA property mismatch: " + key)
    require(device.get("name") == before["gpu"]["name"], "raw CUDA device name differs")
    for key in ("cuda_driver_version", "cuda_runtime_version", "runtime_header_version"):
        require(type(device.get(key)) is int and device[key] > 0, "missing CUDA version: " + key)
    require(device["runtime_header_version"] == device["cuda_runtime_version"] == 12080, "wrapper runtime differs from pinned CUDA 12.8")
    require(device["cuda_driver_version"] >= device["cuda_runtime_version"], "CUDA driver/runtime incompatible")
    return {"same_device_and_driver_before_after": True, "raw_CUDA_matches_frozen_static_properties": True,
            "performance_stability_qualified": False, "background_GPU_isolation_verified": False,
            "timing_not_qualified": True,
            "reported_gpu_process_count_before": len(before["reported_gpu_processes"]),
            "reported_gpu_process_count_after": len(after["reported_gpu_processes"])}


def qualify_raw(path, context, output):
    raw = read_json(path)
    require(raw.get("schema") == "heterollm.mmvq-wrapper-correctness/v1" and
            raw.get("status") == "synthetic_correctness_passed_runtime_equivalence_unverified", "raw wrapper result missing or not numerically passed")
    require(raw.get("cpu_only") is False and raw.get("cuda_api_calls_started") is True and raw.get("gpu_execution_performed") is True,
            "raw evidence does not describe GPU execution")
    require(type(raw.get("timed_runs")) is int and raw["timed_runs"] == 0 and
            type(raw.get("performance_parameters_admitted")) is int and raw["performance_parameters_admitted"] == 0 and
            raw.get("target_llm_latency_used") is False and raw.get("runtime_equivalence_verified") is False,
            "raw result incorrectly claims timing/performance/runtime equivalence")
    require(raw.get("fixture") == FIXTURE and raw.get("main_shim_called") is True and
            raw.get("allocation_guards_intact") is True and raw.get("cuda_errors_or_validation_failures") == [], "raw execution/fixture/cleanup evidence incomplete")
    require(raw.get("tolerance") == context["host"]["tolerance"] and raw.get("fixture_hashes") == context["host"]["fixture_hashes"], "preset input or tolerance changed")
    require(check_ref(raw["identity"]["executable"]) == context["frozen_refs"]["executable"] and
            check_ref(raw["identity"]["build_inputs"]) == context["frozen_refs"]["inputs"], "raw result covers another build")
    require(check_ref(raw["identity"]["ggml_quantize_chunk_provider"]) == context["inputs"]["expected_runtime_dlls"]["ggml-base.dll"], "raw CPU quantizer provider mismatch")
    loaded = raw["identity"].get("loaded_modules")
    require(isinstance(loaded, list) and loaded, "raw loaded-module identity missing")
    required_modules = {"ggml-base.dll", "cudart64_12.dll"}
    for item in loaded:
        check_ref(item); name = Path(item["path"]).name.lower()
        require(name in context["inputs"]["expected_runtime_dlls"] and item == context["inputs"]["expected_runtime_dlls"][name], "unmatched runtime module")
        required_modules.discard(name)
    require(not required_modules, "required loaded runtime modules not evidenced")
    expected_artifacts = {
        "packed_weight_q5_0": (".weights.q5_0.bin", 8650752, "packed_q5_0"),
        "input_f32": (".input.f32.bin", 16384, "input_f32"),
        "expected_q8_1": (".expected.q8_1.bin", 4608, "expected_q8_1"),
        "reference_f64": (".reference.f64.bin", 24576, "reference_f64"),
        "bounds_f64": (".bounds.f64.bin", 24576, "bounds_f64"),
        "actual_q8_1": (".actual.q8_1.bin", 4608, "expected_q8_1"),
        "actual_output_f32": (".actual.output.f32.bin", 12288, None),
    }
    artifacts = raw.get("raw_artifacts")
    require(isinstance(artifacts, dict) and set(artifacts) == set(expected_artifacts), "raw binary evidence incomplete")
    for name, (suffix, size, fingerprint) in expected_artifacts.items():
        item = check_ref(artifacts[name])
        require(Path(item["path"]).resolve() == Path(str(path)+suffix).resolve() and Path(item["path"]).parent.resolve() == output.resolve(), "raw artifact escaped run directory")
        require(item["bytes"] == size, "raw artifact byte size differs: " + name)
        if fingerprint:
            require(item["sha256"] == context["host"]["fixture_hashes"][fingerprint], "raw artifact differs from frozen CPU reference: " + name)
    conversion = raw.get("conversion", {})
    require(type(conversion.get("bytes_compared")) is int and conversion["bytes_compared"] == 4608 and
            type(conversion.get("byte_mismatches")) is int and conversion["byte_mismatches"] == 0, "conversion not byte-exact")
    def unpack(name, fmt):
        return [x[0] for x in struct.iter_unpack(fmt, Path(artifacts[name]["path"]).read_bytes())]
    actual, reference, bounds = unpack("actual_output_f32", "<f"), unpack("reference_f64", "<d"), unpack("bounds_f64", "<d")
    max_error = max_ratio = 0.0; worst = -1
    for i, (a, r, b) in enumerate(zip(actual, reference, bounds)):
        require(all(map(math.isfinite, (a, r, b))) and b > 0, "nonfinite output/reference or invalid bound")
        error, ratio = abs(a-r), abs(a-r)/b
        require(error <= b, "independent output check exceeds preset bound at row " + str(i))
        max_error = max(max_error, error)
        if ratio > max_ratio:
            max_ratio, worst = ratio, i
    reported = raw.get("main_output", {})
    require(type(reported.get("tested")) is int and reported["tested"] == 3072 and
            type(reported.get("failed")) is int and reported["failed"] == 0 and reported.get("worst_index") == worst,
            "reported output counts or worst row differ from raw bytes")
    for key, expected_value in (("max_absolute_error", max_error), ("max_bound_ratio", max_ratio)):
        value = reported.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and math.isclose(value, expected_value, rel_tol=1e-12, abs_tol=1e-15), "reported output error differs: " + key)
    return raw, {"conversion_bytes_compared": 4608, "output_rows_checked": 3072,
                 "max_absolute_error": max_error, "max_bound_ratio": max_ratio, "worst_row": worst,
                 "numerical_qualification": True, "runtime_equivalence_qualified": False,
                 "performance_qualified": False, "performance_parameters_admitted": 0,
                 "background_GPU_isolation_verified": False, "timing_not_qualified": True}


def launch_process(argv, env, cwd, stdout, stderr, output):
    # Popen + wait has no implicit kill-on-interruption behavior. If observing the
    # child is interrupted, preserve its PID/state; a live child remains an idle
    # gate conflict and must not be restarted merely because observation ended.
    child = subprocess.Popen(argv, env=env, cwd=cwd, stdout=stdout, stderr=stderr,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        write_new(output/"child_started.json", {"created_utc": now(), "pid": child.pid, "argv": argv})
        code = child.wait()
        write_new(output/"child_exited.json", {"created_utc": now(), "pid": child.pid, "returncode": code,
                                              "natural_terminal_observed": True})
        return code
    except BaseException as error:
        observed_code = child.poll()
        try:
            write_new(output/"child_observation_error.json", {"created_utc": now(), "pid": child.pid,
                "observed_returncode": observed_code, "child_may_be_live": observed_code is None,
                "termination_requested": False, "error": type(error).__name__ + ": " + str(error)})
        finally:
            raise RuntimeError("child observation failed; no termination requested; pid=" + str(child.pid) +
                               ", observed_returncode=" + str(observed_code)) from error


def execute():
    previous = campaign_finished()
    context = verify_build()
    assert_idle()
    output = P / "wrapper_correctness_v2_run.0001"
    output.mkdir(exist_ok=False)
    result_path = output / "correctness.json"
    finish = {"schema": "r26-wrapper-correctness-finish/v2", "status": "rejected", "returncode": None,
              "numerical_qualification": False, "runtime_equivalence_qualified": False,
              "performance_qualified": False, "timed_runs": 0, "performance_parameters_admitted": 0,
              "background_GPU_isolation_verified": False, "timing_not_qualified": True}
    before = after = raw = None
    env = dirs = None
    try:
        env, dirs = environment_for(context["inputs"])
        argv = [str(EXE.resolve()), "--run-correctness", "--output", str(result_path.resolve())]
        write_new(output/"start.json", {"schema": "r26-wrapper-correctness-start/v2", "created_utc": now(),
            "driver_ref": ref(__file__), "prior_campaign_closure_ref": previous, "frozen_refs": context["frozen_refs"],
            "target_capture_prerequisite": context["prerequisite"], "host_evidence_scope": context["host_evidence_scope"],
            "preserved_predecessor": context["predecessor"],
            "background_GPU_isolation_verified": False, "timing_not_qualified": True,
            "background_inventory_policy": "retain all rows without activity-class inference or allowlists; numeric checks only",
            "concurrency_policy": "known project native/simulation/probe process guard remains enforced",
            "argv": argv, "runtime_search_dirs": dirs, "requested_environment": {"GGML_CUDA_DISABLE_GRAPHS": env["GGML_CUDA_DISABLE_GRAPHS"]},
            "environment_policy": "explicit OS minimum; no inherited injection, FORCE flags, profiler settings or secrets",
            "cache_state_verified": False, "observed_graph_policy_verified": False,
            "target_LLM_execution_allowed": False, "timing_calibration_allowed": False})
        before = capture_hardware("hardware_before", output, env, context["hardware"])
        assert_idle()
        # No timeout or process termination: retain the native synthetic child's natural terminal.
        with (output/"stdout.log").open("xb") as stdout, (output/"stderr.log").open("xb") as stderr:
            returncode = launch_process(argv, env, dirs[0], stdout, stderr, output)
        finish["returncode"] = returncode
        require(returncode == 0, "wrapper process failed: " + str(returncode))
        require(result_path.is_file(), "raw correctness result missing")
        raw, independent = qualify_raw(result_path, context, output)
        finish["raw_numerical_checks"] = independent
    except Exception as error:
        finish["execution_error"] = type(error).__name__ + ": " + str(error)
    try:
        if env is not None:
            after = capture_hardware("hardware_after", output, env, context["hardware"])
        require(before is not None and after is not None and raw is not None, "complete hardware/numeric triplet unavailable")
        finish["hardware_verification"] = compare_hardware(before, after, raw.get("device"), context["hardware"])
        finish["hardware_identity_verified"] = True
    except Exception as error:
        finish["hardware_identity_verified"] = False
        finish["hardware_error"] = type(error).__name__ + ": " + str(error)
    try:
        after_context = verify_build()
        require(after_context["frozen_refs"] == context["frozen_refs"] and campaign_finished() == previous, "frozen build or campaign changed")
        assert_idle()
        finish["identity_unchanged"] = True
    except Exception as error:
        finish["identity_unchanged"] = False
        finish["identity_error"] = type(error).__name__ + ": " + str(error)
    if ("execution_error" not in finish and finish.get("raw_numerical_checks", {}).get("numerical_qualification") is True
            and finish["hardware_identity_verified"] and finish["identity_unchanged"]):
        finish["status"] = "synthetic_wrapper_numerically_qualified"
        finish["numerical_qualification"] = True
    finish["finished_utc"] = now()
    finish["retained_files"] = {p.name: ref(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_new(output/"finish.json", finish)
    require(finish["status"] == "synthetic_wrapper_numerically_qualified", "wrapper correctness rejected; saved failure terminal retained")
    print(json.dumps({"status": finish["status"], "performance_qualified": False, "performance_parameters_admitted": 0,
                      "background_GPU_isolation_verified": False, "timing_not_qualified": True,
                      "finish_ref": ref(output/"finish.json")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.check:
        context = verify_build()
        print(json.dumps({"status": "frozen_wrapper_build_verified", "gpu_execution_performed": False,
                          "host_evidence_scope": context["host_evidence_scope"],
                          "background_GPU_isolation_verified": False, "timing_not_qualified": True,
                          "execution_requires_closed_campaign_project_mutex_and_hardware_checks": True}))
    else:
        execute()

if __name__ == "__main__":
    main()
