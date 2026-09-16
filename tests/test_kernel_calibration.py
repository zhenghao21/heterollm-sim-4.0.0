"""Synthetic fixtures only: no GPU, native LLM, or measured production coefficient."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from heterollm_sim.contracts import ResourceDemand, TaskCategory
from heterollm_sim.cost_models import CostPhase
from heterollm_sim.kernel_calibration import (
    DEVICE_BOUNDARY, OWNER_SCHEMA, PROFILE_SCHEMA, SOURCE_KIND,
    apply_kernel_calibration, canonical_kernel_key, load_kernel_calibration,
    resolve_kernel_calibration, runtime_identity_sha256, effective_hardware_sha256,
    create_kernel_execution_context, _distribution,
)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()



def _fixture_application(config, dll_ref):
    count = config["M"] * config["N"]
    samples = [{"m_index": i // config["N"], "n_index": i % config["N"], "actual": 1., "reference": 1.,
        "math_reference": 1., "path_reference": 1., "math_absolute_error": 0., "path_absolute_error": 0.,
        "math_pass": True, "path_pass": True, "pass": True} for i in range(count)]
    numeric = {"finite_all_outputs": True, "passed": True, "sample_count": count, "samples": samples,
               "math_passed": True, "path_max_absolute_error": 0.}
    runs = []
    phases = [("first_call", 0)] + [("warmup", i) for i in range(5)] + [("formal", i) for i in range(30)]
    fields = ["qpc_pre_sync_start", "qpc_pre_sync_end", "qpc_evict_start", "qpc_evict_submit_end", "qpc_evict_end",
        "qpc_nvtx_push_start", "qpc_nvtx_push_end", "qpc_start", "qpc_record_begin_start", "qpc_record_begin_end",
        "qpc_submit_start", "qpc_submit_end", "qpc_record_end_start", "qpc_record_end_end", "qpc_wait_start",
        "qpc_wait_end", "qpc_end", "qpc_nvtx_pop_start", "qpc_nvtx_pop_end", "qpc_validation_start", "qpc_validation_end"]
    for j, (phase, index) in enumerate(phases):
        r = {name: 10000 + j * 10000 + i * 100 for i, name in enumerate(fields)}
        wall = r["qpc_end"] - r["qpc_start"]
        label = (f"operator_surface/v1|phase={phase}|index={index}|op=MUL_MAT|M={config['M']}|N={config['N']}"
            f"|K={config['K']}|quant={config['quant']}|input=F32|output=F32|layout=contiguous2d|expected_path={config['expected_source_path']}")
        r.update(phase=phase, index=index, graph_computations=1, nvtx_label=label, host_wall_ns=wall,
            host_per_graph_ns=wall, event_envelope_ms=.001, correctness=deepcopy(numeric), cuda_query_before_wait=600)
        for name in ("ggml_status", "cuda_submit_status", "cuda_wait_status", "eviction_status", "cuda_begin_record_status",
                     "cuda_end_record_status", "cuda_query_after_wait", "cuda_elapsed_status"):
            r[name] = 0
        runs.append(r)
    modules = [{"path": str(dll_ref["path"]), "bytes": dll_ref["bytes"], "sha256": dll_ref["sha256"]}]
    return {"schema": "single-operator-surface-probe/v2", "timing_contract": {"id": "actual-backend-single-graph-envelope/v2"},
        "status": "measured", "control_mode": False, "M": config["M"], "N": config["N"], "K": config["K"],
        "weight_format": config["quant"], "input_dtype": "F32", "output_dtype": "F32", "layout": "ordinary_contiguous_2d",
        "device": "cuda", "cuda_index": 0, "threads": 1, "graph_computations_per_batch": 1, "graph_compute_calls": 36,
        "warmup_requested": 5, "formal_repeats_requested": 30, "nvtx_enabled": True, "gpu_l2_bytes": 1024,
        "cache_eviction_bytes": 128 << 20, "cache_policy": "untimed_read_write_sweep_at_least_4x_device_L2",
        "expected_source_path": config["expected_source_path"], "environment": {"GGML_CUDA_DISABLE_GRAPHS": "1", "FORCE_MMQ": None},
        "loaded_modules_before": modules, "loaded_modules_after": modules, "compute_capability_major": 12,
        "compute_capability_minor": 0, "qpc_frequency": 1000000000,
        "correctness_contract": {"absolute_tolerance": .05, "relative_tolerance": .03,
            "path_absolute_tolerance": .0001, "path_relative_tolerance": .00001, "reference_mode": "dual_math_and_source_path"},
        "first_call_correctness": numeric, "final_correctness": numeric, "runs": runs,
        "quantization": {"input_sha256": "1" * 64, "packed_weight_sha256": "2" * 64}}


def _fixture_sqlite(path, app, variant):
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE StringIds(id INTEGER,value TEXT)")
        db.executemany("INSERT INTO StringIds VALUES (?,?)", [(1, "cudaLaunchKernel"), (2, "quantize_q8_1"),
            (3, "mul_mat_vec_q"), (4, "void quantize_q8_1(float*)"), (5, variant)])
        db.execute("CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT)")
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER,returnValue INTEGER)")
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,deviceId INTEGER,streamId INTEGER,demangledName INTEGER,shortName INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,blockX INTEGER,blockY INTEGER,blockZ INTEGER,staticSharedMemory INTEGER,dynamicSharedMemory INTEGER)")
        db.execute("CREATE TABLE TARGET_INFO_GPU(id INTEGER,uuid TEXT,smCount INTEGER,l2CacheSize INTEGER,computeMajor INTEGER,computeMinor INTEGER)")
        db.execute("INSERT INTO TARGET_INFO_GPU VALUES (0,'SYNTHETIC',2,1024,12,0)")
        pid = 1 << 24
        for j, r in enumerate(app["runs"]):
            start = 10000 + j * 10000 + int(path.stem[1:]) * 1000000
            db.execute("INSERT INTO NVTX_EVENTS VALUES (?,?,?,?)", (start, start + 2000, pid + 7, r["nvtx_label"]))
            for role, duration in ((0, 10), (1, 100)):
                offset = 100 + role * 100
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?,?)", (start + offset, start + offset + 10, pid + 7, j * 2 + role, 1, 0))
                db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (start + offset + 20, start + offset + 20 + duration, pid, j * 2 + role, 0, 7, 4 + role, 2 + role, 1, 1, 1, 32, 1, 1, 0, 0))


def _fixture_raw_tables(path):
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        tables = {name: [dict(r) for r in db.execute('SELECT rowid AS evidence_rowid,* FROM "' + name + '"')]
                  for name, in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = {r["id"]: r["value"] for r in tables["StringIds"]}
    for r in tables["NVTX_EVENTS"]: r["resolved_text"] = r["text"]
    for r in tables["CUPTI_ACTIVITY_KIND_RUNTIME"]: r["name_text"] = strings[r["nameId"]]
    for r in tables["CUPTI_ACTIVITY_KIND_KERNEL"]:
        r["demangled_name_text"] = strings[r["demangledName"]]
        r["short_name_text"] = strings[r["shortName"]]
    return tables


@pytest.fixture
def bundle(tmp_path):
    refs = []
    def evidence(kind, value=None, *, identity=None, filename=None, binary=None):
        identity = identity or kind
        path = tmp_path / (filename or identity + ".json")
        payload = binary if binary is not None else _json_bytes({"synthetic_fixture": kind} if value is None else value)
        path.write_bytes(payload)
        ref = {"id": identity, "kind": kind, "path": str(path), "sha256": _sha(payload), "bytes": len(payload)}
        refs.append(ref)
        return ref
    def update(identity, value=None, binary=None):
        ref = next(r for r in refs if r["id"] == identity)
        p = Path(ref["path"])
        payload = binary if binary is not None else _json_bytes(value)
        p.write_bytes(payload); ref.update(sha256=_sha(payload), bytes=len(payload))
    def source_ref(ref): return {name: ref[name] for name in ("path", "sha256", "bytes")}
    for kind in ("probe_source", "extractor", "build_manifest"):
        evidence(kind)
    dll = evidence("runtime_binary", filename="fake.dll")
    source = evidence("native_source", filename="mmvq.cu", binary=b"// synthetic fixture only\nvoid mul_mat_vec_q(){ const int blocks_per_row_x = ncols_x / qk; for (int kbx = 0; kbx < blocks_per_row_x; kbx += blocks_per_iter) {} }")
    runtime = {"observed": True, "driver_version": "synthetic-driver", "dll_sha256": {"fake.dll": dll["sha256"]},
        "source_sha256": {"mmvq.cu": source["sha256"]}, "kernel_environment": {"GGML_CUDA_DISABLE_GRAPHS": "1", "FORCE_MMQ": None}}
    runtime["profile_sha256"] = runtime_identity_sha256(runtime)
    facts = {"gpu_uuid": "GPU-SYNTHETIC", "compute_capability": 1200, "sm_count": 2, "l2_bytes": 1024}
    effective = {"schema": "heterollm.kernel-effective-hardware/v1", "observed_device": facts,
        "simulator_configuration": {"compute": {"sm_count": 2}, "memory": {"HBM_GBps": 1, "HBF_GBps": 2},
            "cache": {"L2_bytes": 1024}, "scheduling": {"streams": 1}, "placement": {"device": "gpu0"}}}
    hardware_file = evidence("effective_hardware_profile", effective)
    hardware = {"observed": True, **facts, "effective_profile_sha256": effective_hardware_sha256(effective)}
    config = {"id": "synthetic-fixture-only", "group": "training", "quant": "Q5_0", "M": 4, "N": 2, "K": 896,
              "expected_source_path": "MMVQ_Q8_1_HALF"}
    variant = "void mul_mat_vec_q<(ggml_type)6, 4>()"
    key = {"op": "ggml_mul_mat", "role": "main", "m": 4, "n": 2, "k_logical": 896, "k_executed": 896,
        "activation_dtype": "F32", "weight_format": "Q5_0", "output_dtype": "F32", "accumulator_dtype": "F32",
        "layout": "contiguous_2d", "strides": {"activation_bytes": [4, 3584], "weight_bytes": [22, 616], "output_bytes": [4, 8]},
        "kernel_family": "cuda_mmvq", "kernel_variant": variant, "dispatch_signature": "mmvq-logical-K/v1:" + source["sha256"],
        "cache_protocol": "cold_sweep_ge_4_l2", "effective_hardware_sha256": hardware["effective_profile_sha256"],
        "runtime_sha256": runtime["profile_sha256"], "launch_geometry": {"grid": [1, 1, 1], "block": [32, 1, 1],
        "static_shared_bytes": 0, "dynamic_shared_bytes": 0}}
    policy = {"process_pairs": 3, "formal_calls_per_process": 30, "kernel_formal_p90_div_p10_max": 1.5,
        "profile_process_median_max_relative_deviation": .05, "profile_direct_host_median_max_relative_difference": .2,
        "direct_host_formal_p90_div_p10_max": 1.5, "target_sm_clock_mhz":2400,"sm_clock_tolerance_mhz":30,"maximum_formal_clock_bracket_gap_ms":25,"every_formal_interval_bracketed_required":True}
    evidence("protocol", {"schema": "operator-matrix-collection-protocol/v2", "configs": [config], "quality_policy": policy})
    owner = {"schema": OWNER_SCHEMA, "verified": True, "device_kind": "gpu", "device_id": "gpu0", "stream_id": "main",
        "owner_resource_id": "gpu0.kernel_stream.main", "stream_count": 1, "concurrent_kernels": False,
        "roles": ["conversion", "main", "fixup"], "effective_hardware_sha256": key["effective_hardware_sha256"],
        "runtime_sha256": key["runtime_sha256"], "cache_protocol": key["cache_protocol"]}
    evidence("execution_contract", owner)
    protocol_ref = next(r for r in refs if r["id"] == "protocol")
    freeze = evidence("collection_freeze", {"protocol_ref": source_ref(protocol_ref), "files": [source_ref(protocol_ref)],
        "probe_files": [source_ref(next(r for r in refs if r["id"] == "probe_source"))],
        "critical_tool_files": [source_ref(next(r for r in refs if r["id"] == "extractor"))]})
    clock_out=evidence("tool_log",identity="clock_stdout",binary=b"")
    clock_err=evidence("tool_log",identity="clock_stderr",binary=b"")
    clock=evidence("clock_receipt", {"schema":"operator-clock-control-receipt/v1","gpu_uuid":facts["gpu_uuid"],
        "target_sm_clock_mhz":2400,"sm_clock_tolerance_mhz":30,"requested_lock_min_mhz":2400,"requested_lock_max_mhz":2400,
        "lock_command_returncode":0,"restore_on_exit_planned":True,"created_utc":"2026-09-16T00:00:00Z",
        "command":["nvidia-smi","-lgc","2400,2400"],"stdout_ref":source_ref(clock_out),"stderr_ref":source_ref(clock_err)})
    boundary={"passed":True,"external_approved_sha256":freeze["sha256"],"required_sets_complete":True,"freeze_ref":source_ref(freeze)}
    pairs, reports, app_hashes = [], [], []
    for pair in range(3):
        app = _fixture_application(config, dll)
        record = {"pair": pair}
        report = {"pair": pair, "numerics_all_rows": True, "trace_chain_complete": True,
            "source_path_matches_observed_family": True, "trace_warning_free": True,
            "profile_kernel": _distribution([110.] * 30), "profile_host": _distribution([900.] * 30), "direct_host": _distribution([900.] * 30)}
        for mode in ("profile", "direct"):
            appref = evidence("raw_application", app, identity=f"p{pair}_{mode}")
            record[mode + "_app_id"] = appref["id"]; app_hashes.append(appref["sha256"])
            report[mode + "_raw"] = {"source": source_ref(appref)}
            receipt = evidence("hardware_runtime_observation", {"status": "completed", "returncode": 0,
                "gpu_identity": {"uuid": facts["gpu_uuid"], "driver_version": runtime["driver_version"]},
                "artifacts": [source_ref(appref)]}, identity=f"p{pair}_{mode}_receipt")
            telemetry=evidence("telemetry", {"samples":[{"qpc_ticks":t,"sm_mhz":{"status":0,"value":2400}} for r in app["runs"] for t in (r["qpc_start"],r["qpc_end"])]},identity=f"p{pair}_{mode}_telemetry")
            spec=evidence("process_spec", {"freeze":freeze["path"],"clock_control_binding":{"receipt_ref":source_ref(clock)},"run_identity":f"{pair}_{mode}"}, identity=f"p{pair}_{mode}_spec")
            start=evidence("process_start", {"spec":source_ref(spec),"freeze_before":boundary},identity=f"p{pair}_{mode}_start")
            receipt_doc=json.loads(Path(receipt["path"]).read_text());receipt_doc.update(spec_ref=source_ref(spec),external_approved_sha256=freeze["sha256"],
                freeze_after=boundary,clock_control_binding={"receipt_ref":source_ref(clock)},child_process_exited=True,telemetry_errors=[],
                process_pid=100+pair*2+(mode=="direct"),utc_started=f"2026-09-16T00:0{pair}:0{int(mode=='direct')}Z",qpc_launch_start=1000+pair*10+(mode=="direct"),qpc_process_complete=999999)
            receipt_doc["artifacts"].append(source_ref(telemetry));update(receipt["id"],receipt_doc)
            record[mode+"_telemetry_id"]=telemetry["id"];record[mode+"_spec_id"]=spec["id"];record[mode+"_start_id"]=start["id"]
            record[mode + "_observation_id"] = receipt["id"]
        path = tmp_path / f"p{pair}.sqlite"; _fixture_sqlite(path, app, variant)
        sqlref = {"id": f"p{pair}_sqlite", "kind": "raw_sqlite", "path": str(path), "sha256": _sha(path.read_bytes()), "bytes": path.stat().st_size}; refs.append(sqlref)
        rawref = evidence("raw_events", {"schema": "operator-matrix-raw-nsys/v1", "time_unit": "ns",
            "source_sqlite": source_ref(sqlref), "tables": _fixture_raw_tables(path)}, identity=f"p{pair}_raw")
        record.update(raw_events_id=rawref["id"], raw_sqlite_id=sqlref["id"])
        for log in ("profile_stdout", "profile_stderr", "export_stdout", "export_stderr"):
            logref = evidence("tool_log", identity=f"p{pair}_{log}", binary=b""); record[log + "_id"] = logref["id"]
        report["clock_domain_validated"]=True
        report["clock_domain_gates"]={mode:{"passed":True,"issues":[],"target_sm_clock_mhz":2400,"tolerance_mhz":30,"formal_intervals":30,"bracketed_intervals":30,"sm_clock_min_mhz":2400,"sm_clock_max_mhz":2400} for mode in ("profile","direct")}
        pairs.append(record); reports.append(report)
    evidence("numerical_validation", {"schema": "heterollm.kernel-numeric-recheck/v1", "raw_application_sha256": app_hashes,
        "all_calls": 216, "failed_rows": 0})
    evidence("quality_report", {"config": config, "measurement_cost_eligible": True, "issues": [], "fixed_policy": policy, "pairs": reports})
    evidence("measurement_bundle", {"schema": "heterollm.kernel-measurement-bundle/v1", "config": config,
        "collection_protocol_id": "protocol", "effective_hardware_id": "effective_hardware_profile", "pairs": pairs,
        "numeric_report_id": "numerical_validation", "quality_report_id": "quality_report", "dispatch_source_id": "native_source", "collection_freeze_id":"collection_freeze", "approved_freeze_sha256":freeze["sha256"], "clock_receipt_id":"clock_receipt"})
    entry = {"key": key, "status": "accepted", "device_ns": 100., "sample_count": 90, "rejection_reason": None,
        "evidence_ids": [r["id"] for r in refs], "gates": {"numerical": {"passed": True, "evidence_id": "numerical_validation"},
        "stability": {"passed": True, "evidence_id": "quality_report"}, "profiling": {"passed": True, "evidence_id": "quality_report"}}}
    data = {"schema": PROFILE_SCHEMA, "profile_id": "synthetic-fixture-only", "source_kind": SOURCE_KIND,
        "target_llm_latency_used": False, "measurement_boundary": DEVICE_BOUNDARY, "hardware": hardware, "runtime": runtime,
        "cache": {"observed": True, "protocol": key["cache_protocol"], "l2_bytes": 1024, "sweep_bytes": 128 << 20},
        "evidence_files": refs, "entries": [entry]}
    def write(candidate=None):
        payload = _json_bytes(data if candidate is None else candidate); path = tmp_path / "profile.json"; path.write_bytes(payload)
        return path, _sha(payload)
    def load(candidate=None):
        path, digest = write(candidate); return load_kernel_calibration(path, expected_sha256=digest)
    def owner_update(**changes): owner.update(changes); update("execution_contract", owner)
    def context(phase=None, query=None, effective_doc=None, catalog=None, **changes):
        phase, query = phase or _phase(), query or key
        catalog = catalog or {rid: {"device_id": "gpu0", "device_kind": "gpu", "kind": kind, "physical_owner": rid}
            for rid, kind in (("gpu0.tensor", "tensor"), ("gpu0.memory", "memory"), ("gpu0.kernel_stream.main", "kernel_envelope"))}
        args = dict(effective_hardware=effective_doc or effective, runtime_facts={k: v for k, v in runtime.items() if k != "profile_sha256"},
            device_id="gpu0", stream_id="main", stream_count=1, concurrent_kernels=False, resource_directory=catalog,
            invocations={"invocation": (phase, query)})
        args.update(changes)
        return create_kernel_execution_context(**args)
    return {"data": data, "key": key, "load": load, "write": write, "root": tmp_path, "owner_update": owner_update,
        "context": context, "update": update, "effective": effective, "pairs": pairs, "refs": refs}


def _phase(*, role="main", service=20.0, memory=40.0, **metadata):
    return CostPhase("gpu_gemm", TaskCategory.COMPUTE,
                     (ResourceDemand("gpu0.tensor", service, work_units=32),
                      ResourceDemand("gpu0.memory", memory, bytes_moved=128, energy_pj=64)),
                     {"target_component": "gpu0", "kernel_role": role, "kernel_stream_id": "main",
                      "kernel_measurement_boundary": DEVICE_BOUNDARY, **metadata})


def test_valid_exact_profile_and_same_phase_floor_preserve_resources(bundle):
    profile = bundle["load"]()
    original = _phase()
    result = apply_kernel_calibration(original, profile, bundle["key"], execution_contract_id="execution_contract", execution_context=bundle["context"](), invocation_id="invocation")
    assert result.name == original.name
    assert result.category == original.category
    assert result.demands[:-1] == original.demands
    assert result.demands[-1] == ResourceDemand("gpu0.kernel_stream.main", 100)
    assert result.service_ns == 100  # not 40 + 100
    assert sum(d.bytes_moved for d in result.demands) == 128
    assert result.energy_pj == original.energy_pj
    audit = result.metadata["kernel_calibration"]
    assert audit["conditional"] is True
    assert audit["exact_replacement"] is False
    assert audit["validated_llm_scope"] is False
    assert audit["resource_occupancy_calibrated"] is False
    assert audit["hardware_transfer_validated"] is False
    assert audit["cache_transfer_validated"] is False
    assert audit["prediction_source"] == "microbench_exact_device_floor_plus_analysis"
    assert "kernel_calibration" not in original.metadata


def test_floor_cannot_shorten_analytical_demand(bundle):
    original = _phase(memory=150)
    result = apply_kernel_calibration(original, bundle["load"](), bundle["key"], execution_contract_id="execution_contract", execution_context=bundle["context"](original), invocation_id="invocation")
    assert result.service_ns == 150
    assert result.metadata["kernel_calibration"]["analytical_exceeds_measured"] is True


def test_off_path_keeps_the_same_phase_object():
    original = _phase()
    assert apply_kernel_calibration(original, None, {}) is original
    assert resolve_kernel_calibration(None, {})["reason"] == "profile_disabled"


@pytest.mark.parametrize(("field", "value", "reason"), [
    ("effective_hardware_sha256", "a" * 64, "effective_hardware_mismatch"),
    ("runtime_sha256", "b" * 64, "runtime_mismatch"),
    ("cache_protocol", "hot_same_allocation_repeat", "cache_protocol_mismatch"),
    ("k_logical", 768, "unseen_joint_kernel_shape"),
    ("k_executed", 1024, "unseen_joint_kernel_shape"),
    ("weight_format", "Q8_0", "unseen_joint_kernel_shape"),
    ("role", "conversion", "unseen_joint_kernel_shape"),
    ("kernel_family", "cuda_mmq", "unseen_joint_kernel_shape"),
    ("kernel_variant", "other-template", "unseen_joint_kernel_shape"),
    ("m", 2, "unseen_joint_kernel_shape"),
    ("n", 1792, "unseen_joint_kernel_shape"),
    ("dispatch_signature", "other-rule", "unseen_joint_kernel_shape"),
])
def test_joint_key_mismatches_are_explicit_fallback(bundle, field, value, reason):
    profile = bundle["load"]()
    query = deepcopy(bundle["key"])
    query[field] = value
    result = resolve_kernel_calibration(profile, query)
    assert result["mode"] == "analytical_fallback"
    assert result["reason"] == reason
    assert result["device_ns"] is None
    original = _phase()
    adapted = apply_kernel_calibration(original, profile, query, execution_contract_id="execution_contract", execution_context=bundle["context"](), invocation_id="invocation")
    assert adapted.demands == original.demands
    assert adapted.service_ns == original.service_ns


def test_stride_is_part_of_exact_joint_key(bundle):
    query = deepcopy(bundle["key"])
    query["strides"]["activation_bytes"][1] += 4
    assert resolve_kernel_calibration(bundle["load"](), query)["reason"] == "unseen_joint_kernel_shape"


def test_runtime_digest_changes_with_dll_and_environment(bundle):
    runtime = deepcopy(bundle["data"]["runtime"])
    original = runtime.pop("profile_sha256")
    runtime["dll_sha256"]["fake.dll"] = "d" * 64
    dll_digest = runtime_identity_sha256(runtime)
    assert dll_digest != original
    query = deepcopy(bundle["key"])
    query["runtime_sha256"] = dll_digest
    assert resolve_kernel_calibration(bundle["load"](), query)["reason"] == "runtime_mismatch"
    runtime["kernel_environment"]["GGML_CUDA_DISABLE_GRAPHS"] = "0"
    assert runtime_identity_sha256(runtime) != dll_digest


def test_runtime_sha_cannot_hide_changed_dll_facts(bundle):
    bundle["data"]["runtime"]["dll_sha256"]["fake.dll"] = "c" * 64
    with pytest.raises(ValueError, match="does not bind"):
        bundle["load"]()


def test_hardware_effective_profile_requires_matching_file_evidence(bundle):
    data = bundle["data"]
    data["hardware"]["effective_profile_sha256"] = "a" * 64
    data["entries"][0]["key"]["effective_hardware_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="hardware profile SHA"):
        bundle["load"]()


def test_same_gpu_with_new_bandwidth_profile_falls_back(bundle):
    query = deepcopy(bundle["key"])
    query["effective_hardware_sha256"] = _sha(_json_bytes({"synthetic_bandwidth_gbps": 2}))
    assert resolve_kernel_calibration(bundle["load"](), query)["reason"] == "effective_hardware_mismatch"


@pytest.mark.parametrize("value", [None, True, 0, "false"])
def test_target_llm_latency_false_is_mandatory_and_typed(bundle, value):
    bundle["data"]["target_llm_latency_used"] = value
    with pytest.raises(ValueError, match="explicitly false"):
        bundle["load"]()


@pytest.mark.parametrize("field", ["hardware", "runtime", "cache"])
def test_missing_or_empty_identity_facts_fail_closed(bundle, field):
    bundle["data"][field] = {}
    with pytest.raises(ValueError):
        bundle["load"]()


@pytest.mark.parametrize("name", ["dll_sha256", "source_sha256", "kernel_environment"])
def test_empty_runtime_identity_maps_rejected(bundle, name):
    bundle["data"]["runtime"][name] = {}
    with pytest.raises(ValueError):
        bundle["load"]()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, True, "100", None])
def test_invalid_accepted_duration_rejected(bundle, value):
    bundle["data"]["entries"][0]["device_ns"] = value
    with pytest.raises(ValueError):
        bundle["load"]()


@pytest.mark.parametrize(("field", "value"), [("m", True), ("n", 0), ("k_logical", -1), ("k_executed", 1),
                                               ("runtime_sha256", None), ("layout", ""), ("role", ["main"])])
def test_malformed_query_cannot_hit(bundle, field, value):
    profile = bundle["load"]()
    query = deepcopy(bundle["key"])
    query[field] = value
    result = resolve_kernel_calibration(profile, query)
    assert result["reason"] == "invalid_kernel_query"
    assert result["device_ns"] is None


def test_model_and_prompt_identifiers_are_not_supported_key_dimensions(bundle):
    for field in ("model_name", "model_sha256", "prompt_fingerprint"):
        query = deepcopy(bundle["key"])
        query[field] = "cannot-be-used-to-index-an-answer"
        with pytest.raises(ValueError):
            canonical_kernel_key(query)


def test_duplicate_exact_key_is_rejected_even_with_different_entry_status(bundle):
    item = deepcopy(bundle["data"]["entries"][0])
    item.update(status="rejected", rejection_reason="unstable")
    bundle["data"]["entries"].append(item)
    with pytest.raises(ValueError, match="duplicate exact"):
        bundle["load"]()


def test_rejected_entries_remain_visible_in_profile_and_resolution(bundle):
    item = bundle["data"]["entries"][0]
    item.update(status="rejected", rejection_reason="profiling perturbation not resolved", device_ns=None, sample_count=0)
    item["gates"]["profiling"]["passed"] = False
    profile = bundle["load"]()
    assert len(profile.entries) == 1
    result = resolve_kernel_calibration(profile, bundle["key"])
    assert result["mode"] == "analytical_fallback"
    assert result["reason"] == "entry_rejected"
    assert "not resolved" in result["detail"]


@pytest.mark.parametrize("name", ["numerical", "stability", "profiling"])
@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_all_quality_gates_must_explicitly_pass_for_accepted_entries(bundle, name, value):
    bundle["data"]["entries"][0]["gates"][name]["passed"] = value
    with pytest.raises(ValueError):
        bundle["load"]()


def test_gate_cannot_reference_unrelated_evidence_kind(bundle):
    bundle["data"]["entries"][0]["gates"]["numerical"]["evidence_id"] = "raw_events"
    with pytest.raises(ValueError, match="correctly typed evidence"):
        bundle["load"]()


@pytest.mark.parametrize("mutation", ["empty", "missing_raw", "unknown_reference", "missing_file", "wrong_sha", "wrong_size"])
def test_evidence_refs_and_content_are_mandatory(bundle, mutation):
    data = bundle["data"]
    if mutation == "empty":
        data["evidence_files"] = []
    elif mutation == "missing_raw":
        data["evidence_files"] = [r for r in data["evidence_files"] if r["kind"] != "raw_events"]
    elif mutation == "unknown_reference":
        data["entries"][0]["evidence_ids"].append("missing")
    elif mutation == "missing_file":
        (bundle["root"] / "p0_raw.json").unlink()
    elif mutation == "wrong_sha":
        (bundle["root"] / "p0_raw.json").write_bytes(b"replacement")
    elif mutation == "wrong_size":
        data["evidence_files"][0]["bytes"] += 1
    with pytest.raises((ValueError, OSError)):
        bundle["load"]()


def test_profile_file_must_match_frozen_sha(bundle):
    path, digest = bundle["write"]()
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_kernel_calibration(path, expected_sha256=digest)


def test_duplicate_json_fields_rejected(bundle):
    path, _ = bundle["write"]()
    payload = path.read_bytes().replace(b'{"cache":', b'{"schema":"duplicate","cache":', 1)
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="duplicate JSON"):
        load_kernel_calibration(path, expected_sha256=_sha(payload))


def test_profile_cannot_be_accepted_without_loader_verification(bundle):
    profile = replace(bundle["load"](), evidence_files_verified=False)
    assert resolve_kernel_calibration(profile, bundle["key"])["reason"] == "profile_evidence_not_verified"


def test_loaded_evidence_and_key_maps_are_immutable(bundle):
    profile = bundle["load"]()
    with pytest.raises(TypeError):
        profile.hardware["effective_profile_sha256"] = "f" * 64
    with pytest.raises(TypeError):
        profile.runtime["dll_sha256"]["fake.dll"] = "f" * 64
    with pytest.raises(TypeError):
        profile.entries["new-key"] = None


@pytest.mark.parametrize(("changes", "reason"), [
    ({"verified": False}, "verified_single_stream_required"),
    ({"device_kind": "cpu"}, "gpu_execution_owner_required"),
    ({"stream_count": 2}, "verified_single_stream_required"),
    ({"stream_count": True}, "verified_single_stream_required"),
    ({"concurrent_kernels": True}, "verified_single_stream_required"),
    ({"concurrent_kernels": 0}, "verified_single_stream_required"),
    ({"owner_resource_id": "unowned.stream"}, "unowned_envelope_resource"),
    ({"roles": ["conversion"]}, "execution_role_mismatch"),
    ({"runtime_sha256": "a" * 64}, "execution_identity_mismatch"),
])
def test_phase_adapter_requires_verified_single_stream_owner(bundle, changes, reason):
    bundle["owner_update"](**changes)
    original = _phase()
    result = apply_kernel_calibration(original, bundle["load"](), bundle["key"], execution_contract_id="execution_contract", execution_context=bundle["context"](), invocation_id="invocation")
    assert result.demands == original.demands
    assert result.metadata["kernel_calibration_resolution"]["reason"] == reason


@pytest.mark.parametrize("contract", [None, "missing", "raw_events"])
def test_missing_or_unowned_execution_contract_rejected(bundle, contract):
    result = apply_kernel_calibration(_phase(), bundle["load"](), bundle["key"], execution_contract_id=contract)
    assert result.metadata["kernel_calibration_resolution"]["reason"] == "verified_execution_contract_required"
    assert len(result.demands) == 2


@pytest.mark.parametrize("metadata", [{"target_component": "gpu1"}, {"kernel_role": "fixup"},
                                      {"kernel_stream_id": "other"}, {"kernel_measurement_boundary": "host_graph_wall"}])
def test_phase_device_role_stream_and_boundary_must_match(bundle, metadata):
    original = _phase(**metadata)
    result = apply_kernel_calibration(original, bundle["load"](), bundle["key"], execution_contract_id="execution_contract", execution_context=bundle["context"](), invocation_id="invocation")
    assert result.demands == original.demands
    assert result.metadata["kernel_calibration_resolution"]["mode"] == "analytical_fallback"


@pytest.mark.parametrize("metadata", [{"native_calibration_applied": True}, {"native_memory_calibration_applied": True},
                                      {"cost_model": {"native_launch_calibration_applied": True}}])
def test_native_calibration_cannot_own_same_phase(bundle, metadata):
    original = _phase(**metadata)
    result = apply_kernel_calibration(original, bundle["load"](), bundle["key"], execution_contract_id="execution_contract", execution_context=bundle["context"](), invocation_id="invocation")
    assert result.demands == original.demands
    assert result.metadata["kernel_calibration_resolution"]["reason"] == "native_calibration_ownership_conflict"


def test_envelope_is_not_charged_twice(bundle):
    profile = bundle["load"]()
    first = apply_kernel_calibration(_phase(), profile, bundle["key"], execution_contract_id="execution_contract", execution_context=bundle["context"](), invocation_id="invocation")
    second = apply_kernel_calibration(first, profile, bundle["key"], execution_contract_id="execution_contract", execution_context=bundle["context"](), invocation_id="invocation")
    assert second.demands == first.demands
    assert second.service_ns == first.service_ns
    assert second.metadata["kernel_calibration_resolution"]["reason"] == "kernel_envelope_already_applied"
    assert second.metadata["kernel_calibration"] == first.metadata["kernel_calibration"]


def test_existing_owner_demand_not_duplicated(bundle):
    phase = _phase()
    phase = replace(phase, demands=phase.demands + (ResourceDemand("gpu0.kernel_stream.main", 12),))
    result = apply_kernel_calibration(phase, bundle["load"](), bundle["key"], execution_contract_id="execution_contract", execution_context=bundle["context"](), invocation_id="invocation")
    assert result.demands == phase.demands
    assert result.metadata["kernel_calibration_resolution"]["reason"] == "kernel_envelope_owner_already_demanded"


@pytest.mark.parametrize("name", ["kernel_launch", "launch_ns", "gpu_gemm"])
def test_host_launch_cannot_receive_device_floor(bundle, name):
    original = replace(_phase(), name=name,
                       demands=(ResourceDemand("gpu0.frontend", 17),))
    result = apply_kernel_calibration(original, bundle["load"](), bundle["key"], execution_contract_id="execution_contract", execution_context=bundle["context"](), invocation_id="invocation")
    assert result.demands == original.demands
    assert result.service_ns == 17
    assert result.metadata["kernel_calibration_resolution"]["reason"] == "launch_phase_cannot_receive_device_envelope"


@pytest.mark.parametrize("boundary", ["host_graph_wall", "kernel_span", "operator_wall", "host_submit", "host_wait"])
def test_non_kernel_boundaries_never_load(bundle, boundary):
    bundle["data"]["measurement_boundary"] = boundary
    with pytest.raises(ValueError, match="individual CUDA"):
        bundle["load"]()


def test_roles_without_independent_work_contract_cannot_be_promoted(bundle):
    entry = bundle["data"]["entries"][0]
    entry["key"]["role"] = "conversion"
    with pytest.raises(ValueError, match="executed K unproven"):
        bundle["load"]()


@pytest.mark.parametrize("category", [TaskCategory.COMMUNICATION, TaskCategory.SYNCHRONIZATION])
def test_non_compute_phase_cannot_be_relabelled_as_kernel(bundle, category):
    original = replace(_phase(), category=category)
    result = apply_kernel_calibration(original, bundle["load"](), bundle["key"], execution_contract_id="execution_contract", execution_context=bundle["context"](), invocation_id="invocation")
    assert result.demands == original.demands
    assert result.metadata["kernel_calibration_resolution"]["reason"] == "device_compute_phase_required"


def test_native_measurement_source_cannot_be_declared_synthetic(bundle):
    bundle["data"]["source_kind"] = "native_llm_inference"
    with pytest.raises(ValueError, match="non-independent"):
        bundle["load"]()


def test_missing_explicit_llm_fit_field_rejected(bundle):
    del bundle["data"]["target_llm_latency_used"]
    with pytest.raises(ValueError, match="missing or unsupported"):
        bundle["load"]()


def test_hot_profile_cannot_serve_cold_query(bundle):
    query = deepcopy(bundle["key"])
    bundle["data"]["cache"].update(protocol="hot_same_allocation_repeat", sweep_bytes=0)
    bundle["data"]["entries"][0]["key"]["cache_protocol"] = "hot_same_allocation_repeat"
    bundle["data"]["entries"][0].update(status="rejected", rejection_reason="no hot measurement evidence")
    profile = bundle["load"]()
    assert resolve_kernel_calibration(profile, query)["reason"] == "cache_protocol_mismatch"


def test_cache_sweep_protocol_does_not_accept_insufficient_or_bool_size(bundle):
    for size in (True, 1024):
        bundle["data"]["cache"]["sweep_bytes"] = size
        with pytest.raises(ValueError):
            bundle["load"]()


def test_runtime_source_hashes_must_have_matching_frozen_files(bundle):
    runtime = bundle["data"]["runtime"]
    runtime["source_sha256"]["fake.cu"] = "c" * 64
    facts = {key: value for key, value in runtime.items() if key != "profile_sha256"}
    runtime["profile_sha256"] = runtime_identity_sha256(facts)
    bundle["data"]["entries"][0]["key"]["runtime_sha256"] = runtime["profile_sha256"]
    with pytest.raises(ValueError, match="unverified DLL or source"):
        bundle["load"]()



def _read_evidence(bundle, identity):
    ref = next(r for r in bundle["refs"] if r["id"] == identity)
    return json.loads(Path(ref["path"]).read_text())


@pytest.mark.parametrize("identity,field,value", [
    ("quality_report", "measurement_cost_eligible", False),
    ("quality_report", "issues", ["failed_profiling"]),
    ("numerical_validation", "failed_rows", 1),
    ("numerical_validation", "all_calls", 30),
])
def test_correct_hash_but_failed_semantic_report_rejected(bundle, identity, field, value):
    doc = _read_evidence(bundle, identity); doc[field] = value; bundle["update"](identity, doc)
    with pytest.raises(ValueError): bundle["load"]()


@pytest.mark.parametrize("field,value", [("device_ns", 7), ("sample_count", 30)])
def test_entry_count_and_duration_rederived_from_original_sqlite(bundle, field, value):
    bundle["data"]["entries"][0][field] = value
    with pytest.raises(ValueError, match="original kernel intervals|median device"):
        bundle["load"]()


def test_raw_json_cannot_disagree_with_original_sqlite_even_with_correct_hash(bundle):
    raw = _read_evidence(bundle, "p0_raw")
    raw["tables"]["CUPTI_ACTIVITY_KIND_KERNEL"][1]["end"] += 1000
    bundle["update"]("p0_raw", raw)
    with pytest.raises(ValueError, match="original SQLite"): bundle["load"]()


def test_correct_hash_raw_numerical_failure_cannot_be_hidden_by_passed_report(bundle):
    app = _read_evidence(bundle, "p0_profile")
    app["runs"][0]["correctness"]["samples"][0]["actual"] = 2.
    bundle["update"]("p0_profile", app)
    with pytest.raises(ValueError, match="numeric rederived"): bundle["load"]()


def test_report_must_reference_the_same_raw_file(bundle):
    quality = _read_evidence(bundle, "quality_report")
    quality["pairs"][0]["profile_raw"]["source"]["sha256"] = "a" * 64
    bundle["update"]("quality_report", quality)
    with pytest.raises(ValueError, match="different raw artifact"): bundle["load"]()


def test_hardware_fields_bound_to_actual_sqlite_observation(bundle):
    bundle["data"]["hardware"]["sm_count"] = 9999
    with pytest.raises(ValueError, match="effective hardware observation"): bundle["load"]()


def test_failed_observation_receipt_denied_even_with_correct_hash(bundle):
    receipt = _read_evidence(bundle, "p0_profile_receipt")
    receipt["gpu_identity"]["uuid"] = "GPU-OTHER"
    bundle["update"]("p0_profile_receipt", receipt)
    with pytest.raises(ValueError, match="hardware/driver"): bundle["load"]()


def test_profiler_warning_text_is_not_ignored(bundle):
    bundle["update"]("p0_profile_stderr", binary=b"Warning: CUPTI failed")
    with pytest.raises(ValueError, match="reported diagnostic"): bundle["load"]()


def test_actual_multistream_raw_denied(bundle):
    pair = bundle["pairs"][0]
    ref = next(r for r in bundle["refs"] if r["id"] == pair["raw_sqlite_id"])
    path = Path(ref["path"])
    with sqlite3.connect(path) as db: db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId=8 WHERE rowid=2")
    ref.update(sha256=_sha(path.read_bytes()), bytes=path.stat().st_size)
    raw = _read_evidence(bundle, "p0_raw")
    raw["source_sqlite"] = {k: ref[k] for k in ("path", "sha256", "bytes")}
    raw["tables"] = _fixture_raw_tables(path); bundle["update"]("p0_raw", raw)
    with pytest.raises(ValueError, match="device/stream"): bundle["load"]()


def test_context_is_mandatory_despite_valid_metadata_and_owner(bundle):
    p = _phase()
    result = apply_kernel_calibration(p, bundle["load"](), bundle["key"], execution_contract_id="execution_contract")
    assert result.demands == p.demands
    assert result.metadata["kernel_calibration_resolution"]["reason"] == "actual_planner_runtime_context_required"


def test_cpu_actual_demands_with_gpu_metadata_cannot_construct_context(bundle):
    p = replace(_phase(), demands=(ResourceDemand("cpu0.compute", 30), ResourceDemand("cpu0.memory", 40)))
    with pytest.raises(ValueError, match="actual phase resources"): bundle["context"](p)


def test_real_phase_mutation_after_context_capture_falls_back(bundle):
    context = bundle["context"]()
    p = replace(_phase(), demands=(ResourceDemand("cpu0.compute", 30), ResourceDemand("cpu0.memory", 40)))
    result = apply_kernel_calibration(p, bundle["load"](), bundle["key"], execution_contract_id="execution_contract",
                                      execution_context=context, invocation_id="invocation")
    assert result.demands == p.demands
    assert result.metadata["kernel_calibration_resolution"]["reason"] == "actual_invocation_phase_or_key_mismatch"


def test_frontend_owner_is_denied(bundle):
    bundle["owner_update"](owner_resource_id="gpu0.frontend.calibration")
    p = _phase()
    result = apply_kernel_calibration(p, bundle["load"](), bundle["key"], execution_contract_id="execution_contract",
                                     execution_context=bundle["context"](), invocation_id="invocation")
    assert result.demands == p.demands
    assert result.metadata["kernel_calibration_resolution"]["reason"] == "unowned_envelope_resource"


@pytest.mark.parametrize("field", ["HBM_GBps", "HBF_GBps"])
def test_hardware_scan_cannot_reuse_stale_key_context(bundle, field):
    effective = deepcopy(bundle["effective"]); effective["simulator_configuration"]["memory"][field] *= 2
    with pytest.raises(ValueError, match="actual context identities"):
        bundle["context"](effective_doc=effective)
    key = deepcopy(bundle["key"]); key["effective_hardware_sha256"] = effective_hardware_sha256(effective)
    context = bundle["context"](query=key, effective_doc=effective)
    result = apply_kernel_calibration(_phase(), bundle["load"](), bundle["key"], execution_contract_id="execution_contract",
        execution_context=context, invocation_id="invocation")
    assert result.metadata["kernel_calibration_resolution"]["reason"] == "actual_context_identity_mismatch"


@pytest.mark.parametrize("changes", [{"stream_count": 2}, {"concurrent_kernels": True}, {"stream_count": True}])
def test_actual_runtime_multistream_context_rejected(bundle, changes):
    with pytest.raises(ValueError, match="single-stream"): bundle["context"](**changes)


def test_v1_declarations_only_profile_not_silently_promoted(bundle):
    bundle["data"]["schema"] = "heterollm.synthetic-kernel-profile/v1"
    with pytest.raises(ValueError, match="unsupported profile"): bundle["load"]()


def test_changed_launch_geometry_is_joint_domain_mismatch(bundle):
    key = deepcopy(bundle["key"]); key["launch_geometry"]["block"][0] *= 2
    assert resolve_kernel_calibration(bundle["load"](), key)["reason"] == "unseen_joint_kernel_shape"


def test_no_default_planner_activation():
    assert apply_kernel_calibration(_phase(), None, {}) is not None


def test_cuda_event_not_ready_is_a_valid_observation_not_a_runtime_failure(bundle):
    ref = next(r for r in bundle["refs"] if r["id"] == "p0_sqlite")
    path = Path(ref["path"])
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO StringIds VALUES (6,'cudaEventQuery')")
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (10100,10110,16777223,999999,6,600)")
    ref.update(sha256=_sha(path.read_bytes()), bytes=path.stat().st_size)
    raw = _read_evidence(bundle, "p0_raw")
    raw["source_sqlite"] = {k: ref[k] for k in ("path", "sha256", "bytes")}
    raw["tables"] = _fixture_raw_tables(path); bundle["update"]("p0_raw", raw)
    assert bundle["load"]().semantic_evidence_verified is True


def test_actual_capture_multiple_processes_denied(bundle):
    ref = next(r for r in bundle["refs"] if r["id"] == "p0_sqlite")
    path = Path(ref["path"])
    with sqlite3.connect(path) as db:
        db.execute("UPDATE NVTX_EVENTS SET globalTid=33554439 WHERE rowid=1")
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET globalTid=33554439 WHERE correlationId IN (0,1)")
        db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET globalPid=33554432 WHERE correlationId IN (0,1)")
    ref.update(sha256=_sha(path.read_bytes()), bytes=path.stat().st_size)
    raw = _read_evidence(bundle, "p0_raw")
    raw["source_sqlite"] = {k: ref[k] for k in ("path", "sha256", "bytes")}
    raw["tables"] = _fixture_raw_tables(path); bundle["update"]("p0_raw", raw)
    with pytest.raises(ValueError, match="device/stream"): bundle["load"]()

def test_extra_unowned_gpu_kernel_overlap_is_rejected(bundle):
    ref = next(r for r in bundle['refs'] if r['id']=='p0_sqlite');path=Path(ref['path'])
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (10205,10305,33554432,99999,0,8,5,3,1,1,1,32,1,1,0,0)')
    ref.update(sha256=_sha(path.read_bytes()),bytes=path.stat().st_size)
    raw=_read_evidence(bundle,'p0_raw');raw['source_sqlite']={k:ref[k] for k in ('path','sha256','bytes')};raw['tables']=_fixture_raw_tables(path);bundle['update']('p0_raw',raw)
    with pytest.raises(ValueError,match='extra overlapping'):bundle['load']()


def test_freeze_false_receipt_is_rejected(bundle):
    receipt=_read_evidence(bundle,'p0_profile_receipt');receipt['freeze_after']['passed']=False;bundle['update']('p0_profile_receipt',receipt)
    with pytest.raises(ValueError,match='freeze validation failed'):bundle['load']()


def test_explicit_clock_false_quality_cannot_pass(bundle):
    quality=_read_evidence(bundle,'quality_report');quality['pairs'][0]['clock_domain_validated']=False;bundle['update']('quality_report',quality)
    with pytest.raises(ValueError,match='clock domain failed'):bundle['load']()


def test_one_pair_replayed_three_times_rejected(bundle):
    doc=_read_evidence(bundle,'measurement_bundle');doc['pairs']=[dict(doc['pairs'][0],pair=i) for i in range(3)];bundle['update']('measurement_bundle',doc)
    with pytest.raises(ValueError,match='reuses'):bundle['load']()


def test_clock_original_readback_recomputed(bundle):
    telemetry=_read_evidence(bundle,'p0_profile_telemetry');telemetry['samples'][12]['sm_mhz']['value']=2000;bundle['update']('p0_profile_telemetry',telemetry)
    with pytest.raises(ValueError,match='measured SM clock'):bundle['load']()


def test_specific_json_limits_fit_large_raw_and_keep_small_contract_cap():
    from heterollm_sim.kernel_calibration import _evidence_json_limit
    assert _evidence_json_limit('raw_application') > 46*1024*1024
    assert _evidence_json_limit('execution_contract') == 16*1024*1024
    assert _evidence_json_limit('raw_application') <= 128*1024*1024


def test_sqlite_connections_are_closed_after_loader(bundle):
    bundle['load']()
    path=bundle['root']/'p0.sqlite';renamed=bundle['root']/'closed.sqlite';path.rename(renamed);renamed.rename(path)


@pytest.mark.parametrize("name,accepted", [
    ("cudaEventQuery_v3020", True), ("cudaEventQuery", True),
    ("cudaEventQuery_v3020extra", False), ("prefix_cudaEventQuery", False),
    ("cudaEventQuery_v", False), ("cudaStreamSynchronize_v3020", False),
])
def test_actual_versioned_event_query_status_600_grammar(bundle, name, accepted):
    ref = next(r for r in bundle['refs'] if r['id']=='p0_sqlite');path=Path(ref['path'])
    with sqlite3.connect(path) as db:
        db.execute('INSERT INTO StringIds VALUES (6,?)',(name,))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (10100,10110,16777223,999999,6,600)')
    ref.update(sha256=_sha(path.read_bytes()),bytes=path.stat().st_size)
    raw=_read_evidence(bundle,'p0_raw');raw['source_sqlite']={k:ref[k] for k in ('path','sha256','bytes')};raw['tables']=_fixture_raw_tables(path);bundle['update']('p0_raw',raw)
    if accepted:assert bundle['load']().semantic_evidence_verified
    else:
        with pytest.raises(ValueError,match='runtime API failed'):bundle['load']()


@pytest.mark.parametrize("label,known,accepted", [
    ("Info",True,True),("Verbose",True,True),("Warning",True,False),
    ("Error",True,False),("Fatal",True,False),("Unknown",True,False),("Info",False,False),
])
def test_actual_diagnostic_severity_enum_with_fourteen_events(bundle,label,known,accepted):
    ref = next(r for r in bundle['refs'] if r['id']=='p0_sqlite');path=Path(ref['path'])
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE ENUM_DIAGNOSTIC_SEVERITY_LEVEL(id INTEGER,label TEXT)')
        db.execute('INSERT INTO ENUM_DIAGNOSTIC_SEVERITY_LEVEL VALUES (?,?)',(1,label))
        db.execute('CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,text TEXT)')
        db.executemany('INSERT INTO DIAGNOSTIC_EVENT VALUES (?,?)',[(1 if known else 999,'original diagnostic '+str(i)) for i in range(14)])
    ref.update(sha256=_sha(path.read_bytes()),bytes=path.stat().st_size)
    raw=_read_evidence(bundle,'p0_raw');raw['source_sqlite']={k:ref[k] for k in ('path','sha256','bytes')};raw['tables']=_fixture_raw_tables(path);bundle['update']('p0_raw',raw)
    if accepted:
        profile=bundle['load']();assert len(profile.evidence['p0_raw'].document['tables']['DIAGNOSTIC_EVENT'])==14
    else:
        with pytest.raises(ValueError,match='diagnostic'):bundle['load']()
