"""Fabricated, CPU-only mutation checks; not GPU measurement evidence."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import supplemental_gate as gate

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "stream_event_probe_v2"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def make_raw(cfg, mode, module):
    check = {"passed": True, "finite_all_outputs": True, "sample_count": 1,
             "samples": [{"m_index": 0, "n_index": 0, "actual": 1.0, "reference": 1.0, "pass": True}]}
    rows = []
    for phase, index in [("first_call", 0)] + [("warmup", i) for i in range(5)] + [("formal", i) for i in range(30)]:
        row = {"phase": phase, "index": index, "graph_computations": 64,
               "qpc_start": 0, "qpc_record_begin_start": 1, "qpc_record_begin_end": 2,
               "qpc_submit_start": 3, "qpc_submit_end": 70, "qpc_record_end_start": 71,
               "qpc_record_end_end": 72, "qpc_wait_start": 73, "qpc_wait_end": 100, "qpc_end": 100,
               "host_wall_ns": 100000, "host_per_graph_ns": 1562.5, "correctness": check,
               "event_envelope_ms": .09 if mode == "event" else None, "cuda_query_before_wait": 600}
        row.update({key: 0 for key in ("ggml_status", "cuda_submit_status", "cuda_wait_status", "cuda_begin_record_status", "cuda_end_record_status", "cuda_query_after_wait", "cuda_elapsed_status")})
        rows.append(row)
    doc = {"schema": "backend-stream-event-probe/v2", **{k: cfg[k] for k in ("M", "N", "K", "seed", "group")},
           "weight_format": cfg["quant"], "input_dtype": "F32", "output_dtype": "F32", "layout": "ordinary_contiguous_2d",
           "device": "cuda", "threads": 1, "cuda_index": 0, "status": "measured", "modules_stable": True,
           "supported": True, "wait_api": "ggml_backend_synchronize", "cache_policy": "same_buffers_repeated_hot_cache_no_flush",
           "graph_computations_per_batch": 64, "graph_compute_calls": 2304, "warmup_requested": 5, "formal_repeats_requested": 30,
           "control_mode": mode == "control", "qpc_frequency": 1000000, "driver_version": 13040, "runtime_version": 12080,
           "compute_capability_major": 12, "compute_capability_minor": 0,
           "timing_contract": {"id": "actual-backend-stream-batch-envelope/v2", "host_device_times_additive": False},
           "environment": {"GGML_CUDA_DISABLE_GRAPHS": "1", **{k: None for k in ("GGML_CUDA_FORCE_MMQ", "GGML_CUDA_FORCE_CUBLAS", "CUDA_VISIBLE_DEVICES", "LLAMA_TRACE_ANNOTATIONS", "GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_CUBLAS_COMPUTE_TYPE")}},
           "quantization": {"packed_weight_sha256": "b" * 64, "input_sha256": "c" * 64, "bytes": 32},
           "loaded_modules_before": [module], "loaded_modules_after": [module],
           "correctness_contract": {"absolute_tolerance": .05, "relative_tolerance": .03},
           "first_call_correctness": check, "final_correctness": check, "runs": rows}
    doc.update({k: "fabricated" for k in ("backend_name", "device_name", "device_description", "gpu_name", "pci_bus_id")})
    return doc


def fixture(base):
    probe, directory = base / "probe", base / "run"
    probe.mkdir(); directory.mkdir()
    for name in ("full_raw_audit.py", "assess.py", "summarize_matrix.py"):
        shutil.copyfile(SOURCE / name, probe / name)
    for name in ("invoke.ps1", "run_frozen_matrix.ps1", "stream-event-probe.exe"):
        (probe / name).write_text("fabricated no-execute fixture")
    configs = [{"id": f"cfg{i}", "group": "dev", "M": 1, "N": 1, "K": 896, "quant": "Q5_0", "seed": 20260914} for i in range(12)]
    protocol = {"configs": configs, "quality_gates": {"dispersion_p90_p10_max": 1.5, "event_control_wall_relative_difference_max": .2}}
    write_json(probe / "protocol.json", protocol)
    records = [gate.fingerprint(p) for p in sorted(probe.iterdir())]
    module = gate.fingerprint(probe / "stream-event-probe.exe")
    manifest = {"files": records, "executable": module}
    write_json(probe / "build_manifest.json", manifest)
    for name in ("identity_before.json", "identity_after.json"):
        write_json(directory / name, {"manifest_sha256": gate.fingerprint(probe / "build_manifest.json")["sha256"]})
    for name in ("device_before.csv", "device_after.csv", "processes_before.json", "compute_processes_before.csv", "background_load.json"):
        (directory / name).write_text("fabricated snapshot")
    execution_rows, assessment_rows = [], []
    with gate.frozen_tools(probe) as tools:
        for i, cfg in enumerate(configs):
            for mode in (("event", "control") if i % 2 == 0 else ("control", "event")):
                path = directory / f"{cfg['id']}.{mode}.json"
                write_json(path, make_raw(cfg, mode, module))
                execution_rows.append({"config": cfg["id"], "mode": mode, "output": str(path), "succeeded": True, "raw_exists": True, "error": None})
            assessed = tools["assess"].assess(make_raw(cfg, "event", module), make_raw(cfg, "control", module), cfg, protocol, manifest)
            path = directory / f"{cfg['id']}.assessment.json"
            write_json(path, assessed)
            assessment_rows.append({"config": cfg["id"], "path": str(path), "exists": True, "exit_code": 0})
        write_json(directory / "execution.json", {"runs": execution_rows, "assessments": assessment_rows, "device_after_exit": 0, "post_identity_error": None, "identity_unchanged": True, "full_raw_audit_exit": 0})
        write_json(directory / "full_raw_audit.json", tools["full_raw_audit"].audit(directory, protocol, manifest))
        write_json(directory / "matrix_summary.json", tools["summarize_matrix"].summarize(directory))
    return probe, directory, protocol


class SupplementMutationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name).resolve()
        self.probe, self.run, self.protocol = fixture(self.base)

    def tearDown(self):
        self.temp.cleanup()

    def test_valid_fixture_passes_without_writing_probe(self):
        before = {str(p): gate.fingerprint(p) for p in self.probe.rglob("*") if p.is_file()}
        result = gate.audit(self.probe, self.run)
        self.assertTrue(result["all_diagnostic_gates_passed"], result["integrity_issues"])
        self.assertEqual(result["event_rows_checked"], 432)
        self.assertFalse(result["calibration_eligible"])
        self.assertEqual(before, {str(p): gate.fingerprint(p) for p in self.probe.rglob("*") if p.is_file()})

    def test_impossible_first_warmup_and_formal_events_rejected(self):
        path = self.run / "cfg0.event.json"
        document = gate.load_json(path)
        for index in (0, 1, 6):
            changed = copy.deepcopy(document)
            changed["runs"][index]["event_envelope_ms"] = 1e9
            problems, _ = gate.containment_issues(changed)
            self.assertTrue(any(f"row {index}" in p for p in problems))

    def test_tolerance_and_two_qpc_ticks(self):
        document = {"qpc_frequency": 1000000, "runs": [{"event_envelope_ms": .11, "host_wall_ns": 100000}]}
        self.assertEqual(gate.containment_issues(document)[0], [])
        document["runs"][0]["event_envelope_ms"] = .111
        self.assertTrue(gate.containment_issues(document)[0])
        document["qpc_frequency"] = 100000
        self.assertEqual(gate.containment_issues(document)[0], [])

    def test_post_audit_raw_change_rejected(self):
        path = self.run / "cfg0.event.json"
        document = gate.load_json(path); document["unexpected"] = "mutation"
        write_json(path, document)
        result = gate.audit(self.probe, self.run)
        self.assertFalse(result["integrity_passed"])
        self.assertTrue(any("raw changed after audit" in p for p in result["integrity_issues"]))

    def test_duplicate_audit_mapping_rejected(self):
        path = self.run / "full_raw_audit.json"
        document = gate.load_json(path); document["rows"][1] = document["rows"][0]
        write_json(path, document)
        result = gate.audit(self.probe, self.run)
        self.assertTrue(any("duplicate" in p for p in result["integrity_issues"]))

    def test_forged_assessment_rejected_even_with_updated_summary_digest(self):
        path = self.run / "cfg0.assessment.json"
        document = gate.load_json(path); document["statistics"]["instrumented_to_control_wall_ratio"] = 99
        write_json(path, document)
        summary_path = self.run / "matrix_summary.json"
        summary = gate.load_json(summary_path)
        summary["assessments"][0]["sha256"] = gate.fingerprint(path)["sha256"]
        summary["assessments"][0]["statistics"] = document["statistics"]
        write_json(summary_path, summary)
        result = gate.audit(self.probe, self.run)
        self.assertTrue(any("current extraction" in p for p in result["integrity_issues"]))

    def test_protocol_edit_and_empty_manifest_rejected(self):
        path = self.probe / "protocol.json"
        document = gate.load_json(path); document["changed"] = True
        write_json(path, document)
        result = gate.audit(self.probe, self.run)
        self.assertTrue(any("frozen file changed" in p for p in result["integrity_issues"]))
        self.assertTrue(gate.verify_manifest(self.probe, {"files": []})[0])

    def test_missing_identity_and_wrong_reference_rejected(self):
        write_json(self.run / "identity_after.json", {})
        path = self.run / "execution.json"
        document = gate.load_json(path); document["assessments"][0]["path"] = str(self.run / "wrong.json")
        write_json(path, document)
        result = gate.audit(self.probe, self.run)
        self.assertTrue(any("manifest differs" in p for p in result["integrity_issues"]))
        self.assertTrue(any("execution assessment reference" in p for p in result["integrity_issues"]))

    def test_output_cannot_overwrite_or_enter_probe(self):
        old = self.base / "old.json"; old.write_text("preserve")
        for path in (old, self.probe / "new.json"):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                gate.main(["--probe", str(self.probe), "--run", str(self.run), "--output", str(path)])
        self.assertEqual(old.read_text(), "preserve")
        self.assertFalse((self.probe / "new.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
