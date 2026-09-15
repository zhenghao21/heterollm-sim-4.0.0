"""Synthetic provenance-chain regression tests; never execute native code."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.llama_runtime_source_binding import (
    SourceBindingError, bind_llama_cuda_op_offload_contract,
    derive_llama_cuda_op_offload_contract_from_sources, verify_llama_runtime_source_binding,
)
from heterollm_sim.runtime_adapters import derive_llama_cuda_op_offload_contract


SCHEDULER = """
void choose() {
    sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer);
    ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor);
}
static enum ggml_status ggml_backend_sched_compute_splits() {
    ggml_backend_tensor_copy(input, input_cpy);
    cpy_tensor_async(input_backend, split_backend, input, input_cpy);
}
"""
CUDA = """
static int64_t get_op_batch_size() {
    switch (op->op) { case GGML_OP_MUL_MAT: return op->ne[1]; }
}
static bool ggml_backend_cuda_device_offload_op() {
    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}
static bool ggml_backend_cuda_device_supports_op() {
    switch (op->op) {
    case GGML_OP_MUL_MAT:
        if (a->nb[0] != ggml_element_size(a)) return false;
        switch (a->type) { case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_XS: return true; }
    case GGML_OP_OUT_PROD: return false;
    }
}
const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;
static size_t ggml_backend_cuda_reg_get_device_count() { return ctx->devices.size(); }
static ggml_backend_t ggml_backend_cuda_device_init_backend() { return ggml_backend_cuda_init(ctx->device); }
"""
OPERATORS = """
struct ggml_tensor * ggml_mul_mat() { return ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne); }
struct ggml_tensor * ggml_get_rows() { enum ggml_type type = GGML_TYPE_F32; }
"""
PARSER = """
add_arg(common_arg(
    {"--op-offload"}, {"--no-op-offload"},
    [](params, bool value) { params.no_op_offload = !value; }
));
"""
PARAMS = "cparams.op_offload        = !params.no_op_offload;"
CONTEXT = "cparams.op_offload = params.op_offload; sched.reset(ggml_backend_sched_new(a, b, c, cparams.op_offload));"
MODULES = ("ggml-base.dll", "ggml-cuda.dll", "ggml.dll", "llama.dll", "llama-common.dll", "llama-server-impl.dll", "llama-server.exe")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((value if isinstance(value, str) else json.dumps(value, indent=2)).encode("utf-8"))
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


class Fixture:
    def __init__(self, root):
        self.root = root
        self.base, self.overlay, self.native_root = root/"base", root/"annotation", root/"native"
        self.build, self.abuild, self.nbin = self.base/"build", self.overlay/"build", self.native_root/"build/bin"
        self.evidence = self.overlay/"evidence"
        self.sources = {"scheduler": self.base/"ggml/src/ggml-backend.cpp", "operators": self.base/"ggml/src/ggml.c",
                        "cuda": self.overlay/"ggml/src/ggml-cuda/ggml-cuda.cu"}
        for role, value in (("scheduler", SCHEDULER), ("operators", OPERATORS), ("cuda", CUDA)):
            write(self.sources[role], value)
        self.cli = {"parser": self.base/"common/arg.cpp", "params": self.base/"common/common.cpp"}
        for role, value in (("parser", PARSER), ("params", PARAMS)):
            write(self.cli[role], value)
        self.context = self.overlay/"src/llama-context.cpp"
        write(self.context, CONTEXT)
        self.headers = {"files": {}}
        for relative in ("ggml/include/ggml.h", "ggml/include/ggml-backend.h", "ggml/src/ggml-backend-impl.h",
                         "ggml/src/ggml-cuda/common.cuh", "common/common.h"):
            path = self.base/relative
            value = "bool no_op_offload = false;" if relative == "common/common.h" else "// synthetic fixed header"
            self.headers["files"][str(path)] = write(path, value)["sha256"]
        self.commands = []
        ninja_lines, objects = [], {"ggml-base.dll": [], "llama-common.dll": []}
        for path in [self.sources["scheduler"], self.sources["operators"], *self.cli.values()]:
            module = "llama-common.dll" if path.parent.name == "common" else "ggml-base.dll"
            output = self.build/(module+".dir")/(path.name+".obj")
            self.commands.append({"directory": str(self.build), "file": str(path), "output": str(output),
                                  "arguments": ["compiler", "-c", str(path), "-o", str(output)]})
            relative = output.relative_to(self.build).as_posix()
            ninja_lines.append("build " + relative + ": COMPILER " + path.as_posix())
            objects[module].append(relative)
        for module, values in objects.items():
            ninja_lines.append("build bin/" + module + ": LINK " + " ".join(values))
        self.ninja = "\n".join(ninja_lines)
        self.command_path, self.ninja_path = self.build/"compile_commands.json", self.build/"build.ninja"
        self.header_path = self.evidence/"header_snapshot.json"
        self.cuda_obj = self.abuild/"cuda.obj"
        self.rsp = self.evidence/"cuda.rsp"
        self.response = str(self.cuda_obj) + "\n/out:" + str(self.abuild/"bin/ggml-cuda.dll")
        native_outputs = {str(self.nbin/n): sha(n) for n in MODULES}
        annotation_outputs = {str(self.abuild/"bin"/n): sha(n) for n in MODULES}
        annotation_outputs[str(self.cuda_obj)] = sha("compiled-object")
        self.am = {"base_source": str(self.base), "overlay_source": str(self.overlay),
                   "modified_translation_units": {"ggml/src/ggml-cuda/ggml-cuda.cu": {"after_sha256": sha(CUDA)},
                                                   "src/llama-context.cpp": {"after_sha256": sha(CONTEXT)}},
                   "original_runtime_sha256": {str(self.build/"bin"/n): sha(n) for n in MODULES}}
        self.nm = {"base": str(self.base), "runtime_base": str(self.abuild/"bin"),
                   "protected_sha256": {str(self.abuild/"bin"/n): sha(n) for n in MODULES}}
        argv = ["nvcc", "--generate-code=arch=compute_120a,code=[sm_120a]", "-c", str(self.sources["cuda"]), "-o", "cuda.obj"]
        self.annotation = {"base": str(self.build), "build": str(self.abuild), "status": "complete",
            "old_runtime_preverified": True, "old_runtime_postverified": True, "old_link_inputs_postverified": True,
            "input_sha256": {str(self.sources["cuda"]): sha(CUDA), str(self.context): sha(CONTEXT)}, "output_sha256": annotation_outputs,
            "steps": [{"label": "cuda compile", "returncode": 0, "argv": argv, "cwd": str(self.abuild)},
                      {"label": "cuda link", "returncode": 0, "argv": ["link", "@"+str(self.rsp)], "cwd": str(self.abuild)},
                      {"label": "compile src/llama-context.cpp", "returncode": 0,
                       "argv": ["cc", "-c", str(self.context)], "cwd": str(self.abuild)}]}
        self.native = {"status": "complete", "runtime_base": str(self.abuild/"bin"), "runtime_output": str(self.nbin),
                       "cuda_recompiled": False, "baseline_preverified": True, "baseline_postverified": True,
                       "output_sha256": native_outputs, "unchanged_runtime_sha256": dict(native_outputs)}
        self.base_receipt = {"returncode": 0, "source_unchanged": True, "command": "build "+str(self.build),
            "source_sha256_before": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                     [self.sources["scheduler"], self.sources["operators"], *self.cli.values()]},
            "binary_artifacts": [{"path": str(self.build/"bin/llama-server.exe"), "sha256": sha("llama-server.exe")}]}
        self.base_receipt["source_sha256_after"] = dict(self.base_receipt["source_sha256_before"])
        self.base_receipt_path = root/"base_receipt.json"
        self.audit_path = root/"audit.json"
        self.refresh()
        self.selected = [{"path": k, "sha256": v} for k,v in native_outputs.items()]
        snap = {"status": "captured", "errors": [], "exe": str(self.nbin/"llama-server.exe"),
                "process_identity": "same-process", "actual_modules": deepcopy(self.selected)}
        self.raw = {"status": "complete", "actual_argv": [str(self.nbin/"llama-server.exe"), "-ngl", "0", "--op-offload"],
            "execution_environment": {"GGML_OP_OFFLOAD_MIN_BATCH": {"is_set": False, "value": None}},
            "runtime_before": deepcopy(snap), "runtime_after": deepcopy(snap),
            "state_measurement_before": {"gpu_state": {"returncode": 0, "stdout": "GPU-test, P1, 32"}},
            "state_after": {"gpu_state": {"returncode": 0, "stdout": "GPU-test, P1, 33"}}}
        self.hardware = {"gpu": {"name": "NVIDIA Synthetic GPU", "uuid": "GPU-test", "compute_capability": "12.0"}}

    def refresh(self):
        commands_ref, ninja_ref = write(self.command_path, self.commands), write(self.ninja_path, self.ninja)
        headers_ref = write(self.header_path, self.headers)
        am_ref = write(self.evidence/"source_manifest.json", self.am)
        nm_ref = write(self.native_root/"source_manifest.json", self.nm)
        self.annotation.update(source_manifest_sha256=am_ref["sha256"], compile_commands_sha256=commands_ref["sha256"],
                               build_ninja_sha256=ninja_ref["sha256"], header_snapshot_sha256=headers_ref["sha256"])
        self.annotation["input_sha256"][str(self.rsp)] = write(self.rsp, self.response)["sha256"]
        self.native["source_manifest_sha256"] = nm_ref["sha256"]
        ar = write(self.evidence/"build_receipt.json", self.annotation)
        nr = write(self.native_root/"build_receipt.json", self.native)
        write(self.base_receipt_path, self.base_receipt)
        cuda_ref = {"path": str(self.sources["cuda"]), "sha256": self.annotation["input_sha256"][str(self.sources["cuda"])]}
        self.audit = {"schema": "stable-native-runtime-build-posthoc-audit/v1",
            "scope": {"selected_native_cuda_artifact": {"path": str(self.nbin/"ggml-cuda.dll"), "sha256": sha("ggml-cuda.dll")}},
            "provenance_chain": [
                {"stage": "selection_to_native_output", "native_build_receipt_ref": nr, "artifact_sha256": sha("ggml-cuda.dll")},
                {"stage": "native_to_annotation", "native_source_manifest_ref": nm_ref,
                 "annotation_build_receipt_ref": ar, "runtime_base": str(self.abuild/"bin"), "artifact_sha256": sha("ggml-cuda.dll")},
                {"stage": "annotation_compile_and_link", "source_manifest_ref": am_ref, "cuda_source_ref": cuda_ref,
                 "compile_step_label": "cuda compile", "compile_argv": self.annotation["steps"][0]["argv"],
                 "link_step_label": "cuda link", "recorded_cuda_object_sha256": sha("compiled-object"), "output_cuda_sha256": sha("ggml-cuda.dll")},
                {"stage": "base_configuration_and_preprocessor_guards", "compile_commands_ref": commands_ref,
                 "build_ninja_ref": ninja_ref, "header_snapshot_ref": headers_ref}]}
        write(self.audit_path, self.audit)

    def verify(self):
        return verify_llama_runtime_source_binding(self.audit_path, base_build_receipt_path=self.base_receipt_path, data_root=self.root)

    def bind(self, binding=None, *, hardware=True):
        return bind_llama_cuda_op_offload_contract(binding or self.verify(), native_runtime_refs=self.selected,
            native_record_ref=write(self.root/"raw.json", self.raw), data_root=self.root,
            hardware_evidence_ref=write(self.root/"hardware.json", self.hardware) if hardware else None)


class RuntimeSourceBindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fx = Fixture(Path(self.directory.name))

    def test_verified_chain_uses_real_separate_compile_sources(self):
        binding = self.fx.verify()
        self.assertEqual(binding["status"], "verified_build_chain")
        self.assertEqual(Path(binding["source_paths"]["cuda"]), self.fx.sources["cuda"])
        self.assertEqual(Path(binding["source_paths"]["scheduler"]), self.fx.sources["scheduler"])
        self.assertNotEqual(self.fx.sources["cuda"].parents[3], self.fx.sources["scheduler"].parents[2])
        result = self.fx.bind(binding)
        self.assertEqual(result["status"], "verified")
        self.assertTrue(result["cuda_backend_available"])
        self.assertTrue(result["op_offload_enabled"])
        self.assertEqual(result["source_contract"]["minimum_m"], 32)
        self.assertEqual(result["source_contract"]["environment"]["status"], "captured_absent")
        self.assertEqual(result["required_workload_metadata"], {"llama_cpp_f32_hidden_storage": True})
        self.assertFalse(result["native_dispatch_proven"])
        self.assertFalse(result["staging_evidence"]["actual_tensor_copy_events_observed"])
        self.assertTrue(result["host_weight_buffer_evidence"]["ngl_zero_alone_is_not_proof"])

    def test_no_binary_model_or_object_is_opened(self):
        read = Path.read_bytes
        def guarded(path):
            self.assertNotIn(path.suffix.lower(), {".exe", ".dll", ".gguf", ".obj", ".lib"})
            return read(path)
        with patch.object(Path, "read_bytes", guarded):
            self.assertEqual(self.fx.bind()["status"], "verified")

    def test_current_environment_never_substitutes_for_capture(self):
        with patch.dict(os.environ, {"GGML_OP_OFFLOAD_MIN_BATCH": "999", "GGML_NO_IQ_PANEL": "0"}):
            result = self.fx.bind()
        self.assertEqual(result["source_contract"]["minimum_m"], 32)
        self.assertFalse(result["today_environment_read"])
        self.assertFalse(result["iq_panel_environment_inferred_or_changed"])

    def test_failed_build_or_changed_inherited_module_rejected(self):
        for name, value in (("status", "failed"), ("cuda_recompiled", True), ("baseline_postverified", False)):
            original = self.fx.native[name]
            self.fx.native[name] = value
            self.fx.refresh()
            with self.assertRaises(SourceBindingError): self.fx.verify()
            self.fx.native[name] = original
        self.fx.native["output_sha256"][str(self.fx.nbin/"ggml-cuda.dll")] = sha("wrong")
        self.fx.refresh()
        with self.assertRaises(SourceBindingError): self.fx.verify()

    def test_wrong_original_scheduler_content_rejected(self):
        write(self.fx.sources["scheduler"], SCHEDULER + "// changed")
        with self.assertRaises(SourceBindingError): self.fx.verify()

    def test_missing_historical_operator_source_identity_rejected(self):
        del self.fx.base_receipt["source_sha256_before"][str(self.fx.sources["operators"])]
        self.fx.refresh()
        with self.assertRaises(SourceBindingError): self.fx.verify()

    def test_compile_database_cannot_point_to_uncompiled_copy(self):
        self.fx.commands[0]["file"] = str(self.fx.root/"copied-scheduler.cpp")
        self.fx.refresh()
        with self.assertRaises(SourceBindingError): self.fx.verify()

    def test_scheduler_object_must_be_a_base_module_link_input(self):
        self.fx.ninja = self.fx.ninja.replace("ggml-base.dll.dir/ggml-backend.cpp.obj", "other.obj", 1)
        self.fx.refresh()
        with self.assertRaises(SourceBindingError): self.fx.verify()

    def test_cuda_compile_input_and_success_are_checked(self):
        self.fx.annotation["steps"][0]["argv"][-3] = str(self.fx.root/"wrong.cu")
        self.fx.refresh()
        with self.assertRaises(SourceBindingError): self.fx.verify()

    def test_cuda_link_must_consume_the_built_overlay_object(self):
        self.fx.response = "/out:" + str(self.fx.abuild/"bin/ggml-cuda.dll")
        self.fx.refresh()
        with self.assertRaises(SourceBindingError): self.fx.verify()

    def test_cuda_rule_threshold_change_rejected(self):
        value = CUDA.replace(": 32;", ": 64;")
        write(self.fx.sources["cuda"], value)
        self.fx.annotation["input_sha256"][str(self.fx.sources["cuda"])] = sha(value)
        self.fx.am["modified_translation_units"]["ggml/src/ggml-cuda/ggml-cuda.cu"]["after_sha256"] = sha(value)
        self.fx.refresh()
        with self.assertRaises(SourceBindingError): self.fx.verify()

    def test_header_tampering_rejected(self):
        write(self.fx.base/"common/common.h", "bool no_op_offload = true;")
        with self.assertRaises(SourceBindingError): self.fx.verify()

    def test_selected_and_loaded_runtime_must_both_match(self):
        self.fx.selected[0]["sha256"] = sha("other")
        with self.assertRaises(SourceBindingError): self.fx.bind()
        self.fx.selected[0]["sha256"] = sha(MODULES[0])
        self.fx.raw["runtime_after"]["actual_modules"][1]["sha256"] = sha("other")
        with self.assertRaises(SourceBindingError): self.fx.bind()

    def test_loaded_cuda_alone_does_not_establish_availability(self):
        result = self.fx.bind(hardware=False)
        self.assertEqual(result["status"], "uncovered")
        self.assertFalse(result["cuda_backend_available"])
        self.assertIsNone(result["source_contract"])
        self.assertEqual(result["required_workload_metadata"], {})

    def test_hardware_uuid_and_compiled_architecture_must_match(self):
        for changes in ({"uuid": "GPU-other"}, {"compute_capability": "12.1"}, {"name": "Other GPU"}):
            with self.subTest(changes=changes):
                original = dict(self.fx.hardware["gpu"])
                self.fx.hardware["gpu"].update(changes)
                result = self.fx.bind()
                self.assertEqual(result["status"], "uncovered")
                self.assertFalse(result["cuda_backend_available"])
                self.fx.hardware["gpu"] = original

    def test_missing_environment_is_uncovered_not_assumed_unset(self):
        self.fx.raw["execution_environment"].clear()
        result = self.fx.bind()
        self.assertEqual(result["status"], "uncovered")
        self.assertIn("historical_GGML_OP_OFFLOAD_MIN_BATCH_not_captured", result["uncovered_reasons"])
        self.assertIsNone(result["source_contract"])

    def test_captured_override_and_malformed_presence(self):
        self.fx.raw["execution_environment"]["GGML_OP_OFFLOAD_MIN_BATCH"] = {"is_set": True, "value": "64"}
        self.assertEqual(self.fx.bind()["source_contract"]["minimum_m"], 64)
        self.fx.raw["execution_environment"]["GGML_OP_OFFLOAD_MIN_BATCH"] = {"is_set": False, "value": "64"}
        with self.assertRaises(SourceBindingError): self.fx.bind()

    def test_explicit_disable_preserved(self):
        self.fx.raw["actual_argv"][-1] = "--no-op-offload"
        result = self.fx.bind()
        self.assertEqual(result["status"], "verified")
        self.assertFalse(result["op_offload_enabled"])
        self.assertEqual(result["required_workload_metadata"], {})

    def test_omitted_flag_uses_default_only_with_historical_parser_content(self):
        self.fx.raw["actual_argv"].pop()
        result = self.fx.bind()
        self.assertTrue(result["op_offload_enabled"])
        for path in self.fx.cli.values():
            self.fx.base_receipt["source_sha256_before"].pop(str(path))
            self.fx.base_receipt["source_sha256_after"].pop(str(path))
        self.fx.refresh()
        result = self.fx.bind()
        self.assertEqual(result["status"], "uncovered")
        self.assertIsNone(result["op_offload_enabled"])
        self.fx.raw["actual_argv"].append("--op-offload")
        result = self.fx.bind()
        self.assertEqual(result["status"], "verified")
        self.assertTrue(any("historical CLI parser" in v for v in result["limits"]))

    def test_binding_mutation_and_source_change_after_verification_rejected(self):
        binding = self.fx.verify()
        bad = deepcopy(binding)
        bad["source_contract"]["minimum_m"] = 1
        with self.assertRaises(SourceBindingError): self.fx.bind(bad)
        write(self.fx.sources["cuda"], CUDA + "// changed after verification")
        with self.assertRaises(SourceBindingError): self.fx.bind(binding)

    def test_separate_path_extractor_matches_existing_contract_parser(self):
        # A synthetic one-root fixture compares the extractor's public output;
        # real provenance tests above always preserve the separate source paths.
        original_cuda = self.fx.base/"ggml/src/ggml-cuda/ggml-cuda.cu"
        write(original_cuda, CUDA)
        source_paths = {**self.fx.sources, "cuda": original_cuda}
        environment = {"GGML_OP_OFFLOAD_MIN_BATCH": None}
        self.assertEqual(derive_llama_cuda_op_offload_contract_from_sources(source_paths, runtime_environment=environment),
                         derive_llama_cuda_op_offload_contract(self.fx.base, runtime_environment=environment))


if __name__ == "__main__":
    unittest.main()
