"""Read-only source/build/module binding for recorded llama.cpp overlay runs.

This module never opens executables, DLLs, models or compiled object/library
files. Their identities come from digest-bound receipts and native captures.
The CUDA translation unit and the scheduler/operators retain their real,
different compilation paths; no synthetic source tree is constructed.

Public flow:
    binding = verify_llama_runtime_source_binding(audit_path,
        base_build_receipt_path=receipt_path, data_root=project_root)
    result = bind_llama_cuda_op_offload_contract(binding,
        native_runtime_refs=row["native_runtime_refs"], native_record_ref=raw_ref,
        data_root=project_root)

Only result["status"] == "verified" permits the returned source_contract and
required_workload_metadata to be installed. Missing historical facts return
an uncovered result; contradictory hashes/receipts raise SourceBindingError.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from heterollm_sim.runtime_adapters import LLAMA_CUDA_OP_OFFLOAD_SCHEMA, _source_function

SCHEMA = "llama-recorded-runtime-source-binding/v1"
_REQUIRED_MODULES = ("ggml-base.dll", "ggml-cuda.dll", "ggml.dll", "llama.dll",
                     "llama-common.dll", "llama-server-impl.dll", "llama-server.exe")
_BINARY_SUFFIXES = {".dll", ".exe", ".gguf", ".obj", ".lib", ".so", ".a", ".pdb", ".pch"}
_ENV = "GGML_OP_OFFLOAD_MIN_BATCH"


class SourceBindingError(ValueError):
    """A purported historical source/build/runtime identity is inconsistent."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise SourceBindingError(message)


def _sha(value: object) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None,
             "missing or malformed SHA256 identity")
    return value.lower()


@lru_cache(maxsize=32768)
def _identity(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/").casefold()


def _content_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


class _Evidence:
    def __init__(self, data_root=None):
        self.root = Path(data_root).resolve() if data_root is not None else None
        self.refs: dict[str, dict[str, Any]] = {}

    def read(self, ref: Mapping[str, Any] | str | Path) -> bytes:
        given = ref if isinstance(ref, Mapping) else {"path": str(ref)}
        path = Path(given["path"]).resolve()
        _require(path.suffix.lower() not in _BINARY_SUFFIXES,
                 "binary/model/object artifacts must remain recorded identities, never live reads")
        if self.root is not None:
            _require(path.is_relative_to(self.root), "evidence path escapes the declared data root")
        _require(path.stat().st_size <= 32 * 1024 * 1024, "evidence text exceeds the 32 MiB bound")
        value = path.read_bytes()
        digest = hashlib.sha256(value).hexdigest()
        if "sha256" in given:
            _require(digest == _sha(given["sha256"]), "evidence SHA256 mismatch: " + str(path))
        for size_key in ("bytes", "size_bytes"):
            if size_key in given:
                _require(given[size_key] == len(value), "evidence byte size mismatch: " + str(path))
        self.refs[_identity(path)] = {"path": str(path), "sha256": digest, "size_bytes": len(value)}
        return value

    def document(self, ref):
        value = json.loads(self.read(ref).decode("utf-8-sig"))
        return value

    def text(self, ref):
        return self.read(ref).decode("utf-8-sig")


def _digest_at(records: Mapping[str, str], path: str | Path) -> str:
    values = [v for p, v in records.items() if _identity(p) == _identity(path)]
    _require(len(values) == 1, "recorded artifact/source missing or ambiguous: " + str(path))
    return _sha(values[0])


def _equal_digests(*values: str) -> None:
    _require(len({_sha(v) for v in values}) == 1, "inherited artifact/source identity chain disagrees")


def _step(receipt, label):
    found = [s for s in receipt.get("steps", ()) if s.get("label") == label]
    _require(len(found) == 1 and found[0].get("returncode") == 0,
             "successful unique build step not established: " + label)
    return found[0]


def _tokens(command):
    if isinstance(command, str):
        return [t.strip('"') for t in shlex.split(command, posix=False)]
    _require(isinstance(command, list) and all(isinstance(v, str) for v in command),
             "compiler command must be a token list or recorded command string")
    return command


def _compile_entry(commands, source: Path, build: Path):
    entries = [r for r in commands if _identity(r.get("file", "")) == _identity(source)]
    _require(len(entries) == 1, "compile database source missing or ambiguous: " + str(source))
    entry = entries[0]
    _require(_identity(entry["directory"]) == _identity(build), "compiler working directory mismatch")
    argv = _tokens(entry.get("arguments", entry.get("command")))
    _require("-c" in argv and _identity(argv[argv.index("-c")+1]) == _identity(source),
             "compiler input is not the bound original translation unit")
    output = Path(entry["output"]).resolve()
    _require(output.is_relative_to(build.resolve()), "compiler output is outside its recorded build")
    return entry


def _ninja_source_link(ninja: str, source: Path, output: Path, build: Path, module: str):
    # Only consume literal rule text; never execute Ninja, response files or shell code.
    normalized = ninja.replace("$:", ":").replace("\\", "/").casefold()
    rel = output.relative_to(build).as_posix().casefold()
    src = str(source.resolve()).replace("\\", "/").casefold()
    compile_lines = [line for line in normalized.splitlines()
                     if line.startswith("build " + rel + ":")]
    _require(len(compile_lines) == 1 and src in compile_lines[0],
             "Ninja compiler input disagrees with compile database")
    link_lines = [line for line in normalized.splitlines()
                  if line.startswith("build bin/" + module.casefold() + " ")
                  or line.startswith("build bin/" + module.casefold() + ":")]
    _require(len(link_lines) == 1 and rel in link_lines[0].split(": ", 1)[-1],
             "Ninja does not bind the source object to " + module)


def derive_llama_cuda_op_offload_contract_from_sources(
    source_paths: Mapping[str, str | Path], *, runtime_environment: Mapping[str, str | None] | None = None,
) -> dict[str, Any]:
    """Equivalent source-rule extraction using each actual compilation path.

    This low-level extractor does not establish binary provenance. Use the
    verified build binding and native-record binder before applying its result.
    """
    _require(set(source_paths) == {"scheduler", "cuda", "operators"}, "three explicit source roles are required")
    evidence = _Evidence()
    source = {name: evidence.text(path) for name, path in source_paths.items()}
    scheduler, cuda, operators = source["scheduler"], source["cuda"], source["operators"]
    required = (
        "sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)",
        "ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)",
    )
    _require(all(v in scheduler for v in required), "unrecognized scheduler host-weight offload selection")
    copies = _source_function(scheduler, "static enum ggml_status ggml_backend_sched_compute_splits(")
    _require("ggml_backend_tensor_copy(input, input_cpy)" in copies
             and "cpy_tensor_async(input_backend, split_backend, input, input_cpy)" in copies,
             "unrecognized per-invocation host-weight staging")
    batches = _source_function(cuda, "static int64_t get_op_batch_size(")
    _require(re.search(r"case GGML_OP_MUL_MAT:\s*return op->ne\[1\];", batches) is not None,
             "unrecognized CUDA physical MUL_MAT batch dimension")
    offload = _source_function(cuda, "static bool ggml_backend_cuda_device_offload_op(")
    _require("get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size" in offload,
             "unrecognized CUDA offload threshold comparison")
    default = re.search(r'const int min_batch_size = getenv\("GGML_OP_OFFLOAD_MIN_BATCH"\) \? '
                        r'atoi\(getenv\("GGML_OP_OFFLOAD_MIN_BATCH"\)\) : (\d+);', cuda)
    _require(default is not None, "unrecognized CUDA offload default/environment rule")
    supports = _source_function(cuda, "static bool ggml_backend_cuda_device_supports_op(")
    _require("case GGML_OP_MUL_MAT:" in supports and "case GGML_OP_OUT_PROD:" in supports,
             "unrecognized CUDA matmul support case")
    matmul = supports.split("case GGML_OP_MUL_MAT:", 1)[1].split("case GGML_OP_OUT_PROD:", 1)[0]
    _require("switch (a->type)" in matmul and "a->nb[0] != ggml_element_size(a)" in matmul,
             "unrecognized CUDA weight type/layout support")
    formats = tuple(dict.fromkeys(re.findall(r"case GGML_TYPE_([A-Z0-9_]+):",
                         matmul.split("switch (a->type)", 1)[1].split("return true;", 1)[0])))
    _require(bool(formats), "no recognized CUDA matmul weight formats")
    mul = _source_function(operators, "struct ggml_tensor * ggml_mul_mat(")
    rows = _source_function(operators, "struct ggml_tensor * ggml_get_rows(")
    _require("ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne)" in mul
             and "enum ggml_type type = GGML_TYPE_F32;" in rows,
             "ordinary matmul/get-rows F32 hidden storage not established")
    captured = runtime_environment is not None and _ENV in runtime_environment
    env_value = runtime_environment[_ENV] if captured else None
    minimum = int(default.group(1))
    status = "captured_absent" if captured else "uncaptured_default_assumption"
    if captured and env_value is not None:
        _require(isinstance(env_value, str) and re.fullmatch(r"[+-]?\d+", env_value.strip()) is not None,
                 "unsupported offload threshold environment value; no guessed atoi behavior")
        minimum, status = max(1, int(env_value.strip())), "captured_override"
    return {
        "schema": LLAMA_CUDA_OP_OFFLOAD_SCHEMA, "backend": "CUDA",
        "source_sha256": {ref["path"]: ref["sha256"] for ref in evidence.refs.values()},
        "source_symbols": ["ggml_backend_sched_backend_id_from_cur", "get_op_batch_size",
                           "ggml_backend_cuda_device_offload_op", "ggml_backend_cuda_device_supports_op",
                           "ggml_backend_sched_compute_splits", "ggml_mul_mat", "ggml_get_rows"],
        "minimum_m": minimum, "default_minimum_m": int(default.group(1)),
        "physical_batch_dimension": "MUL_MAT.output.ne[1]", "supported_weight_formats": formats,
        "environment": {"name": _ENV, "value": env_value, "status": status},
        "prediction_provenance": "source_mechanism" if captured else "conditional_development_assumption",
        "accuracy_validated": False,
        "scope": "ordinary contiguous 2D dense text GGUF host-weight MUL_MAT with F32 hidden storage; CUDA backend",
        "staging": "per_invocation_copy_then_temporary_read_clean_discard",
        "unsupported": ["MUL_MAT_ID/expert dispatch", "arbitrary tensor views/layouts", "GPU performance calibration"],
    }


def verify_llama_runtime_source_binding(
    runtime_build_audit_path: str | Path,
    *,
    base_build_receipt_path: str | Path,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the annotation CUDA / unchanged base / native inheritance chain.

    The base source receipt is mandatory: a CUDA-Graphs audit alone does not
    bind the scheduler or F32-producing operators to their original sources.
    Current parser/common.cpp contents are separately labeled when the old
    receipt did not capture their hashes; an omitted CLI flag cannot silently
    borrow a historically unproven default.
    """
    ev = _Evidence(data_root)
    audit = ev.document(runtime_build_audit_path)
    _require(audit.get("schema") == "stable-native-runtime-build-posthoc-audit/v1", "unsupported runtime audit schema")
    stages = {s["stage"]: s for s in audit.get("provenance_chain", ())}
    _require(len(stages) == len(audit.get("provenance_chain", ())), "duplicate runtime audit stages")
    needed = ("selection_to_native_output", "native_to_annotation", "annotation_compile_and_link",
              "base_configuration_and_preprocessor_guards")
    _require(all(k in stages for k in needed), "runtime inheritance chain is incomplete")
    selected, inherited, compiled, base = (stages[k] for k in needed)
    native = ev.document(selected["native_build_receipt_ref"])
    native_manifest = ev.document(inherited["native_source_manifest_ref"])
    annotation = ev.document(inherited["annotation_build_receipt_ref"])
    annotation_manifest = ev.document(compiled["source_manifest_ref"])
    base_receipt = ev.document(base_build_receipt_path)
    _require(native.get("status") == "complete" and native.get("cuda_recompiled") is False
             and native.get("baseline_preverified") is True and native.get("baseline_postverified") is True,
             "native overlay inheritance was not fully verified")
    _require(annotation.get("status") == "complete" and all(annotation.get(k) is True for k in (
        "old_runtime_preverified", "old_runtime_postverified", "old_link_inputs_postverified")),
        "annotation overlay compile/link inputs were not fully verified")
    _require(base_receipt.get("returncode") == 0 and base_receipt.get("source_unchanged") is True,
             "base source build did not complete with unchanged source")
    _equal_digests(native["source_manifest_sha256"], inherited["native_source_manifest_ref"]["sha256"])
    _equal_digests(annotation["source_manifest_sha256"], compiled["source_manifest_ref"]["sha256"])
    original_root = Path(annotation_manifest["base_source"]).resolve()
    overlay_root = Path(annotation_manifest["overlay_source"]).resolve()
    base_build = Path(annotation["base"]).resolve()
    annotation_build = Path(annotation["build"]).resolve()
    native_bin = Path(native["runtime_output"]).resolve()
    annotation_bin = annotation_build / "bin"
    _require(_identity(native["runtime_base"]) == _identity(annotation_bin)
             and _identity(native_manifest["runtime_base"]) == _identity(annotation_bin)
             and _identity(inherited["runtime_base"]) == _identity(annotation_bin), "native did not inherit the annotation runtime")
    _require(_identity(native_manifest["base"]) == _identity(original_root), "native base source root disagrees")
    _require(str(base_build).replace("\\", "/").casefold() in
             str(base_receipt.get("command", "")).replace("\\", "/").casefold(),
             "base receipt command does not target the inherited build")
    modules = {}
    for name in _REQUIRED_MODULES:
        output_path = native_bin / name
        digest = _digest_at(native["output_sha256"], output_path)
        _equal_digests(digest, _digest_at(native["unchanged_runtime_sha256"], output_path),
                      _digest_at(native_manifest["protected_sha256"], annotation_bin / name),
                      _digest_at(annotation["output_sha256"], annotation_bin / name))
        modules[name] = {"path": str(output_path), "sha256": digest, "basis": "historical_output_and_unchanged_runtime_receipts"}
    for name in ("ggml-base.dll", "ggml.dll", "llama-common.dll"):
        _equal_digests(modules[name]["sha256"],
                      _digest_at(annotation_manifest["original_runtime_sha256"], base_build / "bin" / name))
    artifact = audit["scope"]["selected_native_cuda_artifact"]
    _require(_identity(artifact["path"]) == _identity(modules["ggml-cuda.dll"]["path"]), "audit selected CUDA path is not the native output")
    _equal_digests(artifact["sha256"], modules["ggml-cuda.dll"]["sha256"],
                  selected["artifact_sha256"], inherited["artifact_sha256"], compiled["output_cuda_sha256"])
    # Connect the successful original build to the unmodified runtime snapshot.
    original_server = [r for r in base_receipt.get("binary_artifacts", ())
                       if _identity(r.get("path", "")) == _identity(base_build / "bin" / "llama-server.exe")]
    _require(len(original_server) == 1, "base build executable identity missing")
    _equal_digests(original_server[0]["sha256"], _digest_at(
        annotation_manifest["original_runtime_sha256"], base_build / "bin" / "llama-server.exe"))

    commands = ev.document(base["compile_commands_ref"])
    ninja = ev.text(base["build_ninja_ref"])
    headers = ev.document(base["header_snapshot_ref"])
    for key, ref_key in (("compile_commands_sha256", "compile_commands_ref"),
                         ("build_ninja_sha256", "build_ninja_ref"), ("header_snapshot_sha256", "header_snapshot_ref")):
        _equal_digests(annotation[key], base[ref_key]["sha256"])
    source_paths = {"scheduler": original_root / "ggml/src/ggml-backend.cpp",
                    "operators": original_root / "ggml/src/ggml.c",
                    "cuda": overlay_root / "ggml/src/ggml-cuda/ggml-cuda.cu"}
    source_compilation = {}
    for role in ("scheduler", "operators"):
        path = source_paths[role]
        before = _digest_at(base_receipt["source_sha256_before"], path)
        _equal_digests(before, _digest_at(base_receipt["source_sha256_after"], path))
        ev.read({"path": str(path), "sha256": before})
        entry = _compile_entry(commands, path, base_build)
        _ninja_source_link(ninja, path, Path(entry["output"]).resolve(), base_build, "ggml-base.dll")
        source_compilation[role] = {"source": str(path), "object": entry["output"],
                                    "module": modules["ggml-base.dll"], "source_content_bound": True}
    # The exact CUDA overlay source, successful argv, output object and link
    # response form one chain. A source copy elsewhere is not interchangeable.
    cuda_path = source_paths["cuda"]
    _require(_identity(compiled["cuda_source_ref"]["path"]) == _identity(cuda_path), "CUDA source is not the compiled overlay path")
    cuda_digest = _digest_at(annotation["input_sha256"], cuda_path)
    modified = annotation_manifest["modified_translation_units"].get("ggml/src/ggml-cuda/ggml-cuda.cu")
    _require(isinstance(modified, Mapping), "overlay CUDA source delta missing from manifest")
    _equal_digests(cuda_digest, modified["after_sha256"], compiled["cuda_source_ref"]["sha256"])
    cuda_text = ev.text({"path": str(cuda_path), "sha256": cuda_digest})
    step = _step(annotation, compiled["compile_step_label"])
    argv = _tokens(step["argv"])
    _require(argv == compiled["compile_argv"] and _identity(step["cwd"]) == _identity(annotation_build),
             "bound CUDA compile argv or working directory differs")
    _require("-c" in argv and _identity(argv[argv.index("-c")+1]) == _identity(cuda_path),
             "CUDA compiler did not consume the declared overlay source")
    _require("-o" in argv, "CUDA compiler object output missing")
    obj = (annotation_build / argv[argv.index("-o")+1]).resolve()
    _equal_digests(_digest_at(annotation["output_sha256"], obj), compiled["recorded_cuda_object_sha256"])
    link = _step(annotation, compiled["link_step_label"])
    responses = [arg[1:] for arg in link["argv"] if arg.startswith("@")]
    _require(len(responses) == 1 and _identity(link["cwd"]) == _identity(annotation_build), "CUDA link response is missing or ambiguous")
    response = Path(responses[0]).resolve()
    response_text = ev.text({"path": str(response), "sha256": _digest_at(annotation["input_sha256"], response)})
    link_argv = _tokens(response_text)
    _require(any(_identity(arg) == _identity(obj) for arg in link_argv), "CUDA linked object is not the compiled overlay object")
    outputs = [v[5:] for v in link_argv if v.lower().startswith("/out:")]
    _require(len(outputs) == 1 and _identity(outputs[0]) == _identity(annotation_bin / "ggml-cuda.dll"),
             "CUDA link output differs from inherited module")
    recorded_link_inputs = {**annotation["input_sha256"], **annotation["output_sha256"]}
    linked_project_inputs = 0
    for token in link_argv:
        if Path(token).suffix.lower() not in {".obj", ".lib"} or not Path(token).is_absolute():
            continue
        if Path(token).resolve().is_relative_to(original_root) or Path(token).resolve().is_relative_to(overlay_root):
            _digest_at(recorded_link_inputs, token)
            linked_project_inputs += 1
    _require(linked_project_inputs >= 1, "CUDA project link dependencies were not recorded")
    source_compilation["cuda"] = {"source": str(cuda_path), "object": str(obj),
                                   "module": modules["ggml-cuda.dll"], "source_content_bound": True,
                                   "recorded_project_link_inputs": linked_project_inputs}
    # Verify the original headers directly relevant to these dispatch rules.
    header_paths = [original_root / relative for relative in (
        "ggml/include/ggml.h", "ggml/include/ggml-backend.h", "ggml/src/ggml-backend-impl.h",
        "ggml/src/ggml-cuda/common.cuh", "common/common.h")]
    for header in header_paths:
        ev.read({"path": str(header), "sha256": _digest_at(headers["files"], header)})
    for flag in argv:
        if flag.startswith("-I") and len(flag) > 2:
            include = Path(flag[2:]).resolve()
            if include.is_relative_to(overlay_root):
                # The overlay's extra include directory is only safe if absent
                # or byte-identical to the captured base headers it shadows.
                if include.exists():
                    for header in include.rglob("*.h"):
                        recorded = [v for k, v in annotation["input_sha256"].items() if _identity(k) == _identity(header)]
                        baseline = original_root / header.relative_to(overlay_root)
                        expected = recorded[0] if len(recorded) == 1 else _digest_at(headers["files"], baseline)
                        ev.read({"path": str(header), "sha256": expected})
    context_path = overlay_root / "src/llama-context.cpp"
    context_digest = _digest_at(annotation["input_sha256"], context_path)
    context_delta = annotation_manifest["modified_translation_units"].get("src/llama-context.cpp", {})
    _equal_digests(context_digest, context_delta.get("after_sha256"))
    context_text = ev.text({"path": str(context_path), "sha256": context_digest})
    context_compile = _step(annotation, "compile src/llama-context.cpp")
    context_args = _tokens(context_compile["argv"])
    _require("-c" in context_args and _identity(context_args[context_args.index("-c")+1]) == _identity(context_path),
             "context compiler input is not the bound overlay source")
    _require("cparams.op_offload = params.op_offload;" in context_text
             and re.search(r"ggml_backend_sched_new\([^;]+cparams\.op_offload\)", context_text) is not None,
             "native context does not pass the offload flag to the scheduler")
    source_contract = derive_llama_cuda_op_offload_contract_from_sources(source_paths)
    _require(source_contract["default_minimum_m"] == 32, "bound runtime default offload threshold is not 32")
    # Source registration plus separately captured NVIDIA device evidence may
    # prove availability. Loaded DLL presence alone is deliberately insufficient.
    init_supported = all(fragment in cuda_text for fragment in (
        "ggml_backend_cuda_reg_get_device_count", "return ctx->devices.size();",
        "ggml_backend_cuda_device_init_backend", "return ggml_backend_cuda_init(ctx->device);"))
    common_header = (original_root / "common/common.h").read_text(encoding="utf-8")
    cli_paths = {"parser": original_root / "common/arg.cpp", "params": original_root / "common/common.cpp"}
    cli_text = {}
    cli_historical = True
    for role, path in cli_paths.items():
        entry = _compile_entry(commands, path, base_build)
        _ninja_source_link(ninja, path, Path(entry["output"]).resolve(), base_build, "llama-common.dll")
        expected = [value for name, value in base_receipt["source_sha256_before"].items() if _identity(name) == _identity(path)]
        if expected:
            _equal_digests(expected[0], _digest_at(base_receipt["source_sha256_after"], path))
            cli_text[role] = ev.text({"path": str(path), "sha256": expected[0]})
        else:
            cli_historical = False
            cli_text[role] = ev.text(path)
    parser = cli_text["parser"]
    start = parser.find('{"--op-offload"}')
    end = parser.find('));', start)
    block = parser[start:end+3] if start >= 0 and end >= 0 else ""
    cli_rule = (bool(re.search(r"bool\s+no_op_offload\s*=\s*false\s*;", common_header))
                and '{"--no-op-offload"}' in block and 'params.no_op_offload = !value;' in block
                and '.set_env(' not in block
                and 'cparams.op_offload        = !params.no_op_offload;' in cli_text["params"])
    report = {
        "schema": SCHEMA, "status": "verified_build_chain", "source_contract": source_contract,
        "source_paths": {role: str(path) for role, path in source_paths.items()},
        "source_compilation": source_compilation, "runtime_modules": modules,
        "cuda_source_device_registration_supported": init_supported,
        "compiled_cuda_architectures": sorted({int(v) for v in re.findall(r"(?:compute|sm)_(\d+)", " ".join(argv))}),
        "cli_rule": {"recognized": cli_rule, "op_offload_default": True if cli_rule else None,
                     "parameter_header_historically_bound": True,
                     "parser_and_conversion_cpp_historically_bound": cli_historical,
                     "source_paths": {role: str(path) for role, path in cli_paths.items()}},
        "evidence_refs": sorted(ev.refs.values(), key=lambda r: r["path"]),
        "scope": "historical overlay source/build consistency; not binary disassembly or measured dispatch",
        "uncovered": [] if cli_historical else ["historical CLI parser/common.cpp content hashes not recorded; omitted CLI flag remains uncovered"],
        "limits": ["No executable/DLL/GGUF/OBJ/LIB bytes are read or hashed",
                   "SDK/system-header contents and kernel-level dispatch are not proven by these receipts",
                   "Actual host buffer types and individual tensor copy/dispatch events require per-operator evidence",
                   "Historical IQ panel switch is independent and is neither read nor changed"],
    }
    report["content_sha256"] = _content_hash(report)
    return report


def _native_module_map(refs: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    _require(isinstance(refs, (list, tuple)), "native module references must be a list")
    result = {}
    for ref in refs:
        _require(isinstance(ref, Mapping) and isinstance(ref.get("path"), str), "malformed native module reference")
        name = Path(ref["path"]).name.casefold()
        _require(name not in result, "duplicate/ambiguous native module basename: " + name)
        _sha(ref.get("sha256"))
        result[name] = ref
    return result


def bind_llama_cuda_op_offload_contract(
    binding: Mapping[str, Any],
    *,
    native_runtime_refs: Sequence[Mapping[str, Any]],
    native_record_ref: Mapping[str, Any],
    data_root: str | Path | None = None,
    hardware_evidence_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind verified source rules to captured argv/environment/modules/hardware.

    GPU availability requires both the matching loaded CUDA module and frozen
    NVIDIA device/architecture evidence tied to that run's captured GPU UUID.
    This proves a source-supported execution path, not that every eligible
    operator was observed on CUDA. Tensor buffer/layout evidence stays local
    to the planner; ``-ngl 0`` alone never proves host buffer types.
    """
    report = dict(binding)
    digest = report.pop("content_sha256", None)
    _require(report.get("schema") == SCHEMA and report.get("status") == "verified_build_chain"
             and digest == _content_hash(report), "runtime source binding was not verified or was mutated")
    ev = _Evidence(data_root)
    _require(isinstance(native_record_ref, Mapping) and "sha256" in native_record_ref,
             "the native record must have a frozen SHA256 reference")
    record = ev.document(native_record_ref)
    _require(record.get("status") == "complete", "native record is not a completed capture")
    selected = _native_module_map(native_runtime_refs)
    _require(all(name in selected for name in _REQUIRED_MODULES), "selected native module identity set is incomplete")
    for name, expected in report["runtime_modules"].items():
        actual = selected[name]
        _require(_identity(actual["path"]) == _identity(expected["path"]), "selected native runtime path is not the inherited output: " + name)
        _equal_digests(actual["sha256"], expected["sha256"])
    argv = record.get("actual_argv")
    _require(isinstance(argv, list) and argv and all(isinstance(v, str) for v in argv), "captured actual native argv is missing")
    _require(_identity(argv[0]) == _identity(selected["llama-server.exe"]["path"]), "actual native executable differs from selected runtime")
    snapshots = []
    for key in ("runtime_before", "runtime_after"):
        snap = record.get(key)
        _require(isinstance(snap, Mapping) and snap.get("status") == "captured" and not snap.get("errors"),
                 "complete loaded-module capture missing: " + key)
        _require(_identity(snap.get("exe", "")) == _identity(argv[0]), "captured process executable differs from actual argv")
        captured = _native_module_map(snap.get("actual_modules", ()))
        for name in _REQUIRED_MODULES:
            _require(name in captured, "required runtime module not captured as loaded: " + name)
            _require(_identity(captured[name]["path"]) == _identity(selected[name]["path"]),
                     "loaded runtime module path differs from selected identity: " + name)
            _equal_digests(captured[name]["sha256"], selected[name]["sha256"])
        snapshots.append(snap)
    if snapshots[0].get("process_identity") is not None or snapshots[1].get("process_identity") is not None:
        _require(snapshots[0].get("process_identity") == snapshots[1].get("process_identity"), "native process identity changed during capture")
    reasons = []
    env = record.get("execution_environment", {})
    captured_environment = None
    if isinstance(env, Mapping) and _ENV in env:
        observed = env[_ENV]
        if isinstance(observed, Mapping):
            present, value = observed.get("is_set"), observed.get("value")
            _require(type(present) is bool and ((present and isinstance(value, str)) or (not present and value is None)),
                     "captured offload environment presence/value disagree")
            captured_environment = {_ENV: value}
        elif observed is None or isinstance(observed, str):
            captured_environment = {_ENV: observed}
        else:
            raise SourceBindingError("malformed captured offload environment")
    else:
        reasons.append("historical_GGML_OP_OFFLOAD_MIN_BATCH_not_captured")
    cli = report["cli_rule"]
    op_offload = None
    cli_basis = "uncovered"
    if not cli["recognized"]:
        reasons.append("fixed_cli_offload_parser_rule_not_recognized")
    elif any(v.startswith(("--op-offload=", "--no-op-offload=")) for v in argv):
        reasons.append("unrecognized_op_offload_cli_encoding")
    else:
        switches = [v for v in argv if v in ("--op-offload", "--no-op-offload")]
        if switches:
            op_offload, cli_basis = switches[-1] == "--op-offload", "captured_explicit_cli_switch"
        elif cli["parser_and_conversion_cpp_historically_bound"]:
            op_offload, cli_basis = cli["op_offload_default"], "historically_bound_header_parser_and_parameter_default"
        else:
            reasons.append("op_offload_cli_omitted_and_historical_parser_content_unbound")
    available = False
    device_evidence = {"status": "uncovered", "loaded_cuda_module": True}
    if hardware_evidence_ref is None:
        reasons.append("frozen_cuda_hardware_evidence_missing")
    else:
        _require(isinstance(hardware_evidence_ref, Mapping) and "sha256" in hardware_evidence_ref,
                 "hardware evidence requires a frozen SHA256 reference")
        hardware = ev.document(hardware_evidence_ref)
        gpu = hardware.get("gpu", {})
        uuid = gpu.get("uuid")
        capability = str(gpu.get("compute_capability", ""))
        matched = re.fullmatch(r"(\d+)\.(\d+)", capability)
        architecture = int(matched[1])*10 + int(matched[2]) if matched else None
        states = [record.get(key, {}).get("gpu_state", {}) for key in ("state_measurement_before", "state_after")]
        same_gpu = bool(isinstance(uuid, str) and uuid.startswith("GPU-") and all(
            state.get("returncode") == 0 and any(line.split(",", 1)[0].strip() == uuid
                for line in str(state.get("stdout", "")).splitlines()) for state in states))
        available = (report["cuda_source_device_registration_supported"] is True
                     and str(gpu.get("name", "")).startswith("NVIDIA ")
                     and architecture in report["compiled_cuda_architectures"] and same_gpu)
        if not available:
            reasons.append("loaded_cuda_lacks_matching_frozen_device_or_compiled_architecture_evidence")
        device_evidence = {"status": "source_and_captured_hardware" if available else "uncovered",
                           "loaded_cuda_module": True, "gpu_uuid": uuid, "compute_capability": capability,
                           "compiled_cuda_architectures": report["compiled_cuda_architectures"],
                           "captured_device_uuid_before_and_after_matches": same_gpu,
                           "individual_cuda_operator_execution_observed": False}
    # Recheck only small rule sources, so a reused binding cannot silently read
    # modified source later. Build receipts stay in the frozen evidence list.
    for path, sha in report["source_contract"]["source_sha256"].items():
        ev.read({"path": path, "sha256": sha})
    contract = derive_llama_cuda_op_offload_contract_from_sources(
        report["source_paths"], runtime_environment=captured_environment)
    _require(contract["default_minimum_m"] == 32, "default threshold changed after source binding")
    if not reasons:
        contract.update(
            backend_artifact={"path": selected["ggml-cuda.dll"]["path"], "sha256": selected["ggml-cuda.dll"]["sha256"]},
            source_runtime_binding="verified_annotation_cuda_and_original_base_inherited_by_selected_native_runtime",
            source_binding_content_sha256=digest,
            op_offload_enabled=op_offload,
            native_dispatch_proven=False,
        )
    result = {
        "schema": "llama-recorded-cuda-op-offload-contract/v1",
        "status": "verified" if not reasons else "uncovered",
        "source_contract": contract if not reasons else None,
        "cuda_backend_available": bool(available), "op_offload_enabled": op_offload,
        "op_offload_cli_basis": cli_basis, "device_evidence": device_evidence,
        "native_record_ref": ev.refs[_identity(native_record_ref["path"])],
        "required_workload_metadata": {"llama_cpp_f32_hidden_storage": True} if not reasons and op_offload else {},
        "uncovered_reasons": reasons,
        "evidence_refs": sorted({**{_identity(r["path"]): r for r in report["evidence_refs"]}, **ev.refs}.values(), key=lambda r:r["path"]),
        "per_operator_requirements": {
            "operator": "MUL_MAT", "physical_model_weight": True,
            "weight_buffer_is_host": True, "weight_layout": "ordinary_contiguous_2d",
            "activation_and_output_dtype": "F32", "minimum_m": contract["minimum_m"],
            "cuda_supports_the_specific_operator": True,
        },
        "host_weight_buffer_evidence": {
            "status": "requires_per_tensor_source_or_capture_binding",
            "ngl_zero_alone_is_not_proof": True,
            "needed": ["source-bound model tensor placement and actual host buffer type",
                       "ordinary contiguous matrix shape/type/stride", "runtime scheduler backend selection"],
        },
        "staging_evidence": {"per_invocation_copy_rule_source_verified": True,
                             "actual_tensor_copy_events_observed": False,
                             "needed_for_dispatch_proof": ["backend copy and compute events for the same invocation"]},
        "native_dispatch_proven": False, "performance_accuracy_validated": False,
        "today_environment_read": False, "iq_panel_environment_inferred_or_changed": False,
        "limits": report["limits"] + report["uncovered"],
    }
    result["content_sha256"] = _content_hash(result)
    return result


__all__ = ["SourceBindingError", "verify_llama_runtime_source_binding",
           "bind_llama_cuda_op_offload_contract", "derive_llama_cuda_op_offload_contract_from_sources"]
