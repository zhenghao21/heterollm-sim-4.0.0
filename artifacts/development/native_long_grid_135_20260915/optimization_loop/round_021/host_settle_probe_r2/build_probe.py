"""Review-gated compile/freeze entry for the R21 host-settle probe. Never executes GPU work."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SOURCE_FILES = [
    "host_settle_main.cpp", "host_settle_probe.cpp", "math_reference.h", "frozen_module_guard.h",
    "reference_host.cpp", "protocol.json", "source_provenance.json", "build_probe.py",
    "invoke.ps1", "reference_check.py", "README.md",
]
FORBIDDEN_BUILD_OUTPUTS = [
    "prepared_identity.h", "compile.cmd", "compile.log", "host-settle-probe.exe",
    "host-settle-probe.obj", "reference-host.exe", "reference-host.obj",
    "host_reference_validation.json", "build_manifest.json",
]


def digest(path: Path | str) -> str:
    hash_value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hash_value.update(block)
    return hash_value.hexdigest()


def ref(path: Path | str) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise RuntimeError(f"Missing file: {resolved}")
    return {"path": str(resolved), "sha256": digest(resolved), "bytes": resolved.stat().st_size}


def verify(items: list[dict[str, object]]) -> None:
    if not items:
        raise RuntimeError("Empty identity set")
    seen: set[str] = set()
    for item in items:
        path = item.get("path")
        sha256 = item.get("sha256")
        size = item.get("bytes")
        if not isinstance(path, str) or not re.fullmatch(r"[a-f0-9]{64}", str(sha256)) or not isinstance(size, int) or size <= 0:
            raise RuntimeError("Incomplete identity")
        if path in seen:
            raise RuntimeError(f"Duplicate identity: {path}")
        seen.add(path)
        if ref(path) != item:
            raise RuntimeError(f"Changed frozen input: {path}")


def write_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def cstr(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def compiler_header(files: list[dict[str, object]], modules: list[dict[str, object]], environment: dict[str, object], configs: list[dict[str, object]], protocol_hash: str, expected: dict[str, object]) -> str:
    major, minor = map(int, str(expected["compute_capability"]).split("."))
    identity_material = "".join(f"{item['path']}\\n{item['sha256']}\\n" for item in sorted(files, key=lambda item: str(item["path"])))
    identity_set_hash = hashlib.sha256(identity_material.encode("utf-8")).hexdigest()
    return (
        "#pragma once\n"
        "struct FileIdentity {const char *path; const char *sha256;};\n"
        "static const FileIdentity frozen_files[]={\n"
        + "".join(f" {{{cstr(str(item['path']))},{cstr(str(item['sha256']))}}},\n" for item in sorted(files, key=lambda item: str(item["path"])))
        + "};\nstruct ModuleIdentity {const char *path; const char *hash;};\nstatic const ModuleIdentity frozen_modules[]={\n"
        + "".join(f" {{{cstr(str(item['path']))},{cstr(str(item['sha256']))}}},\n" for item in modules)
        + "};\nstruct EnvIdentity {const char *name; const char *value;};\nstatic const EnvIdentity frozen_environment[]={\n"
        + "".join(f" {{{cstr(name)},{'nullptr' if value is None else cstr(str(value))}}},\n" for name, value in environment.items())
        + "};\nstruct ProbeConfig {const char *id; long long elements; int nodes;};\nstatic const ProbeConfig frozen_configs[]={\n"
        + "".join(f" {{{cstr(str(item['id']))},{int(item['elements'])},{int(item['nodes'])}}},\n" for item in configs)
        + "};\n"
        + f"static const char *protocol_sha256={cstr(protocol_hash)};\n"
        + f"static const char *frozen_file_set_sha256={cstr(identity_set_hash)};\n"
        + f"static const char *expected_gpu_uuid={cstr(str(expected['uuid']))};\n"
        + f"static const char *expected_gpu_name={cstr(str(expected['name']))};\n"
        + f"static const int expected_cc_major={major};\nstatic const int expected_cc_minor={minor};\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="compile only after review; never runs the probe")
    parser.add_argument("--reviewed", action="store_true", help="explicit parent review gate")
    options = parser.parse_args()
    protocol = json.loads((PACKAGE / "protocol.json").read_text(encoding="utf-8-sig"))
    provenance = json.loads((PACKAGE / "source_provenance.json").read_text(encoding="utf-8-sig"))
    if protocol["execution"]["pilot"]["conditions"] != [{"id":"short","settle_ms":0},{"id":"settled","settle_ms":1000}]:
        raise ValueError("complete fixed condition domain required before compiling")
    verify(provenance["files"])
    verify(provenance["native_modules"])
    for copied in provenance["copied_sources"]:
        source = copied["from"]
        target = copied["to"]
        if ref(source["path"]) != source or ref(target["path"]) != target or source["sha256"] != target["sha256"]:
            raise RuntimeError("Copied source identity mismatch")
    if not options.compile:
        print(json.dumps({
            "status": "source_dependencies_verified_not_compiled",
            "configs": len(protocol["configs"]),
            "conditions": protocol["execution"]["pilot"]["conditions"],
            "gpu_access": False,
        }))
        return
    if not options.reviewed:
        raise RuntimeError("Parent source review required before compiling this new probe")
    if any((PACKAGE / output).exists() for output in FORBIDDEN_BUILD_OUTPUTS):
        raise RuntimeError("Refusing to overwrite a build attempt; preserve it and prepare a new revision")

    toolchain = provenance["toolchain"]
    compiler = Path(toolchain["compiler"])
    root = next(parent for parent in PACKAGE.parents if (parent / "pyproject.toml").exists() and (parent / "source").is_dir())
    base_lib = root / "source/llama.cpp-semantic/build-semantic-direct/ggml/src/ggml-base.lib"
    cuda_lib = root / "source/llama.cpp-semantic/build-semantic-direct/ggml/src/ggml-cuda/ggml-cuda.lib"
    common = [
        str(compiler), "/nologo", "/showIncludes", "/std:c++17", "/EHsc", "/O2", "/fp:strict", "/MD", "/utf-8",
        "/DGGML_SHARED", "/DGGML_BACKEND_SHARED", f"/I{root / 'source/llama.cpp-semantic/ggml/include'}", f"/I{toolchain['cuda_include']}",
    ]
    probe = common + [str(PACKAGE / "host_settle_main.cpp"), f"/Fe:{PACKAGE / 'host-settle-probe.exe'}", f"/Fo:{PACKAGE / 'host-settle-probe.obj'}", "/link", str(base_lib), str(cuda_lib), toolchain["cuda_library"], "bcrypt.lib"]
    host = [str(compiler), "/nologo", "/showIncludes", "/std:c++17", "/EHsc", "/O2", "/fp:strict", "/MD", "/utf-8", str(PACKAGE / "reference_host.cpp"), f"/Fe:{PACKAGE / 'reference-host.exe'}", f"/Fo:{PACKAGE / 'reference-host.obj'}"]
    command = (
        "@echo off\ncall " + subprocess.list2cmdline([toolchain["vcvars"]]) + " >nul\nif errorlevel 1 exit /b %errorlevel%\n"
        + subprocess.list2cmdline(probe) + "\nif errorlevel 1 exit /b %errorlevel%\n"
        + subprocess.list2cmdline(host) + "\nexit /b %errorlevel%\n"
    )
    write_new(PACKAGE / "compile.cmd", command)

    tools = [Path(toolchain["vcvars"]), compiler, compiler.parent / "link.exe", Path(toolchain["cuda_library"]), Path(protocol["profiler"]["path"])]
    frozen = {str(item["path"]): item for item in [*provenance["files"], *provenance["native_modules"]]}
    for path in [*(PACKAGE / name for name in SOURCE_FILES), PACKAGE / "compile.cmd", *tools]:
        item = ref(path)
        frozen[str(item["path"])] = item
    environment = dict(protocol["runtime"]["environment"])
    for name in protocol["runtime"]["extra_clear_environment"]:
        environment[name] = None
    header = compiler_header(list(frozen.values()), provenance["native_modules"], environment, protocol["configs"], digest(PACKAGE / "protocol.json"), protocol["runtime"]["gpu_expected"])
    write_new(PACKAGE / "prepared_identity.h", header)

    with (PACKAGE / "compile.log").open("xb") as log:
        result = subprocess.run(["cmd.exe", "/d", "/c", str(PACKAGE / "compile.cmd")], cwd=PACKAGE, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError("Compile failed; preserve compile.log and all outputs; no build manifest published")
    verify(list(frozen.values()))
    raw_log = (PACKAGE / "compile.log").read_bytes()
    try:
        text = raw_log.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_log.decode("cp936")
    includes = {str(Path(match.group(1).strip()).resolve()) for line in text.splitlines() if (match := re.search(r"(?:including file:|包含文件:)\s*(.*)$", line))}
    if len(includes) < 50:
        raise RuntimeError("Incomplete compiler include closure; no build manifest published")
    host_result = subprocess.run([str(PACKAGE / "reference-host.exe")], cwd=PACKAGE, capture_output=True, text=True, check=False)
    try:
        host_validation = json.loads(host_result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Host reference did not return structured evidence") from error
    if host_result.returncode or host_validation.get("pass") is not True or host_validation.get("gpu_access") is not False or host_validation.get("mock_loader_pass") is not True:
        raise RuntimeError("Host mathematical/mock-loader validation failed")
    write_new(PACKAGE / "host_reference_validation.json", json.dumps(host_validation, ensure_ascii=False, indent=2) + "\n")

    files = dict(frozen)
    for path in [*map(Path, includes), PACKAGE / "prepared_identity.h", PACKAGE / "compile.log", PACKAGE / "host-settle-probe.exe", PACKAGE / "reference-host.exe", PACKAGE / "host_reference_validation.json"]:
        item = ref(path)
        files[str(item["path"])] = item
    verify(list(files.values()))
    manifest = {
        "schema": "host-settle-probe-build/v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "compiled_host_tested_not_gpu_executed",
        "gpu_access": False,
        "config_count": len(protocol["configs"]),
        "conditions": protocol["execution"]["pilot"]["conditions"],
        "timed_runs": 0,
        "native_binary_modified": False,
        "native_bin": str(Path(provenance["native_modules"][0]["path"]).parent),
        "executable": ref(PACKAGE / "host-settle-probe.exe"),
        "profiler": ref(protocol["profiler"]["path"]),
        "protocol": ref(PACKAGE / "protocol.json"),
        "compile_header_count": len(includes),
        "host_reference": host_validation,
        "files": sorted(files.values(), key=lambda item: str(item["path"])),
        "source_runtime_equivalence_proven": False,
        "calibration_ready": False,
        "limitations": [
            "Probe is a new binary; original native application binary equivalence is not proven",
            "Actual graph dispatch/counts and replay state await approved CUPTI trace",
            "No GPU timing, hardware diagnosis, inference, coefficient fit, or LLM-time calibration occurred during compile",
        ],
    }
    write_new(PACKAGE / "build_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "frozen_files": len(files), "header_count": len(includes), "exe_sha256": manifest["executable"]["sha256"], "gpu_access": False}))


if __name__ == "__main__":
    main()