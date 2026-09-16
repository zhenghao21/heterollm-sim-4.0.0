"""Build/link only. This command never invokes either resulting executable."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(r"F:\codex_project\37_LLMsim\heterollm-sim-4.0.0")
HERE = Path(__file__).resolve().parent
R6 = ROOT / "artifacts/development/native_long_grid_135_20260915/optimization_loop/round_024/mmvq_device_probe/r6_shared_abi"
SEMANTIC = ROOT / "source/llama.cpp-semantic"
NATIVE = ROOT / "source/llama.cpp-native-thread-control"
NATIVE_BIN = NATIVE / "build-native-thread-control/bin"
LIBDIR = SEMANTIC / "build-semantic-direct/ggml/src"
CUDA = Path(r"E:\cuda")
MSVC = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64")
VCVARS = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}
def write_json(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
def run(argv, log):
    command = 'call "{}" >nul && {}'.format(VCVARS, subprocess.list2cmdline([str(v) for v in argv]))
    result = subprocess.run(command, shell=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    Path(log).write_bytes(result.stdout)
    record = {"argv": [str(v) for v in argv], "shell_prelude": str(VCVARS), "returncode": result.returncode, "log": ref(log)}
    if result.returncode:
        raise RuntimeError("compile/link/static inspection failed: " + str(log))
    return record

def main():
    r6_receipt=json.loads((R6/"source_slice_receipt.json").read_text(encoding="utf8"))
    assert r6_receipt["schema"]=="heterollm.mmvq-shared-abi-link/v1"
    assert sha(R6/"mmvq_probe_abi.h")==r6_receipt["shared_abi"]["sha256"]
    assert sha(SEMANTIC/"ggml/src/ggml-cuda/mmvq.cu")==r6_receipt["source_slice_proof"]["sha256"]
    assert sha(SEMANTIC/"ggml/src/ggml-cuda/quantize.cu")==r6_receipt["source_files"]["quantize.cu"]["sha256"]
    assert sha(NATIVE_BIN/"ggml-cuda.dll")==r6_receipt["target_dll"]["sha256"]
    r6_before={str(p):sha(p) for p in R6.iterdir() if p.is_file()}
    expected={name:ref(NATIVE_BIN/name) for name in ["ggml-base.dll","ggml-cpu.dll","ggml.dll"]}
    expected["cudart64_12.dll"]=ref(CUDA/"bin/cudart64_12.dll")
    # Native-thread-control is an overlay: ggml-base is copied unchanged.
    # Do not invent a second complete source tree or claim file equality to it.
    native_build_path=NATIVE/"evidence/build_receipt.json"
    native_build=json.loads(native_build_path.read_text(encoding="utf8"))
    base_native=NATIVE_BIN/"ggml-base.dll"
    base_semantic=SEMANTIC/"build-semantic-direct/bin/ggml-base.dll"
    assert sha(base_native)==sha(base_semantic)
    assert native_build["unchanged_runtime_sha256"][str(base_native)]==sha(base_native)
    cpu_provider_proof={"native_build_receipt":ref(native_build_path),"semantic_ggml_base":ref(base_semantic),"native_ggml_base":ref(base_native),"bytes_equal":True,"source_tree_owner":str(SEMANTIC)}
    objects=[R6/name for name in ["mmvq_source_slice.obj","quantize_same_source.obj","mmvq_runtime_support.obj"]]
    direct_sources=[HERE/"correctness_driver.cpp",HERE/"cpu_reference.h",HERE/"build_driver.py",
        R6/"mmvq_probe_abi.h",R6/"mmvq_source_slice.cu",R6/"mmvq_runtime_support.cu",R6/"source_slice_receipt.json",R6/"device_identity.json",
        SEMANTIC/"ggml/include/ggml.h",SEMANTIC/"ggml/src/ggml-common.h",SEMANTIC/"ggml/src/ggml-quants.c",
        SEMANTIC/"ggml/src/ggml-cuda/mmvq.cu",SEMANTIC/"ggml/src/ggml-cuda/quantize.cu",SEMANTIC/"ggml/src/ggml-cuda/vecdotq.cuh",
        SEMANTIC/"vendor/nlohmann/json.hpp",CUDA/"include/cuda_runtime_api.h",CUDA/"include/cuda_fp16.h",
        SEMANTIC/"build-semantic-direct/compile_commands.json",LIBDIR/"ggml-base.lib",LIBDIR/"ggml-cpu.lib",LIBDIR/"ggml.lib",
        CUDA/"lib/x64/cudart.lib",CUDA/"lib/x64/cuda.lib",MSVC/"cl.exe",MSVC/"link.exe",native_build_path] + objects
    inputs={"schema":"heterollm.mmvq-correctness-build-inputs/v1", "created_at_utc":datetime.now(timezone.utc).isoformat(),
        "input_refs":[ref(p) for p in direct_sources],"expected_runtime_dlls":expected,"locked_target_cuda":ref(NATIVE_BIN/"ggml-cuda.dll"),
        "cpu_quantizer_provider_proof":cpu_provider_proof,"fixture":{"weight_type":"Q5_0","M":1,"K":4096,"N":3072,"padded_K":4096},
        "original_r6_objects_recompiled":False,"gpu_execution_performed":False,"timed_runs":0,"target_llm_latency_used":False}
    inputs_path=HERE/"build_inputs.json";write_json(inputs_path,inputs)
    header='#pragma once\n#define BUILD_INPUTS_PATH R"IDENT({})IDENT"\n#define BUILD_INPUTS_SHA256 "{}"\n'.format(inputs_path.as_posix(),sha(inputs_path))
    (HERE/"build_identity.h").write_text(header,encoding="utf8")
    result={"schema":"heterollm.mmvq-correctness-build/v1","started_at_utc":datetime.now(timezone.utc).isoformat(),
        "status":"building_not_runtime_qualified","build_inputs":ref(inputs_path),"build_identity_header":ref(HERE/"build_identity.h"),
        "gpu_driver_executed":False,"gpu_execution_performed":False,"native_execution_performed":False,"timed_runs":0,"variants":{}}
    try:
        for label,cpu in [("gpu_correctness",False),("cpu_self_test",True)]:
            obj=HERE/(label+".obj");exe=HERE/("mmvq_"+label+".exe");deps=HERE/(label+".dependencies.json")
            includes=[R6,HERE,SEMANTIC/"ggml/include",SEMANTIC/"ggml/src",SEMANTIC/"vendor",CUDA/"include"]
            compile_args=[MSVC/"cl.exe","/nologo","/std:c++17","/EHsc","/MD","/O2","/fp:strict","/DGGML_SHARED"]
            if cpu:compile_args += ["/DMMVQ_CPU_SELF_TEST_ONLY"]
            compile_args += ["/I"+str(v) for v in includes]+["/sourceDependencies",deps,"/c",HERE/"correctness_driver.cpp","/Fo"+str(obj)]
            compile_record=run(compile_args,HERE/(label+".compile.log"))
            link_args=[MSVC/"link.exe","/nologo","/OPT:REF","/OUT:"+str(exe),obj,"/LIBPATH:"+str(LIBDIR),"ggml-base.lib","bcrypt.lib"]
            if not cpu:link_args += objects+["/LIBPATH:"+str(CUDA/"lib/x64"),"cudart.lib","cuda.lib","ggml-cpu.lib","ggml.lib"]
            link_record=run(link_args,HERE/(label+".link.log"))
            dependency_record=run([MSVC/"dumpbin.exe","/DEPENDENTS",exe],HERE/(label+".imports.log"))
            imported=(HERE/(label+".imports.log")).read_text(encoding="utf8",errors="replace").lower()
            if cpu:assert "cudart" not in imported and "nvcuda" not in imported and "ggml-cuda" not in imported,"CPU self-test imports CUDA"
            actual_deps=json.loads(deps.read_text(encoding="utf-8-sig"))["Data"]["Includes"]
            result["variants"][label]={"compile":compile_record,"link":link_record,"import_inspection":dependency_record,
                "executable":ref(exe),"object":ref(obj),"compiler_dependency_file":ref(deps),"compiler_discovered_includes":[ref(p) for p in actual_deps],
                "cuda_imports_absent":cpu,"executed":False}
        result["status"]="compiled_and_linked_runtime_unverified"
    finally:
        r6_after={str(p):sha(p) for p in R6.iterdir() if p.is_file()}
        result["r6_files_unchanged"]=r6_before==r6_after
        result["finished_at_utc"]=datetime.now(timezone.utc).isoformat();write_json(HERE/"build_receipt.json",result)
        assert r6_before==r6_after,"read-only R6 changed during this build"
    print(json.dumps({"status":result["status"],"r6_files_unchanged":True,"gpu_execution_performed":False}))

if __name__=="__main__":main()
