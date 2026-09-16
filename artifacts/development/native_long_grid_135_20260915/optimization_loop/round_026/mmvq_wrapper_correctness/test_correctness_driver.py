"""CPU-only and static acceptance. This file never invokes the GPU executable."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import pytest

HERE=Path(__file__).resolve().parent
ROOT=Path(r"F:\codex_project\37_LLMsim\heterollm-sim-4.0.0")
NATIVE_BIN=ROOT/"source/llama.cpp-native-thread-control/build-native-thread-control/bin"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inputs():return json.loads((HERE/"build_inputs.json").read_text(encoding="utf8"))
def build():return json.loads((HERE/"build_receipt.json").read_text(encoding="utf8"))
def run_cpu(output, mode="--self-test"):
    rec=build()["variants"]["cpu_self_test"]["executable"]
    exe=Path(rec["path"]);assert sha(exe)==rec["sha256"]
    imports=(HERE/"cpu_self_test.imports.log").read_text(encoding="utf8").lower()
    assert all(v not in imports for v in ("cudart","nvcuda","ggml-cuda"))
    env=os.environ.copy();env["PATH"]=str(NATIVE_BIN)+os.pathsep+env.get("PATH","")
    return subprocess.run([str(exe),mode,"--output",str(output)],cwd=HERE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)

@pytest.fixture(scope="module")
def cpu_run(tmp_path_factory):
    directory=tmp_path_factory.mktemp("mmvq_cpu_only")
    output=directory/"result.json"
    proc=run_cpu(output)
    (directory/"process.log").write_bytes(proc.stdout)
    assert proc.returncode==0,proc.stdout.decode("utf8",errors="replace")
    doc=json.loads(output.read_text(encoding="utf8"))
    # Retain an identity-bearing copy of the CPU-only evidence for review. No GPU executable is invoked.
    (HERE/"cpu_self_test_result.json").write_bytes(output.read_bytes())
    return directory,output,doc

def test_build_is_compile_only_and_r6_unchanged():
    receipt=build()
    assert receipt["status"]=="compiled_and_linked_runtime_unverified"
    assert receipt["gpu_driver_executed"] is False
    assert receipt["gpu_execution_performed"] is False and receipt["timed_runs"]==0
    assert receipt["r6_files_unchanged"] is True
    for entry in receipt["variants"].values():
        assert entry["compile"]["returncode"]==entry["link"]["returncode"]==0
        assert sha(entry["executable"]["path"])==entry["executable"]["sha256"]
    assert receipt["variants"]["cpu_self_test"]["cuda_imports_absent"] is True

def test_cpu_reference_fixture_and_negative_cases(cpu_run):
    _,_,d=cpu_run
    assert d["status"]=="cpu_reference_self_test_passed_gpu_unverified"
    assert d["cuda_api_calls_started"] is False and d["gpu_execution_performed"] is False
    assert d["cpu_self_test_cuda_modules_absent"] is True
    assert d["runtime_equivalence_verified"] is False and d["performance_parameters_admitted"]==0
    assert d["fixture"]==dict(weight_type="Q5_0",ggml_type_id=6,K=4096,N=3072,M=1,padded_K=4096,q8_1_stride_blocks=128,seed=0x51354d31)
    assert d["cpu_self_test"]["packed_weight_bytes"]==8650752
    assert d["cpu_self_test"]["expected_q8_1_bytes"]==4608
    assert d["cpu_self_test"]["all_input_elements_nonzero"] and d["cpu_self_test"]["all_weight_elements_nonzero"]
    assert len(d["cpu_self_test"]["negative_checks"])==5
    assert d["tolerance"]["selected_before_observing_gpu_output"] is True

def test_independent_python_q8_1_layout(cpu_run):
    _,_,d=cpu_run
    raw=bytearray();raw_f32=bytearray()
    for b in range(128):
        delta=math.ldexp(1.,-10+b%4)
        q=[]
        for p in range(15):
            v=1+(b*17+p*29)%120;q.extend([v,-v])
        q.extend([127,-(64+b%63)])
        x=[v*delta for v in q]
        assert all(v!=0 for v in x)
        assert max(map(abs,x))/127==delta
        assert sum(x)==delta*sum(q)!=0
        raw.extend(struct.pack("<ee32b",delta,sum(x),*q))
        raw_f32.extend(struct.pack("<32f",*x))
    assert hashlib.sha256(raw).hexdigest()==d["fixture_hashes"]["expected_q8_1"]
    assert hashlib.sha256(raw_f32).hexdigest()==d["fixture_hashes"]["input_f32"]

def test_output_creation_is_exclusive(cpu_run):
    _,output,_=cpu_run
    before=output.read_bytes()
    proc=run_cpu(output)
    assert proc.returncode==2
    assert b"exclusive output creation failed" in proc.stdout
    assert output.read_bytes()==before

def test_cpu_binary_rejects_gpu_mode_before_output_creation(tmp_path):
    output=tmp_path/"not_created.json"
    proc=run_cpu(output,"--run-correctness")
    assert proc.returncode==2 and b"CPU executable supports only --self-test" in proc.stdout
    assert not output.exists()
