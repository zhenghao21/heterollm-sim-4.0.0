"""Static ABI regression; never executes the CUDA probe."""
import hashlib
import json
from pathlib import Path
import subprocess
HERE = Path(__file__).resolve().parent
ROOT = Path(r"F:\codex_project\37_LLMsim\heterollm-sim-4.0.0")

def test_shared_header_is_used_by_both_definition_and_driver():
    receipt = json.loads((HERE / "source_slice_receipt.json").read_text(encoding="utf8"))
    header = HERE / "mmvq_probe_abi.h"
    assert receipt["shared_abi"]["sha256"] == hashlib.sha256(header.read_bytes()).hexdigest()
    for filename in ("mmvq_source_slice.cu", "link_driver.cpp"):
        assert '#include "mmvq_probe_abi.h"' in (HERE / filename).read_text(encoding="utf8")
    assert receipt["gpu_execution_performed"] is False
    assert receipt["timed_runs"] == 0
    assert receipt["link_record"]["returncode"] == 0
    assert receipt["link_record"]["executed"] is False
    proof = receipt["source_slice_proof"]
    prefix = b"".join(Path(proof["path"]).read_bytes().splitlines(keepends=True)[:proof["included_line_end"]])
    assert (HERE / "mmvq_source_slice.cu").read_bytes().startswith(prefix)

def test_legacy_missing_int_prototype_is_compile_error():
    # This was the R4 link-only declaration. The shared header must reject it
    # before link, because C linkage alone does not encode parameter types.
    source = HERE / "negative_legacy_abi.cpp"
    source.write_text('#include "mmvq_probe_abi.h"\nextern "C" void heterollm_mmvq_probe_convert_q8_1(const float *, void *, int, int, int, cudaStream_t);\n', encoding="utf8")
    vcvars = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    command = r'call "{}" >nul && cl /nologo /std:c++17 /EHsc /MD /I"E:\cuda\include" /c "{}" /Fo"{}"'.format(vcvars, source, HERE / "negative_legacy_abi.obj")
    proc = subprocess.run(command, shell=True, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (HERE / "negative_legacy_abi.log").write_bytes(proc.stdout)
    assert proc.returncode != 0, "old mismatched prototype unexpectedly compiled"
    assert b"C2733" in proc.stdout or b"C conflicting" in proc.stdout
