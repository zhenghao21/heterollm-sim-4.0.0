"""cuobjdump extraction only. Never starts the compiled correctness executable."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=Path(r"F:\codex_project\37_LLMsim\heterollm-sim-4.0.0")
R6=ROOT/"artifacts/development/native_long_grid_135_20260915/optimization_loop/round_024/mmvq_device_probe/r6_shared_abi"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)

def main():
    receipt=json.loads((HERE/"build_receipt.json").read_text(encoding="utf8"))
    executable=receipt["variants"]["gpu_correctness"]["executable"]
    exe=Path(executable["path"]);assert sha(exe)==executable["sha256"]
    inputs=json.loads((HERE/"build_inputs.json").read_text(encoding="utf8"))
    target=Path(inputs["locked_target_cuda"]["path"]);assert sha(target)==inputs["locked_target_cuda"]["sha256"]
    expected=json.loads((R6/"device_identity.json").read_text(encoding="utf8"))
    directory=HERE/"static_device_code";directory.mkdir(exist_ok=False)
    command=[r"E:\cuda\bin\cuobjdump.exe","--extract-elf","all",str(exe)]
    p=subprocess.run(command,cwd=directory,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (HERE/"static_device_code.log").write_bytes(p.stdout);assert p.returncode==0
    hashes={path:sha(path) for path in directory.glob("*.cubin")}
    result={"schema":"heterollm.mmvq-correctness-static-device-identity/v1","executable":ref(exe),"target_dll":ref(target),"cuobjdump_argv":command,
            "gpu_execution_performed":False,"native_execution_performed":False,"timed_runs":0,"performance_parameters_admitted":0,"runtime_equivalence_verified":False,"pairs":{}}
    for name,item in expected["pairs"].items():
        target_cubin=Path(item["target_path"]);assert sha(target_cubin)==item["target_sha256"]
        matches=[path for path,digest in hashes.items() if digest==item["target_sha256"]]
        assert len(matches)==1,(name,len(matches))
        wrapper=matches[0];assert wrapper.read_bytes()==target_cubin.read_bytes()
        result["pairs"][name]={"target_cubin":ref(target_cubin),"wrapper_cubin":ref(wrapper),"bytes_equal":True}
    with (HERE/"static_device_identity.json").open("x",encoding="utf8") as f:json.dump(result,f,indent=2);f.write("\n")
    print("Correctness executable retains both byte-identical device-code segments; no runtime claim.")
if __name__=="__main__":main()
