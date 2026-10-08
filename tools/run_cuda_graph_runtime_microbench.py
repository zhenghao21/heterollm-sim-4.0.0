"""Build and run the standalone CUDA Graph runtime microbenchmark.

Example (from a VS developer shell):
  python tools/run_cuda_graph_runtime_microbench.py --output graph_runtime.json

The output contains synthetic graph measurements only; it never loads a model.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "cuda_graph_runtime_microbench.cu"


def build_microbenchmark(directory, nvcc, vsdevcmd):
    directory = Path(directory)
    exe = directory / "cuda_graph_runtime_microbench.exe"
    build_bat = directory / "build.cmd"
    build_bat.write_text(f'@call "{vsdevcmd}" -arch=amd64 -host_arch=amd64 >nul\r\n'
                        f'@"{nvcc}" -O2 -std=c++17 -arch=native "{SOURCE}" -o "{exe}"\r\n', encoding="utf-8")
    try:
        build = subprocess.run(["cmd.exe", "/d", "/c", str(build_bat)], cwd=directory, text=True,
                               capture_output=True, encoding="utf-8", errors="replace")
    finally:
        build_bat.unlink(missing_ok=True)
    if build.returncode:
        raise RuntimeError(f"nvcc build failed ({build.returncode}):\n{build.stdout}\n{build.stderr}")
    return exe


def complete_identity(data):
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as cpu_key:
            data["cpu_id"] = winreg.QueryValueEx(cpu_key, "ProcessorNameString")[0].strip()
    data["os_id"] = platform.platform()
    driver = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=driver_version", "--format=csv,noheader"],
                            text=True, capture_output=True, check=True)
    data["driver_version"] = driver.stdout.strip()
    data["runtime_id"] = f"CUDA-{data['runtime_version']}-driver-{data['driver_version']}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train", default="8,32,128,256,512,1024,2048,4096,8192")
    p.add_argument("--holdout", default="64,192,384,768,1536,3072,6144")
    p.add_argument("--repetitions", type=int, default=45)
    p.add_argument("--topologies", default="chain,fork_join")
    p.add_argument("--nvcc", default=os.environ.get("CUDA_PATH", r"E:\cuda") + r"\bin\nvcc.exe")
    p.add_argument("--vsdevcmd", default=r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat")
    args = p.parse_args()
    if args.repetitions < 4:
        p.error("--repetitions must be at least 4")
    with tempfile.TemporaryDirectory(prefix="heterollm-graphbench-") as td:
        exe = build_microbenchmark(td, args.nvcc, args.vsdevcmd)
        outputs = []
        identity = None
        for topology in [x.strip() for x in args.topologies.split(",") if x.strip()]:
            run = subprocess.run([str(exe), "--train", args.train, "--holdout", args.holdout,
                                  "--topology", topology, "--repetitions", str(args.repetitions)],
                                 cwd=ROOT, text=True, capture_output=True, encoding="utf-8", errors="replace")
            if run.returncode:
                raise SystemExit(f"CUDA benchmark failed for {topology} ({run.returncode}):\n{run.stderr}")
            result = json.loads(run.stdout)
            current = {k: v for k, v in result.items() if k != "samples"}
            if identity is None:
                identity = current
            elif identity != current:
                raise SystemExit("hardware/runtime identity changed during measurements")
            outputs.extend(result["samples"])
    data = dict(identity or {})
    # cudaDriverGetVersion reports supported CUDA API level, not the installed
    # driver release. Record the actual release to prevent cross-driver reuse.
    complete_identity(data)
    data["samples"] = outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"saved {len(outputs)} graph-size/topology samples to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
