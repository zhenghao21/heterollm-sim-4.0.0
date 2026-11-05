"""Achieved device-memory bandwidth of each case's selected native point."""
import glob, json, sys
from pathlib import Path

scan = Path(sys.argv[1]); out = Path(sys.argv[2])
NOMINAL = {"HBM4": 12800., "CONV": 1600. + 2450., "HBF": 1600. + 11200., "CONV+": 2800., "HBF+": 12800.}
data = json.loads((out / "data.json").read_text(encoding="utf-8"))
for row in data["rows"]:
    L = row["layout"]; hw = row["hardware"].replace("+", "_plus")
    stem = "native__%s__%s__%s__G8__DP%d_TP%d_PP%d_EP%d" % (row["model"], hw, row["workload"], L["dp"], L["tp"], L["pp"], L["ep"])
    files = [f for f in glob.glob(str(scan / "summaries" / (stem + "*__B%d.json" % row["batch_per_replica"])))]
    assert len(files) == 1, (stem, files)
    d = json.loads(Path(files[0]).read_text(encoding="utf-8"))
    budget = d["approximation"]["error_budget"]; ranks = 8 // L["dp"]
    total = context = 0.
    for c in budget["step"]["classes"]:
        io = json.loads(c["class_key"])["native"]["owner_io_shape"]; layers = int(c["layers"])
        total += layers * sum(v["read_bytes"] + v["write_bytes"] for v in io.values())
        context += layers * sum(v.get("context_read_bytes", 0) for v in io.values())
    step = budget["step_service_ns"]["lower"]
    row["native_step_bytes_per_gpu"] = total / ranks
    row["native_context_bytes_per_gpu"] = context / ranks
    row["native_achieved_tb_s"] = total / ranks / step / 1e3
    row["nominal_tb_s"] = NOMINAL[row["hardware"]] / 1e3
    row["native_bandwidth_fraction"] = row["native_achieved_tb_s"] / row["nominal_tb_s"]
(out / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
for r in data["rows"]:
    print("%-16s %-5s %-5s %6.1f GB/step/GPU ctx %5.1f  %5.2f TB/s of %5.2f = %.2f" % (r["model"], r["hardware"], r["workload"],
          r["native_step_bytes_per_gpu"] / 1e9, r["native_context_bytes_per_gpu"] / 1e9, r["native_achieved_tb_s"], r["nominal_tb_s"], r["native_bandwidth_fraction"]))
