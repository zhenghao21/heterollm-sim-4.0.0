"""Post-barrier R30 group and paired report; never run predictions or native."""
from pathlib import Path
from collections import Counter
import importlib.util
import json

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("r30_reporting_runner",P/"run_candidate.py")
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
s=r.s

def paired_rows(before,after):
    s.require(set(before)==set(after),"paired scene universes differ")
    rows=[]
    for ident in sorted(before):
        for metric in s.METRICS:
            a,b=before[ident]["metrics"][metric],after[ident]["metrics"][metric]
            common=a["status"]==b["status"]=="scored"
            row={"cell_id":ident,"metric":metric,"before_status":a["status"],"after_status":b["status"],
                 "outcome":"unscored","native_equal":None}
            if common:
                s.require(a["native_median_ms"]==b["native_median_ms"] and a["native_run_medians_ms"]==b["native_run_medians_ms"],"paired native observations differ")
                delta=b["absolute_percentage_error_pct"]-a["absolute_percentage_error_pct"]
                row.update(native_equal=True,ape_delta_percentage_points=delta,
                    predicted_time_delta_ms=b["simulator_median_ms"]-a["simulator_median_ms"],
                    outcome="improved" if delta<0 else "regressed" if delta>0 else "unchanged")
            rows.append(row)
    return {"fixed_metric_denominator":len(before)*len(s.METRICS),
            "outcomes":dict(Counter(x["outcome"] for x in rows)),"rows":rows}

def summarize_group(rows):
    result=s.summarize(rows)
    result["group_denominator"]=len(rows)
    result["full_campaign_denominator"]=result.pop("full131_denominator")
    result["scored_cells"]=len(rows)-result["failed_or_unscored_cells"]
    return result

def render(report):
    lines=["# R30 paired nominal-HBM evaluation", "", "Development only; fixed native 131 cells. Independent B: unvalidated.", "",
           "|Group|Arm|Cells|Scored|All three <10%|TTFT median/P90/max %|TPOT median/P90/max %|E2E median/P90/max %|",
           "|---|---|---:|---:|---:|---|---|---|"]
    for group,data in report["groups"].items():
        for arm in r.ARMS:
            summary=data["arms"][arm]
            def values(metric):
                d=summary["metrics"][metric]["absolute_percentage_error_pct"]
                return "/".join("NA" if d[k] is None else f"{d[k]:.3f}" for k in ("median","p90","worst"))
            lines.append("|"+"|".join([group,arm,str(summary["group_denominator"]),str(summary["scored_cells"]),str(summary["strict_all3_below10_cells"]),*[values(m) for m in s.METRICS]])+"|")
    lines += ["", "Pairwise outcomes use only common scored metrics; all unscored entries remain in the fixed denominator.",
              "Signed error and absolute milliseconds (median/P90/worst), all failures, and exact paired deltas are in the JSON.",
              "No latency targets were used by this summarizer to select scenes, alter predictions or fit coefficients.", ""]
    return "\n".join(lines)

def main():
    s.require((P/"report.json").is_file(),"completed two-arm scoring required")
    base,bref=s.read_json(P/"report.json")
    barrier,barrier_ref=r.barrier()
    s.require(s.verify_reference(base["barrier_ref"])==barrier_ref,"scoring/barrier identity differs")
    scores={};bundles={}
    for arm in r.ARMS:
        bundles[arm]=s.load_bundle(P/arm,"retained",barrier["arms"][arm],full=True)
        scores[arm]=s.load_score(bundles[arm],"errors.0001.json",bundles[arm]["cells"],base["arms"][arm]["score_ref"])
        s.require(s.timestamp(scores[arm]["created_utc"])>=s.timestamp(barrier["created_utc"]),"score predates barrier")
    rows={arm:scores[arm]["rows"] for arm in r.ARMS}
    grouped={}
    for ident,cell in bundles["on"]["cells"].items():
        key=cell["model_key"]+" / "+cell["deployment"]
        grouped.setdefault(key,[]).append(ident)
    groups={}
    for key,ids in sorted(grouped.items()):
        subset={arm:{i:rows[arm][i] for i in ids} for arm in r.ARMS}
        groups[key]={"arms":{arm:summarize_group(subset[arm]) for arm in r.ARMS},
                     "paired":paired_rows(subset["off"],subset["on"])}
    all_summary={arm:summarize_group(rows[arm]) for arm in r.ARMS}
    report={"schema":"r30-grouped-paired-errors/v1","created_utc":r.now(),"scoring_report_ref":bref,"barrier_ref":barrier_ref,
            "helper_refs":[s.reference(__file__),s.reference(P/"run_candidate.py"),s.reference(r.f.LOOP/"round_023/summarize_ablation.py")],
            "fixed_cells":131,"fixed_metrics_per_arm":393,"arms":all_summary,"groups":groups,"paired":paired_rows(rows["off"],rows["on"]),
            "gate_A":"passed" if all_summary["on"]["strict_all3_below10_cells"]==131 else "not_passed",
            "gate_B":"unvalidated","formal_success":False,"native_remeasurement":False,"target_latency_fitting":False,
            "prior_campaigns":"R27 and earlier preserved; no retroactive acceptance"}
    r.barrier();s.verify_reference(bref)
    r.f.write_new(P/"grouped_paired_report.json",report)
    with (P/"grouped_paired_report.md").open("x",encoding="utf-8") as out:out.write(render(report))
    print(json.dumps({"gate_A":report["gate_A"],"candidate_passed":all_summary["on"]["strict_all3_below10_cells"],"paired_outcomes":report["paired"]["outcomes"]}))
if __name__=="__main__":main()
