"""Presentation only, after existing R30 full262 barrier AND both scores.
Never predicts, scores, modifies native, or selects scenarios. Imports plotting
libraries only in the heatmaps action. Archive entry is archive_verified.py.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

ROUND = Path(__file__).resolve().parent.parent
METRICS = ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms")
LABELS = ("Engine TTFT", "Engine TPOT", "Engine E2E")
PROMPTS, OUTPUTS, CONCURRENCY = (128, 512, 1536), (32, 128, 256), (1, 2, 4)


def require(value, message):
    if not value: raise ValueError(message)


def now(): return datetime.now(timezone.utc).isoformat()


def ref(path):
    path = Path(path).resolve(); h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write("\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def frozen_scorer_ref(s, freeze):
    # source_content verifies the complete snapshot map and each actual byte SHA.
    closure = s.source_content(freeze)
    relative = "tools/predict_stable_native_dataset.py"
    require(relative in closure, "frozen scorer source missing")
    actual = ref(Path(freeze["source"]["root"])/relative)
    require(closure[relative] == {"sha256": actual["sha256"], "bytes": actual["bytes"]}, "frozen scorer identity differs")
    return actual


def validate_scoring_receipt(base):
    # R30's frozen scorer receipt declares gate_B, not a blind field.
    require(base.get("schema") == "r30-two-arm-scoring-receipt/v1" and base.get("denominator_per_arm") == 131 and
            set(base.get("arms", {})) == {"off", "on"} and base.get("formal_success") is False and
            base.get("gate_B") == "unvalidated", "unexpected completed two-arm scoring receipt")


def load_completed():
    # Missing completion files fail before importing campaign logic or opening a
    # score/prediction payload. A currently running campaign is never inspected.
    for name in ("predictions_complete.json", "report.json", "grouped_paired_report.json"):
        require((ROUND/name).is_file(), "existing completed scoring/group report required: " + name)
    r = load_module("r30_postprocess_runner", ROUND/"run_candidate.py")
    s = r.s
    barrier, barrier_ref = r.barrier()  # validates all 262 retained terminals/identities
    base, base_ref = s.read_json(ROUND/"report.json")
    validate_scoring_receipt(base)
    require(s.verify_reference(base["barrier_ref"]) == barrier_ref, "score/barrier identity differs")
    grouped, grouped_ref = s.read_json(ROUND/"grouped_paired_report.json")
    require(grouped.get("schema") == "r30-grouped-paired-errors/v1" and grouped.get("fixed_cells") == 131 and
            grouped.get("fixed_metrics_per_arm") == 393 and grouped.get("gate_B") == "unvalidated" and grouped.get("formal_success") is False,
            "grouped report scope differs")
    require(s.verify_reference(grouped["scoring_report_ref"]) == base_ref and
            s.verify_reference(grouped["barrier_ref"]) == barrier_ref, "grouped report bound to another scoring/barrier")
    helper_refs = [s.verify_reference(x) for x in grouped["helper_refs"]]
    expected_helpers = {str((ROUND/"summarize_groups.py").resolve()), str((ROUND/"run_candidate.py").resolve()),
                        str((ROUND.parent/"round_023/summarize_ablation.py").resolve())}
    require({x["path"] for x in helper_refs} == expected_helpers, "unexpected grouped-report helper identity")
    bundles, rows, scores, scorers = {}, {}, {}, {}
    for arm in ("off", "on"):
        bundles[arm] = s.load_bundle(ROUND/arm, "retained", barrier["arms"][arm], full=True)
        scorers[arm] = frozen_scorer_ref(s, bundles[arm]["freeze"])
        checked = s.load_score(bundles[arm], "errors.0001.json", bundles[arm]["cells"], base["arms"][arm]["score_ref"])
        require(s.timestamp(checked["created_utc"]) >= s.timestamp(barrier["created_utc"]), "score predates full262 barrier")
        rows[arm], scores[arm] = checked["rows"], checked["ref"]
    require(set(bundles["off"]["cells"]) == set(bundles["on"]["cells"]), "paired fixed131 mask differs")
    g = load_module("r30_postprocess_grouped", ROUND/"summarize_groups.py")
    require(grouped["paired"] == g.paired_rows(rows["off"], rows["on"]), "saved paired deltas differ from existing summarizer")
    evidence = {"barrier_ref": barrier_ref, "scoring_report_ref": base_ref, "grouped_report_ref": grouped_ref,
                "score_refs": scores, "frozen_scorer_refs": scorers, "grouped_helper_refs": helper_refs,
                "native_selection_sha256": s.SELECTION_SHA,
                "scorer_identity_scope": "actual frozen scorer source bytes verified; saved scores and arithmetic revalidated by existing load_score; no new scorer process is executed"}
    r.barrier(); s.verify_reference(base_ref); s.verify_reference(grouped_ref)
    return {"cells": bundles["on"]["cells"], "rows": rows, "paired": grouped["paired"], "evidence": evidence}


def check_unchanged(evidence):
    require(load_completed()["evidence"] == evidence, "completed inputs changed during postprocessing")


def coordinates(ident):
    match = re.search(r"_p(128|512|1536)_o(32|128|256)_c(1|2|4)(?:__|$)", ident)
    require(match is not None, "cell outside fixed display grid: " + ident)
    p, o, c = map(int, match.groups())
    return CONCURRENCY.index(c), PROMPTS.index(p)*3 + OUTPUTS.index(o)


def prepare_panels(context):
    cells, scores = context["cells"], context["rows"]
    require(len(cells) == 131 and set(scores) == {"off", "on"}, "fixed131/two-arm data required")
    require(all(set(scores[a]) == set(cells) for a in scores), "score mask differs from frozen131")
    pairs = context["paired"]
    require(pairs.get("fixed_metric_denominator") == 393 and len(pairs.get("rows", [])) == 393, "complete paired metric mask required")
    pairmap = {(r["cell_id"], r["metric"]): r for r in pairs["rows"]}
    require(set(pairmap) == {(i,m) for i in cells for m in METRICS} and len(pairmap) == 393, "duplicate/missing paired metric identity")
    grouped = {}
    for ident, cell in cells.items():
        group = cell["model_key"] + " / " + cell["deployment"]
        y, x = coordinates(ident)
        grouped.setdefault(group, {})
        require((y,x) not in grouped[group], "two selected cells occupy one display coordinate")
        grouped[group][y,x] = ident
    require(len(grouped) == 6, "exactly six fixed model/deployment groups required")
    views = {name: {} for name in ("off", "on", "delta")}
    for group, selected in sorted(grouped.items()):
        for view in views:
            views[view][group] = {}
            for metric in METRICS:
                panel = {}
                for y in range(3):
                    for x in range(9):
                        ident = selected.get((y,x))
                        item = {"cell_id": ident, "selected": ident is not None, "value": None, "label": "-"}
                        if ident is not None:
                            a, b = scores["off"][ident]["metrics"][metric], scores["on"][ident]["metrics"][metric]
                            require(a["status"] in {"scored","unscored"} and b["status"] in {"scored","unscored"}, "unknown metric status")
                            pair = pairmap[ident,metric]
                            require(pair["before_status"] == a["status"] and pair["after_status"] == b["status"], "paired status differs")
                            value = None
                            if view == "delta":
                                if a["status"] == b["status"] == "scored":
                                    require(pair.get("native_equal") is True, "paired native observations differ")
                                    value = pair["ape_delta_percentage_points"]
                                    require(value == b["absolute_percentage_error_pct"] - a["absolute_percentage_error_pct"], "paired delta differs")
                            else:
                                record = a if view == "off" else b
                                if record["status"] == "scored": value = record["absolute_percentage_error_pct"]
                            require(value is None or type(value) in (int,float) and math.isfinite(value), "nonfinite display value")
                            if value is not None and view != "delta": require(value >= 0, "negative absolute error")
                            label = "X" if value is None else (f"{value:+.1f}" if view == "delta" else "<10" if value < 10 and round(value,1) == 10 else f"{value:.1f}")
                            item.update(value=value,label=label,off_status=a["status"],on_status=b["status"])
                        panel[y,x] = item
                views[view][group][metric] = panel
    return views


def heatmaps(context, output):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
    panels = prepare_panels(context)
    output = Path(output).resolve(); require(output.is_relative_to(ROUND.resolve()), "figures must stay inside this round")
    output.mkdir(exist_ok=False)
    ape_cmap = LinearSegmentedColormap.from_list("r30_error", [(0,"#e8f4ef"),(.09999,"#6bb69a"),(.1,"#fce4a3"),(.2,"#f3b15e"),(.5,"#d76c5e"),(1,"#913446")])
    ape_cmap.set_bad("#e4e7eb")
    delta_cmap = LinearSegmentedColormap.from_list("r30_delta", ["#17614e", "#f7f7f7", "#a63240"]); delta_cmap.set_bad("#e4e7eb")
    delta_top = max([abs(i["value"]) for g in panels["delta"].values() for p in g.values() for i in p.values() if i["value"] is not None]+[1.0])
    outputs = []; failed_masks = {}
    for view, groups in panels.items():
        is_delta = view == "delta"; norm = TwoSlopeNorm(0,-delta_top,delta_top) if is_delta else Normalize(0,100)
        fig, axes = plt.subplots(len(groups),3,figsize=(18,3.15*len(groups)),squeeze=False)
        failed_masks[view] = {m: [] for m in METRICS}
        for ri,(group,data) in enumerate(groups.items()):
            for ci,metric in enumerate(METRICS):
                ax = axes[ri,ci]; values = np.full((3,9),np.nan)
                for (y,x),item in data[metric].items():
                    if item["value"] is not None: values[y,x] = item["value"]
                    if item["selected"] and item["value"] is None: failed_masks[view][metric].append(item["cell_id"])
                image = ax.imshow(values,cmap=delta_cmap if is_delta else ape_cmap,norm=norm,aspect="auto")
                for (y,x),item in data[metric].items():
                    value=item["value"]; white=value is not None and (abs(value)>delta_top*.6 if is_delta else value>45)
                    ax.text(x,y,item["label"],ha="center",va="center",fontsize=8,color="white" if white else "#24313d")
                ax.set_xticks(range(9),[f"{p}/{o}" for p in PROMPTS for o in OUTPUTS],rotation=55,ha="right",fontsize=8)
                ax.set_yticks(range(3),["C1","C2","C4"]);ax.set_title(group+" | "+LABELS[ci],loc="left",fontsize=10,fontweight="bold")
                ax.set_xlabel("Prompt / Output tokens",fontsize=8)
                ax.set_xticks(np.arange(-.5,9,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True);ax.grid(which="minor",color="white",linewidth=1);ax.tick_params(which="minor",bottom=False,left=False)
        title = "APE change: on - off (percentage points)" if is_delta else f"{view.upper()} | Engine absolute relative error (%)"
        fig.suptitle("R30 MMVQ HBM ablation | "+title,fontsize=15,y=.995)
        fig.subplots_adjust(left=.045,right=.94,top=.965,bottom=.105,hspace=.78,wspace=.2)
        cax=fig.add_axes([.955,.15,.013,.7]);fig.colorbar(image,cax=cax,label="Negative = improved; positive = regressed" if is_delta else "Absolute error (%) | color capped at 100; strict target <10")
        fig.text(.045,.012,"Fixed native development mask: 131 cells. - = outside fixed mask; X = selected but failed/unscored"+(" in either arm." if is_delta else ".")+"\nNo failed cells removed; no new scoring or fitting. Independent acceptance B remains unvalidated.",fontsize=10,color="#354557")
        stem = "delta_engine_error_heatmap" if is_delta else view+"_engine_error_heatmap"
        for suffix in ("png","svg"):
            target=output/(stem+"."+suffix)
            with target.open("xb") as file: fig.savefig(file,format=suffix,dpi=170,facecolor="white")
            outputs.append(ref(target))
        plt.close(fig)
    check_unchanged(context["evidence"])
    receipt={"schema":"r30-presentation-only-heatmaps/v1","created_utc":now(),"source_evidence":context["evidence"],"postprocessor_ref":ref(__file__),
             "fixed_cell_denominator":131,"fixed_metric_denominator_per_arm":393,"fixed_deployment_groups":6,"fixed_mask":sorted(context["cells"]),
             "fixed_mask_sha256":hashlib.sha256(json.dumps(sorted(context["cells"]),separators=(",",":")).encode()).hexdigest(),
             "failed_or_unscored_display_masks":failed_masks,"delta_definition":"existing paired APE_on - APE_off, percentage points; unscored is X",
             "changes_scores_or_acceptance":False,"new_statistical_aggregation":False,"outputs":outputs}
    write_new(output/"heatmaps.provenance.json",receipt)
    return {"figure_dir":str(output),"figure_count":6,"provenance_ref":ref(output/"heatmaps.provenance.json")}



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("check","heatmaps"))
    parser.add_argument("--figure-dir",type=Path,default=ROUND/"heatmaps.0001")
    args=parser.parse_args()
    context=load_completed()
    result={"completed_inputs_verified":True,"prediction_or_scoring_executed":False}
    if args.action=="heatmaps":result["heatmaps"]=heatmaps(context,args.figure_dir)
    print(json.dumps(result))


if __name__=="__main__":main()
