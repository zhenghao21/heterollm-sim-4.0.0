"""Pure synthetic host tests; never open current campaign scores/predictions,
render a figure, invoke a scorer, or create an evidence archive.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import sys
import pytest

spec=importlib.util.spec_from_file_location("r30_postprocess_tests",Path(__file__).with_name("postprocess.py"))
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def synthetic_scored_context():
    candidates=[]
    for group in range(6):
        for prompt in p.PROMPTS:
            for output in p.OUTPUTS:
                for c in p.CONCURRENCY:
                    ident=f"synthetic{group}_p{prompt}_o{output}_c{c}"
                    candidates.append((ident,dict(cell_id=ident,model_key=f"synthetic{group}",deployment="test")))
    excluded={5*i for i in range(31)}
    cells=dict(v for i,v in enumerate(candidates) if i not in excluded)
    assert len(cells)==131
    rows={arm:{} for arm in ("off","on")}
    paired=[]
    first=sorted(cells)[0]
    for ident in cells:
        for arm in rows:
            records={m:dict(status="scored",absolute_percentage_error_pct=20.0 if arm=="off" else 15.0) for m in p.METRICS}
            if ident==first and arm=="off":records[p.METRICS[0]]=dict(status="unscored",reason="synthetic failure")
            rows[arm][ident]={"metrics":records}
        for metric in p.METRICS:
            a,b=rows["off"][ident]["metrics"][metric],rows["on"][ident]["metrics"][metric]
            r=dict(cell_id=ident,metric=metric,before_status=a["status"],after_status=b["status"],outcome="unscored",native_equal=None)
            if a["status"]==b["status"]=="scored":r.update(native_equal=True,ape_delta_percentage_points=-5.0,outcome="improved")
            paired.append(r)
    return dict(cells=cells,rows=rows,paired=dict(fixed_metric_denominator=393,rows=paired)),first


def test_incomplete_campaign_refused_before_loading_runtime_or_scores(monkeypatch,tmp_path):
    monkeypatch.setattr(p,"ROUND",tmp_path)
    monkeypatch.setattr(p,"load_module",lambda *args:pytest.fail("must not import campaign or read predictions"))
    with pytest.raises(ValueError,match="existing completed scoring"):p.load_completed()
    (tmp_path/"predictions_complete.json").write_text("not parsed")
    with pytest.raises(ValueError,match="report.json"):p.load_completed()
    (tmp_path/"report.json").write_text("not parsed")
    with pytest.raises(ValueError,match="grouped_paired_report"):p.load_completed()


def test_fixed131_mask_failed_x_and_outside_dash_are_distinct():
    context,failed=synthetic_scored_context();views=p.prepare_panels(context)
    for view in views:
        for metric in p.METRICS:
            panels=[entry for group in views[view].values() for entry in group[metric].values()]
            assert len(panels)==162
            assert sum(x["selected"] for x in panels)==131
            assert sum(x["label"]=="-" for x in panels)==31
    y,x=p.coordinates(failed);group=context["cells"][failed]["model_key"]+" / test"
    assert views["off"][group][p.METRICS[0]][y,x]["label"]=="X"
    assert views["on"][group][p.METRICS[0]][y,x]["value"]==15.0
    assert views["delta"][group][p.METRICS[0]][y,x]["label"]=="X"
    assert views["delta"][group][p.METRICS[1]][y,x]["value"]==-5.0


def test_paired_mask_or_value_drift_is_rejected():
    context,_=synthetic_scored_context()
    bad=copy.deepcopy(context);bad["paired"]["rows"].pop()
    with pytest.raises(ValueError,match="complete paired"):p.prepare_panels(bad)
    bad=copy.deepcopy(context)
    row=next(r for r in bad["paired"]["rows"] if r["outcome"]=="improved")
    row["ape_delta_percentage_points"]=-4.0
    with pytest.raises(ValueError,match="paired delta differs"):p.prepare_panels(bad)
    bad=copy.deepcopy(context);bad["rows"]["off"].pop(next(iter(bad["cells"])))
    with pytest.raises(ValueError,match="score mask differs"):p.prepare_panels(bad)


def test_panel_never_drops_wholly_unscored_model_group():
    context,_=synthetic_scored_context()
    ids={i for i,c in context["cells"].items() if c["model_key"]=="synthetic5"}
    for ident in ids:
        for arm in context["rows"]:
            for metric in p.METRICS:context["rows"][arm][ident]["metrics"][metric]=dict(status="unscored",reason="fixture whole group failure")
    for row in context["paired"]["rows"]:
        if row["cell_id"] in ids:row.update(before_status="unscored",after_status="unscored",native_equal=None,outcome="unscored")
    views=p.prepare_panels(context)
    for view in views:
        assert "synthetic5 / test" in views[view]
        for metric in p.METRICS:
            assert sum(x["label"]=="X" for x in views[view]["synthetic5 / test"][metric].values())==len(ids)


def test_precise_threshold_display_does_not_round_passing_ape_to_failure():
    context,failed=synthetic_scored_context();ident=next(i for i in context["cells"] if i!=failed);metric=p.METRICS[0]
    context["rows"]["on"][ident]["metrics"][metric]["absolute_percentage_error_pct"]=9.96
    row=next(r for r in context["paired"]["rows"] if r["cell_id"]==ident and r["metric"]==metric)
    row["ape_delta_percentage_points"]=9.96-20
    view=p.prepare_panels(context);group=context["cells"][ident]["model_key"]+" / test";y,x=p.coordinates(ident)
    assert view["on"][group][metric][y,x]["value"]==9.96
    assert view["on"][group][metric][y,x]["label"]=="<10"


def test_missing_or_changed_frozen_scorer_source_is_rejected(tmp_path):
    root=tmp_path/"source";tool=root/"tools/predict_stable_native_dataset.py";tool.parent.mkdir(parents=True);tool.write_text("synthetic frozen scorer")
    reference=p.ref(tool);mapping={"tools/predict_stable_native_dataset.py":{k:reference[k] for k in ("sha256","bytes")}}
    s=SimpleNamespace(source_content=lambda freeze:mapping)
    freeze={"source":{"root":str(root)}}
    assert p.frozen_scorer_ref(s,freeze)==reference
    tool.write_text("changed fixture")
    with pytest.raises(ValueError,match="scorer identity differs"):p.frozen_scorer_ref(s,freeze)
    s.source_content=lambda freeze:{}
    with pytest.raises(ValueError,match="scorer source missing"):p.frozen_scorer_ref(s,freeze)



def test_R30_receipt_uses_declared_gate_without_inventing_blind_field():
    receipt={"schema":"r30-two-arm-scoring-receipt/v1","denominator_per_arm":131,
             "arms":{"off":{},"on":{}},"gate_B":"unvalidated","formal_success":False}
    p.validate_scoring_receipt(receipt)
    for key,value in (("gate_B","passed"),("formal_success",True),("denominator_per_arm",130)):
        with pytest.raises(ValueError,match="scoring receipt"):
            p.validate_scoring_receipt({**receipt,key:value})
