import copy
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location("r27_groups_test",Path(__file__).with_name("summarize_groups.py"));d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
def fixture():
    return {"a":{"cell_id":"a","prediction_status":"predicted","metrics":{m:{"status":"scored","native_median_ms":100.0,"native_run_medians_ms":[100.0],"simulator_median_ms":90.,"absolute_percentage_error_pct":10.,"absolute_error_ms":10.,"signed_error_pct":-10.} for m in d.s.METRICS}}}
def test_exact10_is_not_pass_and_failure_stays_in_group_denominator():
    rows=fixture();rows["b"]={"cell_id":"b","prediction_status":"failed","prediction_failure_reason":"identity mismatch","metrics":{m:{"status":"unscored","reason":"missing"} for m in d.s.METRICS}}
    report=d.summarize_group(rows)
    assert report["group_denominator"]==2 and report["scored_cells"]==1 and report["strict_all3_below10_cells"]==0
    assert len(report["failures"])==1

def test_paired_requires_identical_native_targets():
    a=fixture();b=copy.deepcopy(a);b['a']['metrics'][d.s.METRICS[0]]['native_median_ms']=99.
    with pytest.raises(ValueError,match='native'):d.paired_rows(a,b)

def test_unscored_pair_keeps_three_metric_entries():
    a=fixture();b=fixture();b['a']['metrics'][d.s.METRICS[0]]={'status':'unscored','reason':'missing'}
    out=d.paired_rows(a,b)
    assert out['fixed_metric_denominator']==3 and out['outcomes']=={'unscored':1,'unchanged':2}

def test_cannot_read_scores_before_completed_report(monkeypatch,tmp_path):
    monkeypatch.setattr(d,'P',tmp_path)
    with pytest.raises(ValueError,match='completed two-arm'):d.main()
