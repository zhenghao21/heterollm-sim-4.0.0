import pytest
from tools.native_variance_experiment import summarize_runs, validate_pairs, file_ref, verify


def pair(begin, first, last, slot=0):
    return ({'id_slot':slot,'truncated':False,'timings':{'engine_request_begin_us':begin,
             'engine_prompt_last_us':first,'engine_last_token_us':last,
             'engine_token_times_us':[first,last],'predicted_n':2,'prompt_n':5,
             'cache_n':0,'engine_timepoints_complete':True}},
            {'request_start_monotonic_s':1.,'last_token_monotonic_s':1.1})


def test_pooled_variation_preserves_between_block_drift_and_outlier():
    runs=[{'pairs':[pair(0,t,t+1000)]} for t in [1000,1000,10000,10000]]
    result=summarize_runs(runs)
    assert result['metrics']['ttft']['batch_medians_ms']==[1.,1.,10.,10.]
    assert result['metrics']['ttft']['sample_cv_pct']>90
    assert summarize_runs(runs[:2])['metrics']['ttft']['sample_cv_pct']==0


def test_split_means_a_later_request_starts_after_an_earlier_first_token():
    runs=[{'pairs':[pair(0,1000,10000,0),pair(2000,3000,12000,1)]}]
    assert summarize_runs(runs)['split_batches']==1
    validate_pairs(runs[0]['pairs'],2,2,5)
    with pytest.raises(ValueError):validate_pairs(runs[0]['pairs'],2,17,5)


def test_freeze_rejects_changed_and_empty_sources(tmp_path):
    path=tmp_path/'test.py';path.write_text('a')
    refs=[file_ref(path)];verify(refs)
    path.write_text('b')
    with pytest.raises(RuntimeError):verify(refs)
    with pytest.raises(ValueError):verify([])
