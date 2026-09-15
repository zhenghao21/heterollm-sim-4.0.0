from types import SimpleNamespace
import pytest
from tools.native_llama_compare import _client_batch_makespan, _completion_payload, _native_execution_environment

def test_batch_counts_start_skew_and_excludes_done():
    rows=[({},dict(request_start_monotonic_s=10.,last_token_monotonic_s=10.1,stream_end_monotonic_s=11.)),({},dict(request_start_monotonic_s=10.08,last_token_monotonic_s=10.15,stream_end_monotonic_s=12.))]
    assert _client_batch_makespan(rows)==pytest.approx(150.)
    assert _client_batch_makespan([({},dict(request_to_end_ms=100))]) is None

def test_warmup_has_same_output_policy():
    a=SimpleNamespace(warmup_predict=2,predict=17,temperature=0,top_k=1,seed=42,request_timing='stream',output_mode='fixed',stop=['stop'])
    w=_completion_payload(a,'hello',warmup=True);f=_completion_payload(a,'hello')
    assert w['ignore_eos'] is True and w['stop']==f['stop']
    assert w['n_predict']==2 and f['n_predict']==17
    assert w['stream'] is False and f['stream'] is True

def test_environment_allowlist(monkeypatch):
    monkeypatch.setenv('SECRET_API_KEY','do-not-capture');monkeypatch.setenv('GGML_OP_OFFLOAD_MIN_BATCH','32')
    e=_native_execution_environment()
    assert 'SECRET_API_KEY' not in e
    assert e['GGML_OP_OFFLOAD_MIN_BATCH']=={'is_set':True,'value':'32'}

from tools.native_llama_compare import _native_measurement_policy, _native_measurement_flags, _collect_warmup_batches


def test_noise_controls_are_explicit_in_flags_and_policy():
    args=SimpleNamespace(native_client='preconnect_http',native_log_verbosity=0,native_priority=0,
                         warmup_batches=8,warmup_predict=17,skip_counter_snapshots=True)
    policy=_native_measurement_policy(args)
    assert _native_measurement_flags(args)==['--log-verbosity','0']
    assert policy['connection_policy']=='preconnect_each_batch'
    assert policy['warmup_stream'] is True
    assert policy['diagnostic_counter_snapshots'] is False
    assert policy['priority']==0
    args.native_log_verbosity=None
    assert _native_measurement_flags(args)==[]


def test_multiple_warmups_use_same_stream_client_and_retain_all_responses():
    calls=[]
    class Client:
        def batch(self,url,payload):
            calls.append(dict(payload));return [({'counter':len(calls)}, {}) for _ in range(4)]
    payload={'prompt':'hi','n_predict':17,'ignore_eos':True,'stream':False,'cache_prompt':False}
    batches=_collect_warmup_batches('http://local',payload,4,3,Client())
    assert len(batches)==3 and all(len(batch)==4 for batch in batches)
    assert all(call['stream'] and call['n_predict']==17 and call['ignore_eos'] for call in calls)
    assert payload['stream'] is False
    assert [batch[0]['counter'] for batch in batches]==[1,2,3]
