"""Streaming framing regressions; all responses are in memory, never native."""
import io
import json

import pytest
from tools import native_llama_compare as native
from tools import native_repeatability_experiment as experiment


def event(value):
    return ('data: '+json.dumps(value,separators=(',', ':'))+'\n\n').encode()


def stream_bytes(prefix=b':\n\n',middle=b': keepalive\n\n',done=True):
    timings={'cache_n':0,'prompt_n':1536,'predicted_n':2,
             'engine_token_times_us':[5000,9000], 'engine_request_begin_us':1000,
             'engine_prompt_last_us':5000,'engine_last_token_us':9000,
             'engine_timepoints_complete':True}
    return (prefix+event({'content':' the','tokens':[279],'stop':False,'id_slot':-1})+
            middle+event({'content':' season','tokens':[3098],'stop':False,'id_slot':-1})+
            event({'content':'','tokens':[],'stop':True,'id_slot':1,'truncated':False,
                   'tokens_evaluated':1536,'tokens_predicted':2,'timings':timings})+
            (b'data: [DONE]\n\n' if done else b'')),timings


def parse(monkeypatch,wire):
    capture={'raw_received_lines':[]}
    monkeypatch.setattr(native,'urlopen',lambda *_args,**_kwargs:
        experiment._CapturedResponse(io.BytesIO(wire),capture,drain=True))
    result,boundary=native.post_stream_json('http://unused/completion',{'stream':True})
    return result,boundary,capture


@pytest.mark.parametrize('done',[False,True])
def test_leading_and_midstream_heartbeat_keep_all_tokens_and_final_timings(monkeypatch,done):
    wire,timings=stream_bytes(done=done)
    result,boundary,capture=parse(monkeypatch,wire)
    assert result['timings']==timings
    assert result['stop'] is True and result['id_slot']==1
    assert result['content']==' the season'
    assert len(result['raw_stream_events'])==3
    assert boundary['mode']=='sse' and boundary['status']=='measured'
    assert boundary['chunk_count']==3 and len(boundary['token_chunk_times_ms'])==2
    assert boundary['request_to_last_token_ms']>=boundary['request_to_first_token_ms']
    assert capture['raw_received_lines'][0]['line']==':\n'
    assert any(row['line']==': keepalive\n' for row in capture['raw_received_lines'])
    assert 'data:' not in capture.get('trailing_body','')
    checked=experiment.inspect_request(result,boundary,2,1536)
    assert checked['status']=='measured'
    assert checked['metrics_ms']=={'ttft':4.,'tpot':4.,'e2e':8.}


@pytest.mark.parametrize('prefix',[b'event: message\n\nid: 8\nretry: 1000\n\n',b'unparseable preamble\n'])
def test_non_data_noise_does_not_select_plain_json_mode(monkeypatch,prefix):
    wire,timings=stream_bytes(prefix=prefix)
    result,boundary,_=parse(monkeypatch,wire)
    assert result['timings']==timings
    assert boundary['mode']=='sse' and boundary['chunk_count']==3


def test_plain_json_compatibility_still_has_unavailable_client_boundary(monkeypatch):
    response={'content':'finished','stop':True,'timings':{'prompt_n':3,'predicted_n':2}}
    result,boundary,_=parse(monkeypatch,(json.dumps(response)+'\n').encode())
    assert result['timings']==response['timings']
    assert boundary['mode']=='json' and boundary['status']=='unavailable'
    assert boundary['request_to_first_token_ms'] is None


def test_only_heartbeats_cannot_become_a_successful_response(monkeypatch):
    with pytest.raises(ValueError,match='no JSON chunks'):
        parse(monkeypatch,b':\n\n: keepalive\n\n')
