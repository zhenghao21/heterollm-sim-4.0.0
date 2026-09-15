import io,json
import pytest
from tools import native_cohort_client as c

def encoded(items):
    return io.BytesIO(b''.join(b'data: '+json.dumps(x).encode()+b'\n\n' for x in items)+b'data: [DONE]\n\n')

def test_interleaved_cohort_keeps_each_index_and_final_boundary(monkeypatch):
    items=[{'index':1,'tokens':[23],'content':'B'}, {'index':0,'tokens':[12],'content':'A'},
           {'index':0,'timings':{'predicted_n':1},'stop':True}, {'index':1,'timings':{'predicted_n':1},'stop':True}]
    seen=[]
    def opener(req,timeout):
        seen.append(json.loads(req.data));return encoded(items)
    monkeypatch.setattr(c,'urlopen',opener)
    monkeypatch.setattr(c.time,'perf_counter',iter([10.,10.1,10.2,10.3,10.4,10.5]).__next__)
    result=c.post_cohort_stream_json('http://localhost/completion',{'prompt':'hi','n_predict':1},2)
    assert seen[0]['prompt']==['hi','hi']
    assert [r['content'] for r,b in result]==['A','B']
    assert result[0][1]['request_to_last_token_ms']==pytest.approx(200.)
    assert result[0][1]['batch_client_makespan_ms']==pytest.approx(200.)
    assert result[1][1]['transport_scope']=='shared_post'
    assert result[0][0]['raw_stream_event_times_monotonic_s']==[10.2,10.3]

@pytest.mark.parametrize('items',[[{'index':True}], [{'index':2}], [{'error':'oops'}], [{'index':0,'tokens':[1]}]])
def test_incomplete_or_invalid_cohort_is_rejected(monkeypatch,items):
    monkeypatch.setattr(c,'urlopen',lambda *a,**kw:encoded(items))
    with pytest.raises(ValueError):c.post_cohort_stream_json('http://local',{'prompt':'hi'},2)


def test_header_only_null_is_preserved_without_creating_a_token(monkeypatch):
    items=[None, {'index':0,'tokens':[1],'content':'a'}, {'index':0,'timings':{'predicted_n':1},'stop':True}]
    monkeypatch.setattr(c,'urlopen',lambda *a,**kw:encoded(items))
    result=c.post_cohort_stream_json('http://local',{'prompt':'hi'},1)
    assert len(result[0][0]['shared_stream_control_events'])==1
    assert result[0][1]['chunk_count']==2
    assert len(result[0][1]['token_chunk_times_ms'])==1


def test_canonical_n_cmpl_cannot_enable_child_prefix_sharing():
    with pytest.raises(ValueError):
        c.post_cohort_stream_json('http://local', {'prompt':'hi','n_cmpl':4},4)
