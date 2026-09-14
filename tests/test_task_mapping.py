from types import SimpleNamespace
from heterollm_sim.task_mapping import map_native_events_to_tasks, task_stage

def test_explicit_stage_and_ambiguous_mapping_are_visible():
    tasks=[SimpleNamespace(task_id='a',metadata={'event_kind':'ffn_projection'}), SimpleNamespace(task_id='b',metadata={'event_kind':'ffn_projection'})]
    assert task_stage(tasks[0])=='ffn'
    result=map_native_events_to_tasks([{'stage':'ffn'}],tasks)
    assert result['status']=='partial' and result['unmatched_count']==1
    result=map_native_events_to_tasks([{'stage':'ffn','task_id':'a'}],tasks)
    assert result['matched_count']==1 and result['matched'][0]['confidence']==1.0

def test_qkv_name_is_not_kv_and_generic_event_kind_falls_back():
    task=SimpleNamespace(task_id='q',name='prefill.layer-000.qkv.kernel_launch',metadata={'event_kind':'kernel_launch'})
    assert task_stage(task)=='attention_qkv'
    report=map_native_events_to_tasks([{'stage':'attention_qkv'}],[task])
    assert report['matched_count']==0 and report['candidate_count']==1
