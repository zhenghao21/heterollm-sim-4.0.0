from dataclasses import replace

from heterollm_sim.reference import build_llama_default_scenario
from heterollm_sim.reporting import report_dict, run_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.serving import BatchItem, _batch_item_main_tokens, _batch_item_draft_tokens


def test_ordinary_decode_is_not_reported_as_speculative_drafting():
    base = build_llama_default_scenario()
    workload = replace(
        base.workload,
        requests=(replace(base.workload.requests[0], prompt_tokens=4, output_tokens=3),),
        prompt_tokens=4,
        output_tokens=3,
    )
    scenario = replace(
        base,
        workload=workload,
        llama_cpp_config=LlamaCppRuntimeConfig(gpu_layers=0, batch=4, ubatch=4, context=7),
    )
    report = report_dict(run_scenario(scenario))
    events = [event["details"] for event in report["scheduler_events"] if event["event_type"] == "tokens_committed"]
    assert sum(event["main_tokens"] for event in events) == 3
    assert sum(event["draft_tokens"] for event in events) == 0
    assert sum(event["accepted_draft_tokens"] for event in events) == 0


def test_legacy_mtp_item_keeps_one_main_token_and_draft_prefix():
    item = BatchItem("request", "mtp", 4, 10, proposed_tokens=4)
    assert _batch_item_main_tokens(item) == 1
    assert _batch_item_draft_tokens(item) == 3
