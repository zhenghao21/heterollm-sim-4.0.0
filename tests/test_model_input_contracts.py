from dataclasses import replace

from heterollm_sim.config import HostOutputContract, scenario_from_dict
from heterollm_sim.llama_scenario import prepare_llama_scenario
from heterollm_sim.planner import validate_scenario
from heterollm_sim.reference import build_llama_default_scenario, build_reference_scenario
from heterollm_sim.runtime_adapters import LlamaCppRuntimeConfig
from heterollm_sim.web import scenario_to_payload


def test_normalized_runtime_payload_is_authoring_round_trippable():
    source = replace(
        build_llama_default_scenario(),
        llama_cpp_config=LlamaCppRuntimeConfig(),
    )
    prepared = prepare_llama_scenario(source)
    payload = scenario_to_payload(prepared)

    # Runtime placement is derived state and must not leak into V4 authoring.
    assert payload["placement"]["op_to_component"] == {}
    assert payload["placement"]["tensor_to_component"] == {}
    assert payload["placement"]["tensor_bytes"] == {}

    reloaded = scenario_from_dict(payload)
    reparsed = prepare_llama_scenario(reloaded)
    assert reparsed.placement.op_to_component
    assert reparsed.placement.tensor_bytes
    assert validate_scenario(reparsed).is_valid


def test_host_output_vocabulary_mismatch_is_rejected_before_costing():
    source = build_reference_scenario()
    scenario = replace(
        source,
        host_output_contract=HostOutputContract(
            target_component_id="cpu0",
            vocabulary_size=1,
            logits_dtype="int8",
            logits_element_bytes=1,
        ),
    )
    report = validate_scenario(scenario)
    assert not report.is_valid
    assert any("vocabulary_size" in message for message in report.errors_en)
