"""Public callers must use the current report, batch, and request contracts."""
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from heterollm_sim.contracts import RunManifest, ANALYTICAL_MODEL_VERSION
from heterollm_sim.llama_trace_diff import diff_scheduler_traces
from heterollm_sim.serving import BatchItem
from heterollm_sim.web import HttpError, _ui_score_request


def test_model_file_request_requires_the_current_envelope():
    from heterollm_sim.web import _model_file_request

    model = {"schema_version": "4.0.0", "graph": {}}
    assert _model_file_request({"model": model, "provenance": {}}) is model
    with pytest.raises(HttpError) as error:
        _model_file_request(model)
    assert error.value.code == "unknown_fields"
    with pytest.raises(HttpError) as error:
        _model_file_request({"artifact_id": "missing-model"})
    assert error.value.code == "missing_model"


def test_mtp_report_attributes_only_current_explicit_batch_items():
    from types import SimpleNamespace
    from heterollm_sim.reporting import _online_mtp_token_events
    from heterollm_sim.serving import BatchCohort, RequestStatus, ServingEvent

    def item(request_id, phase, drafts):
        return BatchItem(request_id, phase, 1, 8, main_tokens=1,
                         draft_tokens=drafts, verifier_tokens=1,
                         committed_tokens=1, logit_tokens=1)

    batch = BatchCohort("mixed", "mtp", 0., (
        item("mtp", "mtp", 2), item("decode", "decode", 0),
        item("failed", "mtp", 2)), metadata={"mtp": True})
    events = tuple(ServingEvent(1., "tokens_committed", request_id, "mixed",
        {"source": "mtp", "mtp": True, "proposed": 2})
        for request_id in ("mtp", "decode", "failed"))
    serving = SimpleNamespace(batches=(batch,), events=events, request_metrics={
        name: SimpleNamespace(status=status) for name, status in (
            ("mtp", RequestStatus.FINISHED), ("decode", RequestStatus.FINISHED),
            ("failed", RequestStatus.RUNNING))})
    assert _online_mtp_token_events(serving) == (events[0],)


def test_browser_import_and_health_use_current_contracts():
    from heterollm_sim.reference import build_reference_scenario
    from heterollm_sim.web import hardware_input_payload

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser contract checks")
    source = (Path(__file__).parents[1] / "src/heterollm_sim/webui/app.js").read_text(encoding="utf-8")
    names = ("requireScenarioSchemaV4", "normalizedComponentKind", "isDedicatedHbm",
             "costProfileKeyForComponentKind", "validateHardwareInputForImport",
             "runtimeHealthView", "mappingFingerprintFrom", "currentMappingFingerprintFrom")
    functions = []
    for name in names:
        match = re.search(r"^function " + name + r"\([^\n]*\) \{.*?^\}", source, re.M | re.S)
        assert match, name
        functions.append(match.group())
    payload = hardware_input_payload(build_reference_scenario())
    script = """
const assert = require('node:assert/strict');
const AUTHORING_SCHEMA_VERSION = '4.0.0';
const asObject = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const asArray = value => Array.isArray(value) ? value : [];
""" + "\n".join(functions) + "\nconst hardware = " + json.dumps(payload) + ";\n" + """
assert.doesNotThrow(() => validateHardwareInputForImport(hardware));
for (const mutate of [
  value => value.schema_version = '4.0',
  value => value.hardware.parameters.components = {},
  value => value.hardware.components[0].cost_profile_id = 'old',
  value => value.hardware.components.find(c => c.execution_profile).execution_profile.legacy = true,
]) {
  const retired = structuredClone(hardware);
  mutate(retired);
  assert.throws(() => validateHardwareInputForImport(retired));
}
const current = runtimeHealthView({ok: true, runtime: {python_version: '3.13', architecture_bits: 64},
  ortools: {available: true, probe_ok: true, version: '9.0'}, available_solvers: ['cp_sat']});
assert.equal(current.pythonVersion, '3.13');
assert.equal(current.architectureBits, 64);
assert.equal(current.ortoolsAvailable, true);
assert.deepEqual(current.availableSolvers, ['cp_sat']);
const retired = runtimeHealthView({python: {version: 'old'}, python_version: 'old',
  cp_sat_available: true, solvers: {available: ['cp_sat']}});
assert.notEqual(retired.pythonVersion, 'old');
assert.equal(retired.ortoolsAvailable, false);
assert.deepEqual(retired.availableSolvers, []);
assert.equal(mappingFingerprintFrom({input_fingerprint: 'current'}), 'current');
assert.equal(mappingFingerprintFrom({mapping_fingerprint: 'old', mapping_input_fingerprint: 'old'}), '');
assert.equal(currentMappingFingerprintFrom({current_input_fingerprint: 'current'}), 'current');
assert.equal(currentMappingFingerprintFrom({current_mapping_fingerprint: 'old'}), '');
"""
    result = subprocess.run([node], input=script, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr


def test_browser_validation_reads_current_fields_and_targets():
    from heterollm_sim.reference import build_reference_scenario
    from heterollm_sim.web import validation_payload_for_scenario

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser contract checks")
    source = (Path(__file__).parents[1] / "src/heterollm_sim/webui/app.js").read_text(encoding="utf-8")
    functions = []
    for name in ("inferValidationIssueFields", "normalizeIssue", "normalizeValidation", "validationIssueTarget"):
        match = re.search(r"^function " + name + r"\([^\n]*\) \{.*?^\}", source, re.M | re.S)
        assert match, name
        functions.append(match.group())
    payload = validation_payload_for_scenario(build_reference_scenario())
    script = """
const assert = require('node:assert/strict');
const asObject = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const asArray = value => Array.isArray(value) ? value : [];
const hasChineseText = value => /[\u4e00-\u9fff]/u.test(value);
const state = {scenario: {hardware: {components: [{component_id: 'hbm0'}]}}};
""" + "\n".join(functions) + "\nconst payload = " + json.dumps(payload) + ";\n" + """
const current = normalizeValidation(payload);
assert.equal(current.valid, payload.valid);
assert.equal(current.errors.length, Object.values(payload.errors).flat().length);
assert.equal(current.information.length, Object.values(payload.information).flat().length);
const retired = normalizeValidation({is_valid: true, validation_errors: ['old'], infos: ['old']});
assert.equal(retired.valid, false);
assert.deepEqual(retired.errors, []);
assert.deepEqual(retired.information, []);
const target = validationIssueTarget({field_path: 'hardware.components.0.capacity_bytes'});
assert.equal(target.componentId, 'hbm0');
assert.equal(target.field, 'capacity_bytes');
assert.equal(validationIssueTarget({field: 'column_same_group_ns', component_id: 'hbm0'}).field,
  'column_same_group_ns');
assert.equal(validationIssueTarget({json_pointer: '/hardware/components/0/capacity_bytes'}).path, '');
assert.equal(validationIssueTarget({field_name: 'old'}).field, '');
"""
    result = subprocess.run([node], input=script, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr


def test_report_names_the_analytical_model_without_a_calibration_alias():
    fields = dict(schema_version="4.0.0", run_id="current-contract", random_seed=0,
                  simulator_version="4.0.0", model_name="model", hardware_name="hardware",
                  workload_name="workload")
    payload = asdict(RunManifest(**fields))
    assert payload["analytical_model_version"] == ANALYTICAL_MODEL_VERSION
    assert "calibration_version" not in payload
    with pytest.raises(TypeError, match="calibration_version"):
        RunManifest(**fields, calibration_version="old")


def test_batches_require_explicit_token_accounting():
    with pytest.raises(TypeError, match="keyword-only"):
        BatchItem("request", "decode", 1, 8)
    item = BatchItem("request", "decode", 1, 8, main_tokens=1, draft_tokens=0,
                     verifier_tokens=1, committed_tokens=1, logit_tokens=1)
    assert item.committed_tokens == item.logit_tokens == 1


@pytest.mark.parametrize("old_key", ["native_report", "r0_reference"])
def test_score_request_rejects_old_reference_names(old_key):
    with pytest.raises(HttpError) as error:
        _ui_score_request({old_key: {}})
    assert error.value.code == "unknown_fields"


def test_trace_comparison_requires_the_current_trace_key():
    assert diff_scheduler_traces({"schedule_trace": []}, [])["equal"]
    with pytest.raises(KeyError, match="schedule_trace"):
        diff_scheduler_traces({"scheduler_trace": []}, [])


def test_trace_exports_allocation_identities_without_javascript_rounding():
    from heterollm_sim.reporting import _trace_metadata

    generation = int.from_bytes(b"\x01request-with-a-full-identity", "big")
    metadata = {
        "physical_allocations": [{"buffer_id": "x", "generation": generation, "size_bytes": 64}],
        "memory_accesses": [{"allocation_generation": generation, "alias_generation": 0}],
        "physical_external_allocations": (("owner", "x", generation),),
        "physical_memory_config": {"generation": "HBM3"},
        "_engine_private": True,
    }
    encoded = _trace_metadata(metadata)
    assert encoded["physical_allocations"][0] == {"buffer_id": "x", "generation": str(generation), "size_bytes": 64}
    assert encoded["memory_accesses"][0] == {"allocation_generation": str(generation), "alias_generation": "0"}
    assert encoded["physical_external_allocations"] == [["owner", "x", str(generation)]]
    assert encoded["physical_memory_config"]["generation"] == "HBM3"
    assert "_engine_private" not in encoded
    assert metadata["physical_allocations"][0]["generation"] == generation


def test_browser_reads_current_catalog_response_shapes():
    from heterollm_sim.component_presets import component_preset_page, component_preset_detail
    from heterollm_sim.architecture_presets import architecture_preset_page, architecture_preset_detail

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser contract checks")
    source = (Path(__file__).parents[1] / "src/heterollm_sim/webui/app.js").read_text(encoding="utf-8")
    names = ("componentPresetId", "componentPresetEnvelopeItems", "componentPresetDetailItem",
             "componentPresetSpec", "architecturePresetEnvelopeItems", "architecturePresetDetailItem",
             "architecturePresetNames", "architecturePresetCategory", "architecturePresetLevel",
             "architecturePresetVendor", "architecturePresetSupport", "architecturePresetIsLoadable")
    functions = []
    for name in names:
        match = re.search(r"^function " + name + r"\([^\n]*\) \{.*?^\}", source, re.M | re.S)
        assert match, name
        functions.append(match.group())
    components = component_preset_page()
    architectures = architecture_preset_page()
    payload = dict(components=components, architectures=architectures,
                   component_details=[component_preset_detail(row["id"]) for row in components["items"]],
                   architecture_details=[architecture_preset_detail(row["id"]) for row in architectures["items"]])
    script = """
const assert = require('node:assert/strict');
const asObject = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const asArray = value => Array.isArray(value) ? value : [];
const deepClone = structuredClone;
""" + "\n".join(functions) + "\nconst payload = " + json.dumps(payload) + ";\n" + """
assert.equal(componentPresetEnvelopeItems(payload.components).length, payload.component_details.length);
for (const raw of payload.component_details) {
  const detail = componentPresetDetailItem(raw);
  assert.equal(componentPresetId(detail), raw.preset.id);
  if (raw.component) assert.equal(componentPresetSpec(detail).kind, raw.component.kind);
  else assert.equal(detail.components.length, raw.components.length);
}
assert.equal(architecturePresetEnvelopeItems(payload.architectures).length, payload.architecture_details.length);
for (const raw of payload.architecture_details) {
  const detail = architecturePresetDetailItem(raw);
  assert.equal(architecturePresetNames(detail).zh, raw.preset.name);
  assert.equal(architecturePresetCategory(detail), raw.preset.topology_class);
  assert.equal(architecturePresetLevel(detail), raw.preset.scale);
  assert.equal(architecturePresetVendor(detail), raw.preset.vendor);
  assert.equal(architecturePresetSupport(detail), raw.preset.support_level);
  assert.equal(architecturePresetIsLoadable(detail), raw.preset.loadable);
  assert.equal(detail.hardware.components.length, raw.hardware.components.length);
}
assert.equal(componentPresetId({preset_id: 'old'}), '');
assert.deepEqual(componentPresetEnvelopeItems({presets: []}), []);
assert.equal(architecturePresetIsLoadable({can_load: true}), false);
"""
    result = subprocess.run([node], input=script, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr


def test_browser_new_gddr_uses_complete_current_physical_config():
    from heterollm_sim.physical_contract import parse_required_physical_config

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser contract checks")
    source = (Path(__file__).parents[1] / "src/heterollm_sim/webui/app.js").read_text(encoding="utf-8")
    start = source.index("  const gddrDefaultCapacityBytes", source.index("function addComponent("))
    end = source.index("  if (!defaults)", start)
    functions = []
    for name in ("gddrGeneration", "sharedMemoryLinkComponent"):
        match = re.search(r"^function " + name + r"\([^\n]*\) \{.*?^\}", source, re.M | re.S)
        assert match, name
        functions.append(match.group())
    script = """
const assert = require('node:assert/strict');
const asObject = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const asArray = value => Array.isArray(value) ? value : [];
const kind = 'gddr';
""" + source[start:end] + "\n" + "\n".join(functions) + """
assert.equal(gddrGeneration(defaults), 'GDDR6');
assert.equal(gddrGeneration({metadata: {generation: 'GDDR7', technology: {generation: 'GDDR7'}}}), '');
assert.equal(sharedMemoryLinkComponent({metadata: {bandwidth_source: 'memory_component'}}, {hardware: {components: []}}), null);
assert.equal(defaults.metadata.generation, undefined);
console.log(JSON.stringify(defaults.metadata.physical_memory_config));
"""
    result = subprocess.run([node], input=script, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr
    config = parse_required_physical_config("gddr0", "gddr", json.loads(result.stdout))
    assert config.generation == "GDDR6"
    assert config.control_mode == "analytical"
