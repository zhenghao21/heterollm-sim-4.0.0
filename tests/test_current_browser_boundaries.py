"""Exercise current backend output through browser error/import callers."""
from dataclasses import replace
from types import SimpleNamespace

from heterollm_sim.gguf_model_catalog import GGUFModelCatalog
from heterollm_sim.planner import ScenarioValidationError
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.web import HeteroLLMRequestHandler, HttpError, validation_payload_for_scenario
from test_current_report_callers import browser_check
from test_current_view_boundaries import run_browser


def test_http_and_background_errors_reach_current_browser_diagnostics():
    scenario = build_reference_scenario()
    scenario = replace(scenario, placement=replace(
        scenario.placement, op_to_component={"*": "missing-component"}))
    validation = validation_payload_for_scenario(scenario)
    assert not validation["valid"]
    captured = {}
    handler = SimpleNamespace(_send_json=lambda status, payload: captured.update(
        status=status, response=payload))
    HeteroLLMRequestHandler._send_error(handler, HttpError(
        422, "invalid_scenario", "场景校验失败", details=validation,
        diagnostic_id="current-diagnostic", exception_type="ScenarioValidationError"))
    captured["job_details"] = ScenarioValidationError(
        "后台场景校验失败", errors=("场景内存不足", "缺少目标组件")).details
    browser_check(("apiRequest", "hasChineseText", "localizedBackendValue",
                   "collectBackendDiagnosticItems", "backendChineseMessage",
                   "normalizeValidation", "normalizeIssue", "inferValidationIssueFields",
                   "showOperationError", "operationErrorIssues"), captured, """
(async () => {
  responsePayload = payload.response;
  let current;
  await assert.rejects(apiRequest('/validate'), error => { current = error; return true; });
  assert.equal(current.code, 'invalid_scenario');
  assert.equal(current.exception_type, 'ScenarioValidationError');
  assert.equal(current.diagnostic_id, 'current-diagnostic');
  assert.deepEqual(current.details, payload.response.error.details);
  showOperationError('校验失败', current);
  assert.equal(state.validation.errors.length, 1);
  assert.match(state.validation.errors[0].message_en, /missing-component/);
  assert.match(lastToast[1], /current-diagnostic/);

  // Background jobs deliver their current error object directly to the UI.
  showOperationError('后台失败', {message: '后台场景校验失败', code: 'run_job_failed',
    details: payload.job_details, diagnostic_id: 'job-diagnostic'});
  assert.equal(state.validation.errors.length, 2);
  assert.match(lastToast[1], /job-diagnostic/);

  responsePayload = {code: 'retired', message_zh: '旧顶层错误',
    details: {diagnostic_id: 'retired'}, exception_type: 'Retired'};
  await assert.rejects(apiRequest('/validate'), error => {
    assert.equal(error.code, 'http_422');
    assert.equal(error.details, null);
    assert.equal(error.diagnostic_id, '');
    assert.equal(error.exception_type, '');
    return true;
  });
  contentType = 'text/html'; responsePayload = '<html>Bad request</html>';
  await assert.rejects(apiRequest('/validate'), error => {
    assert.match(error.message, /HTTP 422/); return true;
  });
  const wrapped = normalizeValidation({validation: payload.response.error.details});
  assert.deepEqual(wrapped.errors, []);
  assert.equal(wrapped.valid, false);
})().catch(error => { console.error(error); process.exitCode = 1; });
""", """
const backendConnection = {request: (...args) => fetch(...args), snapshot: () => ({generation: 1}), assertGeneration: () => {}};
const ISSUE_MESSAGE_BY_CODE = {};
const state = {settings: {language: 'zh-CN'}};
const dom = {};
let responsePayload, lastToast, contentType = 'application/json';
const fetch = async () => ({ok: false, status: 422,
  headers: {get: () => contentType}, json: async () => responsePayload,
  text: async () => responsePayload});
const setConnection = () => {};
const runtimeConnectionLabel = () => ({zh: '在线', en: 'Online'});
const uiText = zh => zh;
const reconcileMappingFingerprint = () => {};
const renderSteps = () => {}, renderDiagnostics = () => {}, openDiagnostics = () => {};
const toast = (...args) => { lastToast = args; };
""")


def test_preset_import_selects_the_id_from_current_catalog_detail(tmp_path):
    detail = GGUFModelCatalog(tmp_path).detail("qwen3-0_6b")
    browser_check(("importModelPreset", "presetId", "restorePresetActionButton",
                   "chineseMessage", "hasChineseText"), detail, """
(async () => {
  responsePayload = payload;
  await importModelPreset('remote/repository');
  assert.equal(state.selectedPresetId, payload.preset.id);
  assert.equal(selectedId, payload.preset.id);
  assert.equal(dom.presetSearchInput.value, payload.preset.id);
  assert.equal(applied, payload);
  for (const retired of [payload.preset, {id: 'old'}]) {
    responsePayload = retired; applied = null;
    await importModelPreset('remote/repository');
    assert.equal(applied, null);
    assert.equal(lastToast[2], 'error');
    assert.match(lastToast[1], /preset.id/);
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
""", """
const state = {};
const dom = {remotePresetStatus: {}, presetSearchInput: {},
  presetList: {querySelector: () => null}, modelPresetsDialog: {close: () => {}}};
const CSS = {escape: value => value};
let responsePayload, selectedId, applied, lastToast;
const apiRequest = async () => responsePayload;
const setPresetMode = () => {};
const loadModelPresets = async options => { selectedId = options.selectedId; };
const presetMaterializationReadiness = () => ({ready: true});
const applyPresetDetailToScenario = value => { applied = value; return {nextModel: value.model}; };
const presetAppliedToastText = () => '已应用';
const toast = (...args) => { lastToast = args; };
""")


def test_fullscreen_uses_current_state_and_retains_focus_on_escape():
    run_browser("""
const initial = {open: false, mode: 'overlay', escapeCloses: true, previousFocusId: 'playback'};
const opened = TraceView.toggleFullscreen(initial);
assert.equal(opened.open, true);
assert.equal(initial.open, false);
const escaped = TraceView.fullscreenEscapeAction(opened, {key: 'Escape'});
assert.equal(escaped.handled, true);
assert.equal(escaped.state.open, false);
assert.equal(escaped.state.previousFocusId, 'playback');
for (const retired of [true, {active: true}, {fullscreen: true}]) {
  assert.equal(TraceView.normalizeFullscreenState(retired).open, false);
}
for (const retired of ['connectPorts', 'disconnectPort', 'modelPortCompatibility']) {
  assert.equal(ModelGraph[retired], undefined);
}
""")
