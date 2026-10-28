"""Execution fidelity stays explicit across JSON, command and browser inputs."""

from dataclasses import asdict, replace

import pytest

from heterollm_sim import cli
from heterollm_sim.config import scenario_from_dict
from heterollm_sim.reference import build_reference_scenario
from heterollm_sim.simulation_options import FastDecodeConfig, SimulationConfig, simulation_from_dict
from heterollm_sim.web import scenario_to_payload
from test_current_report_callers import browser_check


@pytest.mark.parametrize("options", [{}, {"mode": "detailed"},
    {"mode": "fast", "native_window": {}},
    {"mode": "fast", "native_window": {"pd_concurrency_envelope": False}}, {
    "mode": "fast", "fast_decode": {
        "warmup_tokens": 0, "calibration_tokens": 4, "validation_tokens": 2,
        "max_sample_tokens": 12, "target_relative_error": .05,
        "max_context_growth": .10, "segment_tokens": 64,
    }}])
def test_simulation_options_roundtrip_through_authoring_payload(options):
    simulation = simulation_from_dict(options)
    original = replace(build_reference_scenario(), simulation=simulation)
    payload = scenario_to_payload(original)
    assert payload["simulation"] == asdict(simulation)
    assert scenario_from_dict(payload).simulation == simulation
    del payload["simulation"]
    assert scenario_from_dict(payload).simulation == SimulationConfig()


@pytest.mark.parametrize("options", [
    None, [], {"mode": None}, {"mode": True}, {"mode": "FAST"},
    {"mode": "approximate"}, {"typo": 1}, {"fast_decode": None},
    {"fast_decode": []}, {"fast_decode": {"tokens": 16}},
    {"fast_decode": {"warmup_tokens": -1}}, {"fast_decode": {"warmup_tokens": True}},
    {"fast_decode": {"calibration_tokens": 1}}, {"fast_decode": {"calibration_tokens": 2.}},
    {"fast_decode": {"validation_tokens": 1}},
    {"fast_decode": {"sampling_strategy": "fixed", "max_sample_tokens": 55}},
    {"fast_decode": {"sampling_strategy": "unknown"}},
    {"fast_decode": {"sampling_strategy": []}},
    {"fast_decode": {"segment_tokens": 0}}, {"fast_decode": {"target_relative_error": .10001}},
    {"fast_decode": {"target_relative_error": 0}}, {"fast_decode": {"target_relative_error": True}},
    {"fast_decode": {"target_relative_error": float("nan")}},
    {"fast_decode": {"max_context_growth": .20001}},
    {"fast_decode": {"max_context_growth": float("inf")}},
])
def test_invalid_or_unknown_simulation_options_are_rejected(options):
    with pytest.raises(ValueError):
        simulation_from_dict(options)


@pytest.mark.parametrize("command", ["run", "demo", "compare-gpu"])
@pytest.mark.parametrize("override,expected", [(None, "fast"), ("detailed", "detailed"), ("fast", "fast")])
def test_cli_mode_override_preserves_sampling_parameters(monkeypatch, capsys, command, override, expected):
    from heterollm_sim import llama_scenario

    configured = replace(build_reference_scenario(), simulation=SimulationConfig(
        mode="fast", fast_decode=FastDecodeConfig(segment_tokens=128)))
    seen = []
    monkeypatch.setattr(cli, "load_scenario", lambda path: configured)
    monkeypatch.setattr(cli, "build_reference_scenario", lambda: configured)
    monkeypatch.setattr(llama_scenario, "prepare_llama_scenario", lambda value: value)
    monkeypatch.setattr(cli, "run_scenario", lambda scenario, **kwargs: seen.append(scenario) or object())
    monkeypatch.setattr(cli, "compare_with_gpu_baseline", lambda scenario: seen.append(scenario) or {})
    monkeypatch.setattr(cli, "report_dict", lambda result: {})
    argv = [command] + ([] if command == "demo" else ["scenario.json"]) + ["--json"]
    if override is not None:
        argv += ["--simulation-mode", override]
    assert cli.main(argv) == 0
    assert len(seen) == 1
    assert seen[0].simulation.mode == expected
    assert seen[0].simulation.fast_decode == configured.simulation.fast_decode
    capsys.readouterr()


def test_browser_shape_defaults_and_preserves_selected_fidelity():
    payload = scenario_to_payload(build_reference_scenario())
    browser_check(("requireScenarioSchemaV4", "objectArray", "ensureScenarioShape"), payload, """
delete payload.simulation;
assert.equal(ensureScenarioShape(payload).simulation.mode, 'detailed');
payload.simulation = {mode: 'fast', fast_decode: {segment_tokens: 64}};
assert.deepEqual(ensureScenarioShape(payload).simulation, {mode: 'fast', fast_decode: {segment_tokens: 64}});
for (const simulation of [null, [], 'fast', {mode: 'unknown'}]) {
  assert.throws(() => ensureScenarioShape({...payload, simulation}));
}
""", """
const AUTHORING_SCHEMA_VERSION = '4.0.0';
const normalizeV4ModelAuthoring = () => {}, rejectLegacyComponentProfiles = () => {};
""")


def test_browser_mode_change_reestimates_and_blocks_active_run_edits():
    browser_check(("changeRunSimulationMode",), None, """
void (async () => {
  await changeRunSimulationMode();
  assert.deepEqual(state.scenario.simulation, {mode: 'fast', fast_decode: {segment_tokens: 64}});
  assert.equal(changes, 1);
  assert.equal(estimates, 1);
  assert.deepEqual(changeOptions, {mappingImpact: false, preserveModelArtifact: true});
  dom.runSimulationMode.value = 'detailed';
  active = true;
  await changeRunSimulationMode();
  assert.equal(state.scenario.simulation.mode, 'fast');
  assert.equal(estimates, 1);
})().catch(error => { console.error(error); process.exitCode = 1; });
""", """
const state = {scenario: {simulation: {mode: 'detailed', fast_decode: {segment_tokens: 64}}}};
const dom = {runSimulationMode: {value: 'fast'}};
let active = false, changes = 0, estimates = 0, changeOptions;
const runJobIsActive = () => active;
const markScenarioChanged = (message, options) => { changes += 1; changeOptions = options; };
const runScenario = async () => { estimates += 1; };
""")


def test_fast_result_banner_keeps_estimates_distinct_from_detailed_statistics():
    report = {"simulation_mode": "fast", "trace_fidelity": "approximate", "summary": {"task_count": None},
              "approximation": {"effective_mode": "fast", "target_relative_error": .10,
                  "validation_status": "not_validated_against_full_detailed",
                  "sampled_decode_steps": 56, "predicted_decode_steps": 456}}
    browser_check(("formatResultNumber", "runManifestMarkup"), report, """
let html = runManifestMarkup(payload);
assert.match(html, /快速估算/);
assert.match(html, /误差目标≤10%/);
assert.match(html, /尚无完整详细对照验证/);
assert.match(html, /详细采样 Decode：56 步/);
assert.match(html, /预测 Decode：456 步/);
assert.match(html, /物理统计只覆盖已详细执行的前缀/);
assert.match(html, /全量任务数未统计/);
assert.doesNotMatch(html, /0 个任务/);
assert.match(html, /全量资源利用率未估算/);
state.settings.language = 'en';
html = runManifestMarkup(payload);
assert.match(html, /Fast estimate/);
assert.match(html, /has not been validated against a complete detailed run/);
assert.match(html, /Full-run task count not estimated/);
payload.approximation.target_relative_error = .05;
assert.match(runManifestMarkup(payload), /Error target ≤5%/);
""", """
const state = {settings: {language: 'zh-CN'}};
const escapeHtml = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
""")


def test_fast_selection_with_detailed_fallback_shows_reason_and_real_task_count():
    report = {"simulation_mode": "fast", "trace_fidelity": "aggregate", "summary": {"task_count": 27},
              "approximation": {"effective_mode": "detailed", "validation_status": "detailed_execution",
                  "sampled_decode_steps": 8, "predicted_decode_steps": 0,
                  "fallback_reasons": ["output_shorter_than_calibration_budget", "<unrecognized>"]}}
    browser_check(("formatResultNumber", "runManifestMarkup"), report, """
const html = runManifestMarkup(payload);
assert.match(html, /本场景使用详细执行/);
assert.match(html, /输出较短，详细执行比采样预测更合适/);
assert.match(html, /27 个任务/);
assert.match(html, /&lt;unrecognized&gt;/);
assert.doesNotMatch(html, /is-approximate/);
assert.doesNotMatch(html, /尚无完整详细对照验证/);
assert.doesNotMatch(html, /全量任务数未统计/);
payload.summary.task_count = null;
assert.match(runManifestMarkup(payload), /任务数不可用/);
assert.doesNotMatch(runManifestMarkup(payload), /0 个任务/);
""", """
const state = {};
const escapeHtml = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
""")


def test_fast_missing_utilization_is_not_presented_as_zero_or_idle():
    browser_check(("renderUtilization", "traceFidelityLabel"), {
        "resource_utilization": {}, "summary": {"bottleneck_resource": None},
        "approximation": {"effective_mode": "fast"}}, """
renderUtilization(payload);
assert.match(dom.bottleneckMeta.innerHTML, /全量资源利用率未估算/);
assert.match(dom.utilizationList.innerHTML, /不能代表完整请求/);
assert.doesNotMatch(dom.bottleneckMeta.innerHTML, /无资源占用|0%/);
assert.equal(traceFidelityLabel('approximate'), '快速估算');
""", """
const dom = {bottleneckMeta: {}, utilizationList: {}};
const uiText = zh => zh, escapeHtml = String;
""")


def test_actual_fast_report_shape_reaches_all_result_page_branches():
    from heterollm_sim.reporting import report_dict
    from test_fast_decode_reporting import forecast_result

    report = report_dict(forecast_result())
    browser_check(("escapeHtml", "slug", "percentile", "formatResultNumber", "resultWithUnit",
        "formatResultDurationNs", "formatResultRate", "metricLabel", "metricCell",
        "unavailableResult", "resultText", "appendResult", "combineResults", "formatResultBytes",
        "formatResultPercent", "runtimeValue", "runtimeStat", "runtimeGroup", "renderRuntime",
        "runManifestMarkup", "normalizeComponentTimeseries", "renderComponentTimeseries",
        "renderUtilization", "renderCategories", "renderRequestResults", "renderResults"), report, """
state.report = payload;
renderResults();
assert.equal(dom.resultsEmpty.hidden, true);
assert.equal(dom.resultsContent.hidden, false);
assert.match(dom.runManifestBar.innerHTML, /尚无完整详细对照验证/);
assert.match(dom.metricGrid.innerHTML, /TPOT/);
assert.doesNotMatch(dom.metricGrid.innerHTML, /NaN|undefined/);
assert.doesNotMatch(dom.runtimeSummary.innerHTML, /NaN|undefined/);
assert.match(dom.requestResultBody.innerHTML, /预测完成/);
assert.match(dom.requestResultBody.innerHTML, /未生成完整逐 Token 间隔/);
assert.match(dom.utilizationList.innerHTML, /不能代表完整请求/);
assert.equal(dom.componentTimeseriesPanel.hidden, true);
""", r"""
const state = {settings: {language: 'zh-CN'}, componentTimeseriesView: {}};
const dom = new Proxy({}, {get: (target, key) => target[key] ??= {}});
const uiText = (zh, en, values = {}) => String(zh).replace(/\{(\w+)\}/g, (_, key) => String(values[key] ?? ''));
const hydrateConceptHelp = () => {}, renderComparison = () => {};
const bindRuntimeTrackResizeObserver = () => {}, scheduleRuntimeSharedTracks = () => {};
""")


@pytest.mark.parametrize("reason,zh,en", [
    ("fast_mode_requires_bf16_unquantized_layers", "未量化的 BF16", "unquantized BF16"),
    ("fast_mode_requires_dense_layers", "稠密模型层", "dense model layers"),
    ("fast_mode_requires_resident_weights", "权重预先驻留", "resident weights"),
    ("fast_mode_requires_flash_attention", "启用 Flash Attention", "requires Flash Attention"),
    ("fast_mode_requires_one_gpu_for_model_operators", "同一张 GPU", "one GPU"),
    ("fast_mode_requires_physical_hbm_or_hbf", "HBM 或 HBF 物理配置", "physical HBM or HBF"),
    ("fast_mode_requires_bf16_kv", "BF16 KV 缓存", "BF16 KV cache"),
    ("fast_mode_requires_ocp_hbf_protocol", "OCP HBF 0.7 协议", "OCP HBF 0.7 protocol"),
    ("stateful_gpu_cache_requires_detailed_mode", "有状态 GPU 缓存", "stateful GPU cache"),
    ("calibration_does_not_cover_hbf_kv_page_period", "完整 HBF KV 页写入周期", "complete HBF KV page-write period"),
    ("detailed_decode_step_count_does_not_match_runtime", "当前运行状态不一致", "does not match runtime state"),
])
def test_fast_accuracy_scope_fallback_reasons_are_readable_in_both_languages(reason, zh, en):
    browser_check(("runManifestMarkup",), {"simulation_mode": "fast",
        "approximation": {"effective_mode": "detailed", "fallback_reasons": [reason]},
        "zh": zh, "en": en}, """
assert.ok(runManifestMarkup(payload).includes(payload.zh));
state.settings.language = 'en';
assert.ok(runManifestMarkup(payload).includes(payload.en));
""", """
const state = {settings: {language: 'zh-CN'}};
const escapeHtml = String, formatResultNumber = value => ({html: String(value)});
""")
