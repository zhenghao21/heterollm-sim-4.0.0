"""Exercise report and catalog producers with their current browser consumers."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from heterollm_sim.protocol_presets import protocol_preset_detail, protocol_preset_page
from heterollm_sim.reporting import OnlineScenarioResult, report_dict
from heterollm_sim.workload_presets import workload_preset_page
from test_async_kv_prefetch import runtime
from test_backend_execution_semantics import manifest


def browser_check(names, payload, assertions, setup=""):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser contract checks")
    source = (Path(__file__).parents[1] / "src/heterollm_sim/webui/app.js").read_text(encoding="utf-8")
    functions = []
    for name in names:
        match = re.search(r"^(?:async )?function " + name + r"\([^\n]*\) \{.*?^\}", source, re.M | re.S)
        assert match, name
        functions.append(match.group())
    script = """
const assert = require('node:assert/strict');
const asObject = value => value && typeof value === 'object' && !Array.isArray(value) ? value : {};
const asArray = value => Array.isArray(value) ? value : [];
const deepClone = structuredClone;
""" + setup + "\n" + "\n".join(functions) + "\nconst payload = " + json.dumps(payload) + ";\n" + assertions
    result = subprocess.run([node], input=script, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr


def test_protocol_and_workload_consumers_use_current_catalog_envelopes():
    protocols = protocol_preset_page()
    browser_check(("protocolPresetEnvelopeItems", "protocolPresetDetailItem", "syncProtocolPresetFilters",
                   "normalizeWorkloadPresetCatalogItem"),
        dict(protocols=protocols, details=[protocol_preset_detail(row["id"]) for row in protocols["items"]],
             workloads=workload_preset_page()), """
assert.deepEqual(protocolPresetEnvelopeItems(payload.protocols), payload.protocols.items);
syncProtocolPresetFilters(payload.protocols);
assert.deepEqual(dom.protocolPresetProtocolFilter.values, payload.protocols.filters.protocol);
assert.deepEqual(dom.protocolPresetOrganizationFilter.values, payload.protocols.filters.organization);
for (const detail of payload.details) {
  const actual = protocolPresetDetailItem(detail);
  assert.equal(actual.id, detail.preset.id);
  assert.deepEqual(actual.simulation_defaults, detail.simulation_defaults);
}
for (const retired of [payload.protocols.items, {presets: payload.protocols.items}, {results: payload.protocols.items}]) {
  assert.deepEqual(protocolPresetEnvelopeItems(retired), []);
}
for (const retired of [payload.details[0].preset, {item: payload.details[0].preset}]) {
  assert.throws(() => protocolPresetDetailItem(retired));
}
for (const item of payload.workloads.items) {
  const actual = normalizeWorkloadPresetCatalogItem(item);
  assert.equal(actual.id, item.id);
  assert.equal(actual.sourceId, item.sourceId);
  assert.deepEqual(actual.label, item.label);
  assert.equal(actual.promptTokens, item.promptTokens);
  const {id, sourceId, ...other} = item;
  assert.equal(normalizeWorkloadPresetCatalogItem({...other, preset_id: id}), null);
  assert.equal(normalizeWorkloadPresetCatalogItem({...other, id, source_id: sourceId}).sourceId, '');
}
""", """
const WORKLOAD_PRESET_SOURCE = 'current';
const dom = {protocolPresetProtocolFilter: {}, protocolPresetOrganizationFilter: {}};
const replaceFilterOptions = (target, title, values) => { target.values = values; };
""")


@pytest.fixture(scope="module")
def current_report():
    runner = runtime()
    return report_dict(OnlineScenarioResult(runner.plan.scenario, runner.run(), manifest(), (), "aggregate"))


def test_online_report_has_one_kv_contract_and_analysis_uses_selected_boundary(current_report):
    kv = current_report["kv_cache"]
    assert {"layout_mode", "capacity_bytes_by_component", "peak_bytes_by_component", "layer_owner",
            "effective_physical_capacity_bytes", "batch_retry_count"} <= kv.keys()
    assert not {"kv_layout_mode", "kv_capacity_bytes_by_component", "kv_peak_bytes_by_component",
                "kv_layer_owner", "kv_layer_components", "effective_physical_capacity",
                "logical_decode_write_bytes", "physical_decode_write_bytes"} & kv.keys()
    assert "result_semantics" in current_report
    assert "measurement_semantics" not in current_report
    browser_check(("kvAnalysisReportMetrics",), current_report, """
const actual = kvAnalysisReportMetrics(payload);
assert.equal(actual.peak_bytes, payload.kv_cache.peak_used_bytes);
assert.equal(actual.capacity_bytes, payload.kv_cache.effective_physical_capacity_bytes);
assert.equal(actual.ttft_ns, payload.summary.ttft_ns.p50);
assert.equal(actual.e2e_ns, payload.summary.e2e_ns.p50);
assert.equal(actual.traffic_bytes, ['physical_prefill_read_bytes', 'physical_prefill_write_bytes',
  'physical_decode_read_bytes', 'physical_decode_append_bytes'].reduce((sum, key) => sum + payload.kv_cache[key], 0));
const selected = kvAnalysisReportMetrics({summary: {ttft_ns: {p50: 7}, engine_ttft_ns: {p50: 99}, makespan_ns: 88},
  kv_cache: {peak_used_bytes: 0, effective_physical_capacity_bytes: 10, effective_physical_capacity: 999}});
assert.equal(selected.ttft_ns, 7);
assert.equal(selected.e2e_ns, null);
assert.equal(selected.peak_bytes, 0);
assert.equal(selected.capacity_bytes, 10);
assert.equal(selected.traffic_bytes, 0);
assert.equal(kvAnalysisReportMetrics({report: payload}).capacity_bytes, null);
""")


def test_runtime_ui_displays_current_kv_owners_capacity_and_event_counts(current_report):
    browser_check(("renderRuntime",), current_report, """
payload.kv_cache.batch_retry_count = 11;
payload.kv_cache.context_shift_count = 12;
payload.kv_cache.idle_slots_cleared = 13;
payload.kv_cache.capacity_failure_count = 14;
renderRuntime(payload);
const html = dom.runtimeSummary.innerHTML;
assert.ok(html.includes('批次重试 11；上下文偏移 12；空闲槽位清理 13；容量失败 14'));
assert.ok(html.includes('>' + payload.kv_cache.layout_mode + '<'));
assert.ok(html.includes('>' + payload.kv_cache.effective_physical_capacity_bytes + '<'));
for (const [layer, owner] of Object.entries(payload.kv_cache.layer_owner)) {
  assert.ok(html.includes('<td>' + layer + '</td><td>' + owner + '</td>'));
}
assert.equal(dom.runtimeModeMeta.textContent, '连续批处理');
assert.equal(executionStats.find(row => row[2] === 'tp_degree')[1].text,
  String(payload.summary.parallel.tp_degree));
state.scenario = {workload: {scheduler: {mode: 'continuous'}},
  placement: {parallel: {tp_degree: 999}}};
delete payload.execution_mode;
delete payload.summary.parallel.tp_degree;
renderRuntime(payload);
assert.equal(dom.runtimeModeMeta.textContent, '—');
assert.equal(executionStats.find(row => row[2] === 'tp_degree')[1].text, '—');
""", r"""
const state = {};
const dom = {runtimeSummary: {}, runtimeModeMeta: {}};
const escapeHtml = String;
const uiText = (zh, en, vars = {}) => zh.replace(/\{(\w+)\}/g, (_, key) => String(vars[key]));
const resultText = value => ({html: String(value), text: String(value)});
const unavailableResult = () => resultText('—');
const formatResultNumber = resultText, formatResultBytes = resultText, formatResultRate = resultText;
const formatResultDurationNs = resultText, formatResultPercent = resultText;
const appendResult = (value, label) => resultText(value.text + label);
const combineResults = values => resultText(values.map(v => v.text).join(' / '));
const runtimeValue = (obj, key, formatter = resultText) => formatter(obj[key]);
let executionStats;
const runtimeGroup = (title, rows) => { if (title === '执行') executionStats = rows; return ''; };
const bindRuntimeTrackResizeObserver = () => {};
const scheduleRuntimeSharedTracks = () => {};
""")


def test_manifest_source_comes_from_current_metadata(current_report):
    browser_check(("runManifestMarkup",), current_report, """
payload.manifest.metadata.source = 'current-source';
payload.manifest.source = 'retired-manifest-source';
payload.source = 'retired-report-source';
const html = runManifestMarkup(payload);
assert.ok(html.includes('current-source'));
assert.ok(!html.includes('retired-manifest-source'));
assert.ok(!html.includes('retired-report-source'));
""", """
const state = {};
const escapeHtml = String;
const formatResultNumber = value => ({html: String(value)});
""")


def test_timeseries_consumer_keeps_actual_series_and_ignores_retired_wrapping(current_report):
    browser_check(("normalizeComponentTimeseries", "timeseriesMetricLabelPair"), current_report, """
const actual = normalizeComponentTimeseries(payload);
assert.equal(actual.components.length, payload.component_timeseries.components.length);
assert.ok(actual.series.length > 0);
for (const component of payload.component_timeseries.components) {
  for (const series of component.series) {
    const normalized = actual.series.find(s => s.owner_id === component.component_id && s.series_id === series.series_id);
    assert.ok(normalized);
    assert.equal(normalized.component_kind, component.component_kind);
    assert.equal(normalized.label_cn, series.label_cn);
    assert.equal(normalized.fidelity, series.fidelity);
    assert.deepEqual(normalized.points, series.points);
  }
}
assert.deepEqual(normalizeComponentTimeseries({report: payload}).series, []);
const retired = normalizeComponentTimeseries({component_timeseries: {components: [{component_id: 'm', kind: 'old',
  series: [{series_id: 'x', metric: 'unknown_metric', label_zh: 'old'}]}]}});
assert.equal(retired.components[0].component_kind, 'unknown');
assert.equal(retired.series[0].label_cn, 'unknown_metric');
""", "const TIMESERIES_METRIC_LABELS = {};")
