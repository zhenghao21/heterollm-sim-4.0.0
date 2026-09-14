"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const appPath = path.join(webui, "app.js");
const appSource = fs.readFileSync(appPath, "utf8");
const I18n = require(path.join(webui, "ui-i18n.js"));
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));

const CJK = /[\u3400-\u9fff]/u;

function fakeElement() {
  return {
    className: "",
    dataset: {},
    hidden: false,
    innerHTML: "",
    textContent: "",
    classList: { add() {}, remove() {}, toggle() {} },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    removeAttribute() {},
    setAttribute() {},
    style: { removeProperty() {}, setProperty() {} },
  };
}

function loadRuntime() {
  const context = vm.createContext({
    AbortController,
    CSS: { escape: (value) => String(value) },
    Intl,
    Map,
    ModelGraphCore,
    Option: class Option {},
    Promise,
    Set,
    TopologyCore,
    TraceViewCore,
    URL,
    URLSearchParams,
    UiI18n: I18n,
    cancelAnimationFrame() {},
    clearInterval,
    clearTimeout,
    console,
    document: {
      addEventListener() {},
      body: { appendChild() {} },
      createElement() { return fakeElement(); },
      documentElement: { clientHeight: 720, clientWidth: 1280, dataset: {}, style: { removeProperty() {}, setProperty() {} } },
      querySelector() { return null; },
      querySelectorAll() { return []; },
    },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    requestAnimationFrame() { return 1; },
    setInterval,
    setTimeout,
    structuredClone,
  });
  vm.runInContext(`${appSource}\n;globalThis.__pass03 = {
    applyTraceFilters,
    dom,
    normalizeComponentTimeseries,
    numericalBackendLabel,
    renderCategories,
    renderComparison,
    renderControlPlaneStatus,
    renderRequestResults,
    renderRunJobDialog,
    renderRuntime,
    renderUtilization,
    renderTracePageState,
    retentionPolicyLabel,
    runEstimateWarningText,
    runProgressCountText,
    runProgressDetailText,
    runProgressMessageText,
    runStageLabel,
    runStatusLabel,
    state,
    syncRunButtons,
    updateUiSettings,
    timeseriesChartMarkup,
    timeseriesControlsMarkup,
    traceFidelityLabel,
    traceLimitationText,
    traceStepSummary,
  };`, context, { filename: appPath });
  return context.__pass03;
}

function setLanguage(runtime, language) {
  I18n.setLanguage(language, null);
  runtime.state.settings.language = language;
}

test("unknown Chinese prose is never fragmented into mixed-language token substitutions", () => {
  const source = "连续批处理曲线是批次包络聚合或代表性峰值，不能还原为逐算子时间线。";
  assert.equal(I18n.localizeText(source, "en"), source);
  assert.equal(I18n.localizeText("65 条后端曲线 · 每条最多 500 点 · X 轴统一为线性时间", "en"), "65 条后端曲线 · 每条最多 500 点 · X 轴统一为线性时间");
});

test("Results dynamic renderers emit complete English UI copy and preserve raw values", () => {
  const runtime = loadRuntime();
  Object.assign(runtime.dom, {
    runtimeModeMeta: fakeElement(),
    runtimeSummary: fakeElement(),
    comparisonStrip: fakeElement(),
    bottleneckMeta: fakeElement(),
    utilizationList: fakeElement(),
    categoryList: fakeElement(),
    requestResultMeta: fakeElement(),
    requestResultBody: fakeElement(),
  });
  runtime.state.scenario = { placement: { parallel: { tp_degree: 1, pp_degree: 1, ep_degree: 1 } }, workload: { scheduler: {} } };
  const report = {
    execution_mode: "continuous_batching",
    summary: {
      parallel: { tp_degree: 1, pp_degree: 1, ep_degree: 1 },
      mtp: { proposed_tokens: 6, accepted_tokens: 4, committed_tokens: 4, rejected_tokens: 2, effective_acceptance_rate: 2 / 3 },
      goodput: { requests_per_s: 2, visible_output_tokens_per_s: 8, qualified_requests: 1 },
      bottleneck_resource: { resource_id: "gpu0.frontend", utilization: 0.5 },
    },
    scheduler: { total_batches: 4, max_batch_sequences: 1, max_batch_tokens: 32, preemptions: 0, priority_preemptions: 0, memory_preemptions: 0 },
    kv_cache: {
      peak_used_bytes: 81920,
      max_live_tokens_per_request: 67,
      swap_events: 0,
      swap_bytes: 0,
      capacity_pages: 1048576,
      logical_prefill_read_bytes: 32768,
      physical_prefill_read_bytes: 32768,
      prefetch_distance_modeled: false,
    },
    resource_utilization: { "gpu0.frontend": 0.5 },
    category_time_ns: { communication: 100 },
    critical_path_category_ns: { communication: 90 },
  };

  setLanguage(runtime, "en");
  runtime.renderRuntime(report);
  runtime.renderUtilization?.(report);
  runtime.renderCategories(report);
  runtime.renderRequestResults({
    "request-0000": { status: "finished", rejection_reason: "backend-raw-reason", arrival_ns: 0, ttft_ns: 10, tbt_ns: [2, 3], tpot_ns: 3, e2e_ns: 20, visible_output_tokens: 4 },
  });
  const english = [
    runtime.dom.runtimeModeMeta.textContent,
    runtime.dom.runtimeSummary.innerHTML,
    runtime.dom.bottleneckMeta.innerHTML,
    runtime.dom.utilizationList.innerHTML,
    runtime.dom.categoryList.innerHTML,
    runtime.dom.requestResultMeta.innerHTML,
    runtime.dom.requestResultBody.innerHTML,
  ].join("\n");
  assert.doesNotMatch(english, CJK);
  assert.match(english, /Logical 32 KiB \/ Physical 32 KiB/u);
  assert.match(english, /Not explicitly modeled \(policy metadata only\)/u);
  assert.match(english, /backend-raw-reason/u);

  setLanguage(runtime, "zh-CN");
  runtime.renderRuntime(report);
  assert.match(runtime.dom.runtimeSummary.innerHTML, /逻辑 32 KiB \/ 物理 32 KiB/u);
});

test("playback and time-series helpers use complete bilingual phrases", () => {
  const runtime = loadRuntime();
  runtime.dom.tracePageBar = fakeElement();
  runtime.dom.tracePageStatus = fakeElement();
  runtime.dom.tracePagePreviousButton = fakeElement();
  runtime.dom.tracePageNextButton = fakeElement();
  runtime.state.tracePlayback.mode = "task";
  runtime.state.tracePlayback.page = { offset: 0, returned: 168, total: 168, previous_offset: null, has_more: false };
  runtime.state.tracePlayback.batchFilter = "cohort-000000";

  setLanguage(runtime, "en");
  assert.equal(runtime.traceFidelityLabel("representative"), "Representative events");
  assert.equal(runtime.traceLimitationText("内存区间是组件本地的确定性逻辑字节区间，不是 JEDEC 物理地址。"), "Memory intervals are deterministic component-local logical byte ranges, not JEDEC physical addresses.");
  assert.equal(runtime.traceLimitationText("内存偏移是组件本地的确定性逻辑字节区间；未建模 JEDEC bank/row/column 物理寻址。"), "Memory offsets are deterministic component-local logical byte ranges; JEDEC bank/row/column physical addressing is not modeled.");
  assert.equal(runtime.traceLimitationText("可扩展在线推理仅公开已实现的批次包络和聚合资源计数，无法据此反推算子执行区间。"), "Scalable online inference exposes implemented batch envelopes and aggregate resource counts only; per-operator execution intervals cannot be reconstructed from them.");
  assert.equal(runtime.traceLimitationText("每个批次最多序列化 32 个代表性条目。"), "At most 32 representative items are serialized per batch.");
  assert.equal(runtime.traceLimitationText("聚合路由 hop 保留拓扑、链路和协议标识，但不声称逐 hop 的开始/结束时间。"), "Aggregate route hops preserve topology, link, and protocol identity without claiming per-hop start or end times.");
  assert.match(runtime.traceStepSummary().label, /task events/u);
  runtime.renderTracePageState();
  assert.doesNotMatch(runtime.dom.tracePageStatus.textContent, CJK);
  assert.match(runtime.dom.tracePageStatus.textContent, /Task replay.*global simulation time/u);

  const normalized = runtime.normalizeComponentTimeseries({ component_timeseries: {
    schema_version: "1.0",
    fidelity: "representative",
    point_limit: 500,
    components: [{ component_id: "gpu0", component_kind: "gpu", series: [{
      series_id: "modeled-compute",
      label_cn: "建模计算利用率",
      metric: "modeled_compute_utilization",
      unit: "ratio",
      fidelity: "representative",
      quality: "modeled",
      points: [{ start_ns: 0, end_ns: 10, value: 0.5 }],
    }] }],
  } });
  runtime.state.componentTimeseriesView.slots = [{ id: 1, ownerKey: "component:gpu0", seriesKey: normalized.series[0].series_key, yScale: "linear", color: "#5eb6c0" }];
  const controls = runtime.timeseriesControlsMarkup(runtime.state.componentTimeseriesView.slots[0], normalized);
  const chart = runtime.timeseriesChartMarkup(normalized.series[0], { slotId: 1, xDomain: [0, 10], controlsMarkup: controls });
  assert.doesNotMatch(`${controls}\n${chart}`, CJK);
  assert.match(chart, /Modeled compute utilization/u);
  assert.match(chart, /Simulation time \(linear\)/u);

  setLanguage(runtime, "zh-CN");
  assert.equal(runtime.traceFidelityLabel("representative"), "代表性事件");
  assert.match(runtime.timeseriesChartMarkup(normalized.series[0], { xDomain: [0, 10] }), /建模计算利用率/u);
});

test("Background Run renders complete English copy while preserving job identifiers", () => {
  const runtime = loadRuntime();
  Object.assign(runtime.dom, {
    runJobDialog: fakeElement(),
    runEstimateRisk: fakeElement(),
    runEstimateSummary: fakeElement(),
    runEstimateWarnings: fakeElement(),
    runJobProgressPanel: fakeElement(),
    runJobStatus: fakeElement(),
    runProgressStage: fakeElement(),
    runProgressCount: fakeElement(),
    runProgressBar: fakeElement(),
    runProgressMessage: fakeElement(),
    startRunJobButton: fakeElement(),
    cancelRunJobButton: fakeElement(),
    dismissRunJobButton: fakeElement(),
    runButton: fakeElement(),
    rerunButton: fakeElement(),
    emptyRunButton: fakeElement(),
    compareButton: fakeElement(),
  });
  runtime.state.runEstimate = {
    schema_version: "1.0",
    risk_level: "low",
    risk_level_zh: "低",
    request_count: 1,
    prompt_tokens: 8,
    output_tokens: 4,
    layer_count: 2,
    world_size: 1,
    estimated_cohort_count: 4,
    estimated_event_task_count: 763,
    recommended_retention_policy: "aggregate",
    warnings: ["本结果不包含墙钟秒数；实际运行时间取决于主机、并发和场景细节。"],
  };
  runtime.state.runJob = {
    job_id: "job-raw-17",
    status: "running",
    progress: {
      stage: "serving_cohorts",
      completed: 2,
      total: 4,
      ratio: 0.5,
      unit: "serving_batches",
      message: "正在推进在线批次",
      detail: { stage: "cohort_tasks", completed: 3, total: 5, unit: "schedule_tasks" },
    },
  };

  setLanguage(runtime, "en");
  runtime.renderRunJobDialog();
  const english = [
    runtime.dom.runEstimateRisk.textContent,
    runtime.dom.runEstimateSummary.innerHTML,
    runtime.dom.runEstimateWarnings.innerHTML,
    runtime.dom.runJobStatus.textContent,
    runtime.dom.runProgressStage.textContent,
    runtime.dom.runProgressCount.textContent,
    runtime.dom.runProgressBar.textContent,
    runtime.dom.runProgressMessage.textContent,
    runtime.dom.cancelRunJobButton.textContent,
    runtime.dom.dismissRunJobButton.textContent,
    runtime.dom.runButton.textContent,
  ].join("\n");
  assert.doesNotMatch(english, CJK);
  assert.match(english, /Risk: Low/u);
  assert.match(english, /wall-clock seconds/u);
  assert.match(english, /Executed 2 \/ 4 online cohorts/u);
  assert.match(english, /Current cohort: 3\/5 topology tasks/u);
  assert.match(runtime.runProgressMessageText("后端保留的进度说明"), /后端保留的进度说明/u);
  assert.equal(runtime.retentionPolicyLabel("aggregate"), "Aggregate metrics only");
  assert.equal(runtime.numericalBackendLabel("auto"), "Automatic selection (Auto)");
  assert.equal(runtime.runStageLabel("completed"), "Completed");
  assert.equal(runtime.runStatusLabel("running"), "Running");

  setLanguage(runtime, "zh-CN");
  runtime.renderRunJobDialog();
  assert.match(runtime.dom.runEstimateRisk.textContent, /风险：低/u);
  assert.match(runtime.dom.runProgressCount.textContent, /在线批次/u);
});

test("language changes refresh topbar run button labels", () => {
  const runtime = loadRuntime();
  Object.assign(runtime.dom, {
    runButton: fakeElement(),
    rerunButton: fakeElement(),
    emptyRunButton: fakeElement(),
    compareButton: fakeElement(),
  });

  I18n.setLanguage("en", null);
  runtime.state.settings.language = "en";
  runtime.syncRunButtons();
  assert.equal(runtime.dom.runButton.textContent, "Run simulation");
  assert.equal(runtime.dom.rerunButton.textContent, "Run again");
  assert.equal(runtime.dom.emptyRunButton.textContent, "Run current scenario");

  runtime.updateUiSettings({ language: "zh-CN" });
  assert.equal(runtime.dom.runButton.textContent, "运行仿真");
  assert.equal(runtime.dom.rerunButton.textContent, "重新运行");
  assert.equal(runtime.dom.emptyRunButton.textContent, "运行当前场景");
});

test("Pass 03 concept-help targets are bound without turning identifiers into prose", () => {
  assert.match(appSource, /run-manifest-summary[^>]*data-concept-help="analytical_report"|data-concept-help="analytical_report"[^>]*run-manifest-summary/u);
  assert.match(appSource, /run-technical-details[\s\S]*?data-concept-help="run_manifest"/u);
  const runtime = loadRuntime();
  Object.assign(runtime.dom, {
    controlPlaneStatus: fakeElement(),
    controlPlaneStatusBadge: fakeElement(),
    controlPlaneStatusSummary: fakeElement(),
    controlPlaneStatusMetrics: fakeElement(),
  });
  runtime.state.scenario = {
    placement: {
      metadata: {
        control_plane: {
          policy: { options: { mode: "heuristic", objective: "balanced" } },
          decision: {
            fully_placed: true,
            objective: "balanced",
            objective_value: 123.5,
            lower_bound: 100,
            gap: 0.19,
            operator_execution_targets: { "dense0.mlp": [{ rank_id: 0, component_id: "gpu0" }] },
            rank_weight_shards: { "dense0.mlp_weights": [{ rank_id: 0, storage_component_id: "hbm0" }] },
          },
          evidence: { fingerprint_schema: "runtime-control-plane-v4", input_fingerprint: "abc123" },
        },
      },
    },
  };
  runtime.renderControlPlaneStatus();
  assert.match(runtime.dom.controlPlaneStatusMetrics.innerHTML, /data-concept-help="objective"/u);
  assert.match(runtime.dom.controlPlaneStatusMetrics.innerHTML, /data-concept-help="lower_bound"/u);
  assert.match(runtime.dom.controlPlaneStatusMetrics.innerHTML, /data-concept-help="optimality_gap"/u);
  assert.match(appSource, /traceDetailFact\(uiText\("运行标记", "Runtime marker"\)[\s\S]*?"event_marker"/u);
  assert.match(appSource, /traceDetailFact\(uiText\("经过的路径", "Path"\)[\s\S]*?"protocol_path"/u);
  assert.match(appSource, /function refreshConceptHelpLanguage[\s\S]*?aria-label[\s\S]*?Show concept help for/u);
  assert.doesNotMatch(appSource, /data-concept-help="(?:request|rank|operator|tensor)"[^>]*data-(?:request|rank|operator|tensor)-id=/u);
});
