"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const appSource = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const htmlSource = fs.readFileSync(path.join(webui, "index.html"), "utf8");
const ModelGraphCore = require(path.join(webui, "model-graph-core.js"));
const TopologyCore = require(path.join(webui, "topology-core.js"));
const TraceViewCore = require(path.join(webui, "trace-view-core.js"));

function conceptRuntime() {
  const context = vm.createContext({
    AbortController,
    CSS: { escape: String },
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
    clearTimeout,
    console,
    document: {
      addEventListener() {},
      createElement() { return {}; },
      querySelector() { return null; },
      querySelectorAll() { return []; },
    },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    setTimeout,
    structuredClone,
  });
  vm.runInContext(`${appSource}\n;globalThis.__conceptHelp192 = {
    CONCEPT_HELP_ZH,
    CONCEPT_HELP_EN,
    CONCEPT_HELP_DETAIL_PROFILES,
    CONCEPT_HELP_DETAIL_OVERRIDES,
    CONCEPT_HELP_COVERAGE_BY_VIEW,
    CONCEPT_HELP_VISIBLE_LABEL_BINDINGS,
    CONCEPT_HELP_SHARED_LABEL_ALLOWLIST,
    CONCEPT_TERM_PATTERNS,
    cimCostProfileMarkup,
    componentProfileBindingMarkup,
    cpuCostProfileMarkup,
    conceptHelpSections,
    gpuCostProfileMarkup,
    hydrateConceptHelp,
    runtimeStat,
    state,
  };`, context);
  return context.__conceptHelp192;
}

function normalized(value) {
  return String(value || "").replace(/\s+/gu, " ").trim().toLocaleLowerCase("en-US");
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/gu, "\\$&");
}

function sourceBetween(startName, endName) {
  const start = appSource.indexOf(`function ${startName}`);
  const end = appSource.indexOf(`function ${endName}`, start + 1);
  assert.ok(start >= 0, `${startName} must exist`);
  assert.ok(end > start, `${endName} must follow ${startName}`);
  return appSource.slice(start, end);
}

class ConceptHelpTitle {
  constructor(text) {
    this.textContent = text;
    this.dataset = {};
    this.attributes = {};
    this.classList = { add() {} };
  }

  closest() { return null; }
  hasAttribute(name) { return Object.hasOwn(this.attributes, name); }
  insertAdjacentHTML() {}
  setAttribute(name, value) { this.attributes[name] = value; }
}

function conceptHelpRoot(titles) {
  return {
    querySelectorAll(selector) {
      if (selector === "[data-concept-help]") return titles.filter((title) => Object.hasOwn(title.dataset, "conceptHelp"));
      if (selector.includes("[data-field-help]") || selector.startsWith(".field-help-popover")) return [];
      return titles;
    },
  };
}

test("2.2.0 concept coverage is bilingual, page-scoped, and has no orphan professional term", () => {
  const runtime = conceptRuntime();
  const zhKeys = Object.keys(runtime.CONCEPT_HELP_ZH).sort();
  const enKeys = Object.keys(runtime.CONCEPT_HELP_EN).sort();
  assert.deepEqual(zhKeys, enKeys);
  assert.ok(zhKeys.length >= 230, `expected governed glossary, received ${zhKeys.length} keys`);

  const dictionaryKeys = new Set(zhKeys);
  const coveredByPage = new Set();
  for (const [view, keys] of Object.entries(runtime.CONCEPT_HELP_COVERAGE_BY_VIEW)) {
    assert.ok(keys.length >= 18, `${view} needs material professional-term coverage`);
    for (const key of keys) {
      assert.ok(dictionaryKeys.has(key), `${view} references missing glossary key ${key}`);
      coveredByPage.add(key);
    }
  }
  for (const key of [
    "hardware_topology", "hbm_stack", "physical_link", "repeat_pattern", "group_collapse", "layer_override",
    "mapping_diagnostic", "fully_placed", "infeasible_mapping", "continuous_batching", "token_budget",
    "resource_contention", "change_point", "route_hop", "dma_bandwidth", "dma_energy", "dma_resource",
    "outstanding_requests",
  ]) assert.ok(coveredByPage.has(key), `expanded coverage misses ${key}`);

  const patternKeys = new Set(runtime.CONCEPT_TERM_PATTERNS.map(([key]) => key));
  const explicitKeys = new Set([
    ...htmlSource.matchAll(/data-concept-help="([a-z0-9_]+)"/gu),
    ...appSource.matchAll(/data-concept-help="([a-z0-9_]+)"/gu),
  ].map((match) => match[1]));
  for (const key of explicitKeys) assert.ok(dictionaryKeys.has(key), `explicit binding references missing glossary key ${key}`);
  const orphanKeys = zhKeys.filter((key) => !patternKeys.has(key) && !explicitKeys.has(key));
  assert.deepEqual(orphanKeys, [], `professional terms need a pattern or explicit binding: ${orphanKeys.join(", ")}`);
});

test("2.2.0 static visible labels allow only declared canonical sharing", () => {
  const runtime = conceptRuntime();
  const labelsByKey = new Map();
  for (const match of htmlSource.matchAll(/<([a-z0-9-]+)[^>]*data-concept-help="([a-z0-9_]+)"[^>]*>([\s\S]*?)<\/\1>/giu)) {
    const key = match[2];
    const label = match[3].replace(/<[^>]+>/gu, " ").replace(/\s+/gu, " ").trim();
    if (!label || label.includes("${")) continue;
    if (!labelsByKey.has(key)) labelsByKey.set(key, new Set());
    labelsByKey.get(key).add(label);
  }
  for (const [key, labels] of labelsByKey) {
    if (labels.size <= 1) continue;
    assert.deepEqual(
      [...labels].sort(),
      [...(runtime.CONCEPT_HELP_SHARED_LABEL_ALLOWLIST[key] || [])].sort(),
      `${key} shares materially different visible labels without an alias declaration`,
    );
  }
});

test("link inspector latency and bidirectional labels bind existing concept keys", () => {
  const runtime = conceptRuntime();
  assert.ok(runtime.CONCEPT_HELP_ZH.link_latency, "link_latency glossary entry exists");
  assert.ok(runtime.CONCEPT_HELP_ZH.bidirectional_link, "bidirectional_link glossary entry exists");

  const linkInspector = sourceBetween("renderLinkInspector", "bindInspectorFields");
  const latencyLine = linkInspector.split("\n").find((line) => line.includes('"延迟（Latency, ns）"')) || "";
  const bidirectionalLine = linkInspector.split("\n").find((line) => line.includes("双向传输（Bidirectional）")) || "";
  assert.ok(latencyLine.includes('helpKey: "link_latency"'), "Latency uses the existing link_latency concept key");
  assert.ok(
    bidirectionalLine.includes('fieldTitleMarkup("双向传输（Bidirectional）", "bidirectional_link")')
      || bidirectionalLine.includes('data-concept-help="bidirectional_link"'),
    "Bidirectional uses the existing bidirectional_link concept key",
  );
  assert.match(linkInspector, /hydrateConceptHelp\(dom\.inspectorContent\)/u);
});

test("V4 cost-profile markup exposes microarchitecture fields without flat throughput knobs", () => {
  const runtime = conceptRuntime();
  const titles = [
    new ConceptHelpTitle("GEMM 吞吐（GOP/s）"),
    new ConceptHelpTitle("逐元素吞吐（GOP/s）"),
    new ConceptHelpTitle("归约吞吐（GOP/s）"),
    new ConceptHelpTitle("Request Throughput"),
  ];
  runtime.hydrateConceptHelp(conceptHelpRoot(titles));
  assert.deepEqual(
    titles.map((title) => title.dataset.conceptHelp),
    ["gemm_throughput", "elementwise_throughput", "reduction_throughput", "request_throughput"],
  );

  const renderedProfiles = `${runtime.gpuCostProfileMarkup({})}${runtime.cpuCostProfileMarkup({})}`;
  for (const field of [
    "tensor_core.sm_count",
    "tensor_core.tensor_cores_per_sm",
    "scalar_ops_per_cycle",
    "special_function_ops_per_cycle",
    "pipeline.decode_width",
    "pipeline.issue_width",
    "pipeline.retire_width",
    "pipeline.reorder_buffer_entries",
    "pipeline.load_store_queue_entries",
  ]) assert.match(renderedProfiles, new RegExp(`data-cost-profile-field="${field.replaceAll(".", "\\.")}"`));
  assert.doesNotMatch(renderedProfiles, /data-cost-profile-field="(?:gemm_gops|elementwise_gops|reduction_gops|peak_tops)"/u);
});

const costProfileHelpLabels = Object.freeze([
  ["成本 Profile ID", "cost_profile"],
  ["Cache Level 1", "cache_hierarchy"],
  ["Tensor Core 频率（GHz）", "tensor_core"],
  ["占用率（ratio 0–1）", "occupancy"],
  ["可达效率（ratio 0–1）", "attainable_efficiency"],
  ["Pipeline 资源 ID", "pipeline"],
  ["NoC 带宽（GB/s）", "noc"],
]);

test("V4 cost-profile concept keys resolve from visible labels and are architecture-covered", () => {
  const runtime = conceptRuntime();
  const architectureCoverage = new Set(runtime.CONCEPT_HELP_COVERAGE_BY_VIEW.architecture);
  const renderedProfiles = `${runtime.componentProfileBindingMarkup("gpu", {})}${runtime.gpuCostProfileMarkup({})}${runtime.cpuCostProfileMarkup({})}${runtime.cimCostProfileMarkup({})}`;

  for (const [label, key] of costProfileHelpLabels) {
    assert.ok(runtime.CONCEPT_HELP_ZH[key], `${key} has Chinese help`);
    assert.ok(runtime.CONCEPT_HELP_EN[key], `${key} has English help`);
    assert.ok(architectureCoverage.has(key), `${key} is covered by the architecture view`);
    assert.match(renderedProfiles, new RegExp(escapeRegExp(label), "u"), `${label} remains a representative visible label`);
  }

  const titles = costProfileHelpLabels.map(([label]) => new ConceptHelpTitle(label));
  runtime.hydrateConceptHelp(conceptHelpRoot(titles));
  assert.deepEqual(titles.map((title) => title.dataset.conceptHelp), costProfileHelpLabels.map(([, key]) => key));
});

const governedLabels = Object.freeze([
  ["请求生成", "request_generation"], ["合成请求数", "synthetic_request_count"],
  ["合成提示 Token 数", "synthetic_prompt_tokens"], ["合成输出 Token 数", "synthetic_output_tokens"],
  ["预填充分块 Token 数", "prefill_chunk_tokens"], ["张量并行度（TP Degree）", "tp_degree"],
  ["流水线并行度（PP Degree）", "pp_degree"], ["专家并行度（EP Degree）", "ep_degree"],
  ["批次数（Batches）", "total_batches"], ["峰值批次（Peak Batch）", "peak_batch"],
  ["峰值占用（Peak）", "kv_peak_occupancy"], ["页容量（Capacity Pages）", "kv_capacity_pages"],
  ["迁移合计（Migration）", "kv_migration_total"], ["重计算（Recompute）", "kv_recompute"],
  ["Token 率（Tokens / s）", "goodput_token_rate"], ["请求率（Requests / s）", "goodput_request_rate"],
  ["请求吞吐（Request Throughput）", "request_throughput"], ["Token 吞吐（Token Throughput）", "token_throughput"],
  ["每输出 Token 时间（TPOT）", "tpot"], ["相对时间（Relative Time）", "relative_time"],
  ["算子执行目标（Operator Targets）", "operator_targets"], ["权重张量分片（Weight Tensor Shards）", "weight_tensor_shards"],
  ["KV 读写流量（KV Traffic）", "kv_traffic_summary"], ["KV 迁移与限制（KV Movement）", "kv_movement_summary"],
]);

test("2.2.0 visible semantic labels resolve to their canonical keys", () => {
  const runtime = conceptRuntime();
  const titles = governedLabels.map(([label]) => new ConceptHelpTitle(label));
  runtime.hydrateConceptHelp(conceptHelpRoot(titles));
  assert.deepEqual(titles.map((title) => title.dataset.conceptHelp), governedLabels.map(([, key]) => key));
  for (const [, key] of governedLabels) assert.ok(runtime.CONCEPT_HELP_ZH[key], `missing governed key ${key}`);

  const runtimeBindings = [
    ["批次数（Batches）", "total_batches"], ["峰值批次（Peak Batch）", "peak_batch"],
    ["峰值占用（Peak）", "kv_peak_occupancy"], ["页容量（Capacity Pages）", "kv_capacity_pages"],
    ["迁移合计（Migration）", "kv_migration_total"], ["重计算（Recompute）", "kv_recompute"],
    ["Token 率（Tokens / s）", "goodput_token_rate"],
  ];
  for (const [label, key] of runtimeBindings) {
    const sourceLine = appSource.split("\n").find((line) => line.includes(`["${label}",`));
    assert.ok(sourceLine?.includes(`"${key}"`), `${label} must be explicitly bound to ${key}`);
  }
  assert.match(appSource, /runtimeGroup\(uiText\("KV 读写流量", "KV Traffic"\)[\s\S]*?\], "is-kv-traffic", "kv_traffic_summary"\)/u);
  assert.match(appSource, /runtimeGroup\(uiText\("KV 迁移与限制", "KV Movement"\)[\s\S]*?\], "is-kv-movement", "kv_movement_summary"\)/u);
});

test("2.2.0 semantic-neighbor audit catches wrong sharing beyond exact duplicate strings", () => {
  const runtime = conceptRuntime();
  const semanticFamilies = [
    [["请求生成", "request_generation"], ["合成请求数", "synthetic_request_count"]],
    [["批次数（Batches）", "total_batches"], ["峰值批次（Peak Batch）", "peak_batch"]],
    [["每页 Token 数（Tokens per Page）", "page_size"], ["页容量（Capacity Pages）", "kv_capacity_pages"]],
    [["张量并行度（TP Degree）", "tp_degree"], ["每输出 Token 时间（TPOT）", "tpot"]],
    [["到达时间（Arrival, ns）", "arrival_time"], ["相对时间（Relative Time）", "relative_time"]],
    [["Token 吞吐（Token Throughput）", "token_throughput"], ["Token 率（Tokens / s）", "goodput_token_rate"]],
    [["算子执行目标（Operator Targets）", "operator_targets"], ["权重张量分片（Weight Tensor Shards）", "weight_tensor_shards"]],
  ];
  for (const family of semanticFamilies) {
    const titles = family.map(([label]) => new ConceptHelpTitle(label));
    runtime.hydrateConceptHelp(conceptHelpRoot(titles));
    const actual = titles.map((title) => title.dataset.conceptHelp);
    assert.deepEqual(actual, family.map(([, key]) => key));
    assert.equal(new Set(actual).size, family.length, `semantic neighbors must not share ${actual.join(", ")}`);
  }
  assert.deepEqual(
    Object.keys(runtime.CONCEPT_HELP_SHARED_LABEL_ALLOWLIST).sort(),
    ["arrival_time", "request"],
    "only declared true aliases may share a canonical key",
  );
});

test("2.2.0 governed canonical keys have independent bilingual six-section detail", () => {
  const runtime = conceptRuntime();
  const governedKeys = new Set([
    ...governedLabels.map(([, key]) => key),
    "cpu", "link_bandwidth", "parallel_strategy", "collective_algorithm", "colocated_ranks",
    "kv_cache_component", "kv_offload_component", "backing_component", "model_weight_capacity",
    "mtp_enabled", "simulation_results", "runtime_summary", "kv_traffic_summary", "kv_movement_summary", "goodput", "category_time", "request_metrics", "visible_tokens",
    "preemption_count", "preemption_breakdown", "kv_max_live_tokens", "kv_swap_summary",
    "kv_prefill_read_traffic", "kv_prefill_write_traffic", "kv_decode_read_traffic", "kv_decode_append_traffic",
    "kv_offload_total", "kv_prefetch_total", "kv_swap_transfer_time", "kv_prefetch_distance",
    "mtp_proposed_tokens", "mtp_accepted_tokens", "mtp_committed_tokens", "mtp_rejected_tokens", "mtp_effective_rate",
    "goodput_qualified_requests", "request_throughput", "token_throughput",
  ]);
  for (const key of governedKeys) {
    assert.ok(runtime.CONCEPT_HELP_DETAIL_OVERRIDES[key], `${key} needs a term-specific detail override`);
    for (const language of ["zh-CN", "en"]) {
      runtime.state.settings.language = language;
      const sections = runtime.conceptHelpSections(key);
      assert.equal(sections.length, 6, `${key}/${language} must have six sections`);
      assert.equal(new Set(sections.map((section) => normalized(section.body))).size, 6, `${key}/${language} sections must be independent`);
    }
  }
});

for (const language of ["zh-CN", "en"]) {
  test(`2.2.0 ${language} detailed concept explanations are term-specific`, () => {
    const runtime = conceptRuntime();
    runtime.state.settings.language = language;
    const fullFingerprints = new Map();
    const sectionFingerprints = Array.from({ length: 6 }, () => new Map());
    for (const key of Object.keys(runtime.CONCEPT_HELP_ZH).sort()) {
      const sections = runtime.conceptHelpSections(key);
      assert.equal(sections.length, 6, `${key} must retain all structured sections`);
      const bodies = sections.map((section) => normalized(section.body));
      assert.ok(bodies.every(Boolean), `${key} has an empty detailed section`);
      const full = bodies.join("\n");
      assert.equal(fullFingerprints.has(full), false, `${key} duplicates the complete explanation of ${fullFingerprints.get(full)}`);
      fullFingerprints.set(full, key);
      bodies.forEach((body, index) => {
        assert.equal(
          sectionFingerprints[index].has(body),
          false,
          `${key} duplicates section ${index + 1} of ${sectionFingerprints[index].get(body)}`,
        );
        sectionFingerprints[index].set(body, key);
      });
    }
  });
}
