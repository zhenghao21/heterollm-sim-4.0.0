"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const ModelGraphCore = require("../src/heterollm_sim/webui/model-graph-core.js");
const TopologyCore = require("../src/heterollm_sim/webui/topology-core.js");
const TraceViewCore = require("../src/heterollm_sim/webui/trace-view-core.js");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const app = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");
const css = fs.readFileSync(path.join(webui, "styles.css"), "utf8");

function helpers() {
  const context = vm.createContext({
    AbortController,
    CSS: { escape: String },
    Intl,
    ModelGraphCore,
    Promise,
    URL,
    Option: class Option { constructor(text, value) { this.text = text; this.value = value; } },
    TopologyCore,
    TraceViewCore,
    clearTimeout,
    console,
    document: { addEventListener() {} },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    setTimeout,
  });
  vm.runInContext(`${app}\n;globalThis.__v070 = {
    state,
    dom,
    formatBandwidthGbps,
    parseBandwidthToGbps,
    componentPresetEnvelopeItems,
    componentPresetDetailItem,
    componentPresetVendor,
    componentPresetEvidenceLabel,
    componentPresetDetailsMarkup,
    componentPresetSourceMarkup,
    componentPresetFactLabel,
    normalizePresetBandwidthText,
    syncComponentPresetFilters,
    materializeComponentPreset,
    renderUtilization,
    bindFieldHelp,
    closeFieldHelp,
    positionFieldHelp,
    conceptHelpText,
    runtimeSharedTrackMeasurements,
    applyRuntimeSharedTracks,
    bindToolbarMenus,
    toolbarMenuItems,
    componentInspectorEvidence,
    componentInspectorSource,
    componentInspectorProfile,
    componentPresetCardMarkup,
    componentPresetType,
    isTopologyBundlePreset,
    materializeTopologyBundle,
    protocolPresetEnvelopeItems,
    protocolPresetDetailItem,
    protocolPresetBandwidthMarkup,
    protocolPresetCardMarkup,
    syncProtocolManualControls,
    currentProtocolConnectionDefaults,
    runProgressCountText,
    runProgressDetailText,
    operatorTargetRows,
    tensorShardRows,
    groupEffectiveMappingRows,
    filterEffectiveMappingGroups,
    effectiveMappingPage,
    effectiveMappingGroupMarkup,
    syncTraceSelect,
    traceVisualizationFromReport,
  };`, context);
  context.__v070.__context = context;
  return context.__v070;
}

class FakeElement {
  constructor({ selectors = {}, lists = [], disabled = false, hidden = false } = {}) {
    this.selectors = selectors;
    this.lists = lists;
    this.disabled = disabled;
    this.hidden = hidden;
    this.dataset = {};
    this.attributes = {};
    this.listeners = {};
    this.children = [];
    this.ownerDocument = null;
  }

  querySelector(selector) {
    return this.selectors[selector] || null;
  }

  querySelectorAll(selector) {
    return this.lists[selector] || [];
  }

  addEventListener(type, listener) {
    (this.listeners[type] ||= []).push(listener);
  }

  dispatch(type, event = {}) {
    for (const listener of this.listeners[type] || []) listener({ target: this, ...event });
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  getAttribute(name) {
    return this.attributes[name] ?? null;
  }

  contains(node) {
    return node === this || this.children.includes(node);
  }

  closest() {
    return this.closestResult || null;
  }

  focus() {
    this.ownerDocument.activeElement = this;
  }
}

test("bandwidth UI uses decimal byte-per-second units at the *_gbps boundary", () => {
  const ui = helpers();
  assert.equal(ui.formatBandwidthGbps(8), "1 GB/s");
  assert.equal(ui.formatBandwidthGbps(0.008), "1 MB/s");
  assert.equal(ui.formatBandwidthGbps(8000), "1 TB/s");
  assert.equal(ui.parseBandwidthToGbps("1 GB/s"), 8);
  assert.equal(ui.parseBandwidthToGbps("1 TB/s"), 8000);
  assert.equal(ui.parseBandwidthToGbps("125 MB/s"), 1);
  assert.equal(ui.parseBandwidthToGbps("2"), 16, "unitless inspector input defaults to GB/s");
  assert.equal(ui.parseBandwidthToGbps("2 Gb/s"), null, "bit-rate labels are not accepted by the UI");
  assert.doesNotMatch(app, /JSON 的 \*_gbps/, "help text must not expose internal bit-rate field names");
});

test("background serving progress explains primary batches and nested topology tasks", () => {
  const ui = helpers();
  const progress = {
    stage: "serving_cohorts",
    completed: 7,
    total: null,
    ratio: null,
    unit: "serving_batches",
    detail: {
      stage: "cohort_tasks",
      completed: 256,
      total: 640,
      ratio: 0.4,
      unit: "schedule_tasks",
      scope: "nested",
    },
  };
  assert.equal(ui.runProgressCountText(progress), "已执行 7 个在线批次（总数未知）");
  assert.equal(ui.runProgressDetailText(progress), "当前批次内 256/640 个拓扑任务");
  assert.equal(
    ui.runProgressCountText({ stage: "cohort_tasks", completed: 2, total: 5 }),
    "2 / 5",
    "V4 progress without an explicit unit uses generic work-unit rendering",
  );
});

test("component preset adapters accept arrays, items envelopes, and merged detail envelopes", () => {
  const ui = helpers();
  assert.equal(ui.componentPresetEnvelopeItems([{ id: "a" }]).length, 1);
  assert.equal(ui.componentPresetEnvelopeItems({ items: [{ id: "b" }], total: 1 }).length, 1);
  const detail = ui.componentPresetDetailItem({
    preset: { id: "gpu-x", name: "GPU X", evidence_level: "S2_VENDOR_DECLARED" },
    component: { component_id: "vendor-id", kind: "gpu", ports: [] },
  });
  assert.equal(detail.id, "gpu-x");
  assert.equal(detail.component.kind, "gpu");
  assert.equal(ui.componentPresetEvidenceLabel(detail.evidence_level), "S2 · 厂商声明");
});

test("vendor filtering derives values from source publishers instead of family facets", () => {
  const ui = helpers();
  const control = () => ({ value: "", options: [], replaceChildren(...options) { this.options = options; } });
  ui.dom.componentPresetKindFilter = control();
  ui.dom.componentPresetVendorFilter = control();
  ui.dom.componentPresetEvidenceFilter = control();
  const items = [
    { id: "a", family: "Hopper", component_kind: "gpu", evidence_level: "S2_VENDOR_DECLARED", sources: [{ publisher: "NVIDIA" }] },
    { id: "b", family: "PCIe", component_kind: "ssd", evidence_level: "S1_STANDARD", sources: [{ publisher: "Kioxia" }] },
  ];
  ui.syncComponentPresetFilters({ filters: { family: ["Hopper", "PCIe"], component_kind: ["gpu", "ssd"] } }, items);
  assert.equal(ui.componentPresetVendor(items[0]), "NVIDIA");
  assert.deepEqual(Array.from(ui.dom.componentPresetVendorFilter.options, (option) => option.value), ["", "Kioxia", "NVIDIA"]);
});

test("loading a component preset regenerates numeric component and port IDs", () => {
  const ui = helpers();
  ui.state.scenario = { hardware: { components: [{ component_id: "gpu0", kind: "gpu", ports: [] }], links: [] } };
  const source = {
    id: "vendor-gpu",
    component: {
      schema_version: "4.0.0",
      component_id: "vendor-component",
      kind: "gpu",
      die_id: "vendor-die",
      ports: [{ port_id: "fabric-a", protocol: "NVLink" }, { port_id: "fabric-a", protocol: "NVLink" }],
      metadata: {},
    },
  };
  const component = ui.materializeComponentPreset(source);
  assert.equal(component.component_id, "gpu1");
  assert.equal(component.die_id, "gpu1_die");
  assert.deepEqual(Array.from(component.ports, (port) => port.port_id), ["fabric-a", "port0"]);
  assert.equal(source.component.component_id, "vendor-component", "catalog payload remains immutable");
});

test("meaningful preset port IDs survive while only missing and duplicate IDs are regenerated", () => {
  const ui = helpers();
  ui.state.scenario = { hardware: { components: [], links: [] } };
  const component = ui.materializeComponentPreset({
    id: "ports",
    component: {
      schema_version: "4.0.0",
      kind: "gpu",
      ports: [
        { port_id: "hbm0" }, { port_id: "nvlink0" }, { port_id: "pcie0" }, { port_id: "ucie0" },
        { port_id: "nvlink0" }, { port_id: "" }, { port_id: "port0" },
      ],
    },
  });
  assert.deepEqual(Array.from(component.ports, (port) => port.port_id), ["hbm0", "nvlink0", "pcie0", "ucie0", "port1", "port2", "port0"]);
});

test("topology bundles remap components, links, package, and preset GPU-root collapse state atomically before append", () => {
  const ui = helpers();
  ui.state.scenario = { hardware: {
    components: [
      { component_id: "gpu0", kind: "gpu", package_id: "package0", ports: [] },
      { component_id: "hbm0", kind: "hbm", package_id: "package0", ports: [] },
      { component_id: "hbm2", kind: "hbm", package_id: "package0", ports: [] },
    ],
    links: [{ link_id: "hbm_link0" }],
  } };
  ui.state.topologyView = { groups: [{ group_id: "sxm_group0" }] };
  const source = {
    id: "bundle",
    preset_type: "topology_bundle",
    components: [
      { component_id: "gpu0", kind: "gpu", package_id: "package0", ports: [{ port_id: "hbm0" }, { port_id: "hbm1" }] },
      {
        component_id: "hbm0",
        kind: "hbm",
        package_id: "package0",
        ports: [{ port_id: "host" }],
        metadata: {
          physical_composition: {
            controller_component_id: "gpu0",
            memory_subsystem_id: "gpu0",
            source_basis: "gpu0",
          },
        },
      },
      {
        component_id: "hbm1",
        kind: "hbm",
        package_id: "package0",
        ports: [{ port_id: "host" }],
        metadata: {
          physical_composition: {
            controller_component_id: "gpu0",
            memory_subsystem_id: "gpu0-memory",
            source_basis: "gpu0",
          },
        },
      },
    ],
    links: [
      { link_id: "hbm_link0", source_component: "gpu0", source_port: "hbm0", target_component: "hbm0", target_port: "host" },
      { link_id: "hbm_link1", source_component: "gpu0", source_port: "hbm1", target_component: "hbm1", target_port: "host" },
    ],
    group: { group_id: "sxm_group0", label: "SXM", members: ["gpu0", "hbm0", "hbm1"], root: "gpu0", collapsed: true },
  };
  const bundle = ui.materializeTopologyBundle(source);
  assert.deepEqual(Array.from(bundle.components, (item) => item.component_id), ["gpu1", "hbm1", "hbm3"]);
  assert.ok(bundle.components.every((item) => item.package_id === "package1"));
  assert.deepEqual(Array.from(bundle.links, (item) => item.link_id), ["hbm_link1", "hbm_link2"]);
  assert.equal(bundle.links[0].source_component, "gpu1");
  assert.equal(bundle.links[0].target_component, "hbm1");
  assert.equal(bundle.group.group_id, "sxm_group1");
  assert.equal(bundle.group.root, "gpu1");
  assert.equal(bundle.group.collapsed, true);
  const hbm1 = bundle.components.find((item) => item.component_id === "hbm1");
  assert.equal(hbm1.metadata.physical_composition.controller_component_id, "gpu1");
  assert.equal(hbm1.metadata.physical_composition.memory_subsystem_id, "gpu1");
  assert.equal(hbm1.metadata.physical_composition.source_basis, "gpu0", "ordinary physical metadata text is not rewritten");
  const hbm3 = bundle.components.find((item) => item.component_id === "hbm3");
  assert.equal(hbm3.metadata.physical_composition.controller_component_id, "gpu1");
  assert.equal(hbm3.metadata.physical_composition.memory_subsystem_id, "gpu0-memory", "derived namespaces are not rewritten as text");
  assert.equal(source.group.collapsed, true, "catalog payload remains immutable");
  assert.equal(source.components[1].metadata.physical_composition.controller_component_id, "gpu0", "catalog payload metadata remains immutable");

  ui.state.scenario.hardware.components = [...ui.state.scenario.hardware.components, ...bundle.components];
  ui.state.scenario.hardware.links = [...ui.state.scenario.hardware.links, ...bundle.links];
  ui.state.topologyView.groups = [...ui.state.topologyView.groups, bundle.group];
  const secondBundle = ui.materializeTopologyBundle(source);
  assert.deepEqual(Array.from(secondBundle.components, (item) => item.component_id), ["gpu2", "hbm4", "hbm5"]);
  assert.ok(secondBundle.components.every((item) => item.package_id === "package2"));
  assert.deepEqual(Array.from(secondBundle.links, (item) => item.link_id), ["hbm_link3", "hbm_link4"]);
  assert.equal(secondBundle.group.group_id, "sxm_group2");
  assert.equal(secondBundle.group.root, "gpu2");
  const hbm4 = secondBundle.components.find((item) => item.component_id === "hbm4");
  assert.equal(hbm4.metadata.physical_composition.controller_component_id, "gpu2");
  assert.equal(hbm4.metadata.physical_composition.memory_subsystem_id, "gpu2");
});

test("component details escape provenance text and disclose all evidence categories", () => {
  const ui = helpers();
  const markup = ui.componentPresetDetailsMarkup({
    id: "safe",
    family: "Vendor",
    evidence_level: "A_ANALYTICAL",
    sources: [{ publisher: "Lab", title: "<img src=x onerror=alert(1)>", url: "https://example.invalid" }],
    limitations: ["仅限分析"],
    notes: "假设冷启动",
    component: { kind: "hbm", read_bandwidth_gbps: 8000, metadata: { measurement_basis: "推导" } },
  });
  assert.match(markup, /事实（Facts）/);
  assert.match(markup, /厂商（Vendor）/);
  assert.match(markup, /推导（Derived）/);
  assert.match(markup, /假设（Assumptions）/);
  assert.match(markup, /限制（Limitations）/);
  assert.match(markup, /1 TB\/s/);
  assert.doesNotMatch(markup, /<img/);
  assert.match(markup, /&lt;img/);
  assert.match(markup, /href="https:\/\/example\.invalid\/"/);
  assert.match(markup, /target="_blank" rel="noopener noreferrer"/);
  assert.doesNotMatch(ui.componentPresetSourceMarkup({ sources: [{ title: "bad", url: "javascript:alert(1)" }] }), /href=/);
});

test("provenance labels are bilingual and bandwidth normalization handles case safely", () => {
  const ui = helpers();
  assert.equal(ui.componentPresetFactLabel("measurement_basis"), "测量依据（Measurement Basis）");
  assert.equal(ui.componentPresetFactLabel("derived_formula"), "推导公式（Derived Formula）");
  assert.equal(ui.componentPresetFactLabel("evidence_level"), "证据等级（Evidence Level）");
  assert.equal(ui.normalizePresetBandwidthText("峰值 8000 GBPS"), "峰值 1 TB/s");
  assert.equal(ui.normalizePresetBandwidthText("峰值 8000 gbps"), "峰值 1 TB/s");
  assert.equal(ui.normalizePresetBandwidthText("现有 1 GB/s"), "现有 1 GB/s", "byte-rate labels must not be divided again");
});

test("utilization bars clamp non-finite values and expose accessible progress state", () => {
  const ui = helpers();
  ui.dom.bottleneckMeta = { innerHTML: "" };
  ui.dom.utilizationList = { innerHTML: "" };
  ui.renderUtilization({
    resource_utilization: { tiny: 0.000001, overflow: 2, negative: -1, invalid: "not-a-number" },
    summary: { bottleneck_resource: { resource_id: "overflow", utilization: 2 } },
  });
  assert.match(ui.dom.utilizationList.innerHTML, /role="progressbar"/);
  assert.match(ui.dom.utilizationList.innerHTML, /aria-valuenow="100"/);
  assert.match(ui.dom.utilizationList.innerHTML, /tiny[\s\S]*width:0\.5%/);
  assert.doesNotMatch(ui.dom.utilizationList.innerHTML, /NaN|Infinity/);
  assert.match(ui.dom.utilizationList.innerHTML, /is-critical is-bottleneck/);
});

test("field-help click and Escape close while focused, then blur restores focus behavior", () => {
  const ui = helpers();
  const trigger = new FakeElement();
  const help = new FakeElement({ selectors: { ".field-help-trigger": trigger } });
  const document = {
    activeElement: null,
    querySelectorAll(selector) { return selector === "[data-field-help]" ? [help] : []; },
    addEventListener() {},
  };
  trigger.ownerDocument = document;
  help.ownerDocument = document;
  ui.__context.document = document;
  ui.bindFieldHelp();

  trigger.focus();
  trigger.dispatch("focus");
  assert.equal(trigger.getAttribute("aria-expanded"), "true", "focus-visible help must expose its expanded state");
  trigger.dispatch("blur");
  assert.equal(trigger.getAttribute("aria-expanded"), "false", "un-pinned help must collapse on blur");

  trigger.dispatch("click", { preventDefault() {}, stopPropagation() {} });
  assert.equal(help.dataset.open, "true");
  assert.equal(trigger.getAttribute("aria-expanded"), "true");

  trigger.dispatch("click", { preventDefault() {}, stopPropagation() {} });
  assert.equal(help.dataset.open, undefined);
  assert.equal(help.dataset.focusClosed, "true");
  assert.equal(trigger.getAttribute("aria-expanded"), "false");

  trigger.dispatch("blur");
  assert.equal(help.dataset.focusClosed, undefined);

  trigger.focus();
  trigger.dispatch("click", { preventDefault() {}, stopPropagation() {} });
  let prevented = false;
  trigger.dispatch("keydown", {
    key: "Escape",
    preventDefault() { prevented = true; },
    stopPropagation() {},
  });
  assert.equal(prevented, true);
  assert.equal(help.dataset.open, undefined);
  assert.equal(help.dataset.focusClosed, "true");
  assert.equal(trigger.getAttribute("aria-expanded"), "false");
  trigger.dispatch("blur");
  assert.equal(help.dataset.focusClosed, undefined);

  trigger.focus();
  trigger.dispatch("focus");
  let enterPrevented = false;
  trigger.dispatch("keydown", {
    key: "Enter",
    preventDefault() { enterPrevented = true; },
    stopPropagation() {},
  });
  assert.equal(enterPrevented, true);
  assert.equal(help.dataset.open, "true");
  assert.equal(trigger.getAttribute("aria-expanded"), "true");
  let spacePrevented = false;
  trigger.dispatch("keydown", {
    key: " ",
    preventDefault() { spacePrevented = true; },
    stopPropagation() {},
  });
  assert.equal(spacePrevented, true);
  assert.equal(help.dataset.open, undefined);
  assert.equal(trigger.getAttribute("aria-expanded"), "false");
});

test("concept help uses one dictionary and viewport placement avoids right and bottom clipping", () => {
  const ui = helpers();
  ui.__context.innerWidth = 320;
  ui.__context.innerHeight = 240;
  ui.__context.document.documentElement = { clientWidth: 320, clientHeight: 240 };
  const trigger = { getBoundingClientRect: () => ({ left: 296, right: 312, top: 210, bottom: 226, width: 16, height: 16 }) };
  const portal = { style: {}, getBoundingClientRect: () => ({ width: 180, height: 90 }) };
  ui.positionFieldHelp(trigger, portal);
  assert.equal(portal.style.left, "130px");
  assert.equal(portal.style.top, "113px", "tooltip flips above when the viewport has no room below");
  assert.match(ui.conceptHelpText("peak_ops"), /cost_profile_id 绑定的 profiles\.components\.cpu[\s\S]*fabric 与 I\/O die 不执行算子/);
  assert.equal(ui.conceptHelpText("unknown fallback"), "unknown fallback");
});

test("runtime shared tracks use measured content maxima for each visual stat row", () => {
  const ui = helpers();
  const measured = (height) => ({
    scrollHeight: height,
    getBoundingClientRect() { return { height }; },
  });
  const style = () => ({
    values: {},
    setProperty(name, value) { this.values[name] = value; },
    removeProperty(name) { delete this.values[name]; },
  });
  const stat = (row, labelHeight, valueHeight) => {
    const label = measured(labelHeight);
    const value = measured(valueHeight);
    return {
      dataset: { runtimeStatRow: String(row) },
      style: style(),
      querySelector(selector) {
        if (selector === ".runtime-stat-label-copy") return label;
        if (selector === ".runtime-stat-value-copy") return value;
        return null;
      },
      querySelectorAll() { return []; },
    };
  };
  const group = (titleHeight, stats) => ({
    querySelector(selector) { return selector === ".runtime-group-title-copy" ? measured(titleHeight) : null; },
    querySelectorAll(selector) { return selector === "[data-runtime-stat-row]" ? stats : []; },
  });
  const stats = [stat(0, 10, 30), stat(1, 15, 20), stat(0, 18, 24), stat(1, 11, 40)];
  const groups = [group(20, stats.slice(0, 2)), group(32, stats.slice(2))];
  const rootStyle = style();
  const root = {
    style: rootStyle,
    querySelectorAll(selector) {
      if (selector === ".runtime-group") return groups;
      if (selector === ".runtime-stat[data-runtime-stat-row]") return stats;
      return [];
    },
  };

  const measuredTracks = ui.applyRuntimeSharedTracks(root);
  assert.equal(measuredTracks.titleHeight, 32);
  assert.deepEqual(Array.from(measuredTracks.labelHeights), [18, 15]);
  assert.deepEqual(Array.from(measuredTracks.valueHeights), [30, 40]);
  assert.equal(rootStyle.values["--runtime-title-track"], "32px");
  assert.equal(stats[0].style.values["--runtime-label-track"], "18px");
  assert.equal(stats[1].style.values["--runtime-value-track"], "40px");
});

test("toolbar roving keys leave protocol selects native and move only menuitems", () => {
  const ui = helpers();
  const trigger = new FakeElement();
  const first = new FakeElement();
  const second = new FakeElement();
  first.closestResult = first;
  second.closestResult = second;
  const panel = new FakeElement({ lists: {
    'button[role="menuitem"]:not([disabled])': [first, second],
    "button:not(.toolbar-menu-trigger)": [first, second],
  } });
  panel.children = [first, second];
  const section = new FakeElement({ selectors: {
    ".toolbar-menu-trigger": trigger,
    ".toolbar-menu-panel": panel,
  } });
  const select = new FakeElement();
  const document = {
    activeElement: null,
    querySelectorAll(selector) {
      if (selector === "[data-toolbar-menu]") return [section];
      return [];
    },
    addEventListener() {},
  };
  for (const element of [trigger, first, second, panel, section, select]) element.ownerDocument = document;
  ui.__context.document = document;
  ui.__context.window = { innerHeight: 800, addEventListener() {} };
  ui.bindToolbarMenus();

  let selectPrevented = false;
  panel.dispatch("keydown", {
    target: select,
    key: "ArrowDown",
    preventDefault() { selectPrevented = true; },
  });
  assert.equal(selectPrevented, false);
  const items = ui.toolbarMenuItems(panel);
  assert.equal(items.length, 2);
  assert.equal(items[0], first);
  assert.equal(items[1], second);

  let itemPrevented = false;
  panel.dispatch("keydown", {
    target: first,
    key: "ArrowDown",
    preventDefault() { itemPrevented = true; },
  });
  assert.equal(itemPrevented, true);
  assert.equal(document.activeElement, second);
});

test("component inspector accepts evidence_level and source arrays without mutating metadata", () => {
  const ui = helpers();
  const metadata = {
    evidence_level: "S2_VENDOR_DECLARED",
    sources: [{ publisher: "NVIDIA", title: "H200 datasheet" }],
  };
  assert.equal(ui.componentInspectorEvidence(metadata), "S2_VENDOR_DECLARED");
  assert.equal(ui.componentInspectorSource(metadata), "NVIDIA · H200 datasheet");
  assert.deepEqual(metadata, {
    evidence_level: "S2_VENDOR_DECLARED",
    sources: [{ publisher: "NVIDIA", title: "H200 datasheet" }],
  });
  assert.equal(ui.componentInspectorEvidence({ evidence_status: "S1_STANDARD", evidence_level: "S2_VENDOR_DECLARED" }), "S1_STANDARD");
  assert.equal(ui.componentInspectorSource({ source: "catalog", sources: [{ publisher: "ignored" }] }), "catalog");
});

test("component inspector gives CPU profile-backed compute while fabric remains transport-only", () => {
  const ui = helpers();
  const gpu = ui.componentInspectorProfile("gpu", { peak_ops_per_s: 1 });
  const hbm = ui.componentInspectorProfile("hbm", { capacity_bytes: 1 });
  const cpu = ui.componentInspectorProfile("cpu", { capacity_bytes: 9, peak_ops_per_s: 8, read_bandwidth_gbps: 7 });
  assert.equal(cpu.transportOnly, false);
  assert.equal(cpu.capacity, false);
  assert.equal(cpu.peakOps, true);
  assert.equal(cpu.componentBandwidth, false);
  assert.equal(cpu.latencyDma, false);
  for (const kind of ["fabric_switch", "io_die"]) {
    const profile = ui.componentInspectorProfile(kind, { capacity_bytes: 9, peak_ops_per_s: 8, read_bandwidth_gbps: 7 });
    assert.equal(profile.transportOnly, true);
    assert.equal(profile.capacity, false);
    assert.equal(profile.peakOps, false);
    assert.equal(profile.componentBandwidth, false);
    assert.equal(profile.latencyDma, false);
  }
  assert.equal(gpu.peakOps, true);
  assert.equal(hbm.capacity, true);
  assert.equal(hbm.componentBandwidth, true);
  assert.doesNotMatch(app, /当前 Kind 的隐藏字段/);
  assert.match(app, /物理容量与传输能力/);
  assert.match(app, /Profile 是执行\/内存成本权威/);
  assert.match(app, /运行时 metadata（延迟、DMA、传输粒度与并发上限）会保留/);
  assert.match(app, /data-inspector-port-field="bandwidth_gbps"/);
});

test("component preset summaries omit zero capabilities and retain positive OPS", () => {
  const ui = helpers();
  const gpuCard = ui.componentPresetCardMarkup({
    id: "gpu-h200",
    name: "H200",
    family: "Hopper",
    component: {
      kind: "gpu",
      capacity_bytes: 50 * 1024 ** 2,
      read_bandwidth_gbps: 0,
      peak_ops_per_s: 989.5e12,
    },
  });
  assert.match(gpuCard, /50 MiB/);
  assert.match(gpuCard, /989\.5 TOPS/);
  assert.doesNotMatch(gpuCard, /0 MB\/s/);

  const memoryCard = ui.componentPresetCardMarkup({
    id: "hbm3e",
    name: "HBM3E",
    component: {
      kind: "hbm",
      capacity_bytes: 16 * 1024 ** 3,
      read_bandwidth_gbps: 4096,
      peak_ops_per_s: 0,
    },
  });
  assert.match(memoryCard, /512 GB\/s/);
  assert.doesNotMatch(memoryCard, /0 OPS/);
});

test("protocol catalog keeps direction scopes explicit and applies one-way manual defaults", () => {
  const ui = helpers();
  assert.equal(ui.protocolPresetEnvelopeItems({ items: [{ id: "p" }] }).length, 1);
  const detail = ui.protocolPresetDetailItem({
    preset: { id: "pcie", protocol: "PCIe", version: "5.0" },
    simulation_defaults: { link: { bandwidth_gbps: 504.123 } },
  });
  assert.equal(detail.simulation_defaults.link.bandwidth_gbps, 504.123);
  const markup = ui.protocolPresetBandwidthMarkup({ bandwidth: {
    raw_gbps: 512,
    effective_one_way_gbps: 504.123,
    aggregate_bidirectional_gbps: 1008.246,
  } });
  assert.match(markup, /原始线路/);
  assert.match(markup, /单向有效/);
  assert.match(markup, /双向聚合/);
  assert.match(markup, /64 GB\/s/);

  ui.dom.protocolVersionInput = { value: "" };
  ui.dom.protocolUnitsInput = { value: "" };
  ui.dom.protocolBandwidthInput = { value: "" };
  ui.dom.protocolLatencyInput = { value: "" };
  ui.dom.protocolPayloadInput = { value: "" };
  ui.dom.protocolPresetSelection = { textContent: "" };
  ui.dom.protocolSelect = { value: "PCIe" };
  ui.syncProtocolManualControls({ version: "5.0", lanes: 16, bandwidth_gbps: 504.123, latency_ns: 150 }, "pcie-5_0-x16");
  assert.equal(ui.dom.protocolBandwidthInput.value, "63.02 GB/s");
  assert.equal(ui.state.selectedProtocolPresetId, "pcie-5_0-x16");
  ui.dom.protocolBandwidthInput.value = "50 GB/s";
  assert.equal(ui.currentProtocolConnectionDefaults().bandwidth_gbps, 400);
});

test("all catalog protocol families are selectable and visible copy stays byte-based", () => {
  for (const protocol of ["NVLink-C2C", "InfinityFabric", "RoCE", "LPDDR5X"]) {
    assert.match(html, new RegExp(`<option value="${protocol}">`));
    assert.match(app, new RegExp(protocol.replace("-", "\\-")));
  }
  assert.match(app, /带宽统一按单向 MB\/s、GB\/s 或 TB\/s 展示/);
  assert.doesNotMatch(app, /仿真字段使用单向 bit\/s 边界/);
});

test("1.0.0 UI contracts wire the fullscreen catalog, dropdown toolbar, help, and progressbars", () => {
  for (const id of [
    "hardwarePresetsButton", "hardwarePresetsDialog", "hardwarePresetComponentTab", "hardwarePresetArchitectureTab", "componentPresetsDialog", "componentPresetSearchInput",
    "componentPresetKindFilter", "componentPresetVendorFilter", "componentPresetEvidenceFilter",
    "topologyEditMenuButton", "topologyGroupMenuButton", "topologyConnectMenuButton",
    "protocolCatalogButton", "protocolPresetsDialog", "protocolVersionInput", "protocolUnitsInput", "protocolBandwidthInput", "protocolLatencyInput", "protocolPayloadInput",
  ]) assert.match(html, new RegExp(`id="${id}"`));
  assert.match(app, /apiRequest\("\/component-presets"/);
  assert.match(app, /`\/component-presets\/\$\{encodeURIComponent\(id\)\}`/);
  assert.match(app, /apiRequest\("\/protocol-presets"/);
  assert.match(app, /`\/protocol-presets\/\$\{encodeURIComponent\(id\)\}`/);
  assert.match(app, /role="progressbar"/);
  assert.match(app, /aria-valuemin="0"/);
  assert.match(app, /data-field-help/);
  assert.match(css, /\.hardware-presets-dialog,[\s\S]*?100dvw[\s\S]*?100dvh/s);
  assert.match(css, /\.canvas-toolbar\s*\{[^}]*flex-wrap:\s*nowrap[^}]*overflow-x:\s*auto/s);
  assert.match(css, /\.bar-fill\s*\{[^}]*display:\s*block/s);
  assert.match(css, /\.bar-track,[\s\S]*height:\s*clamp\(7px,\s*calc\(5px \* var\(--layout-scale\)\),\s*18px\)/);
  assert.match(css, /data-font-band="extreme"\] \.util-row\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)/s);
  assert.match(css, /\.field-help-viewport-popover\s*\{[^}]*position:\s*fixed[^}]*z-index:\s*2000/s);
  assert.match(css, /\.field-help-trigger\s*\{[^}]*cursor:\s*help/s);
  assert.match(css, /\.field-help-trigger\s*\{[^}]*text-decoration:\s*none/s);
  assert.doesNotMatch(css, /text-decoration-style:\s*dotted/);
  assert.match(app, /const CONCEPT_HELP = Object\.freeze/);
  assert.match(app, /positionFieldHelp\(trigger, portal\)/);
  assert.match(app, /event\.key === "Escape" && !panel\.hidden/);
  assert.match(html, /id="protocolSelect"/);
  assert.match(html, /role="group" aria-label="连接协议/);
  for (const id of [
    "selectModeButton", "undoTopologyButton", "redoTopologyButton", "copyTopologyButton", "pasteTopologyButton",
    "createGroupButton", "setGroupRootButton", "toggleGroupButton", "releaseGroupButton", "connectModeButton",
    "fitCanvasButton",
  ]) assert.match(html, new RegExp(`id="${id}"`));
  assert.doesNotMatch(html, /id="topologyViewMenuButton"/);
  assert.doesNotMatch(html, /id="panModeButton"/);
  assert.doesNotMatch(html, /id="zoomOutButton"|id="zoomInButton"/);
  assert.match(html, /id="topologyEditMenu"[\s\S]*id="fitCanvasButton"/);
  assert.match(app, /modelGraphCanvas\.addEventListener\("wheel", \(event\) => \{\s*if \(!\(event\.ctrlKey \|\| event\.metaKey\)\) return;\s*event\.preventDefault\(\)/);
  assert.match(app, /event\.code === "Space"/);
  assert.match(app, /event\.button === 1/);
});

test("rank-aware mapping tables preserve TP targets and rank-local weight shards", () => {
  const ui = helpers();
  const ranks = [
    { rank: 0, tp_rank: 0, pp_rank: 0, ep_rank: 0, component_id: "gpu0" },
    { rank: 1, tp_rank: 1, pp_rank: 0, ep_rank: 0, component_id: "gpu1" },
  ];
  const metadata = {
    operator_execution_targets: {
      "layer-000.mlp": [
        { rank_id: 0, component_id: "gpu0", partition: "tp_shard" },
        { rank_id: 1, component_id: "gpu1", partition: "tp_shard" },
      ],
    },
    weight_tensor_details: {
      "layer-000.mlp_weights": {
        shard_policy: "tp_shard",
      },
    },
    rank_weight_shards: {
      "layer-000.mlp_weights": [
        { rank_id: 0, compute_component_id: "gpu0", component_id: "hbm0", storage_component_id: "hbm0", shard_index: 0, shard_count: 2, residency: "resident", logical_bytes: 1024, physical_bytes: 1152, shard_kind: "tp_shard" },
        { rank_id: 1, compute_component_id: "gpu1", component_id: "hbm1", storage_component_id: "hbm1", shard_index: 1, shard_count: 2, residency: "cold_stream", logical_bytes: 1024, physical_bytes: 1152, shard_kind: "tp_shard" },
      ],
    },
  };
  const operators = ui.operatorTargetRows(metadata, ranks);
  const shards = ui.tensorShardRows(metadata, ranks);
  assert.deepEqual(Array.from(operators, (item) => item.rank.rank), [0, 1]);
  assert.deepEqual(Array.from(operators, (item) => item.componentId), ["gpu0", "gpu1"]);
  assert.deepEqual(Array.from(shards, (item) => item.storage), ["hbm0", "hbm1"]);
  assert.deepEqual(Array.from(shards, (item) => item.compute), ["gpu0", "gpu1"]);
  assert.deepEqual(Array.from(shards, (item) => item.physicalBytes), [1152, 1152]);
  assert.deepEqual(Array.from(shards, (item) => item.shardIndex), [0, 1]);
  assert.deepEqual(Array.from(shards, (item) => item.shardCount), [2, 2]);
  assert.deepEqual(Array.from(shards, (item) => item.residency), ["resident", "cold_stream"]);
  const operatorGroups = ui.groupEffectiveMappingRows(operators, "operator");
  const tensorGroups = ui.groupEffectiveMappingRows(shards, "tensorId");
  assert.equal(operatorGroups.length, 1);
  assert.equal(operatorGroups[0].rows.length, 2, "grouping keeps every per-rank operator target");
  assert.equal(tensorGroups[0].rows.length, 2, "grouping keeps every per-rank tensor shard");
  assert.equal(ui.tensorShardRows({ weight_tensor_details: {
    legacy: { rank_shards: [{ rank_id: 0, storage_component_id: "wrong" }] },
  } }, ranks).length, 0, "removed V4 nested rank_shards must not shadow the canonical field");
  const operatorTable = ui.effectiveMappingGroupMarkup(operatorGroups[0], "operator");
  const tensorTable = ui.effectiveMappingGroupMarkup(tensorGroups[0], "tensor");
  assert.match(operatorTable, /rank-mapping-table-operator[\s\S]*?<colgroup>[\s\S]*?coordinate-col/);
  assert.match(tensorTable, /rank-mapping-table-tensor[\s\S]*?<colgroup>[\s\S]*?compute-component-col[\s\S]*?storage-component-col[\s\S]*?physical-bytes-col[\s\S]*?shard-index-col[\s\S]*?residency-col/);
  assert.match(tensorTable, /gpu0[\s\S]*?hbm0[\s\S]*?0 \/ 2[\s\S]*?resident/);
  assert.match(tensorTable, /gpu1[\s\S]*?hbm1[\s\S]*?1 \/ 2[\s\S]*?cold_stream/);
  assert.match(css, /\.rank-mapping-table\s*\{[^}]*table-layout:\s*fixed/);
  assert.match(css, /\.rank-mapping-table-tensor\s*\{[^}]*min-width:\s*calc\(1040px \* var\(--layout-scale\)\)/);
  assert.match(css, /\.rank-mapping-table th,[\s\S]*?\.rank-mapping-table td\s*\{[^}]*white-space:\s*normal;[^}]*overflow-wrap:\s*anywhere/);
  assert.equal(ui.filterEffectiveMappingGroups(operatorGroups, "operator", { query: "", rank: "1", component: "" })[0].rows[0].rank.rank, 1);
  assert.equal(ui.filterEffectiveMappingGroups(tensorGroups, "tensor", { query: "cold_stream", rank: "", component: "gpu1" })[0].rows[0].storage, "hbm1");
  const elevenGroups = Array.from({ length: 11 }, (_, index) => ({ label: `op-${index}`, rows: [{ rank: { rank: index } }] }));
  const secondPage = ui.effectiveMappingPage(elevenGroups, 1, 6);
  assert.equal(secondPage.pageCount, 2);
  assert.equal(secondPage.items.length, 5);
  assert.doesNotMatch(html, /manualMappingDisclosure|manual-mapping-disclosure|addOpMappingButton|addTensorMappingButton/);
});

test("trace normalization keeps logical memory offsets, protocol hops, and raw rank filter values", () => {
  const ui = helpers();
  ui.state.scenario = {
    hardware: { components: [{ component_id: "gpu0", kind: "gpu" }] },
    placement: { parallel: { tp_degree: 1, pp_degree: 1, ep_degree: 1, rank_mapping: [
      { rank: 0, tp_rank: 0, pp_rank: 0, ep_rank: 0, component_id: "gpu0" },
    ] } },
  };
  const trace = ui.traceVisualizationFromReport({
    retention_policy: "exact",
    visualization: {
      schema_version: "1.0",
      fidelity: "exact",
      time_range_ns: [10, 30],
      memory_layout: { components: { hbm0: [{ tensor_id: "w", offset_bytes: 4096, length_bytes: 2048 }] } },
      events: [{
        event_id: "e0",
        start_ns: 10,
        end_ns: 30,
        operator_id: "layer-000.mlp",
        rank: { rank_id: 0, tp_rank: 0, pp_rank: 0, ep_rank: 0, component_id: "gpu0" },
        tensor: { logical_id: "w", physical_id: "w#rank-0", component_id: "hbm0", offset_bytes: 4096, length_bytes: 2048 },
        transfer: { source_component: "hbm0", target_component: "gpu0", bytes: 2048, hops: [
          { link_id: "hbm-link0", protocol: "HBM3E", source_component: "hbm0", target_component: "gpu0" },
        ] },
      }],
    },
  });
  assert.equal(trace.fidelity, "exact");
  assert.equal(trace.events[0].tensor.offset_bytes, 4096);
  assert.equal(trace.events[0].transfer.hops[0].protocol, "HBM3E");
  assert.equal(trace.events[0].rank.rank, 0);

  const select = { innerHTML: "" };
  ui.syncTraceSelect(select, [{ value: "0", label: "Rank 0" }], "全部 Rank", "0");
  assert.match(select.innerHTML, /value="0" selected>Rank 0/);
  assert.doesNotMatch(select.innerHTML, /value="Rank 0"/);
});

test("playback UI binds wall-clock transport, local event filters, and topology-only memory summaries", () => {
  for (const id of [
    "playbackStatus", "playbackCount", "playbackFidelity", "traceEmpty", "traceRunButton", "traceContent",
    "traceResetButton", "tracePreviousButton", "tracePlayButton", "traceNextButton", "traceTimeline", "playbackTime",
    "traceRequestFilter", "traceBatchFilter", "traceRankFilter", "traceEventMeta", "traceArrangeFitButton", "traceTopologyCanvas",
    "traceLinkLayer", "traceNodeLayer", "traceEventDetails", "traceEventBody", "traceLimitations",
    "traceEventKeywordFilter", "traceEventCategoryFilter", "traceEventPhaseFilter", "traceEventTemporalFilter",
  ]) {
    assert.match(html, new RegExp(`id="${id}"`));
    assert.match(app, new RegExp(`"${id}"`));
  }
  assert.match(app, /traceRunButton\.addEventListener\("click", runScenario\)/);
  assert.match(app, /tracePlayButton\.addEventListener\("click", toggleTracePlayback\)/);
  assert.match(app, /traceTimeline\.addEventListener\("input"/);
  assert.doesNotMatch(html, /id="traceSpeed"|id="traceMemoryLayout"/);
  assert.match(app, /globalThis\.setInterval\(traceAnimationStep, TRACE_PLAYBACK_STEP_MS\)/);
  assert.match(app, /traceNodeMemorySummary\(componentId\)/);
});
