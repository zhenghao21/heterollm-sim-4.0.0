"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const ModelGraphCore = require("../src/heterollm_sim/webui/model-graph-core.js");
const TopologyCore = require("../src/heterollm_sim/webui/topology-core.js");
const TraceViewCore = require("../src/heterollm_sim/webui/trace-view-core.js");
const UiI18n = require("../src/heterollm_sim/webui/ui-i18n.js");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const app = fs.readFileSync(path.join(webui, "app.js"), "utf8");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");
const css = fs.readFileSync(path.join(webui, "styles.css"), "utf8");

function helpers(language = "zh-CN") {
  UiI18n.setLanguage(language, null);
  const context = vm.createContext({
    AbortController,
    CSS: { escape: String },
    Intl,
    Map,
    ModelGraphCore,
    Option: class Option { constructor(text, value) { this.text = text; this.value = value; } },
    Promise,
    Set,
    TopologyCore,
    TraceViewCore,
    UiI18n,
    URL,
    URLSearchParams,
    clearTimeout,
    console,
    document: { addEventListener() {} },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    setTimeout,
  });
  vm.runInContext(`${app}\n;globalThis.__trace = {
    state,
    dom,
    finiteTraceNumber,
    normalizeTraceEvent,
    traceVisualizationFromReport,
    tracePageEndpoint,
    tracePageData,
    renderTracePageState,
    tracePagePagination,
    tracePageCacheKey,
    tracePageCacheGet,
    tracePageCacheSet,
    tracePagePendingGetOrCreate,
    traceEventRequestIds,
    setTraceSelectedIndex,
    setTraceTime,
    selectedTraceEvent,
    activeTraceEvents,
    traceActiveSignature,
    traceIntervalActive,
    activeTraceNodeRoles,
    activeTraceLinkIds,
    traceTransferKind,
    traceLinkFlowDirections,
    traceLinkIsActive,
    traceParticleRouteForDirection,
    renderTraceParticles,
    renderTraceProtocolLegend,
    traceNodeMemorySummary,
    showLinkTooltip,
    topologyLayoutMetrics,
    buildTraceTopologyLayout,
    measureTraceTopologyNodes,
    traceNodeMemoryCapacityLabel,
    traceNodeSizes,
    traceTopologyContentBounds,
    traceFitMetrics,
    traceRevealTargetRect,
    traceRevealScrollTarget,
    traceRevealScrollBehavior,
    tracePrefersReducedMotion,
    tracePlaybackControlTarget,
    revealTracePlaybackControls,
    applyAggregateTraceData,
    traceStepSummary,
    traceFriendlyTerm,
    traceEventSubject,
    traceEventSemantics,
  };`, context);
  return context.__trace;
}

function component(component_id, kind) {
  return { component_id, kind, ports: [] };
}

function link(link_id, source_component, target_component, protocol = "NVLink") {
  return { link_id, source_component, target_component, protocol, bidirectional: true };
}

function sourceBetween(startName, endName) {
  const start = app.indexOf(`function ${startName}`);
  const end = app.indexOf(`function ${endName}`, start + 1);
  assert.ok(start >= 0, `${startName} must exist`);
  assert.ok(end > start, `${endName} must follow ${startName}`);
  return app.slice(start, end);
}

function expectMatch(source, pattern, message) {
  assert.equal(pattern.test(source), true, message || `expected ${pattern}`);
}

function expectNoMatch(source, pattern, message) {
  assert.equal(pattern.test(source), false, message || `did not expect ${pattern}`);
}

function displayLinkId(item) {
  return String(item.display_id || item.link_id);
}

function routeFor(layout, linkValue) {
  return layout.routes.get(displayLinkId(linkValue)) || layout.routes.get(String(linkValue.link_id));
}

function assertLayoutRoutesAreOrthogonalAndClear(layout, label) {
  const rectsById = new Map(layout.rects.map((rect) => [String(rect.id), rect]));
  for (const linkValue of layout.links) {
    const route = routeFor(layout, linkValue);
    assert.ok(route?.points.length >= 2, `${label}: ${displayLinkId(linkValue)} has a route`);
    route.points.forEach((point, index) => {
      if (!index) return;
      const previous = route.points[index - 1];
      assert.ok(point.x === previous.x || point.y === previous.y, `${label}: ${displayLinkId(linkValue)} segment ${index} is orthogonal`);
      for (const obstacle of layout.rects) {
        if ([linkValue.source_component, linkValue.target_component].map(String).includes(String(obstacle.id))) continue;
        assert.equal(
          TopologyCore.segmentHitsRect(previous, point, obstacle),
          false,
          `${label}: ${displayLinkId(linkValue)} crosses ${obstacle.id}`,
        );
      }
    });
    const source = rectsById.get(String(linkValue.source_component));
    const target = rectsById.get(String(linkValue.target_component));
    assert.ok(source, `${label}: ${displayLinkId(linkValue)} source is visible`);
    assert.ok(target, `${label}: ${displayLinkId(linkValue)} target is visible`);
  }
}

test("trace numbers reject nullish, empty, and boolean values instead of inventing rank zero", () => {
  const ui = helpers();
  assert.equal(ui.finiteTraceNumber(null, undefined, "", "   ", false, true), null);
  assert.equal(ui.finiteTraceNumber(null, "0"), 0);
  ui.state.scenario = { hardware: { components: [] }, placement: { parallel: {} } };
  const event = ui.normalizeTraceEvent({
    start_ns: 10,
    end_ns: 20,
    rank: { rank: null, tp_rank: null, pp_rank: null, ep_rank: null },
  }, 0);
  assert.equal(event.rank, null);
});

test("V4 playback ignores reports without the canonical visualization object", () => {
  const ui = helpers();
  ui.state.scenario = { hardware: { components: [] }, placement: { parallel: {} } };
  const trace = ui.traceVisualizationFromReport({
    execution_backend: "scalable_serving",
    batch_history: [{ cohort_id: "batch-0", kind: "decode", start_ns: 100, end_ns: 150 }],
  });
  assert.equal(trace.fidelity, "unavailable");
  assert.deepEqual(Array.from(trace.events), []);
  assert.deepEqual(Array.from(trace.batch_trace_index), []);
});

test("trace normalization preserves server semantics and unknown extensions without inventing aggregate hop timing", () => {
  const ui = helpers();
  ui.state.scenario = { hardware: { components: [] }, placement: { parallel: {} } };
  const event = ui.normalizeTraceEvent({
    event_id: "aggregate:0",
    start_ns: 100,
    end_ns: 200,
    marker: "token_emit",
    token_index: 7,
    request_ids: ["r0", "r1"],
    request_ids_truncated: true,
    request_id_limit: 2,
    representative_items: [{ request_id: "r0", custom_item_field: "kept" }],
    representative_items_truncated: true,
    detail_semantics: "aggregate_batch_with_selected_items",
    future_event_extension: { version: 2 },
    rank: { rank: 3, component_id: "gpu3", future_rank_extension: "rank-kept" },
    tensor: { logical_id: "kv", physical_id: "kv:3", future_tensor_extension: "tensor-kept" },
    transfer: {
      source_component: "hbm3",
      target_component: "gpu3",
      future_transfer_extension: "transfer-kept",
      hops: [{
        link_id: "h3",
        start_ns: null,
        end_ns: null,
        interval_semantics: "aggregate_route_without_hop_timing",
        future_hop_extension: "hop-kept",
      }],
    },
    resources: [{ resource_id: "gpu3.compute", start_ns: 100, end_ns: 200, interval_semantics: "aggregate_busy_within_batch_envelope", future_resource_extension: "resource-kept" }],
  }, 0);
  assert.equal(event.marker, "token_emit");
  assert.equal(event.token_index, 7);
  assert.deepEqual(Array.from(event.request_ids), ["r0", "r1"]);
  assert.equal(event.request_ids_truncated, true);
  assert.equal(event.representative_items_truncated, true);
  assert.equal(event.detail_semantics, "aggregate_batch_with_selected_items");
  assert.equal(event.future_event_extension.version, 2);
  assert.equal(event.rank.future_rank_extension, "rank-kept");
  assert.equal(event.tensor.future_tensor_extension, "tensor-kept");
  assert.equal(event.transfer.future_transfer_extension, "transfer-kept");
  assert.equal(event.transfer.hops[0].future_hop_extension, "hop-kept");
  assert.equal(event.transfer.hops[0].start_ns, null);
  assert.equal(event.transfer.hops[0].end_ns, null);
  assert.equal(event.transfer.hops[0].interval_semantics, "aggregate_route_without_hop_timing");
  assert.equal(event.resources[0].future_resource_extension, "resource-kept");
  assert.deepEqual(Array.from(ui.traceEventRequestIds(event)), ["r0", "r1"]);

  const visualization = ui.traceVisualizationFromReport({
    visualization: {
      future_trace_extension: { contract: "kept" },
      events: [{ event_id: "e0", start_ns: 0, end_ns: 1, future_event_extension: true }],
    },
  });
  assert.equal(visualization.future_trace_extension.contract, "kept");
  assert.equal(visualization.events[0].future_event_extension, true);
});

test("batch task endpoint uses the completed job id and bounded pagination query", () => {
  const ui = helpers();
  assert.equal(
    ui.tracePageEndpoint("job / 7", "batch?0", 5000, 5000),
    "/run-jobs/job%20%2F%207/trace?batch_id=batch%3F0&offset=5000&limit=5000",
  );
  const page = ui.tracePageData({
    fidelity: "representative",
    trace_source: "exact_cohort_replay",
    scope: { batch_id: "batch-0", start_ns: 1_000_000, end_ns: 2_000_000 },
    events: [{ task_id: "t0", start_ns: 1_250_000, end_ns: 1_500_000 }],
    pagination: { offset: 5000, limit: 5000, returned: 1, total: 9001, has_more: true, next_offset: 10000 },
  }, "batch-0", { memory_layout: {}, batch_trace_index: [] });
  assert.equal(page.start_ns, 1_000_000, "page events retain global simulation time");
  assert.equal(page.events[0].start_ns, 1_250_000);
  assert.equal(page.pagination.total, 9001);
  assert.equal(page.pagination.previous_offset, 0);
  assert.equal(page.pagination.next_offset, 10000);

  const fallbackIdPage = ui.tracePageData({
    events: [{ start_ns: 2_000_000, end_ns: 2_000_001 }],
    pagination: { offset: 5000, limit: 5000, total: 5001 },
  }, "batch-0", { memory_layout: {}, batch_trace_index: [] });
  assert.equal(fallbackIdPage.events[0].event_id, "trace-event-5000", "fallback ids include the stable backend page offset");
});

test("trace task pages use an eight-entry LRU and same-key failed requests can retry", async () => {
  const ui = helpers();
  const playback = { pageCache: new Map(), pagePending: new Map() };
  for (let index = 0; index < 8; index += 1) {
    ui.tracePageCacheSet(playback, ui.tracePageCacheKey("batch", index * 5000), { index });
  }
  assert.equal(playback.pageCache.size, 8);
  assert.equal(ui.tracePageCacheGet(playback, ui.tracePageCacheKey("batch", 0)).index, 0, "cache hit refreshes recency");
  ui.tracePageCacheSet(playback, ui.tracePageCacheKey("batch", 40_000), { index: 8 });
  assert.equal(playback.pageCache.size, 8);
  assert.equal(playback.pageCache.has(ui.tracePageCacheKey("batch", 5000)), false, "least-recently used page is evicted");
  assert.equal(playback.pageCache.has(ui.tracePageCacheKey("batch", 0)), true, "refreshed page remains cached");

  let calls = 0;
  let rejectRequest;
  const createFailure = () => {
    calls += 1;
    return new Promise((resolve, reject) => { rejectRequest = reject; });
  };
  const first = ui.tracePagePendingGetOrCreate(playback, "same-page", 7, createFailure);
  const second = ui.tracePagePendingGetOrCreate(playback, "same-page", 7, createFailure);
  assert.equal(first.promise, second.promise, "same page and generation share one pending promise");
  await Promise.resolve();
  assert.equal(calls, 1);
  rejectRequest(new Error("temporary failure"));
  await assert.rejects(first.promise, /temporary failure/);
  assert.equal(playback.pagePending.has("same-page"), false, "failed requests leave no poisoned pending entry");

  const retry = ui.tracePagePendingGetOrCreate(playback, "same-page", 7, async () => {
    calls += 1;
    return { ok: true };
  });
  assert.deepEqual(await retry.promise, { ok: true });
  assert.equal(calls, 2, "a failed page can be fetched again");
  assert.equal(playback.pagePending.has("same-page"), false);
});

test("overlapping TP tasks expose every active GPU, HBM, and route at the same global time", () => {
  const ui = helpers();
  ui.state.tracePlayback.filteredEvents = [
    {
      event_id: "tp0", start_ns: 10, end_ns: 30,
      rank: { rank: 0, tp_rank: 0, component_id: "gpu0" },
      tensor: { logical_id: "w", component_id: "hbm0" },
      transfer: { source_component: "hbm0", target_component: "gpu0", hops: [{ link_id: "hbm-link0", start_ns: 10, end_ns: 20 }] },
      resources: [{ resource_id: "gpu0.compute", start_ns: 10, end_ns: 30 }],
    },
    {
      event_id: "tp1", start_ns: 12, end_ns: 32,
      rank: { rank: 1, tp_rank: 1, component_id: "gpu1" },
      tensor: { logical_id: "w", component_id: "hbm1" },
      transfer: { source_component: "hbm1", target_component: "gpu1", hops: [{ link_id: "hbm-link1", start_ns: 12, end_ns: 22 }] },
      resources: [{ resource_id: "gpu1.compute", start_ns: 12, end_ns: 32 }],
    },
  ];
  const active = ui.activeTraceEvents(15);
  assert.deepEqual(Array.from(active, (event) => event.event_id), ["tp0", "tp1"]);
  const roles = ui.activeTraceNodeRoles(active, 15, new Set(["gpu0", "gpu1", "hbm0", "hbm1"]));
  for (const id of ["gpu0", "gpu1", "hbm0", "hbm1"]) assert.equal(roles.get(id).has("is-active"), true, `${id} is active`);
  assert.deepEqual(Array.from(ui.activeTraceLinkIds(active, 15)).sort(), ["hbm-link0", "hbm-link1"]);
  assert.notEqual(ui.traceActiveSignature(15), ui.traceActiveSignature(31), "highlight signature changes when one overlapping task finishes without a new task starting");
});

test("an explicit transfer link id activates only that parallel route", () => {
  const ui = helpers();
  const events = [{
    event_id: "parallel-transfer",
    start_ns: 10,
    end_ns: 30,
    transfer: {
      link_id: "parallel-a",
      source_component: "gpu0",
      target_component: "hbm0",
    },
  }];
  const activeIds = ui.activeTraceLinkIds(events, 15);
  assert.deepEqual(Array.from(activeIds), ["parallel-a"]);
  assert.equal(ui.traceLinkIsActive({ link_id: "parallel-a", source_component: "gpu0", target_component: "hbm0" }, events, activeIds, 15), true);
  assert.equal(ui.traceLinkIsActive({ link_id: "parallel-b", source_component: "gpu0", target_component: "hbm0" }, events, activeIds, 15), false);

  const ambiguous = [{
    event_id: "endpoint-only",
    start_ns: 10,
    end_ns: 30,
    transfer: { source_component: "gpu0", target_component: "hbm0" },
  }];
  assert.equal(ui.traceLinkIsActive({ link_id: "parallel-b", source_component: "gpu0", target_component: "hbm0" }, ambiguous, new Set(), 15), true, "endpoint fallback remains available when no exact route identity exists");
});

test("trace particles preserve exact and endpoint-only transfer direction", () => {
  const ui = helpers();
  const bidirectional = { link_id: "lane", source_component: "a", target_component: "b", bidirectional: true };
  const directed = { ...bidirectional, bidirectional: false };
  const reverse = [{
    event_id: "reverse-exact",
    start_ns: 10,
    end_ns: 30,
    transfer: { link_id: "lane", source_component: "b", target_component: "a" },
  }];
  const exactIds = ui.activeTraceLinkIds(reverse, 15);
  assert.deepEqual(
    Array.from(ui.traceLinkFlowDirections(bidirectional, reverse, exactIds, 15), (flow) => flow.direction),
    [-1],
    "an exact reverse transfer travels back along a bidirectional route",
  );
  assert.deepEqual(
    Array.from(ui.traceLinkFlowDirections(directed, reverse, exactIds, 15)),
    [],
    "an exact reverse transfer cannot activate a directed link",
  );

  const endpointOnly = [{
    event_id: "reverse-fallback",
    start_ns: 10,
    end_ns: 30,
    transfer: { source_component: "b", target_component: "a" },
  }];
  assert.deepEqual(
    Array.from(ui.traceLinkFlowDirections(bidirectional, endpointOnly, new Set(), 15), (flow) => flow.direction),
    [-1],
    "endpoint fallback keeps the reverse direction",
  );
  assert.equal(
    ui.traceLinkIsActive(directed, endpointOnly, new Set(), 15),
    false,
    "endpoint fallback does not activate a directed link in reverse",
  );

  const hopIdentityOnly = [{
    event_id: "reverse-hop-identity",
    start_ns: 10,
    end_ns: 30,
    transfer: {
      source_component: "b",
      target_component: "a",
      hops: [{ link_id: "lane", start_ns: 10, end_ns: 30 }],
    },
  }];
  assert.deepEqual(
    Array.from(ui.traceLinkFlowDirections(bidirectional, hopIdentityOnly, new Set(["lane"]), 15), (flow) => flow.direction),
    [-1],
    "an exact hop inherits reverse endpoints from its parent transfer",
  );
  assert.deepEqual(
    Array.from(ui.traceLinkFlowDirections(directed, hopIdentityOnly, new Set(["lane"]), 15)),
    [],
    "an exact hop without endpoints must not force a reverse transfer onto a directed link",
  );
});

test("one bidirectional route can expose simultaneous opposite particle groups", () => {
  const ui = helpers();
  const linkValue = { link_id: "lane", source_component: "a", target_component: "b", bidirectional: true };
  const events = [
    { event_id: "forward", start_ns: 10, end_ns: 30, transfer: { link_id: "lane", source_component: "a", target_component: "b" } },
    { event_id: "reverse", start_ns: 10, end_ns: 30, transfer: { link_id: "lane", source_component: "b", target_component: "a" } },
  ];
  const flows = ui.traceLinkFlowDirections(linkValue, events, ui.activeTraceLinkIds(events, 15), 15);
  assert.deepEqual(Array.from(flows, (flow) => flow.direction).sort((left, right) => left - right), [-1, 1]);
  const route = { points: [{ x: 0, y: 0 }, { x: 10, y: 0 }] };
  assert.deepEqual(
    Array.from(flows, (flow) => ui.traceParticleRouteForDirection(route, flow.direction).points),
    [[{ x: 0, y: 0 }, { x: 10, y: 0 }], [{ x: 10, y: 0 }, { x: 0, y: 0 }]],
    "opposite flows receive opposite point orders",
  );
});

test("trace particle flows preserve instruction/data kinds, inherit hop kinds, and keep fixed decoration", () => {
  const ui = helpers();
  const linkValue = { link_id: "lane", source_component: "a", target_component: "b", bidirectional: true };
  const events = [
    { event_id: "instruction", start_ns: 10, end_ns: 30, transfer: { kind: "instruction", link_id: "lane", source_component: "a", target_component: "b", bytes: 1 } },
    { event_id: "data", start_ns: 10, end_ns: 30, transfer: { transfer_kind: "data", link_id: "lane", source_component: "b", target_component: "a", bytes: 10 ** 12 } },
    { event_id: "unknown", start_ns: 10, end_ns: 30, transfer: { kind: "future_kind", link_id: "lane", source_component: "a", target_component: "b" } },
  ];
  const flows = ui.traceLinkFlowDirections(linkValue, events, ui.activeTraceLinkIds(events, 15), 15);
  assert.deepEqual(
    Array.from(flows, (flow) => [flow.direction, flow.transferKind, flow.transfer_kind]),
    [[1, "instruction", "instruction"], [-1, "data", "data"], [1, "data", "data"]],
    "both accepted spellings survive direction matching while unknown values default to data",
  );

  const identityOnlyHop = ui.normalizeTraceEvent({
    event_id: "instruction-route",
    start_ns: 10,
    end_ns: 30,
    transfer: {
      kind: "instruction",
      source_component: "a",
      target_component: "b",
      hops: [{ link_id: "lane", start_ns: null, end_ns: null, interval_semantics: "topology_route_identity_without_hop_service" }],
    },
  }, 0);
  const hopFlows = ui.traceLinkFlowDirections(linkValue, [identityOnlyHop], ui.activeTraceLinkIds([identityOnlyHop], 15), 15);
  assert.deepEqual(Array.from(hopFlows, (flow) => [flow.direction, flow.transferKind]), [[1, "instruction"]], "identity-only hops inherit the parent command interval and kind");
  assert.equal(identityOnlyHop.transfer.hops[0].start_ns, null, "normalization preserves the backend's raw null hop start");
  assert.equal(identityOnlyHop.transfer.hops[0].end_ns, null, "normalization preserves the backend's raw null hop end");
  assert.deepEqual(
    Array.from(ui.traceLinkFlowDirections(linkValue, [identityOnlyHop], ui.activeTraceLinkIds([identityOnlyHop], 31), 31)),
    [],
    "null hop timing inherits the parent event interval instead of becoming an unbounded route",
  );

  ui.state.view = "playback";
  ui.state.tracePlayback.layoutView = { positions: {}, zoom: 1, offsetX: 0, offsetY: 0, autoFitKey: "" };
  ui.dom.traceParticleLayer = { innerHTML: "" };
  ui.renderTraceParticles([
    { points: [{ x: 0, y: 0 }, { x: 10, y: 0 }], direction: 1, transferKind: "instruction", bytes: 1 },
    { points: [{ x: 0, y: 0 }, { x: 10, y: 0 }], direction: -1, transfer_kind: "data", bytes: 10 ** 12 },
  ]);
  assert.equal((ui.dom.traceParticleLayer.innerHTML.match(/data-trace-particle/g) || []).length, 6, "particle count is fixed per active direction/kind, not byte-scaled");
  assert.equal((ui.dom.traceParticleLayer.innerHTML.match(/is-instruction/g) || []).length, 3);
  assert.equal((ui.dom.traceParticleLayer.innerHTML.match(/is-data/g) || []).length, 3);
  assert.match(ui.dom.traceParticleLayer.innerHTML, /data-trace-transfer-kind="instruction"/);
  assert.match(ui.dom.traceParticleLayer.innerHTML, /data-trace-transfer-kind="data"/);
});

test("resource-only link intervals do not synthesize data transfer flows", () => {
  const ui = helpers();
  const linkValue = { link_id: "lane", source_component: "a", target_component: "b", bidirectional: true };
  const resourceOnlyEvents = [{
    event_id: "resource-only-link",
    start_ns: 10,
    end_ns: 30,
    resources: [{ resource_id: "link.lane", start_ns: 10, end_ns: 30 }],
  }];
  const activeIds = ui.activeTraceLinkIds(resourceOnlyEvents, 15);
  assert.deepEqual(Array.from(activeIds), [], "bare resource intervals are not trace transfer identities");
  assert.equal(ui.traceLinkIsActive(linkValue, resourceOnlyEvents, activeIds, 15), false);
  assert.deepEqual(Array.from(ui.traceLinkFlowDirections(linkValue, resourceOnlyEvents, activeIds, 15)), []);

  const roles = ui.activeTraceNodeRoles([{
    event_id: "compute-resource",
    start_ns: 10,
    end_ns: 30,
    resources: [{ resource_id: "gpu0.compute", component_id: "gpu0", start_ns: 10, end_ns: 30 }],
  }], 15, new Set(["gpu0"]));
  assert.equal(roles.get("gpu0").has("is-active"), true, "component resource highlighting remains independent from transfer flows");
});

test("same-start events retain an explicitly selected row for step and table navigation", () => {
  const ui = helpers();
  const playback = ui.state.tracePlayback;
  playback.filteredEvents = [
    { event_id: "same-0", start_ns: 10, end_ns: 20 },
    { event_id: "same-1", start_ns: 10, end_ns: 30 },
    { event_id: "later", start_ns: 40, end_ns: 50 },
  ];
  Object.assign(playback, { startNs: 10, endNs: 50, selectedIndex: 1, selectedEventId: "same-0" });
  playback.activeSignature = ui.traceActiveSignature(10);
  ui.setTraceTime(10, { eventId: "same-0" });
  assert.equal(playback.selectedIndex, 0);
  assert.equal(ui.selectedTraceEvent().event_id, "same-0");
  Object.assign(playback, { selectedIndex: 0, selectedEventId: "same-1" });
  ui.setTraceTime(10, { eventId: "same-1" });
  assert.equal(playback.selectedIndex, 1);
  assert.equal(ui.selectedTraceEvent().event_id, "same-1");
  assert.match(app, /setTraceTime\(event\.start_ns, \{ force: true, eventId: event\.event_id, reveal: true \}\)/);
  assert.match(app, /data-trace-event-id/);
});

test("selected event remains stable by event_id while a timeline gap has no active event", () => {
  const ui = helpers();
  ui.state.tracePlayback.filteredEvents = [
    { event_id: "before", start_ns: 10, end_ns: 20 },
    { event_id: "after", start_ns: 40, end_ns: 50 },
  ];
  ui.setTraceSelectedIndex(1);
  ui.state.tracePlayback.timeNs = 30;
  assert.equal(ui.selectedTraceEvent().event_id, "after");
  assert.equal(ui.state.tracePlayback.selectedEventId, "after");
  assert.deepEqual(Array.from(ui.activeTraceEvents(30)), []);
  assert.match(app, /这是两项工作之间的时间空档/);
  assert.match(app, /它此刻未在运行，拓扑不会把它伪装成活动/);
});

test("active intervals are half-open while zero-duration markers remain visible", () => {
  const ui = helpers();
  ui.state.tracePlayback.filteredEvents = [
    { event_id: "ended", start_ns: 10, end_ns: 20 },
    { event_id: "marker", start_ns: 20, end_ns: 20 },
    { event_id: "started", start_ns: 20, end_ns: 30 },
  ];
  assert.deepEqual(Array.from(ui.activeTraceEvents(20), (event) => event.event_id), ["marker", "started"]);
  assert.equal(ui.traceIntervalActive({ start_ns: 10, end_ns: 20 }, 20), false);
  assert.equal(ui.traceIntervalActive({ start_ns: 20, end_ns: 20 }, 20), true);
});

test("aggregate overview is sticky after the user selects it and cached pages still stop playback", () => {
  const ui = helpers();
  ui.state.tracePlayback.aggregateData = {
    events: [{ event_id: "batch", batch_id: "b0", start_ns: 0, end_ns: 10 }],
    start_ns: 0,
    end_ns: 10,
    batch_trace_index: [{ batch_id: "b0" }],
  };
  ui.state.tracePlayback.batchTraceIndex = [{ batch_id: "b0" }];
  ui.state.tracePlayback.autoLoadFirstBatch = true;
  ui.state.tracePlayback.batchFilter = "b0";
  ui.applyAggregateTraceData();
  assert.equal(ui.state.tracePlayback.mode, "aggregate");
  assert.equal(ui.state.tracePlayback.batchFilter, "");
  assert.equal(ui.state.tracePlayback.autoLoadFirstBatch, false);

  const loadBody = app.slice(app.indexOf("async function loadTraceTaskPage"), app.indexOf("function traceFilterValue"));
  assert.ok(loadBody.indexOf("stopTracePlayback();") < loadBody.indexOf("tracePageCacheGet(playback, cacheKey)"));
});

test("playback step summary distinguishes batch envelopes from task events", () => {
  const ui = helpers();
  ui.state.tracePlayback.events = [
    { event_id: "batch-0" },
    { event_id: "batch-1" },
  ];
  ui.state.tracePlayback.data = { total_events: 2 };
  ui.state.tracePlayback.batchTraceIndex = [{ batch_id: "b0" }, { batch_id: "b1" }];
  ui.state.tracePlayback.mode = "aggregate";
  assert.deepEqual({ ...ui.traceStepSummary() }, { count: 2, label: "2 个批次摘要" });

  ui.state.tracePlayback.mode = "task";
  ui.state.tracePlayback.page = { total: 4931 };
  assert.deepEqual({ ...ui.traceStepSummary() }, { count: 4931, label: "4,931 个任务事件" });
});

test("trace payload enums are presented as deduplicated natural language", () => {
  const zh = helpers("zh-CN");
  const cases = new Map([
    ["kv_read_skipped", "跳过 KV 缓存读取"],
    ["synchronization", "同步"],
    ["kernel launch", "启动计算内核"],
    ["gpu gemm", "GPU 矩阵乘法"],
    ["serving cohort start", "开始服务批次"],
    ["policy", "策略"],
    ["exclusive service", "独占服务区间"],
  ]);
  cases.forEach((expected, raw) => assert.equal(zh.traceFriendlyTerm(raw), expected, raw));
  assert.equal(zh.traceEventSemantics({ phase: "exclusive_service", event_kind: "exclusive service", category: "synchronization" }), "独占服务区间 · 同步");
  assert.equal(zh.traceEventSubject({ event_kind: "gpu_gemm", category: "compute", rank: { component_id: "gpu0" } }), "gpu0 执行 GPU 矩阵乘法");
  assert.equal(zh.traceEventSubject({ layer_id: "layer.0", operator_id: "backend_gemm", rank: { component_id: "gpu0" } }), "gpu0 执行 layer.0 / backend_gemm");
  assert.doesNotMatch(zh.traceEventSemantics({ event_kind: "kernel_launch", category: "compute" }), /kernel launch|compute/iu);

  const en = helpers("en");
  assert.equal(en.traceEventSemantics({ event_kind: "kv_read_skipped", category: "synchronization" }), "KV cache read skipped · Synchronization");
  assert.equal(en.traceEventSubject({ event_kind: "serving_cohort_start", category: "policy", rank: { component_id: "runtime0" } }), "runtime0 performs Serving cohort start");
  assert.equal(en.traceEventSubject({ layer_id: "layer.0", operator_id: "backend_gemm", rank: { component_id: "gpu0" } }), "gpu0 performs layer.0 / backend_gemm");
  assert.doesNotMatch(en.traceEventSemantics({ event_kind: "gpu_gemm", category: "compute" }), /[\u3400-\u9fff]/u);
});

test("explicit event selection reveals its node without making playback or slider steal scrolling", () => {
  const ui = helpers();
  const layout = {
    components: [component("gpu1", "gpu"), component("gpu0", "gpu")],
    rects: [
      { id: "gpu1", x: 80, y: 80, width: 140, height: 70 },
      { id: "gpu0", x: 900, y: 80, width: 140, height: 70 },
    ],
  };
  const active = [
    { rank: { component_id: "gpu1" }, start_ns: 10, end_ns: 30, resources: [] },
    { rank: { component_id: "gpu0" }, start_ns: 10, end_ns: 30, resources: [] },
  ];
  const target = ui.traceRevealTargetRect(layout, active, 15, active[1], 500, 300, 20);
  assert.equal(target.x, 900, "an over-wide active union falls back to the explicitly selected event component");
  const scroll = ui.traceRevealScrollTarget({
    clientWidth: 500, clientHeight: 300, scrollWidth: 1200, scrollHeight: 400, scrollLeft: 0, scrollTop: 0,
  }, target, 20);
  assert.ok(scroll.left > 0, "off-screen gpu0 is revealed from the left-most viewport");
  assert.equal(ui.traceRevealScrollTarget({
    clientWidth: 500, clientHeight: 300, scrollWidth: 1200, scrollHeight: 400, scrollLeft: 650, scrollTop: 0,
  }, target, 20), null, "an already visible target does not move the user's viewport");
  assert.equal(ui.traceRevealScrollBehavior(false, false), "smooth");
  assert.equal(ui.traceRevealScrollBehavior(true, false), "auto");
  assert.equal(ui.traceRevealScrollBehavior(false, true), "auto");

  const animationBody = app.slice(app.indexOf("function traceAnimationStep"), app.indexOf("function toggleTracePlayback"));
  const sliderBody = app.slice(app.indexOf('dom.traceTimeline.addEventListener("input"'), app.indexOf("const updateTraceFilter"));
  assert.doesNotMatch(animationBody, /reveal:\s*true/);
  assert.doesNotMatch(sliderBody, /reveal:\s*true/);
});

test("semantic event selection returns to the playback controls with reduced-motion-safe scrolling", () => {
  const ui = helpers();
  const scrollCalls = [];
  const control = {
    scrollIntoView(options) { scrollCalls.push(options); },
  };
  ui.state.view = "playback";
  ui.state.settings.reduceMotion = false;
  ui.dom.traceContent = { hidden: false };
  ui.dom.traceTimeline = {
    closest(selector) {
      assert.equal(selector, ".trace-control-bar");
      return control;
    },
  };
  assert.equal(ui.tracePlaybackControlTarget(), control);
  assert.equal(ui.revealTracePlaybackControls(), true);
  assert.equal(scrollCalls.length, 1);
  assert.equal(scrollCalls[0].behavior, "smooth");
  assert.equal(scrollCalls[0].block, "start");
  assert.equal(scrollCalls[0].inline, "nearest");

  ui.state.settings.reduceMotion = true;
  assert.equal(ui.revealTracePlaybackControls(), true);
  assert.equal(scrollCalls.at(-1).behavior, "auto");

  ui.state.tracePlayback.playing = true;
  assert.equal(ui.revealTracePlaybackControls(), false, "automatic playback never steals page scrolling");
  assert.equal(scrollCalls.length, 2);
});

test("semantic event rows ignore only their nested details and keep the accessible select button path", () => {
  const streamBody = app.slice(app.indexOf("function renderTraceEventTable"), app.indexOf("function percentile"));
  assert.match(streamBody, /clickEvent\.target\.closest\("\.trace-row-details"\)/);
  assert.doesNotMatch(streamBody, /clickEvent\.target\.closest\("details"\)/);
  assert.match(streamBody, /stopTracePlayback\(\);[\s\S]*setTraceTime\(event\.start_ns, \{ force: true, eventId: event\.event_id, reveal: true \}\);[\s\S]*revealTracePlaybackControls\(\)/);
  assert.match(streamBody, /<button type="button" class="trace-event-select"/);
});

test("trace fit centers content bounds and exposes an explicit arrange-and-fit action", () => {
  const ui = helpers();
  assert.equal(JSON.stringify(ui.traceTopologyContentBounds({
    bounds: { width: 640, height: 480 },
    rects: [
      { id: "left", x: 90, y: 40, width: 120, height: 50 },
      { id: "right", x: 410, y: 170, width: 100, height: 70 },
    ],
  })), JSON.stringify({ x: 90, y: 40, width: 420, height: 200 }));
  assert.equal(JSON.stringify(ui.traceTopologyContentBounds({
    contentBounds: { x: 24, y: 18, width: 730, height: 410 },
    rects: [{ id: "node-only", x: 100, y: 100, width: 80, height: 40 }],
  })), JSON.stringify({ x: 24, y: 18, width: 730, height: 410 }), "fit uses complete node/route/label geometry when available");
  assert.match(html, /id="traceArrangeFitButton">整理并适配<\/button>/);
  assert.match(app, /"traceArrangeFitButton"/);
  assert.match(app, /traceArrangeFitButton\.addEventListener\("click", arrangeAndFitTraceLayout\)/);
  assert.match(app, /function traceFitMetrics[\s\S]*traceTopologyContentBounds\(layout\)/);
  assert.match(app, /Topology\.planOrthogonalRoutes\(links, rects/);
  assert.doesNotMatch(app, /scrollTo\?\.\(\{ left: 0, top: 0, behavior: "auto" \}\)/);
});

test("trace automatic fit stays between 80% and 100% and preserves scrollable overflow", () => {
  const ui = helpers();
  ui.dom.traceTopologyCanvas = { clientWidth: 600, clientHeight: 360 };
  ui.state.tracePlayback.layoutView = { positions: {}, zoom: 1, offsetX: 0, offsetY: 0, autoFitKey: "" };
  const compact = ui.traceFitMetrics({
    autoFitKey: "compact",
    components: [],
    links: [],
    positions: {},
    sizes: {},
    contentBounds: { x: 30, y: 20, width: 300, height: 180 },
  });
  assert.equal(ui.state.tracePlayback.layoutView.zoom, 1);
  assert.equal(compact.worldWidth, 600);
  assert.equal(compact.worldHeight, 360);

  const overflow = ui.traceFitMetrics({
    autoFitKey: "overflow",
    components: [],
    links: [],
    positions: {},
    sizes: {},
    contentBounds: { x: 0, y: 0, width: 1200, height: 700 },
  });
  assert.equal(ui.state.tracePlayback.layoutView.zoom, 0.8);
  assert.ok(overflow.worldWidth > ui.dom.traceTopologyCanvas.clientWidth);
  assert.ok(overflow.worldHeight > ui.dom.traceTopologyCanvas.clientHeight);
});

test("playback viewport-aware layout spreads compact graphs and reroutes without changing node boxes", () => {
  const ui = helpers();
  const hardware = {
    components: [component("gpu0", "gpu"), component("hbm0", "hbm"), component("ssd0", "ssd")],
    links: [link("gh", "gpu0", "hbm0", "HBM"), link("hs", "hbm0", "ssd0", "PCIe")],
  };
  const compact = ui.buildTraceTopologyLayout(hardware, 100, [], {}, null, { width: 560, height: 360 });
  const wide = ui.buildTraceTopologyLayout(hardware, 100, [], {}, null, { width: 1120, height: 560 });
  const extent = (layout, axis) => Math.max(...Object.values(layout.positions).map((point) => point[axis]))
    - Math.min(...Object.values(layout.positions).map((point) => point[axis]));
  assert.ok(extent(wide, "x") > extent(compact, "x"));
  assert.ok(extent(wide, "y") > extent(compact, "y"));
  assert.deepEqual(wide.sizes, compact.sizes, "canvas fill comes from spacing, not enlarged nodes");
  assertLayoutRoutesAreOrthogonalAndClear(wide, "wide viewport playback layout");
});

test("comparison candidate reports cannot reuse a completed job belonging to the previous report", () => {
  const compareBody = app.slice(app.indexOf("async function compareScenario"), app.indexOf("function switchView"));
  assert.match(compareBody, /state\.report = payload\.candidate;\s*state\.runJob = null;\s*state\.runJobScenarioGeneration = null;/);
});

test("trace node sizing follows visible id and memory text instead of hidden kind labels", () => {
  const ui = helpers();
  const hiddenKind = ui.traceNodeSizes([
    { component_id: "x", kind: "extremely_verbose_hidden_component_kind_label_that_is_not_rendered" },
  ], 100).x;
  const visibleMemory = ui.traceNodeSizes([
    { component_id: "gpu0", kind: "gpu", capacity_bytes: 64 * 1024 ** 3 },
  ], 100, {
    components: {
      gpu0: [{ logical_tensor_id: "kv", offset_bytes: 0, length_bytes: 1024 ** 3 }],
    },
  }).gpu0;
  const noMemory = ui.traceNodeSizes([{ component_id: "gpu0", kind: "gpu" }], 100).gpu0;
  assert.ok(hiddenKind.width < 140, "hidden kind labels must not make compact trace nodes wide");
  assert.ok(visibleMemory.height > noMemory.height, "visible memory text and bar reserve vertical space");
  assert.equal(ui.traceNodeMemoryCapacityLabel({ component_id: "empty" }, { components: {} }), "");
});

test("trace node measurement ignores pseudo-element overflow outside the visible copy", () => {
  const ui = helpers();
  const content = {
    offsetWidth: 96,
    scrollWidth: 96,
    offsetHeight: 40,
    scrollHeight: 40,
  };
  const node = {
    dataset: { traceComponent: "gpu0" },
    offsetWidth: 100,
    offsetHeight: 40,
    scrollHeight: 999,
    querySelector(selector) {
      return selector === ".trace-node-copy" ? content : null;
    },
  };
  ui.dom.traceNodeLayer = {
    querySelectorAll(selector) {
      assert.equal(selector, ".trace-node[data-trace-component]");
      return [node];
    },
  };
  ui.state.tracePlayback.measuredNodeSizes = {};

  assert.equal(ui.measureTraceTopologyNodes({ sizes: { gpu0: { width: 100, height: 40 } } }), false);
  assert.deepEqual(ui.state.tracePlayback.measuredNodeSizes, {});
});

test("trace topology layout reuses architecture projection, layout, and routes for playback-local collapse", () => {
  const ui = helpers();
  const hardware = {
    components: [
      component("gpu0", "gpu"),
      component("hbm0", "hbm"),
      component("fabric0", "fabric_switch"),
      component("host0", "host_memory"),
    ],
    links: [
      { ...link("gpu-hbm", "gpu0", "hbm0", "HBM"), bandwidth_gbps: 256 },
      { ...link("hbm-fabric-a", "hbm0", "fabric0", "NVLink"), bandwidth_gbps: 80 },
      { ...link("hbm-fabric-b", "hbm0", "fabric0", "NVLink"), bandwidth_gbps: 96 },
      { ...link("fabric-host", "fabric0", "host0", "PCIe"), bandwidth_gbps: 32 },
    ],
  };
  const architectureGroups = [{ group_id: "gpu0-package", members: ["gpu0", "hbm0"], root: "gpu0", collapsed: false }];
  const playbackGroups = [{ ...architectureGroups[0], collapsed: true }];
  const projection = TopologyCore.collapseProjection(hardware.components, hardware.links, playbackGroups);
  const layout = ui.buildTraceTopologyLayout(hardware, 100, playbackGroups);

  assert.deepEqual(
    layout.components.map((item) => item.component_id).sort(),
    projection.visibleComponentIds.slice().sort(),
    "playback collapse uses topology-core's visible-component projection",
  );
  assert.equal(layout.components.some((item) => item.component_id === "hbm0"), false, "collapsed hidden members are not rendered as trace nodes");
  assert.deepEqual(
    layout.links.map(displayLinkId).sort(),
    projection.links.map(displayLinkId).sort(),
    "trace route identities mirror topology-core projected link identities",
  );
  const aggregate = layout.links.find((item) => Array.isArray(item.original_link_ids) && item.original_link_ids.includes("hbm-fabric-a"));
  assert.ok(aggregate, "external hidden-member links are projected to the group root");
  assert.equal(aggregate.aggregate_count, 2);
  assert.equal(aggregate.bandwidth_gbps, 176);
  assertLayoutRoutesAreOrthogonalAndClear(layout, "collapsed playback layout");

  const expandedLayout = ui.buildTraceTopologyLayout(hardware, 100, architectureGroups);
  assert.ok(expandedLayout.components.some((item) => item.component_id === "hbm0"), "expanded architecture groups do not force playback collapse");
  assert.equal(architectureGroups[0].collapsed, false, "building playback layout does not mutate architecture groups");
  assert.equal(playbackGroups[0].collapsed, true);
});

test("expanded GB200 GPU-HBM group keeps every architecture and playback route", () => {
  const ui = helpers();
  const components = [];
  const links = [];
  const groups = [];
  for (let module = 0; module < 2; module += 1) {
    const grace = `grace${module}`;
    const lpddr = `lpddr${module}`;
    components.push(component(grace, "cpu"), component(lpddr, "host_memory"));
    links.push({
      ...link(`lpddr${module}`, grace, lpddr, "LPDDR5X"),
      source_port: "memory",
      target_port: "host",
    });
    groups.push({ group_id: `${grace}_lpddr`, members: [grace, lpddr], root: grace, collapsed: false });
    for (let local = 0; local < 2; local += 1) {
      const index = module * 2 + local;
      const gpu = `gpu${index}`;
      const hbmIds = Array.from({ length: 8 }, (_, hbmIndex) => `hbm${index}_${hbmIndex}`);
      components.push(component(gpu, "gpu"), ...hbmIds.map((id) => component(id, "hbm")));
      links.push(
        {
          ...link(`c2c${index}`, grace, gpu, "NVLink-C2C"),
          source_port: `gpu${local}`,
          target_port: "cpu",
        },
        {
          ...link(`fabric${index}`, gpu, "fabric0", "NVLink"),
          source_port: "fabric",
          target_port: `endpoint${index}`,
        },
        ...hbmIds.map((hbmId, hbmIndex) => ({
          ...link(`hbm_link${index}_${hbmIndex}`, gpu, hbmId, "HBM"),
          source_port: `hbm${hbmIndex}`,
          target_port: "host",
        })),
      );
      groups.push({ group_id: `${gpu}_hbm`, members: [gpu, ...hbmIds], root: gpu, collapsed: true });
    }
  }
  components.push(component("fabric0", "fabric_switch"));
  groups.push({ group_id: "fabric", members: ["fabric0"], root: "fabric0", collapsed: true });
  const hardware = { components, links };

  const folded = ui.buildTraceTopologyLayout(hardware, 100, groups);
  assert.equal(folded.links.length, 10, "the default folded projection exposes ten external links");
  assert.equal(Array.from(folded.routes.values()).filter((route) => route.path).length, 10);

  const expandedGroups = groups.map((group) => group.group_id === "gpu0_hbm" ? { ...group, collapsed: false } : group);
  const metrics = ui.topologyLayoutMetrics(100);
  assert.equal(metrics.nodeGap, 34, "shared default spacing leaves room for both 16px endpoint escapes");
  assert.ok(metrics.nodeGap >= metrics.routeClearance * 2 + 2);
  const architectureProjection = TopologyCore.collapseProjection(components, links, expandedGroups);
  const architectureVisibleIds = new Set(architectureProjection.visibleComponentIds.map(String));
  const architectureComponents = components.filter((item) => architectureVisibleIds.has(String(item.component_id)));
  const architectureSizes = Object.fromEntries(architectureComponents.map((item) => [item.component_id, {
    width: metrics.nodeW,
    height: metrics.nodeH,
  }]));
  const architectureLayoutGroups = expandedGroups.map((group) => group.collapsed ? { ...group, members: [group.root] } : group);
  const architectureLayout = TopologyCore.layoutGraph(
    architectureComponents,
    architectureProjection.links,
    architectureLayoutGroups,
    architectureSizes,
    { nodeGap: metrics.nodeGap, layerGap: metrics.layerGap, rowGap: metrics.rowGap },
  );
  const architectureRects = architectureComponents.map((item) => (
    TopologyCore.rectForNode(item.component_id, architectureLayout.positions, architectureSizes)
  ));
  const architectureRoutePlan = TopologyCore.planOrthogonalRoutes(architectureProjection.links, architectureRects, {
    clearance: metrics.routeClearance,
    parallelSpacing: metrics.parallelSpacing,
    routeSpacing: metrics.routeSpacing,
    cornerRadius: metrics.cornerRadius,
    shortCurveDistance: metrics.shortCurveDistance,
    renderGeometry: true,
    allowCurves: true,
  });
  assert.equal(architectureProjection.links.length, 18);
  assert.equal(architectureRoutePlan.errors.length, 0, "architecture relayout keeps all expanded GB200 routes");
  assert.equal(architectureRoutePlan.routes.length, architectureProjection.links.length);
  architectureRoutePlan.routes.forEach((route) => {
    assert.ok(route.path, `${route.linkId} keeps non-empty architecture SVG geometry`);
    const obstacles = architectureRects.filter((rect) => ![
      route.link.source_component,
      route.link.target_component,
    ].map(String).includes(String(rect.id)));
    assert.equal(
      TopologyCore.sampledPathClear(route.points, obstacles),
      true,
      `${route.linkId} architecture geometry does not cross another node`,
    );
  });

  const expanded = ui.buildTraceTopologyLayout(hardware, 100, expandedGroups);
  assert.equal(expanded.links.length, 18, "expanding GPU 0 adds its eight physical HBM links");
  assert.equal(expanded.routes.size, expanded.links.length, "every visible link has a route record");
  expanded.links.forEach((edge) => {
    const route = routeFor(expanded, edge);
    assert.ok(route?.path, `${displayLinkId(edge)} keeps non-empty SVG geometry (${route?.error || "missing route"})`);
  });
  assertLayoutRoutesAreOrthogonalAndClear(expanded, "expanded GB200 playback layout");
});

test("collapsed trace activity maps hidden members to root nodes and projected routes", () => {
  const ui = helpers();
  const hardware = {
    components: [
      component("gpu0", "gpu"),
      component("hbm0", "hbm"),
      component("fabric0", "fabric_switch"),
    ],
    links: [
      { ...link("gpu-hbm", "gpu0", "hbm0", "HBM"), bandwidth_gbps: 256 },
      { ...link("hbm-fabric", "hbm0", "fabric0", "NVLink"), bandwidth_gbps: 80 },
    ],
  };
  const groups = [{ group_id: "gpu0-package", members: ["gpu0", "hbm0"], root: "gpu0", collapsed: true }];
  const layout = ui.buildTraceTopologyLayout(hardware, 100, groups);
  const projected = layout.links.find((item) => Array.isArray(item.original_link_ids) && item.original_link_ids.includes("hbm-fabric"));
  assert.ok(projected, "hidden-member transfer route is represented by a projected link");

  const transferEvent = {
    event_id: "hidden-member-transfer",
    start_ns: 10,
    end_ns: 30,
    transfer: {
      source_component: "hbm0",
      target_component: "fabric0",
      hops: [{ link_id: "hbm-fabric", source_component: "hbm0", target_component: "fabric0", start_ns: 10, end_ns: 30 }],
    },
  };
  const activeIds = ui.activeTraceLinkIds([transferEvent], 15);
  assert.equal(ui.traceLinkIsActive(projected, [transferEvent], activeIds, 15), true, "original hop ids activate their collapsed proxy route");

  const tensorEvent = {
    event_id: "hidden-member-tensor",
    start_ns: 10,
    end_ns: 30,
    tensor: { logical_id: "kv", component_id: "hbm0" },
  };
  const target = ui.traceRevealTargetRect(layout, [tensorEvent], 15, tensorEvent, 1, 1, 0);
  assert.equal(target?.id, "gpu0", "hidden-member activity reveals the collapsed group root");
});

test("trace topology collapse controls are local to playback instead of mutating architecture state", () => {
  const groupsBody = sourceBetween("traceTopologyGroups", "traceCollapsedGroupIds");
  const layoutKeyBody = sourceBetween("traceTopologyLayout", "traceIntervalActive");
  const renderGroupsBody = sourceBetween("renderTraceGroups", "bindTraceLinkTooltip");
  const renderBody = sourceBetween("renderTraceTopology", "traceDetailFact");
  const toggleBody = sourceBetween("toggleTraceTopologyGroup", "traceTopologyAutoFitKey");

  expectMatch(html, /id="traceGroupLayer"/, "trace page has a group layer");
  expectMatch(app, /"traceGroupLayer"/, "traceGroupLayer is hydrated into dom");
  expectMatch(app, /tracePlayback:\s*\{[\s\S]*?topologyGroups:\s*null,[\s\S]*?collapsedGroupIds:\s*\[\],[\s\S]*?groupCollapseInitialized:\s*false/, "trace playback owns local group collapse state");
  expectMatch(groupsBody, /playback\.topologyGroups[\s\S]*state\.topologyView\?\.groups/, "playback seeds its local group copy from architecture groups");
  expectMatch(layoutKeyBody, /const groups = traceTopologyGroups\(hardware\)/, "playback layout reads replay-local group state");
  expectMatch(layoutKeyBody, /const collapsedGroupIds = traceCollapsedGroupIds\(hardware,\s*groups\)/, "playback derives collapsed ids from replay-local groups");
  expectMatch(layoutKeyBody, /collapsed/, "cache keys include collapsed state so toggles invalidate the replay layout");
  expectMatch(renderBody, /renderTraceGroups\(layout,\s*nodeRoles,\s*selectedOriginalIds\)/, "trace renderer writes group frames");
  expectMatch(renderGroupsBody, /data-trace-group|data-trace-group-toggle/, "trace group frames expose toggle targets");
  expectMatch(renderGroupsBody, /aria-expanded/, "trace group controls expose expanded state");
  expectMatch(renderGroupsBody, /aria-controls/, "trace group controls identify the frame they expand");
  expectMatch(renderGroupsBody, /按 Enter 或空格|press Enter or Space/, "trace group controls advertise keyboard interaction");
  expectMatch(app, /Topology\.setGroupCollapsed/, "trace group toggles reuse topology-core collapse mutation");
  expectNoMatch(toggleBody, /state\.topologyView\s*=|state\.topologyView\.groups\s*=/, "trace group toggles do not mutate architecture topology state");
  expectMatch(toggleBody, /fitTraceLayout\(\{ render: true \}\)/, "trace group toggles refit the expanded playback projection");
  expectMatch(renderGroupsBody, /addEventListener\("(?:click|keydown)"/, "trace group layer handles mouse and keyboard toggles");
  const groupLayerCss = css.match(/\.playback-view \.trace-group-layer\s*\{([^}]*)\}/)?.[1] || "";
  const groupFrameCss = css.match(/\.trace-group\s*\{([^}]*)\}/)?.[1] || "";
  const groupToggleCss = css.match(/\.trace-group-toggle\s*\{([^}]*)\}/)?.[1] || "";
  expectMatch(groupLayerCss, /\bz-index\s*:\s*4\b/, "transparent group layer keeps toggle buttons above link and node layers");
  expectMatch(groupLayerCss, /pointer-events\s*:\s*none/, "the raised group layer does not block node or link interaction outside its buttons");
  expectNoMatch(groupFrameCss, /\bz-index\s*:/, "group frames must not create a stacking context below their buttons");
  expectMatch(groupFrameCss, /background\s*:\s*transparent/, "raised group frames remain transparent instead of obscuring the graph");
  expectMatch(groupFrameCss, /pointer-events\s*:\s*none/, "group frames themselves do not intercept canvas gestures");
  expectMatch(groupToggleCss, /\bz-index\s*:\s*4\b/, "group toggle buttons stay clickable above trace links and nodes");
  expectMatch(groupToggleCss, /pointer-events\s*:\s*auto/, "group toggle buttons opt back into pointer interaction");
  expectMatch(groupToggleCss, /min-height/, "group toggle buttons have an obvious touch target");
  expectMatch(groupToggleCss, /border:\s*1px/, "group toggle buttons have a visible boundary");
  expectNoMatch(css, /\.playback-view \.trace-node-layer\s*\{[^}]*pointer-events\s*:\s*auto/s, "the full node layer must not intercept link hover targets");
  expectMatch(css, /\.playback-view \.trace-node-layer\s*\{[^}]*pointer-events\s*:\s*none/s, "blank node-layer space passes pointer input to links");
  expectMatch(css, /\.playback-view \.trace-node\s*\{[^}]*pointer-events\s*:\s*auto/s, "individual trace nodes remain draggable and selectable");
});

test("architecture and playback links use protocol styling and tooltips instead of permanent labels", () => {
  const createArchitectureLink = sourceBetween("createLinkElement", "topologyDisplayLinkId");
  const layoutMetrics = sourceBetween("topologyLayoutMetrics", "topologyPlacementSizes");
  const architectureLayout = sourceBetween("ensureNodePositions", "createTopologyNode");
  const renderArchitectureLinks = sourceBetween("renderLinks", "handleTopologyNodeClick");
  const bundleLayout = sourceBetween("appendTopologyBundle", "appendUnconnectedComponent");
  const presetFallbackLayout = sourceBetween("collisionSafeArchitectureTopologyView", "resetPlacementForArchitecturePreset");
  const buildTraceLayout = sourceBetween("buildTraceTopologyLayout", "traceTopologyLayout");
  const renderTrace = sourceBetween("renderTraceTopology", "traceDetailFact");

  expectMatch(layoutMetrics, /nodeGap:\s*Math\.max\([\s\S]*routeClearance \* 2 \+ 2\)/, "shared metrics reserve both endpoint clearance stubs");
  for (const [source, label] of [
    [architectureLayout, "architecture relayout"],
    [bundleLayout, "component bundle provisional layout"],
    [presetFallbackLayout, "architecture preset fallback layout"],
    [buildTraceLayout, "playback layout"],
  ]) expectMatch(source, /nodeGap:\s*metrics\.nodeGap/, `${label} uses the shared safe node gap`);
  expectNoMatch(buildTraceLayout, /nodeGap:\s*Math\.max/, "playback has no page-local node-gap exception");
  expectMatch(buildTraceLayout, /Topology\.collapseProjection\(/, "trace layout reuses topology-core collapseProjection");
  expectMatch(buildTraceLayout, /Topology\.layoutGraph\(/, "trace layout reuses topology-core layoutGraph");
  expectMatch(buildTraceLayout, /Topology\.planOrthogonalRoutes\(/, "trace layout reuses topology-core orthogonal routes");
  expectNoMatch(buildTraceLayout, /TraceView\.(?:directedTraceTopologyLayout|directedTopologyLayout|layoutDirectedTraceTopology|layoutTraceTopology|defaultTraceTopologyLayout)/, "trace layout no longer uses the old directed-only layout");
  expectNoMatch(buildTraceLayout, /collapsed:\s*false/, "trace layout must not force all groups expanded");
  expectNoMatch(buildTraceLayout, /Topology\.placeRouteLabels\(/, "trace layout does not plan permanent link labels");

  expectNoMatch(createArchitectureLink, /topology-link-label|svgElement\("text"/, "architecture link elements do not create permanent label text");
  expectNoMatch(renderArchitectureLinks, /topology-link-label|Topology\.placeRouteLabels|topologyLabelRect/, "architecture render path does not maintain permanent link labels");
  expectNoMatch(renderTrace, /trace-link-label|<text|Topology\.placeRouteLabels/, "trace render path does not create permanent link labels");
  expectNoMatch(html, /class="(?:topology|trace)-link-label"/, "architecture and playback pages do not include permanent link label elements");

  expectMatch(createArchitectureLink, /tabindex/, "architecture links are keyboard focusable");
  expectMatch(createArchitectureLink, /addEventListener\("keydown",[\s\S]*?\["Enter", " "\][\s\S]*?activate\(event\)/, "architecture links support keyboard activation");
  expectMatch(createArchitectureLink, /pointerenter/, "architecture links show a hover tooltip");
  expectMatch(createArchitectureLink, /focus/, "architecture links show a focus tooltip");
  expectMatch(renderArchitectureLinks, /topologyProtocolClass\(link\.protocol\)/, "architecture links use protocol styling");
  expectMatch(renderTrace, /topologyProtocolClass\(link\.protocol\)/, "trace links use protocol styling");
  expectMatch(renderTrace, /tabindex/, "trace links are keyboard focusable");
  expectMatch(renderTrace, /aria-label/, "trace links expose tooltip text to assistive tech");
  expectMatch(app, /topologyLinkTooltipText[\s\S]*protocol[\s\S]*bandwidth_gbps[\s\S]*aggregate_count/, "link tooltip text includes protocol, bandwidth, and aggregate count");
  expectMatch(app, /showLinkTooltip[\s\S]*hideLinkTooltip/, "shared tooltip show/hide helpers are present");
  expectMatch(app, /dom\.topologyLinkTooltip/, "architecture tooltip is wired");
  expectMatch(app, /dom\.traceLinkTooltip/, "trace tooltip is wired");
});

test("architecture and playback share a compact protocol legend with styled line classes", () => {
  for (const id of ["topologyProtocolLegend", "traceProtocolLegend"]) {
    expectMatch(html, new RegExp(`id="${id}"[^>]*role="list"`), `${id} is a compact role=list legend`);
    const legend = html.slice(html.indexOf(`id="${id}"`), html.indexOf("</div>", html.indexOf(`id="${id}"`)));
    expectMatch(legend, /protocol-legend-item/, `${id} has line swatches`);
    expectNoMatch(legend, /<p|<dl|<table/, `${id} is compact`);
  }
  for (const protocol of ["pcie", "cxl", "ucie", "nvlink", "roce", "infinityfabric", "hbm", "lpddr5x", "unknown"]) {
    expectMatch(html, new RegExp(`protocol-${protocol}`), `${protocol} appears in a legend swatch`);
    expectMatch(css, new RegExp(`protocol-${protocol}`), `${protocol} has CSS styling`);
  }
  expectMatch(css, /\.protocol-legend\s*\{[^}]*display:\s*(?:flex|grid)/s, "protocol legend has compact layout CSS");
  expectMatch(css, /\.protocol-legend-line\s*\{/, "protocol legend line swatches are styled");
  expectMatch(css, /(?:\.trace-link-path\.protocol-|:is\([^)]*\.trace-link-path[^)]*\)\[class\*="protocol-"\])/, "trace link protocols affect rendered paths");
  expectMatch(css, /(?:\.topology-link-line\.protocol-|:is\([^)]*\.topology-link-line[^)]*\)\[class\*="protocol-"\])/, "architecture link protocols affect rendered paths");
});

test("trace-only graph layout remains non-overlapping and routable across the supported 80%–200% font scale", () => {
  const ui = helpers();
  const hardware = {
    components: [
      component("hbm0", "hbm"), component("gpu0", "gpu"), component("fabric-switch-with-long-label", "high_io_ssd"),
      component("hbm1", "hbm"), component("gpu1", "gpu"), component("host0", "ssd"),
    ],
    links: [
      link("l0", "hbm0", "gpu0", "HBM3E"), link("l1", "gpu0", "fabric-switch-with-long-label"),
      link("l2", "hbm1", "gpu1", "HBM3E"), link("l3", "gpu1", "fabric-switch-with-long-label"),
      link("l4", "fabric-switch-with-long-label", "host0", "PCIe"),
    ],
  };
  for (const fontScale of [80, 120, 200]) {
    const layout = ui.buildTraceTopologyLayout(hardware, fontScale);
    for (let left = 0; left < layout.rects.length; left += 1) for (let right = left + 1; right < layout.rects.length; right += 1) {
      assert.equal(TopologyCore.rectsIntersect(layout.rects[left], layout.rects[right]), false, `${fontScale}% nodes overlap`);
    }
    for (const edge of hardware.links) {
      const route = layout.routes.get(edge.link_id);
      assert.ok(route?.points.length >= 2, `${fontScale}% ${edge.link_id} has an orthogonal route`);
      route.points.forEach((point, index) => {
        if (!index) return;
        for (const obstacle of layout.rects.filter((rect) => ![edge.source_component, edge.target_component].includes(rect.id))) {
          assert.equal(TopologyCore.segmentHitsRect(route.points[index - 1], point, obstacle), false, `${fontScale}% ${edge.link_id} crosses ${obstacle.id}`);
        }
      });
    }
    const content = layout.contentBounds;
    const contains = (rect) => rect.x >= content.x && rect.y >= content.y
      && rect.x + rect.width <= content.x + content.width
      && rect.y + rect.height <= content.y + content.height;
    layout.rects.forEach((rect) => assert.equal(contains(rect), true, `${fontScale}% content bounds include node ${rect.id}`));
    Array.from(layout.routes.values()).map((route) => TopologyCore.pointsBounds(route.points)).filter(Boolean)
      .forEach((rect) => assert.equal(contains(rect), true, `${fontScale}% content bounds include a routed link`));
    const shuffledHardware = { ...hardware, components: hardware.components.slice().reverse(), links: hardware.links.slice().reverse() };
    const shuffled = ui.buildTraceTopologyLayout(shuffledHardware, fontScale);
    const routePoints = (candidate) => Object.fromEntries(Array.from(candidate.routes.entries()).map(([id, route]) => [id, route.points]));
    assert.equal(JSON.stringify(routePoints(shuffled)), JSON.stringify(routePoints(layout)), `${fontScale}% routing is deterministic under shuffled input`);
  }
});

test("scrolling world and node coordinate rules are scoped to the playback trace canvas", () => {
  assert.match(html, /id="traceTopologyCanvas"[\s\S]*id="traceTopologyWorld"[\s\S]*id="traceLinkLayer"[\s\S]*id="traceNodeLayer"/);
  assert.match(css, /\.playback-view \.trace-topology-canvas\s*\{[^}]*overflow:\s*auto/s);
  assert.match(css, /\.playback-view \.trace-topology-world\s*\{/);
  assert.match(css, /\.playback-view \.trace-node\s*\{[^}]*position:\s*absolute/s);
  assert.doesNotMatch(css, /\.topology-canvas\s*\{[^}]*max-height:\s*min\(72vh/s, "architecture canvas must not inherit playback scrolling constraints");
});

test("trace playback uses selected-event language, collapsed semantic stream, and large-font cards", () => {
  assert.match(html, /id="traceNarrativePrimary"/);
  assert.match(html, /id="traceEventDetailTitle">选中事件/);
  assert.doesNotMatch(html, /SELECTED EVENT/);
  assert.doesNotMatch(html, /CURRENT EVENT|>当前事件</);
  assert.match(html, /<details class="trace-event-table-panel trace-event-stream" id="traceEventStreamDisclosure">/);
  assert.doesNotMatch(html, /id="traceEventStreamDisclosure"[^>]*\sopen(?:\s|>)/);
  assert.match(html, /<caption class="sr-only">[\s\S]*本地筛选不改变全局回放顺序/);
  assert.match(html, /<th scope="col">时序与持续条<\/th>/);
  assert.match(css, /\.trace-duration-bar\s*\{/);
  assert.match(css, /:root\[data-font-band="large"\] \.trace-event-table tr/);
  assert.match(css, /:root\[data-font-band="extreme"\] \.trace-event-table td\s*\{\s*grid-template-columns:\s*minmax\(0, 1fr\)/s);
  const responsiveHeaderRule = Array.from(css.matchAll(/([^{}]+)\{([^{}]*)\}/g)).find(([, selectors]) => (
    selectors.includes(':root[data-font-band="large"] .trace-event-table thead')
    && selectors.includes(':root[data-font-band="extreme"] .trace-event-table thead')
  ));
  assert.ok(responsiveHeaderRule, "large and extreme trace cards share one hidden-header contract");
  assert.match(responsiveHeaderRule[2], /position:\s*absolute\s*!important/);
  assert.match(responsiveHeaderRule[2], /width:\s*1px\s*!important/);
  assert.match(responsiveHeaderRule[2], /height:\s*1px\s*!important/);
  assert.match(responsiveHeaderRule[2], /visibility:\s*hidden\s*!important/);
  const traceTableRenderer = sourceBetween("renderTraceEventTable", "percentile");
  for (const accessibleLabel of [
    "时序：{value}",
    "语义事件：{subject}；{semantics}",
    "计算主体 / 数据流：{rank}；{flow}",
    "请求范围：{value}",
    "状态 / 详情：{value}",
  ]) {
    assert.match(traceTableRenderer, new RegExp(`aria-label="\\$\\{escapeHtml\\(uiText\\("${accessibleLabel.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`));
  }
  assert.doesNotMatch(css, /\.trace-event-table\s*\{[^}]*white-space:\s*nowrap/s);
});

test("semantic event pager IDs stay aligned across markup, DOM caching, and event binding", () => {
  for (const id of ["traceEventPreviousPageButton", "traceEventNextPageButton", "traceEventPageStatus"]) {
    assert.match(html, new RegExp(`id="${id}"`), `${id} exists in the playback markup`);
    assert.match(app, new RegExp(`"${id}"`), `${id} is cached before static events bind`);
  }
  assert.doesNotMatch(app, /"traceEventNextButton"/, "the obsolete pager ID must not interrupt bootstrap");
  assert.match(app, /dom\.traceEventNextPageButton\.addEventListener\("click"/);
});

test("a link tooltip uses its canvas as the coordinate frame on first reveal", () => {
  const ui = helpers();
  const values = new Map();
  const protocolTarget = {};
  const bandwidthTarget = {};
  const tooltip = {
    hidden: true,
    offsetParent: null,
    parentElement: {
      scrollLeft: 35,
      scrollTop: 45,
      getBoundingClientRect: () => ({ left: 200, top: 100 }),
    },
    querySelector(selector) {
      return selector.includes("protocol") ? protocolTarget : bandwidthTarget;
    },
    classList: { [Symbol.iterator]: function* iterator() {}, add() {}, remove() {} },
    setAttribute() {},
    style: { setProperty(name, value) { values.set(name, value); } },
  };
  ui.showLinkTooltip(tooltip, {
    _topologyLinkData: { protocol: "CXL", bandwidth_gbps: 64, aggregate_count: 2 },
  }, { clientX: 260, clientY: 180 });
  assert.equal(values.get("--link-tooltip-x"), "95px");
  assert.equal(values.get("--link-tooltip-y"), "125px");
  assert.equal(protocolTarget.textContent, "CXL");
  assert.match(bandwidthTarget.textContent, /8 GB\/s/);
  assert.equal(tooltip.hidden, false);
});

test("wall-clock playback advances one globally filtered event per second without speed state", () => {
  assert.doesNotMatch(html, /id="traceSpeed"|Playback Speed|回放速度/);
  assert.doesNotMatch(app, /tracePlayback:\s*\{[\s\S]{0,900}\bspeed\s*:/);
  assert.match(app, /const TRACE_PLAYBACK_STEP_MS = 1000/);
  assert.match(app, /setInterval\(traceAnimationStep, TRACE_PLAYBACK_STEP_MS\)/);
  const stepBody = app.slice(app.indexOf("function traceAnimationStep"), app.indexOf("function startTracePlayback"));
  assert.match(stepBody, /selectedIndex \+ 1/);
  assert.match(stepBody, /eventId: event\.event_id/);
  assert.match(stepBody, /loadTraceTaskPage\([\s\S]*resumePlayback: true/);
  assert.doesNotMatch(stepBody, /elapsedMs|1_000_000|playback\.speed/);
});

test("active routes keep continuous particles and compute-only events keep a node pulse", () => {
  const particleBody = app.slice(app.indexOf("function animateTraceParticles"), app.indexOf("function renderTraceTopology"));
  const topologyBody = app.slice(app.indexOf("function renderTraceTopology"), app.indexOf("function traceDetailFact"));
  assert.match(app, /const TRACE_PARTICLES_PER_ROUTE = 3/);
  assert.match(particleBody, /flatMap\(\(route, routeIndex\)/);
  assert.match(particleBody, /data-trace-route-index/);
  assert.match(particleBody, /data-trace-direction/);
  assert.match(particleBody, /traceParticleAnimationCapability\(\)\.mode === "static"/);
  assert.match(topologyBody, /const componentIds = new Set\(layout\.allComponents\.map/);
  assert.match(topologyBody, /const sourceNodeRoles = activeTraceNodeRoles\(events,\s*state\.tracePlayback\.timeNs,\s*componentIds\)/);
  assert.match(topologyBody, /const nodeRoles = projectTraceNodeRoles\([\s\S]*sourceNodeRoles,[\s\S]*layout\.components\.map[\s\S]*layout\.proxyFor/);
  assert.match(topologyBody, /renderTraceParticles\(activeRoutes\)/);
  assert.match(topologyBody, /dashDirections\.forEach\(\(direction\) => activeRoutes\.push/, "particle groups are deduplicated to one per active direction");
  assert.match(css, /\.trace-link-path\.is-active\s*\{[^}]*stroke-dasharray:\s*9 6;[^}]*animation:\s*trace-flow 720ms linear infinite/s);
  const protocolRuleIndex = css.indexOf(':is(.topology-link-line, .trace-link-path)[class*="protocol-"]');
  const activeRuleIndex = css.indexOf(".trace-link.is-active .trace-link-path,\n.trace-link-path.is-active", protocolRuleIndex);
  assert.ok(protocolRuleIndex >= 0, "protocol cascade rule exists");
  assert.ok(activeRuleIndex > protocolRuleIndex, "active dash override is declared after protocol styling");
  const activeRule = css.slice(activeRuleIndex, css.indexOf("}", activeRuleIndex) + 1);
  assert.match(activeRule, /stroke-dasharray:\s*9 6;/, "active links keep a visible dash even when the protocol dash is none");
  assert.match(activeRule, /animation:\s*trace-flow 720ms linear infinite;/, "active links keep the flow animation after protocol styling");
  const hoverRuleIndex = css.indexOf(":is(.topology-link:hover");
  assert.ok(hoverRuleIndex >= 0, "protocol hover/focus cascade rule exists");
  const hoverRule = css.slice(hoverRuleIndex, css.indexOf("}", hoverRuleIndex) + 1);
  assert.doesNotMatch(hoverRule, /\.trace-link\.is-active|\.is-active\)/, "protocol hover styling no longer overwrites active dash arrays");
  assert.match(css, /@keyframes trace-flow-reverse\s*\{[^}]*stroke-dashoffset:\s*30/);
  assert.match(css, /\.trace-link-path\.is-active\.is-reverse[^}]*animation-name:\s*trace-flow-reverse/s);
  assert.match(css, /\.trace-node\.is-active\s*\{[^}]*animation:\s*trace-node-pulse 980ms ease-in-out infinite/s);
  assert.match(css, /:root\[data-reduce-motion="true"\] :is\(\.trace-node, \.trace-link-path, \.trace-flow-particle\)[^}]*animation:\s*none !important/s);
  assert.match(css, /@media \(prefers-reduced-motion: reduce\)[\s\S]*:is\(\.trace-node, \.trace-link-path, \.trace-flow-particle\)\s*\{[^}]*animation:\s*none !important/s);
  assert.match(css, /\.trace-flow-particle\.is-instruction[^}]*background:\s*var\(--copper-bright\)/);
  assert.match(css, /\.trace-flow-particle\.is-data[^}]*background:\s*var\(--cyan-bright\)/);
  assert.match(app, /data-trace-transfer-kind=/);
  assert.match(app, /trace-particle-legend-dot is-instruction/);
  assert.match(app, /trace-particle-legend-dot is-data/);
  assert.match(topologyBody, /if \(active\) \{[\s\S]*particleGroups[\s\S]*dashDirections\.forEach\(\(direction\) => activeRoutes\.push/);
  assert.doesNotMatch(topologyBody, /dashDirections\.forEach\(\(direction\) => activeRoutes\.push\(traceParticleRouteForDirection\(route, direction\)\)\)/, "inactive links do not create decorative particles");
});

test("topology owns its toolbar and application fullscreen overlay", () => {
  const panel = html.slice(html.indexOf('id="traceTopologyPanel"'), html.indexOf('id="traceTopologyCanvas"'));
  for (const id of ["traceArrangeFitButton", "traceAutoLayoutButton", "traceFitButton", "traceZoomOutButton", "traceZoomInButton", "traceFullscreenButton"]) {
    assert.match(panel, new RegExp(`id="${id}"`));
  }
  assert.doesNotMatch(html, /traceFullscreenSurface/);
  assert.match(css, /\.playback-view \.trace-topology-panel\.is-fullscreen\s*\{[^}]*position:\s*fixed/s);
  assert.doesNotMatch(css, /trace-fullscreen-surface\.is-fullscreen/);
});

test("logical memory is compact in topology nodes and raw diagnostics are copy-only", () => {
  assert.doesNotMatch(html, /id="traceMemoryLayout"|id="traceMemoryTitle"/);
  assert.match(html, /id="traceDiagnosticCopyButton">复制诊断信息<\/button>/);
  assert.doesNotMatch(html, /完整规范化字段|trace-raw-details/);
  assert.doesNotMatch(app, /detail_limit=/);
  assert.doesNotMatch(app, /<pre><code>\$\{escapeHtml\(traceNormalizedJson/);
  assert.match(app, /data-trace-memory-summary/);
  assert.match(app, /逻辑地址（不是 JEDEC 物理地址）/);
});

test("trace nodes omit memory warning tooltips when no logical segments exist", () => {
  const ui = helpers();
  ui.state.tracePlayback.data = { memory_layout: { components: {} } };
  assert.equal(ui.traceNodeMemorySummary("gpu0"), "", "components without segments have no fabricated memory warning");
  ui.state.tracePlayback.data = {
    memory_layout: {
      components: {
        gpu0: [{ logical_id: "weights", offset_bytes: 0, length_bytes: 1024 }],
      },
    },
  };
  assert.match(ui.traceNodeMemorySummary("gpu0"), /逻辑地址（不是 JEDEC 物理地址）/);

  const renderBody = sourceBetween("renderTraceTopology", "traceDetailFact");
  assert.match(renderBody, /memoryAttribute = memorySummary \?/);
  assert.match(renderBody, /titleAttribute = title \?/);
  assert.match(css, /\.playback-view \.trace-node:is\(:hover, \.is-selected\)\[data-trace-memory-summary\]::after/);
  assert.doesNotMatch(renderBody, /没有声明逻辑地址区间/);
});


test("static exact and representative report events retain their backend fidelity in sidebar", () => {
  const ui = helpers();
  ui.state.tracePlayback.events = [{ event_id: "task" }];
  ui.state.tracePlayback.batchTraceIndex = [];
  ui.state.tracePlayback.mode = "aggregate";
  for (const [fidelity, label] of [["exact", "3 个精确事件"], ["representative", "3 个代表性事件"], ["aggregate", "3 个聚合事件"]]) {
    ui.state.tracePlayback.data = { total_events: 3, fidelity };
    assert.deepEqual({ ...ui.traceStepSummary() }, { count: 3, label });
  }
});

test("global previous/next controls update even with the semantic event stream collapsed", () => {
  const ui = helpers();
  const playback = ui.state.tracePlayback;
  playback.filteredEvents = [{ event_id: "first" }, { event_id: "last" }];
  playback.semanticStreamOpen = false;
  ui.dom.tracePreviousButton = { disabled: true };
  ui.dom.traceNextButton = { disabled: false };
  ui.setTraceSelectedIndex(1);
  assert.equal(ui.dom.tracePreviousButton.disabled, false);
  assert.equal(ui.dom.traceNextButton.disabled, true);
  ui.setTraceSelectedIndex(0);
  assert.equal(ui.dom.tracePreviousButton.disabled, true);
  assert.equal(ui.dom.traceNextButton.disabled, false);
  playback.page = { previous_offset: 0, has_more: true };
  ui.setTraceSelectedIndex(0);
  assert.equal(ui.dom.tracePreviousButton.disabled, false);
  ui.setTraceSelectedIndex(1);
  assert.equal(ui.dom.traceNextButton.disabled, false);
});


test("in-report static task traces are not described as missing aggregate-only playback", () => {
  const ui = helpers();
  ui.dom.tracePageBar = { classList: { toggle() {} } };
  ui.dom.tracePageStatus = { textContent: "" };
  const playback = ui.state.tracePlayback;
  playback.events = [{ event_id: "task0" }, { event_id: "task1" }];
  playback.data = { fidelity: "exact", total_events: 5 };
  playback.mode = "aggregate";
  ui.renderTracePageState();
  assert.match(ui.dom.tracePageStatus.textContent, /精确事件.*2 \/ 5/u);
  assert.doesNotMatch(ui.dom.tracePageStatus.textContent, /聚合回放|没有可读取/u);
});
