"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const TraceViewCore = require("../src/heterollm_sim/webui/trace-view-core.js");
const TopologyCore = require("../src/heterollm_sim/webui/topology-core.js");

function hardware() {
  return {
    components: [
      { component_id: "gpu0", kind: "gpu" },
      { component_id: "hbm0", kind: "hbm" },
      { component_id: "host0", kind: "host" },
    ],
    links: [
      { link_id: "l0", source_component: "hbm0", target_component: "gpu0", protocol: "HBM3E" },
      { link_id: "l1", source_component: "gpu0", target_component: "host0", protocol: "PCIe" },
    ],
  };
}

test("trace topology normalization uses directed playback defaults and does not mutate its source", () => {
  const input = {
    trace_topology_view: {
      viewport: { x: 12, y: -4, scale: 20 },
      layout: {
        positions: { gpu0: { x: 900, y: 40 }, stale: { x: 1, y: 1 } },
        node_sizes: { gpu0: { width: 210, height: 90 } },
      },
    },
  };
  const before = JSON.stringify(input);
  const { components, links } = hardware();
  const normalized = TraceViewCore.normalizeTraceTopologyView(input, components, links);
  const expected = TraceViewCore.directedTraceTopologyLayout(components, links, {
    nodeSizes: {
      gpu0: { width: 210, height: 90 },
      hbm0: { width: 130, height: 66 },
      host0: { width: 130, height: 66 },
    },
  });
  assert.deepEqual(normalized.layout.positions.hbm0, expected.positions.hbm0);
  assert.deepEqual(normalized.layout.positions.host0, expected.positions.host0);
  assert.deepEqual(normalized.layout.positions.gpu0, { x: 900, y: 40 });
  assert.equal(normalized.layout.positions.stale, undefined);
  assert.equal(normalized.viewport.scale, 4);
  assert.equal(normalized.layout.node_sizes.gpu0.width, 210);
  assert.equal(normalized.layout.algorithm, "trace-directed-layered-v2");
  assert.equal(JSON.stringify(input), before);
});

test("V4 trace topology accepts only canonical fields, signatures, and exports", () => {
  const { components, links } = hardware();
  const camelCaseWrapper = TraceViewCore.normalizeTraceTopologyView({
    traceTopologyView: { layout: { positions: { gpu0: { x: 999, y: 999 } } } },
  }, components, links);
  assert.notDeepEqual(camelCaseWrapper.layout.positions.gpu0, { x: 999, y: 999 });

  const topLevelAliases = TraceViewCore.normalizeTraceTopologyView({
    positions: { gpu0: { x: 888, y: 888 } },
    node_sizes: { gpu0: { width: 999, height: 999 } },
    camera: { x: 12, y: 13, zoom: 2 },
  }, components, links);
  assert.notDeepEqual(topLevelAliases.layout.positions.gpu0, { x: 888, y: 888 });
  assert.notEqual(topLevelAliases.layout.node_sizes.gpu0.width, 999);
  assert.deepEqual(topLevelAliases.viewport, { x: 0, y: 0, scale: 1 });

  const removedSecondArgumentOverload = TraceViewCore.normalizeTraceTopologyView({}, { components, links });
  assert.deepEqual(removedSecondArgumentOverload.components, []);
  const removedPositionMapOverload = TraceViewCore.normalizeTraceTopologyView({}, components, { gpu0: { x: 777, y: 777 } });
  assert.notDeepEqual(removedPositionMapOverload.layout.positions.gpu0, { x: 777, y: 777 });

  const removedFiveArgumentLayout = TraceViewCore.defaultTraceTopologyLayout(
    components,
    links,
    [],
    { gpu0: { width: 999, height: 999 } },
    {},
  );
  assert.notEqual(removedFiveArgumentLayout.node_sizes.gpu0.width, 999);
  for (const name of [
    "normalizeTopologyView", "normalizeTraceView", "directedTopologyLayout", "layoutDirectedTraceTopology",
    "layoutTraceTopology", "arrangeTraceTopology", "fitTraceTopology", "dragNode", "panViewport", "zoomViewport",
    "normalizeTraceMemoryLayout", "mergeMemoryLayout", "memorySegmentsByComponent", "normalizedMemoryComponents",
    "memoryNodeModels", "activeNodeLinkSets", "totalPolylineLength", "polylineLength", "buildParticles",
    "transferParticles", "particlePositions", "getAnimationCapability", "animationStrategy", "selectAnimationStrategy",
    "animationPlan", "handleFullscreenEscape", "computeFitViewport", "pointAtLoopProgress", "closeFullscreenOnEscape",
  ]) assert.equal(TraceViewCore[name], undefined, `${name} must not survive the V4 hard cutoff`);
});

test("arrange, fit, drag, pan, and zoom are immutable view-state helpers", () => {
  const { components, links } = hardware();
  const view = TraceViewCore.normalizeTraceTopologyView({}, components, links);
  const arranged = TraceViewCore.arrangeTraceTopologyView(view, components, links);
  assert.deepEqual(arranged.layout.positions, view.layout.positions);
  const manuallyMoved = TraceViewCore.dragTraceNode(view, "gpu0", { dx: 90, dy: -40 });
  const rearranged = TraceViewCore.arrangeTraceTopologyView(manuallyMoved, components, links);
  assert.deepEqual(rearranged.layout.positions, view.layout.positions, "arrange clears manual positions and restores the directed default");
  const dragged = TraceViewCore.dragTraceNode(view, "gpu0", { dx: 10, dy: 20 });
  assert.notDeepEqual(dragged.layout.positions.gpu0, view.layout.positions.gpu0);
  assert.deepEqual(view.layout.positions.gpu0, TraceViewCore.normalizeTraceTopologyView({}, components, links).layout.positions.gpu0);
  const panned = TraceViewCore.panTraceViewport(view, { dx: 30, dy: -5 });
  assert.deepEqual(panned.viewport, { x: 30, y: -5, scale: 1 });
  const zoomed = TraceViewCore.zoomTraceViewport(view, 2, { x: 100, y: 100 });
  assert.equal(zoomed.viewport.scale, 2);
  const fitted = TraceViewCore.fitTraceViewport(view, { width: 1280, height: 720 });
  assert.deepEqual(view.viewport, { x: 0, y: 0, scale: 1 });
  assert.notEqual(fitted, view);
});

test("fit viewport centers non-zero node bounds with requested padding", () => {
  const view = TraceViewCore.normalizeTraceTopologyView({
    layout: {
      positions: { a: { x: 100, y: 50 } },
      node_sizes: { a: { width: 200, height: 100 } },
      bounds: { x: 100, y: 50, width: 200, height: 100 },
    },
  }, [{ component_id: "a" }], []);
  const viewport = TraceViewCore.fitTraceViewport(view, { width: 500, height: 300 }, { padding: 50, minScale: 0.1, maxScale: 4 }).viewport;
  assert.deepEqual(viewport, { x: -150, y: -50, scale: 2 });
});

test("directed trace layout is deterministic, non-overlapping, and follows source to target", () => {
  const { components, links } = hardware();
  const options = {
    nodeSizes: {
      hbm0: { width: 110, height: 80 },
      gpu0: { width: 180, height: 64 },
      host0: { width: 140, height: 96 },
    },
    margin: 20,
    layerGap: 70,
    rowGap: 30,
  };
  const layout = TraceViewCore.directedTraceTopologyLayout(components, links, options);
  const shuffled = TraceViewCore.directedTraceTopologyLayout(
    [components[2], components[0], components[1]],
    [links[1], links[0]],
    options,
  );
  assert.deepEqual(shuffled.positions, layout.positions, "input ordering cannot change the layout");
  assert.deepEqual(layout.layers, [["hbm0"], ["gpu0"], ["host0"]]);
  assert.ok(layout.positions.hbm0.x < layout.positions.gpu0.x);
  assert.ok(layout.positions.gpu0.x < layout.positions.host0.x);
  const rects = Object.keys(layout.positions).map((id) => TopologyCore.rectForNode(id, layout.positions, layout.node_sizes));
  for (let left = 0; left < rects.length; left += 1) for (let right = left + 1; right < rects.length; right += 1) {
    assert.equal(TopologyCore.rectsIntersect(rects[left], rects[right]), false, `${rects[left].id} overlaps ${rects[right].id}`);
  }
});

test("directed trace layout collapses cycles into a documented shared layer", () => {
  const components = [
    { component_id: "a" },
    { component_id: "b" },
    { component_id: "c" },
  ];
  const links = [
    { link_id: "ab", source_component: "a", target_component: "b" },
    { link_id: "ba", source_component: "b", target_component: "a" },
    { link_id: "bc", source_component: "b", target_component: "c" },
  ];
  const layout = TraceViewCore.directedTraceTopologyLayout(components, links);
  assert.equal(layout.layerById.a, layout.layerById.b);
  assert.ok(layout.layerById.c > layout.layerById.b, "edges leaving a cycle still point forwards");
  assert.deepEqual(layout.cyclicComponentIds, ["a", "b"]);
  assert.deepEqual(layout.cyclicLinkIds, ["ab", "ba"]);
});

test("directed trace layers share a center axis, keep even gaps, and reduce branch crossings", () => {
  const components = ["s0", "s1", "t0", "t1"].map((component_id) => ({ component_id }));
  const links = [
    { link_id: "cross-0", source_component: "s0", target_component: "t1" },
    { link_id: "cross-1", source_component: "s1", target_component: "t0" },
  ];
  const nodeSizes = {
    s0: { width: 90, height: 50 },
    s1: { width: 130, height: 70 },
    t0: { width: 100, height: 60 },
    t1: { width: 150, height: 80 },
  };
  const layout = TraceViewCore.directedTraceTopologyLayout(components, links, {
    nodeSizes,
    rowGap: 36,
    minimumBounds: { width: 700, height: 500 },
  });
  const layerCenter = (members) => {
    const top = Math.min(...members.map((id) => layout.positions[id].y));
    const bottom = Math.max(...members.map((id) => layout.positions[id].y + nodeSizes[id].height));
    return (top + bottom) / 2;
  };
  assert.equal(layerCenter(layout.layers[0]), layerCenter(layout.layers[1]), "all layers share one horizontal center axis");
  layout.layers.forEach((members) => {
    for (let index = 1; index < members.length; index += 1) {
      const previous = members[index - 1];
      const current = members[index];
      assert.equal(layout.positions[current].y - (layout.positions[previous].y + nodeSizes[previous].height), 36);
    }
  });
  assert.ok(layout.positions.s1.y < layout.positions.s0.y, "barycentric ordering aligns s1 with t0 instead of crossing links");
  assert.ok(layout.positions.t0.y < layout.positions.t1.y);
  const shuffled = TraceViewCore.directedTraceTopologyLayout(components.slice().reverse(), links.slice().reverse(), {
    nodeSizes,
    rowGap: 36,
    minimumBounds: { width: 700, height: 500 },
  });
  assert.deepEqual(shuffled.positions, layout.positions);
});

test("memory segments merge by component while preserving logical offsets, fields, and fidelity", () => {
  const layout = TraceViewCore.normalizeMemoryLayout({
    schema_version: "1.0",
    components: {
      hbm1: [{ component_id: "hbm1", logical_id: "kv", offset_bytes: 4096, length_bytes: 128, future: { keep: true } }],
      hbm0: [{ component_id: "hbm0", logical_id: "w", offset_bytes: 0, length_bytes: 64 }],
    },
    component_totals_bytes: { hbm0: 256 },
  }, { fidelity: "aggregate" });
  assert.deepEqual(Object.keys(layout.components), ["hbm0", "hbm1"]);
  assert.equal(layout.components.hbm1[0].offset_bytes, 4096);
  assert.equal(layout.components.hbm1[0].future.keep, true);
  assert.equal(layout.component_models.hbm1.fidelity, "aggregate");
  assert.equal(layout.component_totals_bytes.hbm0, 256);
  assert.equal(layout.not_jedec_addressing, true);
  assert.match(layout.limitations[0], /JEDEC/);
  assert.equal(layout.components.hbm0[0].length_bytes, 64);
});

test("V4 memory layout ignores removed array, wrapper, and segment-field aliases", () => {
  assert.deepEqual(TraceViewCore.mergeMemorySegments([
    { component_id: "hbm0", logical_id: "w0", offset_bytes: 100, length_bytes: 20 },
  ]), {});
  assert.deepEqual(TraceViewCore.mergeMemorySegments({
    components: {
      hbm0: { segments: [{ component_id: "hbm0", logical_id: "w0", offset_bytes: 100, length_bytes: 20 }] },
    },
  }), {});
  assert.deepEqual(TraceViewCore.mergeMemorySegments({
    components: {
      hbm0: [
        { storage_component_id: "hbm0", logical_id: "w0", offset_bytes: 100, length_bytes: 20 },
        { component_id: "hbm0", logical_id: "w1", offset_bytes: 400, physical_bytes: 10 },
      ],
    },
  }), {});
  assert.deepEqual(TraceViewCore.mergeMemorySegments({ allocations: [
    { component_id: "hbm0", logical_id: "w0", offset_bytes: 100, length_bytes: 20 },
  ] }), {});
  assert.deepEqual(TraceViewCore.mergeMemorySegments({}), {});
});

test("orthogonal particle geometry supports multiple transfers and particles", () => {
  const points = [{ x: 0, y: 0 }, { x: 20, y: 0 }, { x: 20, y: 10 }];
  assert.equal(TraceViewCore.polylineTotalLength(points), 30);
  assert.deepEqual(TraceViewCore.pointAtProgress(points, 0), { x: 0, y: 0 });
  assert.deepEqual(TraceViewCore.pointAtProgress(points, 0.5), { x: 15, y: 0 });
  assert.deepEqual(TraceViewCore.pointAtProgress(points, 1), { x: 20, y: 10 });
  const particles = TraceViewCore.buildTransferParticles([
    { transfer_id: "t0", link_id: "l0", progress: 0.5 },
    { transfer_id: "t1", link_id: "l1", progress: 0.25, particle_count: 2 },
  ], {
    l0: { points },
    l1: { points: [{ x: 5, y: 5 }, { x: 5, y: 15 }] },
  });
  assert.equal(particles.length, 3);
  assert.deepEqual(particles[0].point, { x: 15, y: 0 });
  assert.ok(particles.every((particle) => particle.total_length > 0));
});

test("particle loop progress wraps RAF and interval clocks instead of clamping at the endpoint", () => {
  const points = [{ x: 0, y: 0 }, { x: 20, y: 0 }];
  assert.equal(TraceViewCore.loopProgress(1), 0);
  assert.equal(TraceViewCore.loopProgress(2.25), 0.25);
  assert.equal(TraceViewCore.loopProgress(-0.25), 0.75);
  assert.equal(TraceViewCore.particleLoopProgress(600, { periodMs: 1200 }), 0.5, "RAF clock");
  assert.equal(TraceViewCore.particleLoopProgress(1_700_000_000_000, { periodMs: 1200 }), 2 / 3, "Date.now interval clock");
  assert.equal(TraceViewCore.particleLoopProgress(2400, 1200), 0, "the next cycle restarts at the source");
  assert.deepEqual(TraceViewCore.pointAtProgress(points, TraceViewCore.loopProgress(1.25)), { x: 5, y: 0 });
});

test("active node and link sets honor half-open intervals and aggregate hop semantics", () => {
  const events = [{
    event_id: "e0",
    start_ns: 10,
    end_ns: 20,
    rank: { component_id: "gpu0" },
    transfer: {
      source_component: "hbm0",
      target_component: "gpu0",
      hops: [{ link_id: "l0", start_ns: 10, end_ns: 15 }],
    },
    resources: [{ resource_id: "gpu0.compute", start_ns: 10, end_ns: 20 }],
  }];
  const active = TraceViewCore.activeTraceSets(events, 12, ["hbm0", "gpu0"]);
  assert.deepEqual(Array.from(active.nodeIds).sort(), ["gpu0", "hbm0"]);
  assert.deepEqual(Array.from(active.linkIds).sort(), ["gpu0.compute", "l0"]);
  assert.deepEqual(TraceViewCore.activeTraceSets(events, 20).activeEvents, []);
});

test("animation capability falls back from requestAnimationFrame to interval to static", () => {
  assert.equal(TraceViewCore.animationCapability({ requestAnimationFrame() {}, setInterval() {} }).mode, "raf");
  assert.equal(TraceViewCore.animationCapability({ setInterval() {} }).mode, "interval");
  assert.equal(TraceViewCore.animationCapability({}).mode, "static");
  const reduced = TraceViewCore.animationCapability({ requestAnimationFrame() {}, setInterval() {} }, { reduceMotion: true });
  assert.equal(reduced.mode, "static");
  assert.equal(reduced.reducedMotion, true);
});

test("pointer drag state computes zoomed positions and clears on lost capture in Chrome 108", () => {
  const drag = TraceViewCore.beginTracePointerDrag(
    { type: "pointerdown", pointerId: 7, clientX: 100, clientY: 80 },
    "gpu0",
    { x: 300, y: 160 },
  );
  assert.deepEqual(
    TraceViewCore.tracePointerDragPosition(drag, { type: "pointermove", pointerId: 7, clientX: 140, clientY: 100 }, 2),
    { x: 320, y: 170 },
  );
  assert.equal(
    TraceViewCore.tracePointerDragPosition(drag, { type: "pointermove", pointerId: 8, clientX: 140, clientY: 100 }, 2),
    null,
    "another pointer cannot move this drag",
  );
  assert.equal(TraceViewCore.finishTracePointerDrag(drag, { type: "pointerup", pointerId: 8 }), drag);
  assert.equal(TraceViewCore.finishTracePointerDrag(drag, { type: "pointercancel", pointerId: 7 }), null);
  assert.equal(TraceViewCore.finishTracePointerDrag(drag, { type: "lostpointercapture", pointerId: 7 }), null);
});

test("fullscreen overlay state closes only on Escape and node text remains component_id", () => {
  const opened = TraceViewCore.toggleFullscreen({ open: false }, true, { previousFocusId: "trace-button" });
  assert.equal(opened.open, true);
  assert.equal(opened.previousFocusId, "trace-button");
  assert.equal(TraceViewCore.fullscreenEscapeAction(opened, "Enter").state.open, true);
  const closed = TraceViewCore.fullscreenEscapeAction(opened, { key: "Escape" }).state;
  assert.equal(closed.open, false);
  const node = TraceViewCore.nodePresentation({ component_id: "gpu0", kind: "GPU / 计算" });
  assert.equal(node.text, "gpu0");
  assert.match(node.ariaLabel, /GPU/);
  assert.equal(node.title, "GPU / 计算");
});

test("duration-bar temporal states use half-open event intervals independently of selection", () => {
  const event = { start_ns: 10, end_ns: 20 };
  assert.equal(TraceViewCore.traceEventTemporalState(event, 9), "future");
  assert.equal(TraceViewCore.traceEventTemporalState(event, 10), "current");
  assert.equal(TraceViewCore.traceEventTemporalState(event, 19), "current");
  assert.equal(TraceViewCore.traceEventTemporalState(event, 20), "past");
  assert.equal(TraceViewCore.traceEventTemporalState({ start_ns: 20, end_ns: 20 }, 20), "current");
  assert.equal(TraceViewCore.traceEventTemporalState({ start_ns: 20, end_ns: 20 }, 21), "past");
});

test("semantic event filters are local, relevance-ranked for search, and paginated to at most 50", () => {
  const events = Array.from({ length: 125 }, (_, index) => ({
    event_id: `event-${index}`,
    task_id: index === 70 ? "needle" : `task-${index}`,
    category: index % 2 ? "compute" : "communication",
    phase: index < 80 ? "prefill" : "decode",
    start_ns: index,
    end_ns: index + 2,
  }));
  const before = JSON.stringify(events);
  const filtered = TraceViewCore.filterSemanticTraceEvents(events, { category: "compute", phase: "prefill", temporal: "past" }, 100);
  assert.ok(filtered.every((event) => event.category === "compute" && event.phase === "prefill" && event.end_ns <= 100));
  const searched = TraceViewCore.filterSemanticTraceEvents(events, { query: "needle" }, 0);
  assert.equal(searched[0].event_id, "event-70");
  const first = TraceViewCore.paginateSemanticTraceEvents(events, 0, 500);
  const third = TraceViewCore.paginateSemanticTraceEvents(events, 2, 50);
  assert.equal(first.events.length, 50);
  assert.equal(first.pageSize, 50);
  assert.equal(third.events.length, 25);
  assert.equal(JSON.stringify(events), before, "local stream operations cannot mutate the playback event list");
});

test("trace topology view clones group collapse state for playback-local controls", () => {
  const components = [
    { component_id: "gpu0", kind: "gpu" },
    { component_id: "hbm0", kind: "hbm" },
    { component_id: "fabric0", kind: "fabric_switch" },
  ];
  const links = [
    { link_id: "gpu-hbm", source_component: "gpu0", target_component: "hbm0", protocol: "HBM" },
    { link_id: "hbm-fabric", source_component: "hbm0", target_component: "fabric0", protocol: "NVLink" },
  ];
  const source = {
    trace_topology_view: {
      groups: [{ group_id: "gpu0-package", members: ["gpu0", "hbm0"], root: "gpu0", collapsed: true }],
      layout: {
        positions: {
          gpu0: { x: 100, y: 80 },
          hbm0: { x: 260, y: 80 },
          fabric0: { x: 420, y: 80 },
        },
      },
    },
  };
  const before = JSON.stringify(source);
  const normalized = TraceViewCore.normalizeTraceTopologyView(source, components, links);
  assert.equal(normalized.groups[0].collapsed, true);
  normalized.groups[0].collapsed = false;
  assert.equal(source.trace_topology_view.groups[0].collapsed, true, "normalized playback groups cannot mutate source report groups");
  const arranged = TraceViewCore.arrangeTraceTopologyView(source, components, links);
  assert.equal(arranged.groups[0].collapsed, true, "arrange preserves the playback group's collapsed state");
  assert.equal(JSON.stringify(source), before);
});
