"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { performance } = require("node:perf_hooks");
const Core = require("../src/heterollm_sim/webui/topology-core.js");

function component(id, ports = [], extras = {}) {
  return { component_id: id, kind: "gpu", ports, ...extras };
}

function link(id, source, target, extras = {}) {
  return { link_id: id, source_component: source, source_port: `${source}-p`, target_component: target, target_port: `${target}-p`, protocol: "UCIe", bandwidth_gbps: 10, bidirectional: true, ...extras };
}

function orthogonalSegments(points) {
  return points.slice(1).map((point, index) => ({
    x1: points[index].x,
    y1: points[index].y,
    x2: point.x,
    y2: point.y,
  }));
}

function overlapLength(first, second) {
  if (first.x1 === first.x2 && second.x1 === second.x2 && first.x1 === second.x1) {
    return Math.max(0, Math.min(Math.max(first.y1, first.y2), Math.max(second.y1, second.y2)) - Math.max(Math.min(first.y1, first.y2), Math.min(second.y1, second.y2)));
  }
  if (first.y1 === first.y2 && second.y1 === second.y2 && first.y1 === second.y1) {
    return Math.max(0, Math.min(Math.max(first.x1, first.x2), Math.max(second.x1, second.x2)) - Math.max(Math.min(first.x1, first.x2), Math.min(second.x1, second.x2)));
  }
  return 0;
}

function crosses(first, second) {
  const horizontal = first.y1 === first.y2 ? first : second.y1 === second.y2 ? second : null;
  const vertical = first.x1 === first.x2 ? first : second.x1 === second.x2 ? second : null;
  if (!horizontal || !vertical || horizontal === vertical) return false;
  return vertical.x1 > Math.min(horizontal.x1, horizontal.x2)
    && vertical.x1 < Math.max(horizontal.x1, horizontal.x2)
    && horizontal.y1 > Math.min(vertical.y1, vertical.y2)
    && horizontal.y1 < Math.max(vertical.y1, vertical.y2);
}

test("topology_view uses only V4 layout positions and enforces flat groups with member roots", () => {
  const view = Core.normalizeTopologyView({ layout: { positions: { a: { x: 12, y: 34 } } }, groups: [
    { group_id: "g", members: ["a", "b"], root: "missing" },
    { group_id: "overlap", members: ["b", "c"], root: "c" },
  ] }, ["a", "b", "c"]);
  assert.equal(view.version, 1);
  assert.deepEqual(view.layout.positions.a, { x: 12, y: 34 });
  assert.deepEqual(view.groups[0].members, ["a", "b"]);
  assert.equal(view.groups[0].root, "a");
  assert.deepEqual(view.groups[1].members, ["c"]);
  assert.deepEqual(Core.validateGroups(view.groups, ["a", "b", "c"]), []);
  assert.throws(() => Core.createGroup(view, ["b", "c"]), /已经属于另一个分组/);
  assert.throws(() => Core.setGroupRoot(view, "g", "c"), /必须是当前分组的成员/);
});

test("screen/world transforms and marquee selection remain correct after pan and zoom", () => {
  const viewport = { x: 80, y: -20, scale: 2 };
  const world = Core.screenToWorld({ x: 280, y: 180 }, viewport);
  assert.deepEqual(world, { x: 100, y: 100 });
  assert.deepEqual(Core.worldToScreen(world, viewport), { x: 280, y: 180 });
  const selected = Core.marqueeSelection(
    { x: 70, y: -30 },
    { x: 310, y: 210 },
    viewport,
    { a: { x: 0, y: 0, width: 30, height: 30 }, b: { x: 90, y: 90, width: 30, height: 30 }, c: { x: 300, y: 300, width: 20, height: 20 } },
  );
  assert.deepEqual(selected, ["a", "b"]);
});

test("negative world coordinates are normalized without moving their screen position", () => {
  const viewport = { x: 120, y: 80, scale: 1.5 };
  const positions = { a: { x: -90, y: -30 }, b: { x: 50, y: 70 } };
  const before = Core.worldToScreen(positions.a, viewport);
  const normalized = Core.normalizeWorldOrigin(positions, { x: -150, y: -100, width: 700, height: 500 }, viewport);
  assert.deepEqual(normalized.bounds, { x: 0, y: 0, width: 700, height: 500 });
  assert.deepEqual(normalized.shift, { x: 150, y: 100 });
  assert.deepEqual(Core.worldToScreen(normalized.positions.a, normalized.viewport), before);
});

test("collapse hides members/internal edges, aggregates proxies, and preserves directed semantics", () => {
  const components = [component("root"), component("member"), component("outside")];
  const links = [
    link("internal", "root", "member"),
    link("out-a", "member", "outside", { bidirectional: false }),
    link("out-b", "outside", "member", { bidirectional: false }),
    link("out-c", "member", "outside", { bidirectional: false }),
  ];
  const projection = Core.collapseProjection(components, links, [{ group_id: "g", members: ["root", "member"], root: "root", collapsed: true }]);
  assert.deepEqual(projection.visibleComponentIds.sort(), ["outside", "root"]);
  assert.equal(projection.links.length, 2, "opposite directed links must not merge");
  assert.equal(new Set(projection.links.map((item) => item.display_id)).size, 2, "opposite projected directions need unique DOM and route identities");
  assert.ok(projection.links.every((item) => item.display_id.startsWith("proxy:directed:")));
  const forward = projection.links.find((item) => item.source_component === "root");
  assert.equal(forward.aggregate_count, 2);
  assert.deepEqual(forward.original_link_ids, ["out-a", "out-c"]);
  assert.equal(projection.links.some((item) => item.original_link_ids.includes("internal")), false);
  const plan = Core.planOrthogonalRoutes(projection.links, [
    { id: "root", x: 0, y: 0, width: 90, height: 90 },
    { id: "outside", x: 420, y: 0, width: 90, height: 90 },
  ], { clearance: 12 });
  assert.equal(plan.errors.length, 0);
  assert.deepEqual(plan.routes.map((route) => route.linkId).sort(), projection.links.map((item) => item.display_id).sort());
});

test("collapsed bidirectional links aggregate without endpoint ordering", () => {
  const components = [component("root"), component("member"), component("outside")];
  const projection = Core.collapseProjection(components, [
    link("forward", "member", "outside"),
    link("reverse", "outside", "member"),
  ], [{ group_id: "g", members: ["root", "member"], root: "root", collapsed: true }]);
  assert.equal(projection.links.length, 1);
  assert.equal(projection.links[0].aggregate_count, 2);
  assert.deepEqual(projection.links[0].original_link_ids, ["forward", "reverse"]);
  assert.match(projection.links[0].display_id, /^proxy:bidirectional:/);
});

test("GB200 NVL4-style default GPU-HBM group folding routes the initial canvas cleanly", () => {
  const components = [];
  const links = [];
  const groups = [];
  const positions = {};
  const superchips = 2;
  for (let module = 0; module < superchips; module += 1) {
    const baseX = module * 1400;
    const baseY = 280;
    const grace = `grace${module}`;
    const lpddr = `lpddr${module}`;
    components.push({ ...component(grace), kind: "cpu" }, { ...component(lpddr), kind: "host_memory" });
    links.push(link(`lpddr${module}`, grace, lpddr, { protocol: "LPDDR5X" }));
    groups.push({ group_id: `${grace}_lpddr`, members: [grace, lpddr], root: grace, collapsed: false });
    positions[grace] = { x: baseX + 460, y: baseY };
    positions[lpddr] = { x: baseX + 680, y: baseY };

    for (let local = 0; local < 2; local += 1) {
      const index = module * 2 + local;
      const gpu = `gpu${index}`;
      const gpuBaseX = baseX + local * 650;
      const hbmIds = Array.from({ length: 8 }, (_, hbmIndex) => `hbm${index}_${hbmIndex}`);
      components.push({ ...component(gpu), kind: "gpu" });
      hbmIds.forEach((hbmId) => components.push({ ...component(hbmId), kind: "hbm" }));
      links.push(link(`c2c${index}`, grace, gpu, { protocol: "NVLink-C2C" }));
      links.push(link(`fabric${index}`, gpu, "fabric0", { protocol: "NVLink" }));
      hbmIds.forEach((hbmId, hbmIndex) => links.push(link(`hbm${index}_${hbmIndex}`, gpu, hbmId, { protocol: "HBM" })));
      groups.push({ group_id: `${gpu}_hbm`, members: [gpu, ...hbmIds], root: gpu, collapsed: true });
      positions[gpu] = { x: gpuBaseX, y: baseY + 150 };
      hbmIds.forEach((hbmId, hbmIndex) => {
        positions[hbmId] = {
          x: gpuBaseX + (hbmIndex % 4) * 150,
          y: baseY + 270 + Math.floor(hbmIndex / 4) * 90,
        };
      });
    }
  }
  components.push({ ...component("fabric0"), kind: "fabric_switch" });
  groups.push({ group_id: "fabric", members: ["fabric0"], root: "fabric0", collapsed: true });
  positions.fabric0 = { x: 280, y: 20 };

  const projection = Core.collapseProjection(components, links, groups);
  assert.equal(projection.visibleComponentIds.length, 9);
  assert.deepEqual(
    projection.visibleComponentIds.filter((id) => id.startsWith("hbm")),
    [],
    "physical HBM nodes stay in the simulation JSON but are hidden by default groups",
  );
  const sizes = Object.fromEntries(projection.visibleComponentIds.map((id) => [id, { width: 130, height: 66 }]));
  const rects = projection.visibleComponentIds.map((id) => Core.rectForNode(id, positions, sizes));
  for (let left = 0; left < rects.length; left += 1) for (let right = left + 1; right < rects.length; right += 1) {
    assert.equal(Core.rectsIntersect(rects[left], rects[right]), false, `${rects[left].id} overlaps ${rects[right].id}`);
  }

  const parallel = new Map();
  projection.links.forEach((edge) => {
    const key = [edge.source_component, edge.target_component].sort().join("\u001f");
    if (!parallel.has(key)) parallel.set(key, []);
    parallel.get(key).push(edge);
  });
  projection.links.forEach((edge) => {
    const source = Core.rectForNode(edge.source_component, positions, sizes);
    const target = Core.rectForNode(edge.target_component, positions, sizes);
    const peers = parallel.get([edge.source_component, edge.target_component].sort().join("\u001f"));
    const channel = (peers.indexOf(edge) - (peers.length - 1) / 2) * 18;
    const route = Core.routeOrthogonal(source, target, rects, { clearance: 14, channelOffset: channel });
    route.forEach((point, index) => {
      if (!index) return;
      rects.filter((rect) => rect.id !== source.id && rect.id !== target.id).forEach((obstacle) => {
        assert.equal(Core.segmentHitsRect(route[index - 1], point, obstacle), false, `${edge.link_id} crosses ${obstacle.id}`);
      });
    });
  });
});

test("copy/paste keeps only internal topology and remaps component, port, link, and group IDs", () => {
  const hardware = {
    components: [
      component("a", [{ port_id: "pa", protocol: "UCIe" }]),
      component("b", [{ port_id: "pb", protocol: "UCIe" }], {
        kind: "hbm",
        metadata: {
          physical_composition: {
            controller_component_id: "a",
            memory_subsystem_id: "a",
            source_basis: "a",
          },
        },
      }),
      component("outside", [{ port_id: "po", protocol: "UCIe" }]),
    ],
    links: [link("inside", "a", "b", { source_port: "pa", target_port: "pb" }), link("external", "a", "outside", { source_port: "pa", target_port: "po" })],
    metadata: {},
  };
  const view = Core.normalizeTopologyView({
    groups: [{ group_id: "pair", label: "pair", members: ["a", "b"], root: "b", collapsed: true }],
    layout: { positions: { a: { x: 100, y: 200 }, b: { x: 260, y: 240 }, outside: { x: 500, y: 0 } } },
  }, ["a", "b", "outside"]);
  const payload = Core.copySelection(hardware, view, ["a", "b"]);
  assert.deepEqual(payload.origin, { x: 100, y: 200 }, "origin must be the selection's positive-coordinate top-left");
  assert.deepEqual(payload.links.map((item) => item.link_id), ["inside"]);
  assert.equal(Object.hasOwn(payload, "placement"), false);
  const pasted = Core.pasteSelection(hardware, view, payload, { x: 700, y: 600 });
  assert.equal(pasted.pastedIds.length, 2);
  assert.equal(new Set(pasted.pastedIds).size, 2);
  assert.equal(pasted.hardware.links.length, 3);
  const newLink = pasted.hardware.links.at(-1);
  assert.equal(newLink.link_id, "inside1");
  assert.equal(newLink.source_port, "pa1");
  assert.equal(newLink.target_port, "pb1");
  const pastedMemory = pasted.hardware.components.find((item) => item.component_id === pasted.componentMap.b);
  assert.equal(pastedMemory.metadata.physical_composition.controller_component_id, pasted.componentMap.a);
  assert.equal(pastedMemory.metadata.physical_composition.memory_subsystem_id, pasted.componentMap.a);
  assert.equal(pastedMemory.metadata.physical_composition.source_basis, "a", "ordinary physical metadata text is not rewritten");
  const newGroup = pasted.topologyView.groups.at(-1);
  assert.equal(newGroup.group_id, "pair1");
  assert.equal(newGroup.label, "pair1");
  assert.equal(newGroup.collapsed, false);
  assert.ok(newGroup.members.includes(newGroup.root));
  const firstPosition = pasted.topologyView.layout.positions[pasted.componentMap.a];
  assert.deepEqual(firstPosition, { x: 700, y: 600 });

  const pastedAgain = Core.pasteSelection(pasted.hardware, pasted.topologyView, payload, { x: 900, y: 800 });
  assert.deepEqual(pastedAgain.pastedIds, ["a2", "b2"]);
  assert.equal(pastedAgain.hardware.links.at(-1).link_id, "inside2");
  assert.deepEqual(pastedAgain.portMap, { a: { pa: "pa2" }, b: { pb: "pb2" } });
  assert.equal(pastedAgain.topologyView.groups.at(-1).group_id, "pair2");
  assert.equal(pastedAgain.topologyView.groups.at(-1).label, "pair2");
});

test("group copy/paste remaps physical HBM owner metadata after all component IDs are known", () => {
  const hardware = {
    components: [
      component("hbm0", [{ port_id: "host", protocol: "HBM" }], {
        kind: "hbm",
        metadata: {
          physical_composition: {
            controller_component_id: "gpu0",
            memory_subsystem_id: "gpu0",
            source_basis: "gpu0",
            derived_namespace: "gpu0-memory",
          },
        },
      }),
      component("gpu0", [{ port_id: "hbm", protocol: "HBM" }]),
    ],
    links: [link("hbm_link0", "gpu0", "hbm0", { source_port: "hbm", target_port: "host" })],
    metadata: {},
  };
  const view = Core.normalizeTopologyView({
    groups: [{ group_id: "module0", label: "Module", members: ["gpu0", "hbm0"], root: "gpu0", collapsed: true }],
    layout: { positions: { gpu0: { x: 100, y: 100 }, hbm0: { x: 220, y: 100 } } },
  }, ["gpu0", "hbm0"]);

  const payload = Core.copySelection(hardware, view, [], "module0");
  assert.deepEqual(Array.from(payload.components, (item) => item.component_id), ["hbm0", "gpu0"], "copy preserves hardware order");
  const pasted = Core.pasteSelection(hardware, view, payload, { x: 500, y: 400 });
  assert.equal(pasted.componentMap.gpu0, "gpu1");
  assert.equal(pasted.componentMap.hbm0, "hbm1");
  const pastedMemory = pasted.hardware.components.find((item) => item.component_id === "hbm1");
  const physical = pastedMemory.metadata.physical_composition;
  assert.equal(physical.controller_component_id, "gpu1");
  assert.equal(physical.memory_subsystem_id, "gpu1");
  assert.equal(physical.source_basis, "gpu0");
  assert.equal(physical.derived_namespace, "gpu0-memory");
  assert.equal(pasted.topologyView.groups.at(-1).root, "gpu1");
});

test("paste naming increments numeric suffixes and never emits copy labels", () => {
  const hardware = {
    components: [
      component("hbm0", [{ port_id: "host", protocol: "UCIe" }]),
      component("hbm1", [{ port_id: "host", protocol: "UCIe" }]),
    ],
    links: [],
    metadata: {},
  };
  const view = Core.normalizeTopologyView({
    groups: [{ group_id: "group0", label: "Memory group0", members: ["hbm0"], root: "hbm0" }],
    layout: { positions: { hbm0: { x: 10, y: 20 }, hbm1: { x: 200, y: 20 } } },
  }, ["hbm0", "hbm1"]);
  const payload = Core.copySelection(hardware, view, ["hbm0"]);

  const first = Core.pasteSelection(hardware, view, payload, { x: 400, y: 300 });
  assert.equal(first.componentMap.hbm0, "hbm2");
  assert.equal(first.portMap.hbm0.host, "host1");
  assert.equal(first.topologyView.groups.at(-1).group_id, "group1");
  assert.equal(first.topologyView.groups.at(-1).label, "Memory group1");

  const second = Core.pasteSelection(first.hardware, first.topologyView, payload, { x: 500, y: 300 });
  assert.equal(second.componentMap.hbm0, "hbm3");
  assert.equal(second.portMap.hbm0.host, "host2");
  assert.equal(second.topologyView.groups.at(-1).group_id, "group2");
  assert.equal(second.topologyView.groups.at(-1).label, "Memory group2");

  const generated = [
    ...second.pastedIds,
    ...Object.values(second.portMap).flatMap((ports) => Object.values(ports)),
    second.topologyView.groups.at(-1).group_id,
    second.topologyView.groups.at(-1).label,
  ];
  assert.ok(generated.every((value) => !/copy/i.test(value)));
});

test("external clipboard validation rejects duplicate and dangling references atomically", () => {
  const hardware = { components: [component("existing", [{ port_id: "p", protocol: "UCIe" }])], links: [], metadata: {} };
  const view = Core.normalizeTopologyView({ layout: { positions: { existing: { x: 0, y: 0 } } } }, ["existing"]);
  const invalid = {
    type: Core.CLIPBOARD_TYPE,
    version: 1,
    components: [component("a", [{ port_id: "pa" }]), component("a", [{ port_id: "pb" }])],
    links: [link("bad", "a", "missing", { source_port: "pa", target_port: "nope" })],
    groups: [],
    positions: { a: { x: 0, y: 0 } },
    origin: { x: 0, y: 0 },
  };
  const beforeHardware = structuredClone(hardware);
  const beforeView = structuredClone(view);
  assert.throws(() => Core.pasteSelection(hardware, view, invalid), /组件 ID 缺失或重复/);
  assert.deepEqual(hardware, beforeHardware);
  assert.deepEqual(view, beforeView);

  const dangling = Core.copySelection({ components: [component("a", [{ port_id: "pa" }])], links: [] }, Core.normalizeTopologyView({ layout: { positions: { a: { x: 1, y: 2 } } } }, ["a"]), ["a"]);
  dangling.links.push(link("dangling", "a", "a", { source_port: "missing", target_port: "pa" }));
  assert.throws(() => Core.validateClipboardPayload(dangling), /未知的源端口/);
  assert.equal(Core.parseClipboardText(JSON.stringify(dangling)), null);
});

test("collision-aware placement moves a selection as one rigid group to the nearest deterministic free position", () => {
  const positions = { a: { x: 0, y: 0 }, b: { x: 80, y: 0 }, obstacle: { x: 220, y: 0 } };
  const sizes = { a: { width: 60, height: 60 }, b: { width: 60, height: 60 }, obstacle: { width: 80, height: 80 } };
  const desired = { a: { x: 170, y: 0 }, b: { x: 250, y: 0 } };
  const first = Core.resolveCollisionPlacement(positions, sizes, desired, { gap: 10, step: 20, maxRings: 20 });
  const second = Core.resolveCollisionPlacement(positions, sizes, desired, { gap: 10, step: 20, maxRings: 20 });
  assert.deepEqual(first, second);
  assert.equal(first.adjusted, true);
  assert.equal(first.positions.b.x - first.positions.a.x, 80);
  assert.equal(first.positions.b.y - first.positions.a.y, 0);
  const obstacleRect = Core.rectForNode("obstacle", positions, sizes);
  for (const id of ["a", "b"]) assert.equal(Core.rectsIntersect(Core.rectForNode(id, first.positions, sizes), obstacleRect), false);
});

test("deterministic group-aware layout produces non-overlapping node boxes", () => {
  const components = ["a", "b", "c", "d", "e"].map((id) => component(id));
  const links = [link("ab", "a", "b"), link("bc", "b", "c"), link("cd", "c", "d"), link("de", "d", "e")];
  const groups = [{ group_id: "g", members: ["b", "c"], root: "b", collapsed: false }];
  const sizes = { a: { width: 180, height: 90 }, b: { width: 210, height: 100 }, c: { width: 160, height: 140 }, d: { width: 190, height: 80 }, e: { width: 150, height: 70 } };
  const first = Core.layoutGraph(components, links, groups, sizes);
  const second = Core.layoutGraph(components, links, groups, sizes);
  assert.deepEqual(first, second);
  const rects = components.map((item) => Core.rectForNode(item.component_id, first.positions, sizes));
  for (let left = 0; left < rects.length; left += 1) for (let right = left + 1; right < rects.length; right += 1) {
    assert.equal(Core.rectsIntersect(rects[left], rects[right]), false, `${components[left].component_id} overlaps ${components[right].component_id}`);
  }
  assert.ok(first.groupBounds.g.width > sizes.b.width);
  for (const edge of links) {
    const source = Core.rectForNode(edge.source_component, first.positions, sizes);
    const target = Core.rectForNode(edge.target_component, first.positions, sizes);
    const route = Core.routeOrthogonal(source, target, rects, { clearance: 12 });
    route.forEach((point, index) => {
      if (!index) return;
      rects.filter((rect) => rect.id !== source.id && rect.id !== target.id).forEach((obstacle) => {
        assert.equal(Core.segmentHitsRect(route[index - 1], point, obstacle), false, `${edge.link_id} crosses ${obstacle.id}`);
      });
    });
  }
});

test("viewport-aware layout fills spare canvas with spacing without enlarging nodes", () => {
  const components = ["a", "b", "c"].map((id) => component(id));
  const links = [link("ab", "a", "b"), link("bc", "b", "c")];
  const sizes = {
    a: { width: 64, height: 34 },
    b: { width: 88, height: 40 },
    c: { width: 72, height: 36 },
  };
  const compact = Core.layoutGraph(components, links, [], sizes, {
    viewportWidth: 520,
    viewportHeight: 300,
    fillViewport: true,
  });
  const wide = Core.layoutGraph(components, links, [], sizes, {
    viewportWidth: 1100,
    viewportHeight: 560,
    fillViewport: true,
  });
  assert.equal(wide.algorithm, "group-layered-viewport-v2");
  const extent = (layout) => ({
    width: Math.max(...Object.values(layout.positions).map((point) => point.x)) - Math.min(...Object.values(layout.positions).map((point) => point.x)),
    height: Math.max(...Object.values(layout.positions).map((point) => point.y)) - Math.min(...Object.values(layout.positions).map((point) => point.y)),
  });
  assert.ok(extent(wide).width > extent(compact).width);
  assert.ok(extent(wide).height > extent(compact).height);
  for (const item of components) {
    const id = item.component_id;
    assert.deepEqual(Core.rectForNode(id, wide.positions, sizes), {
      id,
      x: wide.positions[id].x,
      y: wide.positions[id].y,
      ...sizes[id],
    });
  }
  const single = Core.layoutGraph([component("only")], [], [], { only: sizes.a }, {
    viewportWidth: 900,
    viewportHeight: 500,
    fillViewport: true,
  });
  assert.ok(Math.abs(single.positions.only.x + sizes.a.width / 2 - 450) <= 1);
  assert.ok(Math.abs(single.positions.only.y + sizes.a.height / 2 - 250) <= 1);
});

test("automatic fit prefers 100%, stops at 80%, and reports remaining scroll overflow", () => {
  const small = Core.computeAutoFitScale(
    { x: 0, y: 0, width: 300, height: 160 },
    { width: 900, height: 500 },
    { padding: 24, minScale: 0.8, maxScale: 1 },
  );
  assert.equal(small.scale, 1, "small diagrams never enlarge their nodes");
  assert.equal(small.fits, true);

  const medium = Core.computeAutoFitScale(
    { x: 0, y: 0, width: 1000, height: 500 },
    { width: 948, height: 548 },
    { padding: 24, minScale: 0.8, maxScale: 1 },
  );
  assert.equal(medium.scale, 0.9);
  assert.equal(medium.fits, true);

  const large = Core.computeAutoFitScale(
    { x: 0, y: 0, width: 1800, height: 900 },
    { width: 900, height: 500 },
    { padding: 24, minScale: 0.8, maxScale: 1 },
  );
  assert.equal(large.scale, 0.8);
  assert.equal(large.fits, false);
  assert.ok(large.overflowX > 0);
  assert.ok(large.overflowY > 0);
});

test("expanded H100 module layout replaces compact preset coordinates with routable measured boxes", () => {
  const moduleIds = ["accelerator0", ...Array.from({ length: 5 }, (_, index) => `memory0_hbm${index}`)];
  const components = [...moduleIds, "fabric0"].map((id) => component(id));
  const links = [
    ...moduleIds.slice(1).map((id, index) => link(`memory_link0_hbm${index}`, "accelerator0", id, { protocol: "HBM3" })),
    link("fabric_link0", "accelerator0", "fabric0", { protocol: "NVLink" }),
  ];
  const groups = [
    { group_id: "module0", members: moduleIds, root: "accelerator0", collapsed: false },
    { group_id: "fabric", members: ["fabric0"], root: "fabric0", collapsed: true },
  ];
  const sizes = Object.fromEntries(components.map((item) => [item.component_id, { width: 130, height: 66 }]));
  const compactPresetPositions = {
    accelerator0: { x: 0, y: 220 },
    ...Object.fromEntries(moduleIds.slice(1).map((id, index) => [id, {
      x: (index % 4) * 56,
      y: 340 + Math.floor(index / 4) * 72,
    }])),
    fabric0: { x: 480, y: 20 },
  };
  assert.equal(
    Core.rectsIntersect(
      Core.rectForNode("memory0_hbm0", compactPresetPositions, sizes),
      Core.rectForNode("memory0_hbm1", compactPresetPositions, sizes),
    ),
    true,
    "the collapsed preset's hidden-member coordinates are intentionally too compact for visible DOM boxes",
  );

  const first = Core.layoutGraph(components, links, groups, sizes, { nodeGap: 28, layerGap: 112, rowGap: 64 });
  const second = Core.layoutGraph(components, links, groups, sizes, { nodeGap: 28, layerGap: 112, rowGap: 64 });
  assert.deepEqual(first, second, "expanded coordinates remain deterministic");
  const rects = components.map((item) => Core.rectForNode(item.component_id, first.positions, sizes));
  for (let left = 0; left < rects.length; left += 1) for (let right = left + 1; right < rects.length; right += 1) {
    assert.equal(Core.rectsIntersect(rects[left], rects[right]), false, `${rects[left].id} overlaps ${rects[right].id}`);
  }
  for (const edge of links) {
    const source = rects.find((rect) => rect.id === edge.source_component);
    const target = rects.find((rect) => rect.id === edge.target_component);
    const route = Core.routeOrthogonal(source, target, rects, { clearance: 14 });
    route.forEach((point, index) => {
      if (!index) return;
      rects.filter((rect) => rect.id !== source.id && rect.id !== target.id).forEach((obstacle) => {
        assert.equal(Core.segmentHitsRect(route[index - 1], point, obstacle), false, `${edge.link_id} crosses ${obstacle.id}`);
      });
    });
  }
});

test("orthogonal routing finds a clearance path through an obstacle maze without crossing nodes", () => {
  const source = { id: "source", x: 0, y: 80, width: 80, height: 50 };
  const target = { id: "target", x: 540, y: 80, width: 80, height: 50 };
  const obstacles = [
    source,
    target,
    { id: "wall-a", x: 160, y: 20, width: 80, height: 180 },
    { id: "wall-b", x: 340, y: -80, width: 80, height: 180 },
  ];
  const route = Core.routeOrthogonal(source, target, obstacles, { clearance: 12, channelOffset: 9 });
  assert.ok(route.length >= 4);
  route.forEach((point, index) => {
    if (!index) return;
    assert.ok(point.x === route[index - 1].x || point.y === route[index - 1].y, "every route segment is orthogonal");
    for (const obstacle of obstacles.slice(2)) assert.equal(Core.segmentHitsRect(route[index - 1], point, obstacle), false, `route crosses ${obstacle.id}`);
  });
});

test("visibility fallback solves a maze after every cheap orthogonal candidate is blocked", () => {
  const source = { id: "source", x: 0, y: 100, width: 80, height: 50 };
  const target = { id: "target", x: 500, y: 100, width: 80, height: 50 };
  const obstacles = [
    source,
    target,
    { id: "wall", x: 250, y: 50, width: 80, height: 150 },
    { id: "cap-top", x: 80, y: 0, width: 100, height: 80 },
    { id: "cap-bottom", x: 80, y: 170, width: 100, height: 80 },
  ];
  const route = Core.routeOrthogonal(source, target, obstacles, { clearance: 12 });
  assert.ok(route.length > 6, "the multi-turn visibility path, not a cheap candidate, is required for this maze");
  orthogonalSegments(route).forEach((segment) => {
    for (const obstacle of obstacles.slice(2)) {
      assert.equal(Core.segmentHitsRect({ x: segment.x1, y: segment.y1 }, { x: segment.x2, y: segment.y2 }, obstacle), false, `route crosses ${obstacle.id}`);
    }
  });
});

test("routing reports no solution instead of returning a path through a covering obstacle", () => {
  const source = { id: "source", x: 0, y: 0, width: 40, height: 40 };
  const target = { id: "target", x: 160, y: 0, width: 40, height: 40 };
  const cover = { id: "cover", x: -30, y: -30, width: 260, height: 100 };
  assert.throws(() => Core.routeOrthogonal(source, target, [source, target, cover], { clearance: 10 }), /找不到避开组件的正交连线路径/);
});

test("parallel edge channel offsets produce distinct orthogonal hit paths", () => {
  const source = { id: "source", x: 0, y: 0, width: 80, height: 50 };
  const target = { id: "target", x: 420, y: 0, width: 80, height: 50 };
  const upper = Core.routeOrthogonal(source, target, [source, target], { clearance: 12, channelOffset: -18 });
  const lower = Core.routeOrthogonal(source, target, [source, target], { clearance: 12, channelOffset: 18 });
  assert.notEqual(Core.pathToSvg(upper), Core.pathToSvg(lower));
  for (const route of [upper, lower]) route.forEach((point, index) => {
    if (index) assert.ok(point.x === route[index - 1].x || point.y === route[index - 1].y);
  });
});

test("real port identities receive stable same-side fractions for fan-out and fan-in", () => {
  const rects = [
    { id: "source", x: 0, y: 100, width: 90, height: 120 },
    { id: "a", x: 420, y: 0, width: 90, height: 60 },
    { id: "b", x: 420, y: 260, width: 90, height: 60 },
    { id: "c", x: 120, y: 420, width: 90, height: 60 },
    { id: "d", x: 120, y: 560, width: 90, height: 60 },
    { id: "sink", x: 560, y: 490, width: 90, height: 120 },
  ];
  const links = [
    link("out-a", "source", "a", { source_port: "shared", target_port: "in" }),
    link("out-b", "source", "b", { source_port: "shared", target_port: "in" }),
    link("out-c", "source", "a", { source_port: "alpha", target_port: "aux" }),
    link("in-c", "c", "sink", { source_port: "out", target_port: "shared-in" }),
    link("in-d", "d", "sink", { source_port: "out", target_port: "shared-in" }),
  ];
  const assigned = Core.assignPortEndpoints(links, rects);
  assert.deepEqual(assigned.links[0].sourcePoint, assigned.links[1].sourcePoint, "fan-out from one real port shares one anchor");
  assert.deepEqual(assigned.links[3].targetPoint, assigned.links[4].targetPoint, "fan-in to one real port shares one anchor");
  assert.notDeepEqual(assigned.links[0].sourcePoint, assigned.links[2].sourcePoint, "different source ports use different slots");
  const alpha = assigned.byPort[Core.portIdentity("source", "alpha")];
  const shared = assigned.byPort[Core.portIdentity("source", "shared")];
  assert.equal(alpha.side, "right");
  assert.equal(shared.side, "right");
  assert.deepEqual([alpha.fraction, shared.fraction], [1 / 3, 2 / 3]);
  assert.deepEqual(Core.assignPortEndpoints(links.slice().reverse(), rects).byPort, assigned.byPort, "slot assignment is independent of input order");
});

test("routeOrthogonal honors exact port endpoints while preserving orthogonality", () => {
  const source = { id: "source", x: 0, y: 0, width: 100, height: 100 };
  const target = { id: "target", x: 420, y: 40, width: 100, height: 100 };
  const sourcePoint = { x: 100, y: 20 };
  const targetPoint = { x: 420, y: 120 };
  const route = Core.routeOrthogonal(source, target, [source, target], { clearance: 12, sourcePoint, targetPoint });
  assert.deepEqual(route[0], sourcePoint);
  assert.deepEqual(route.at(-1), targetPoint);
  orthogonalSegments(route).forEach((segment) => assert.ok(segment.x1 === segment.x2 || segment.y1 === segment.y2));
});

test("route plan permits only the short shared escape stub for one fan-out port", () => {
  const rects = [
    { id: "source", x: 0, y: 80, width: 80, height: 50 },
    { id: "upper", x: 500, y: 0, width: 80, height: 50 },
    { id: "lower", x: 500, y: 160, width: 80, height: 50 },
  ];
  const links = [
    link("to-upper", "source", "upper", { source_port: "shared", target_port: "in" }),
    link("to-lower", "source", "lower", { source_port: "shared", target_port: "in" }),
  ];
  const plan = Core.planOrthogonalRoutes(links, rects, { clearance: 14, parallelOffsets: false });
  assert.equal(plan.errors.length, 0);
  assert.equal(plan.routes.length, 2);
  assert.deepEqual(plan.routes[0].sourcePoint, plan.routes[1].sourcePoint);
  const first = orthogonalSegments(plan.routes[0].points);
  const second = orthogonalSegments(plan.routes[1].points);
  const sharedLength = first.reduce((sum, a) => sum + second.reduce((inner, b) => inner + overlapLength(a, b), 0), 0);
  assert.equal(sharedLength, 14, "the shared geometry stops at the configured escape length");
});

test("occupied crossings receive a medium penalty and route around when a clean lane exists", () => {
  const source = { id: "source", x: 0, y: 0, width: 80, height: 50 };
  const target = { id: "target", x: 420, y: 0, width: 80, height: 50 };
  const occupied = [{ x1: 200, y1: -50, x2: 200, y2: 100 }];
  const unpenalized = Core.routeOrthogonal(source, target, [source, target], { clearance: 12, occupiedSegments: occupied, crossingPenalty: 0 });
  assert.ok(orthogonalSegments(unpenalized).some((segment) => crosses(segment, occupied[0])), "the shortest route crosses without a crossing penalty");
  const routed = Core.routeOrthogonal(source, target, [source, target], { clearance: 12, occupiedSegments: occupied });
  assert.ok(orthogonalSegments(routed).every((segment) => !crosses(segment, occupied[0])), "the default penalty selects the longer clean lane");
});

test("route plans are deterministic across link input ordering and retain parallel channels", () => {
  const rects = [
    { id: "a", x: 0, y: 0, width: 90, height: 90 },
    { id: "b", x: 420, y: 0, width: 90, height: 90 },
  ];
  const links = [
    link("z-link", "a", "b", { source_port: "z", target_port: "z" }),
    link("a-link", "a", "b", { source_port: "a", target_port: "a" }),
  ];
  const snapshot = (plan) => plan.routes.map((route) => ({ id: route.linkId, channelOffset: route.channelOffset, points: route.points }));
  const forward = Core.planOrthogonalRoutes(links, rects, { clearance: 12 });
  const reverse = Core.planOrthogonalRoutes(links.slice().reverse(), rects, { clearance: 12 });
  assert.equal(forward.errors.length, 0);
  assert.equal(reverse.errors.length, 0);
  assert.deepEqual(snapshot(forward), snapshot(reverse));
  assert.deepEqual(forward.routes.map((route) => route.channelOffset), [-9, 9]);
  assert.notEqual(Core.pathToSvg(forward.routes[0].points), Core.pathToSvg(forward.routes[1].points));
});

test("routeIds filters output after full-link endpoint slots and parallel peers are computed", () => {
  const rects = [
    { id: "a", x: 0, y: 0, width: 120, height: 120 },
    { id: "b", x: 480, y: 0, width: 120, height: 120 },
  ];
  const links = [
    link("a-link", "a", "b", { source_port: "a", target_port: "a" }),
    link("m-link", "a", "b", { source_port: "m", target_port: "m" }),
    link("z-link", "a", "b", { source_port: "z", target_port: "z" }),
  ];
  const full = Core.planOrthogonalRoutes(links, rects, { clearance: 12 });
  assert.equal(full.errors.length, 0);
  assert.equal(full.routes.length, 3);
  const fullZ = full.routes.find((route) => route.linkId === "z-link");
  const seed = [{ x1: -200, y1: -200, x2: -100, y2: -200 }];
  const filtered = Core.planOrthogonalRoutes(links, rects, { clearance: 12, routeIds: ["z-link"], occupiedSegments: seed });
  assert.deepEqual(filtered.routes.map((route) => route.linkId), ["z-link"]);
  assert.equal(filtered.errors.length, 0);
  assert.deepEqual(filtered.endpoints, full.endpoints, "all links still participate in real-port fraction slots");
  assert.deepEqual(filtered.routes[0].sourcePoint, fullZ.sourcePoint);
  assert.deepEqual(filtered.routes[0].targetPoint, fullZ.targetPoint);
  assert.equal(filtered.routes[0].channelOffset, fullZ.channelOffset, "all component-pair peers still determine the parallel channel");
  assert.deepEqual(
    { x1: filtered.occupiedSegments[0].x1, y1: filtered.occupiedSegments[0].y1, x2: filtered.occupiedSegments[0].x2, y2: filtered.occupiedSegments[0].y2 },
    seed[0],
    "unaffected routes remain usable as occupied-segment seeds",
  );
  const selectedBySet = Core.planOrthogonalRoutes(links, rects, { clearance: 12, routeIds: new Set(["m-link"]) });
  assert.deepEqual(selectedBySet.routes.map((route) => route.linkId), ["m-link"]);
  assert.equal(selectedBySet.routes[0].channelOffset, 0);
});

test("forty non-congested links stay on the cheap route-plan path", { timeout: 5000 }, () => {
  const rects = [];
  const links = [];
  for (let index = 0; index < 40; index += 1) {
    rects.push(
      { id: `source-${index}`, x: 0, y: index * 120, width: 80, height: 50 },
      { id: `target-${index}`, x: 500, y: index * 120, width: 80, height: 50 },
    );
    links.push(link(`route-${index}`, `source-${index}`, `target-${index}`, { source_port: "out", target_port: "in" }));
  }
  const started = performance.now();
  const plan = Core.planOrthogonalRoutes(links, rects, { clearance: 12 });
  const elapsed = performance.now() - started;
  assert.equal(plan.errors.length, 0);
  assert.equal(plan.routes.length, 40);
  assert.ok(elapsed < 2000, `40-link cheap route plan took ${elapsed.toFixed(1)}ms; expected well below the previous ~3900ms`);
});

test("rank columns share one visual center axis even when layer populations differ", () => {
  const components = ["hub", ...Array.from({ length: 7 }, (_, index) => `leaf-${index}`)].map((id) => component(id));
  const links = components.slice(1).map((item, index) => link(`edge-${index}`, "hub", item.component_id));
  const sizes = Object.fromEntries(components.map((item, index) => [item.component_id, {
    width: 100 + (index % 3) * 12,
    height: 52 + (index % 2) * 10,
  }]));
  const layout = Core.layoutGraph(components, links, [], sizes, { margin: 40, rowGap: 34, layerGap: 120 });
  const rects = components.map((item) => Core.rectForNode(item.component_id, layout.positions, sizes));
  const columns = new Map();
  rects.forEach((rect) => {
    const key = Math.round(rect.x + rect.width / 2);
    if (!columns.has(key)) columns.set(key, []);
    columns.get(key).push(rect);
  });
  assert.equal(columns.size, 2);
  const canvasAxis = layout.bounds.height / 2;
  columns.forEach((column) => {
    const top = Math.min(...column.map((rect) => rect.y));
    const bottom = Math.max(...column.map((rect) => rect.y + rect.height));
    assert.ok(Math.abs((top + bottom) / 2 - canvasAxis) <= 1, "each layer is centered on the canvas axis");
  });
});

test("long dense link labels are placed deterministically outside nodes, labels, and every sampled route", () => {
  const routes = Array.from({ length: 12 }, (_, index) => {
    const points = [{ x: 120, y: 80 + index * 20 }, { x: 880, y: 80 + index * 20 }];
    return {
      linkId: `link-${String(index).padStart(2, "0")}`,
      label: `Ultra-long-protocol-${index}-with-deterministic-bandwidth-800.000-GB-per-second`,
      points,
      segments: Core.routeSegments(points, { routeId: `link-${index}` }),
    };
  });
  const nodes = [
    { id: "left", x: 0, y: 40, width: 100, height: 300 },
    { id: "right", x: 900, y: 40, width: 100, height: 300 },
  ];
  const options = { fontSize: 10, nodeGap: 10, labelGap: 7, routeGap: 5, canvasMargin: 12 };
  const first = Core.placeRouteLabels(routes, nodes, options);
  const second = Core.placeRouteLabels(routes.slice().reverse(), nodes, options);
  assert.deepEqual(first.placements, second.placements, "stable link IDs make placement independent of input order");
  assert.equal(first.placements.length, routes.length);
  assert.ok(first.placements.every((placement) => placement.rect.width > 400), "the deterministic text estimate covers the entire long label");
  const allSegments = routes.flatMap((route) => route.segments);
  first.placements.forEach((placement, index) => {
    nodes.forEach((node) => assert.equal(Core.rectsIntersectWithGap(placement.rect, node, options.nodeGap), false));
    allSegments.forEach((segment) => assert.equal(
      Core.lineSegmentHitsRect({ x: segment.x1, y: segment.y1 }, { x: segment.x2, y: segment.y2 }, placement.rect, options.routeGap),
      false,
      `${placement.linkId} overlaps route geometry`,
    ));
    first.placements.slice(0, index).forEach((prior) => assert.equal(Core.rectsIntersectWithGap(placement.rect, prior.rect, options.labelGap), false));
  });
});

test("short Beziers require sampled clearance proof and fall back to rounded orthogonal geometry", () => {
  const source = { id: "source", x: 0, y: 0, width: 80, height: 60 };
  const target = { id: "target", x: 220, y: 0, width: 80, height: 60 };
  const edge = link("short", "source", "target", { source_port: "out", target_port: "in" });
  const curved = Core.planOrthogonalRoutes([edge], [source, target], {
    clearance: 12,
    renderGeometry: true,
    allowCurves: true,
    shortCurveDistance: 300,
  });
  assert.equal(curved.errors.length, 0);
  assert.equal(curved.routes[0].kind, "curve");
  assert.match(curved.routes[0].path, / C /);
  assert.ok(curved.routes[0].points.length >= 16, "Bezier occupancy uses sampled points instead of hidden SVG geometry");
  assert.equal(curved.routes[0].segments.length, curved.routes[0].points.length - 1);

  const obstacle = { id: "obstacle", x: 120, y: -10, width: 60, height: 80 };
  const fallback = Core.planOrthogonalRoutes([edge], [source, target, obstacle], {
    clearance: 12,
    renderGeometry: true,
    allowCurves: true,
    shortCurveDistance: 300,
    cornerRadius: 9,
  });
  assert.equal(fallback.errors.length, 0);
  assert.equal(fallback.routes[0].kind, "orthogonal");
  assert.doesNotMatch(fallback.routes[0].path, / C /, "an unproved SVG C is never rendered");
  assert.equal(Core.sampledPathClear(fallback.routes[0].points, [obstacle], 12), true);
  assert.equal(fallback.routes[0].segments.length, fallback.routes[0].points.length - 1);
});

test("forty rendered routes and labels remain deterministic within the interaction budget", { timeout: 5000 }, () => {
  const rects = [];
  const links = [];
  for (let index = 0; index < 40; index += 1) {
    rects.push(
      { id: `source-${index}`, x: 0, y: index * 110, width: 80, height: 50 },
      { id: `target-${index}`, x: 500, y: index * 110, width: 80, height: 50 },
    );
    links.push(link(`route-${String(index).padStart(2, "0")}`, `source-${index}`, `target-${index}`, { source_port: "out", target_port: "in" }));
  }
  const snapshot = () => {
    const plan = Core.planOrthogonalRoutes(links, rects, { clearance: 12, renderGeometry: true, shortCurveDistance: 220 });
    const labels = Core.placeRouteLabels(plan.routes.map((route) => ({ ...route, label: `${route.linkId} · 100 GB/s` })), rects, { fontSize: 9 });
    return { plan, labels };
  };
  const started = performance.now();
  const first = snapshot();
  const elapsed = performance.now() - started;
  const second = snapshot();
  assert.equal(first.plan.errors.length, 0);
  assert.equal(first.plan.routes.length, 40);
  assert.equal(first.labels.placements.length, 40);
  assert.deepEqual(first, second);
  assert.ok(elapsed < 2000, `40 rendered routes and labels took ${elapsed.toFixed(1)}ms`);
});

test("reduced motion switches topology transitions to immediate state changes", () => {
  assert.equal(Core.motionDuration(true, 240), 0);
  assert.equal(Core.motionDuration(false, 240), 240);
});
