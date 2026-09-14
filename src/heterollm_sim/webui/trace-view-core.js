"use strict";

// Pure Trace playback view primitives.  This file intentionally has no DOM,
// timer, or scenario dependency.  It can be loaded by Node tests and by the
// browser (after topology-core.js) through the UMD wrapper below.
(function traceViewCoreFactory(root, factory) {
  const api = factory(root);
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root && typeof root === "object") root.TraceViewCore = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function buildTraceViewCore(root) {
  const VIEW_VERSION = 1;
  const DEFAULT_VIEWPORT = Object.freeze({ x: 0, y: 0, scale: 1 });
  const DEFAULT_NODE_SIZE = Object.freeze({ width: 130, height: 66 });
  const DEFAULT_BOUNDS = Object.freeze({ x: 0, y: 0, width: 640, height: 420 });
  const MIN_SCALE = 0.25;
  const MAX_SCALE = 4;

  function clone(value) {
    if (value === undefined) return undefined;
    return JSON.parse(JSON.stringify(value));
  }

  function object(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  function array(value) {
    return Array.isArray(value) ? value : [];
  }

  function finite(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function hasFinite(value) {
    return Number.isFinite(Number(value));
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function asId(value) {
    return value == null ? "" : String(value);
  }

  function normalizedPoint(value, fallback = { x: 0, y: 0 }) {
    const source = object(value);
    return { x: finite(source.x, fallback.x), y: finite(source.y, fallback.y) };
  }

  function componentList(value) {
    return array(value).map(object);
  }

  function linkList(value) {
    return array(value).map(object);
  }

  function componentIds(components) {
    return Array.from(new Set(
      componentList(components)
        .map((item) => asId(item.component_id))
        .filter(Boolean),
    ));
  }

  function componentKind(component) {
    return asId(object(component).kind || "component");
  }

  function normalizeSize(value, component, fallback = DEFAULT_NODE_SIZE) {
    const source = object(value);
    const fromComponent = object(component);
    const width = Math.max(40, finite(
      source.width ?? fromComponent.width,
      fallback.width,
    ));
    const height = Math.max(30, finite(
      source.height ?? fromComponent.height,
      fallback.height,
    ));
    return { width, height };
  }

  function sizeSource(raw) {
    return object(object(raw).node_sizes);
  }

  function normalizeNodeSizes(raw, components = [], options = {}) {
    const fallback = normalizeSize(options.defaultNodeSize, {}, DEFAULT_NODE_SIZE);
    const source = sizeSource(raw);
    const list = componentList(components);
    const ids = componentIds(list);
    const result = {};
    for (const component of list) {
      const id = asId(component.component_id);
      if (!id) continue;
      const rawSize = source[id];
      result[id] = normalizeSize(rawSize, component, fallback);
    }
    return result;
  }

  function compareIds(left, right) {
    const a = String(left);
    const b = String(right);
    return a < b ? -1 : a > b ? 1 : 0;
  }

  // Deterministic left-to-right layout for playback traces.  Unlike
  // TopologyCore.layoutGraph (which treats hardware links as undirected), this
  // helper preserves source_component -> target_component when assigning
  // layers.  Strongly connected components share a layer because no layout
  // can make every edge in a directed cycle point forwards.
  function directedTraceTopologyLayout(components, links, options = {}) {
    const opts = object(options);
    const byId = new Map();
    for (const component of componentList(components)) {
      const id = asId(component.component_id);
      if (id && !byId.has(id)) byId.set(id, component);
    }
    const ids = Array.from(byId.keys()).sort(compareIds);
    const validIds = new Set(ids);
    const edges = linkList(links)
      .map((link, index) => ({
        ...link,
        source_component: asId(link.source_component),
        target_component: asId(link.target_component),
        __order: index,
      }))
      .filter((link) => validIds.has(link.source_component) && validIds.has(link.target_component))
      .sort((left, right) => compareIds(left.source_component, right.source_component)
        || compareIds(left.target_component, right.target_component)
        || compareIds(asId(left.link_id), asId(right.link_id))
        || left.__order - right.__order);
    const adjacency = new Map(ids.map((id) => [id, []]));
    for (const edge of edges) {
      if (!adjacency.get(edge.source_component).includes(edge.target_component)) {
        adjacency.get(edge.source_component).push(edge.target_component);
      }
    }
    adjacency.forEach((targets) => targets.sort(compareIds));

    // Tarjan SCC traversal is lexical, so equivalent graphs produce identical
    // component/layer assignments regardless of input array order.
    let nextIndex = 0;
    const indices = new Map();
    const lowLinks = new Map();
    const stack = [];
    const onStack = new Set();
    const stronglyConnected = [];
    function visit(id) {
      indices.set(id, nextIndex);
      lowLinks.set(id, nextIndex);
      nextIndex += 1;
      stack.push(id);
      onStack.add(id);
      for (const target of adjacency.get(id)) {
        if (!indices.has(target)) {
          visit(target);
          lowLinks.set(id, Math.min(lowLinks.get(id), lowLinks.get(target)));
        } else if (onStack.has(target)) {
          lowLinks.set(id, Math.min(lowLinks.get(id), indices.get(target)));
        }
      }
      if (lowLinks.get(id) !== indices.get(id)) return;
      const members = [];
      let member;
      do {
        member = stack.pop();
        onStack.delete(member);
        members.push(member);
      } while (member !== id);
      stronglyConnected.push(members.sort(compareIds));
    }
    ids.forEach((id) => { if (!indices.has(id)) visit(id); });
    stronglyConnected.sort((left, right) => compareIds(left[0], right[0]));

    const sccFor = new Map();
    stronglyConnected.forEach((members, index) => members.forEach((id) => sccFor.set(id, index)));
    const sccAdjacency = new Map(stronglyConnected.map((_members, index) => [index, new Set()]));
    const indegree = new Map(stronglyConnected.map((_members, index) => [index, 0]));
    for (const edge of edges) {
      const source = sccFor.get(edge.source_component);
      const target = sccFor.get(edge.target_component);
      if (source === target || sccAdjacency.get(source).has(target)) continue;
      sccAdjacency.get(source).add(target);
      indegree.set(target, indegree.get(target) + 1);
    }
    const sccKey = (index) => stronglyConnected[index][0];
    const ready = Array.from(indegree.entries())
      .filter(([, value]) => value === 0)
      .map(([index]) => index)
      .sort((left, right) => compareIds(sccKey(left), sccKey(right)));
    const sccLayer = new Map(stronglyConnected.map((_members, index) => [index, 0]));
    while (ready.length) {
      const source = ready.shift();
      const targets = Array.from(sccAdjacency.get(source)).sort((left, right) => compareIds(sccKey(left), sccKey(right)));
      for (const target of targets) {
        sccLayer.set(target, Math.max(sccLayer.get(target), sccLayer.get(source) + 1));
        indegree.set(target, indegree.get(target) - 1);
        if (indegree.get(target) === 0) {
          ready.push(target);
          ready.sort((left, right) => compareIds(sccKey(left), sccKey(right)));
        }
      }
    }

    const layerById = {};
    const layerMap = new Map();
    for (const id of ids) {
      const layer = sccLayer.get(sccFor.get(id)) || 0;
      layerById[id] = layer;
      if (!layerMap.has(layer)) layerMap.set(layer, []);
      layerMap.get(layer).push(id);
    }
    const layerNumbers = Array.from(layerMap.keys()).sort((left, right) => left - right);
    const layers = layerNumbers.map((layer) => layerMap.get(layer).sort(compareIds));
    const neighbors = new Map(ids.map((id) => [id, new Set()]));
    for (const edge of edges) {
      neighbors.get(edge.source_component).add(edge.target_component);
      neighbors.get(edge.target_component).add(edge.source_component);
    }
    // Reorder nodes within each layer using deterministic two-way barycentric
    // sweeps.  This keeps the source-to-target rank contract while reducing
    // avoidable crossings in branch/merge graphs. Lexical IDs remain the
    // stable tie-breaker, so shuffled input arrays cannot change the result.
    for (let sweep = 0; sweep < 4; sweep += 1) {
      const layerOrder = sweep % 2 === 0
        ? layers.map((_members, index) => index)
        : layers.map((_members, index) => index).reverse();
      const order = new Map();
      layers.forEach((members) => members.forEach((id, index) => order.set(id, index)));
      for (const layerIndex of layerOrder) {
        layers[layerIndex].sort((left, right) => {
          const average = (id) => {
            const adjacent = Array.from(neighbors.get(id))
              .filter((neighbor) => layerById[neighbor] !== layerNumbers[layerIndex] && order.has(neighbor));
            return adjacent.length
              ? adjacent.reduce((sum, neighbor) => sum + order.get(neighbor), 0) / adjacent.length
              : order.get(id);
          };
          return average(left) - average(right) || compareIds(left, right);
        });
        layers[layerIndex].forEach((id, index) => order.set(id, index));
      }
    }
    const sizes = normalizeNodeSizes({ node_sizes: opts.nodeSizes || {} }, Array.from(byId.values()), opts);
    const margin = Math.max(0, finite(opts.margin, 42));
    const layerGap = Math.max(1, finite(opts.layerGap, 120));
    const rowGap = Math.max(1, finite(opts.rowGap, 54));
    const layerWidths = layers.map((members) => Math.max(0, ...members.map((id) => sizes[id].width)));
    const layerHeights = layers.map((members) => members.reduce((total, id, index) => total + sizes[id].height + (index ? rowGap : 0), 0));
    const contentHeight = Math.max(0, ...layerHeights);
    const contentWidth = layerWidths.reduce((total, width) => total + width, 0)
      + Math.max(0, layers.length - 1) * layerGap;
    const minimumBounds = normalizeBounds(opts.minimumBounds, DEFAULT_BOUNDS);
    const canvasWidth = Math.max(minimumBounds.width, contentWidth + margin * 2);
    const canvasHeight = Math.max(minimumBounds.height, contentHeight + margin * 2);
    const positions = {};
    let x = (canvasWidth - contentWidth) / 2;
    layers.forEach((members, layerIndex) => {
      let y = (canvasHeight - layerHeights[layerIndex]) / 2;
      members.forEach((id) => {
        positions[id] = {
          x: Math.round(x + (layerWidths[layerIndex] - sizes[id].width) / 2),
          y: Math.round(y),
        };
        y += sizes[id].height + rowGap;
      });
      x += layerWidths[layerIndex] + layerGap;
    });
    const bounds = {
      x: 0,
      y: 0,
      width: canvasWidth,
      height: canvasHeight,
    };
    const cyclicSccs = stronglyConnected.filter((members) => members.length > 1);
    const cyclicIds = new Set(cyclicSccs.flat());
    const cyclicLinks = edges
      .filter((edge) => edge.source_component === edge.target_component
        || (cyclicIds.has(edge.source_component) && sccFor.get(edge.source_component) === sccFor.get(edge.target_component)))
      .map((edge) => asId(edge.link_id))
      .filter(Boolean);
    return {
      algorithm: "trace-directed-layered-v2",
      direction: "left-to-right",
      components: ids.map((id) => clone(byId.get(id))),
      links: edges.map(({ __order, ...edge }) => clone(edge)),
      positions,
      node_sizes: clone(sizes),
      bounds,
      layers: clone(layers),
      layerById,
      stronglyConnectedComponents: clone(stronglyConnected),
      cyclicComponentIds: Array.from(cyclicIds).sort(compareIds),
      cyclicLinkIds: cyclicLinks,
    };
  }

  function viewportSource(raw) {
    const source = object(raw);
    return object(source.viewport);
  }

  function normalizeViewport(raw, fallback = DEFAULT_VIEWPORT) {
    const source = object(raw);
    return {
      x: finite(source.x, fallback.x),
      y: finite(source.y, fallback.y),
      scale: clamp(finite(source.scale, fallback.scale), MIN_SCALE, MAX_SCALE),
    };
  }

  function normalizeBounds(value, fallback = DEFAULT_BOUNDS) {
    const source = object(value);
    return {
      x: finite(source.x, fallback.x),
      y: finite(source.y, fallback.y),
      width: Math.max(0, finite(source.width, fallback.width)),
      height: Math.max(0, finite(source.height, fallback.height)),
    };
  }

  function positionSource(raw) {
    return object(object(raw).layout).positions || {};
  }

  function validPositions(raw, ids) {
    const source = positionSource(raw);
    const valid = ids.length ? new Set(ids) : null;
    const result = {};
    for (const [id, value] of Object.entries(object(source))) {
      if (valid && !valid.has(String(id))) continue;
      if (!hasFinite(value?.x) || !hasFinite(value?.y)) continue;
      result[String(id)] = normalizedPoint(value);
    }
    return result;
  }

  function defaultTraceTopologyLayout(components, links, options = {}) {
    const normalizedOptions = object(options);
    const list = componentList(components);
    const edges = linkList(links);
    const sizes = normalizeNodeSizes({ node_sizes: normalizedOptions.nodeSizes || {} }, list, normalizedOptions);
    const directedOptions = {
      ...normalizedOptions,
      nodeSizes: sizes,
    };
    const layout = directedTraceTopologyLayout(list, edges, directedOptions);
    return {
      positions: clone(object(layout?.positions)),
      bounds: normalizeBounds(layout?.bounds, DEFAULT_BOUNDS),
      node_sizes: clone(sizes),
      components: clone(list),
      links: clone(edges),
      algorithm: String(layout?.algorithm || "trace-directed-layered-v2"),
      direction: String(layout?.direction || "left-to-right"),
      layers: clone(array(layout?.layers)),
    };
  }

  function unwrapTraceView(raw) {
    const source = object(raw);
    return source.trace_topology_view && typeof source.trace_topology_view === "object" && !Array.isArray(source.trace_topology_view)
      ? source.trace_topology_view
      : source;
  }

  function normalizeTraceTopologyView(raw, components, links, options = {}) {
    const source = unwrapTraceView(raw);
    const opts = object(options);
    const componentInput = components == null ? source.components : components;
    const linkInput = links == null ? source.links : links;
    const list = componentList(componentInput);
    const edges = linkList(linkInput);
    const ids = componentIds(list);
    const sourceLayout = object(source.layout);
    const explicit = validPositions(source, ids);
    const layout = defaultTraceTopologyLayout(list, edges, {
      ...opts,
      groups: source.groups || [],
      nodeSizes: sizeSource(sourceLayout),
    });
    const positions = { ...layout.positions, ...explicit };
    const filteredPositions = {};
    for (const id of ids) if (positions[id]) filteredPositions[id] = normalizedPoint(positions[id]);
    const nodeSizes = normalizeNodeSizes(sourceLayout, list, opts);
    const bounds = normalizeBounds(
      sourceLayout.bounds,
      layout.bounds,
    );
    const viewport = normalizeViewport(viewportSource(source), normalizeViewport(opts.viewport, DEFAULT_VIEWPORT));
    const algorithm = String(sourceLayout.algorithm || layout.algorithm || "trace-directed-layered-v2");
    const groups = array(source.groups).map((group) => clone(group));
    const result = {
      version: VIEW_VERSION,
      components: clone(list),
      links: clone(edges),
      viewport,
      groups,
      layout: {
        algorithm,
        positions: clone(filteredPositions),
        node_sizes: clone(nodeSizes),
        bounds: clone(bounds),
      },
    };
    return result;
  }

  function computeViewBounds(positions, sizes, minimum = DEFAULT_BOUNDS) {
    const ids = Object.keys(object(positions));
    if (!ids.length) return normalizeBounds(minimum, DEFAULT_BOUNDS);
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const id of ids) {
      const point = normalizedPoint(positions[id]);
      const size = normalizeSize(sizes?.[id], {}, DEFAULT_NODE_SIZE);
      minX = Math.min(minX, point.x);
      minY = Math.min(minY, point.y);
      maxX = Math.max(maxX, point.x + size.width);
      maxY = Math.max(maxY, point.y + size.height);
    }
    return {
      x: Math.min(0, minX),
      y: Math.min(0, minY),
      width: Math.max(minimum.width, maxX - Math.min(0, minX)),
      height: Math.max(minimum.height, maxY - Math.min(0, minY)),
    };
  }

  function arrangeTraceTopologyView(view, components, links, options = {}) {
    const source = unwrapTraceView(view);
    const nextOptions = object(options);
    const list = components == null ? source.components : components;
    const edges = links == null ? source.links : links;
    const sourceLayout = object(source.layout);
    const layout = defaultTraceTopologyLayout(list, edges, {
      ...nextOptions,
      groups: source.groups || [],
      nodeSizes: sizeSource(sourceLayout),
    });
    const normalized = normalizeTraceTopologyView({
      ...source,
      layout: {
        ...sourceLayout,
        algorithm: layout.algorithm,
        positions: layout.positions,
        node_sizes: layout.node_sizes,
        bounds: layout.bounds,
      },
    }, list, edges, nextOptions);
    return normalized;
  }

  function fitTraceViewport(view, viewportSize, options = {}) {
    const source = normalizeTraceTopologyView(view, options.components, options.links, options);
    const size = object(viewportSize);
    const width = Math.max(1, finite(size.width ?? size.clientWidth, 1));
    const height = Math.max(1, finite(size.height ?? size.clientHeight, 1));
    const padding = object(options.padding);
    const paddingX = Number.isFinite(Number(options.padding)) ? Math.max(0, Number(options.padding)) : Math.max(0, finite(padding.x ?? padding.left, 24));
    const paddingY = Number.isFinite(Number(options.padding)) ? Math.max(0, Number(options.padding)) : Math.max(0, finite(padding.y ?? padding.top, 24));
    const bounds = normalizeBounds(source.layout.bounds, DEFAULT_BOUNDS);
    const usableWidth = Math.max(1, width - paddingX * 2);
    const usableHeight = Math.max(1, height - paddingY * 2);
    const scale = clamp(Math.min(usableWidth / Math.max(1, bounds.width), usableHeight / Math.max(1, bounds.height)), finite(options.minScale, MIN_SCALE), finite(options.maxScale, MAX_SCALE));
    const viewport = {
      x: (width - bounds.width * scale) / 2 - bounds.x * scale,
      y: (height - bounds.height * scale) / 2 - bounds.y * scale,
      scale,
    };
    const result = clone(source);
    result.viewport = viewport;
    return result;
  }

  function updateViewPosition(view, id, position) {
    const source = normalizeTraceTopologyView(view);
    const next = clone(source);
    const key = asId(id);
    if (!key) return next;
    const point = normalizedPoint(position, next.layout.positions[key] || { x: 0, y: 0 });
    next.layout.positions[key] = clone(point);
    next.layout.bounds = computeViewBounds(next.layout.positions, next.layout.node_sizes, next.layout.bounds);
    return next;
  }

  function dragTraceNode(view, id, target, options = {}) {
    const source = normalizeTraceTopologyView(view);
    const key = asId(id);
    const current = source.layout.positions[key] || { x: 0, y: 0 };
    const desired = object(target);
    const isDelta = options.delta === true || hasFinite(desired.dx) || hasFinite(desired.dy);
    const point = isDelta
      ? { x: current.x + finite(desired.dx, 0), y: current.y + finite(desired.dy, 0) }
      : normalizedPoint(desired.position || desired, current);
    return updateViewPosition(source, key, point);
  }

  function panTraceViewport(view, delta, options = {}) {
    const source = normalizeTraceTopologyView(view);
    const next = clone(source);
    const value = object(delta);
    const dx = finite(value.dx, hasFinite(value.x) ? value.x : 0);
    const dy = finite(value.dy, hasFinite(value.y) ? value.y : 0);
    if (options.absolute === true) {
      next.viewport.x = dx;
      next.viewport.y = dy;
    } else {
      next.viewport.x += dx;
      next.viewport.y += dy;
    }
    return next;
  }

  function zoomTraceViewport(view, factor, anchor, options = {}) {
    const source = normalizeTraceTopologyView(view);
    let amount = factor;
    let point = anchor;
    if (factor && typeof factor === "object") {
      const value = factor;
      amount = value.factor ?? value.scale;
      if (amount == null && hasFinite(value.deltaY)) amount = Math.exp(-Number(value.deltaY) * 0.001);
      point = value.anchor || value.point || anchor;
    }
    amount = finite(amount, 1);
    if (amount <= 0) amount = 1;
    const currentScale = source.viewport.scale;
    const scale = clamp(currentScale * amount, finite(options.minScale, MIN_SCALE), finite(options.maxScale, MAX_SCALE));
    const pivot = normalizedPoint(point, { x: 0, y: 0 });
    const world = {
      x: (pivot.x - source.viewport.x) / currentScale,
      y: (pivot.y - source.viewport.y) / currentScale,
    };
    const next = clone(source);
    next.viewport.scale = scale;
    next.viewport.x = pivot.x - world.x * scale;
    next.viewport.y = pivot.y - world.y * scale;
    return next;
  }

  function setTraceViewport(view, viewport) {
    const next = clone(normalizeTraceTopologyView(view));
    next.viewport = normalizeViewport(viewport, next.viewport);
    return next;
  }

  function applyTraceViewAction(view, action, options = {}) {
    const value = object(action);
    const type = String(value.type || value.action || "").toLowerCase();
    if (type === "arrange" || type === "layout" || type === "auto_layout") return arrangeTraceTopologyView(view, options.components, options.links, options);
    if (type === "fit" || type === "fit_viewport") return fitTraceViewport(view, value.viewport || options.viewport || value, options);
    if (type === "drag" || type === "drag_node") return dragTraceNode(view, value.component_id || value.node_id || value.id, value.position || value, { ...options, delta: value.delta === true || value.dx != null || value.dy != null });
    if (type === "pan") return panTraceViewport(view, value, options);
    if (type === "zoom") return zoomTraceViewport(view, value, value.anchor || options.anchor, options);
    if (type === "set_viewport") return setTraceViewport(view, value.viewport || value);
    return normalizeTraceTopologyView(view);
  }

  // -----------------------------------------------------------------------
  // Logical memory layout normalization
  // -----------------------------------------------------------------------

  function memorySource(raw) {
    return object(raw);
  }

  function memoryEntries(raw) {
    const source = memorySource(raw);
    const result = [];
    const direct = source.components;
    if (direct && typeof direct === "object" && !Array.isArray(direct)) {
      let index = 0;
      for (const [componentId, value] of Object.entries(direct)) {
        array(value).forEach((item) => result.push({ key: String(componentId), segment: object(item), index: index++ }));
      }
    }
    return result;
  }

  function segmentComponentId(segment, key) {
    const componentId = asId(segment.component_id);
    return componentId && componentId === asId(key) ? componentId : "";
  }

  function segmentOffset(segment) {
    const value = segment.offset_bytes;
    return hasFinite(value) ? Math.max(0, Number(value)) : null;
  }

  function segmentLength(segment) {
    const value = segment.length_bytes;
    return hasFinite(value) ? Math.max(0, Number(value)) : null;
  }

  function normalizeMemorySegment(segment, componentId) {
    const next = clone(object(segment));
    next.component_id = componentId;
    return next;
  }

  function mergeMemorySegments(raw) {
    const groups = {};
    for (const entry of memoryEntries(raw)) {
      const componentId = segmentComponentId(entry.segment, entry.key);
      if (!componentId || segmentOffset(entry.segment) == null || segmentLength(entry.segment) == null) continue;
      (groups[componentId] ||= []).push(normalizeMemorySegment(entry.segment, componentId));
    }
    const sorted = {};
    for (const id of Object.keys(groups).sort()) sorted[id] = groups[id];
    return sorted;
  }

  function effectiveSegmentEnd(segment) {
    const offset = segmentOffset(segment);
    const length = segmentLength(segment);
    return offset != null && length != null ? offset + length : null;
  }

  function buildMemoryComponentModels(raw, options = {}) {
    const source = memorySource(raw);
    const segmentsByComponent = mergeMemorySegments(raw);
    const declaredTotals = object(source.component_totals_bytes);
    const fidelity = asId(source.fidelity || options.fidelity || "");
    const models = {};
    for (const [componentId, segments] of Object.entries(segmentsByComponent)) {
      const maximumEnd = Math.max(0, ...segments.map(effectiveSegmentEnd).filter((value) => value != null));
      const declared = hasFinite(declaredTotals[componentId]) ? Math.max(0, Number(declaredTotals[componentId])) : 0;
      const segmentFidelities = Array.from(new Set(segments.map((item) => item.fidelity).filter(Boolean).map(String)));
      models[componentId] = {
        component_id: componentId,
        segments: clone(segments),
        total_bytes: Math.max(maximumEnd, declared),
        declared_total_bytes: declared || null,
        fidelity: segmentFidelities.length === 1 ? segmentFidelities[0] : fidelity || null,
        address_space: asId(source.address_space || "component_local_logical_bytes"),
        physical_addressing: asId(source.physical_addressing || "not_modeled"),
        not_jedec_addressing: source.not_jedec_addressing !== false,
      };
    }
    return models;
  }

  function normalizeMemoryLayout(raw, trace = {}, options = {}) {
    const source = memorySource(raw);
    const components = mergeMemorySegments(raw);
    const models = buildMemoryComponentModels(raw, { ...options, fidelity: options.fidelity || trace.fidelity });
    const declaredTotals = object(source.component_totals_bytes);
    const totals = {};
    const allIds = new Set([...Object.keys(declaredTotals), ...Object.keys(components)]);
    for (const id of Array.from(allIds).sort()) {
      const end = Math.max(0, ...(components[id] || []).map(effectiveSegmentEnd).filter((value) => value != null));
      totals[id] = Math.max(hasFinite(declaredTotals[id]) ? Number(declaredTotals[id]) : 0, end);
    }
    const rawObject = object(source);
    const layoutFidelity = asId(rawObject.fidelity || object(trace).fidelity || options.fidelity || "");
    const segmentCount = hasFinite(rawObject.segment_count) ? Number(rawObject.segment_count) : Object.values(components).reduce((sum, items) => sum + items.length, 0);
    const returnedCount = hasFinite(rawObject.returned_segment_count) ? Number(rawObject.returned_segment_count) : Object.values(components).reduce((sum, items) => sum + items.length, 0);
    const result = {
      schema_version: asId(rawObject.schema_version || "1.0"),
      address_space: asId(rawObject.address_space || "component_local_logical_bytes"),
      physical_addressing: asId(rawObject.physical_addressing || "not_modeled"),
      not_jedec_addressing: rawObject.not_jedec_addressing !== false,
      fidelity: layoutFidelity || null,
      components: clone(components),
      component_models: clone(models),
      component_totals_bytes: totals,
      segment_count: segmentCount,
      returned_segment_count: returnedCount,
      segment_limit: hasFinite(rawObject.segment_limit) ? Number(rawObject.segment_limit) : returnedCount,
      truncated: rawObject.truncated === true,
      limitations: array(rawObject.limitations).map(String),
    };
    if (!result.limitations.some((item) => String(item).includes("JEDEC"))) {
      result.limitations.push("内存偏移仅表示组件本地逻辑字节区间；未建模 JEDEC bank/row/column 物理寻址。");
    }
    return result;
  }

  // -----------------------------------------------------------------------
  // Trace activity and orthogonal particle geometry
  // -----------------------------------------------------------------------

  function intervalActive(item, timeNs) {
    const source = object(item);
    const start = hasFinite(source.start_ns) ? Number(source.start_ns) : null;
    const end = hasFinite(source.end_ns) ? Number(source.end_ns) : null;
    if (timeNs == null || !hasFinite(timeNs)) return true;
    const time = Number(timeNs);
    if (start != null && end != null && start === end) return time === start;
    return (start == null || start <= time) && (end == null || time < end);
  }

  function eventArray(events) {
    if (Array.isArray(events)) return events.map(object);
    if (events && Array.isArray(events.events)) return events.events.map(object);
    return events && typeof events === "object" ? [object(events)] : [];
  }

  function addResourceNodeIds(set, resource, knownIds) {
    const componentId = resource.component_id;
    if (componentId) set.add(String(componentId));
    const resourceId = asId(resource.resource_id);
    for (const id of knownIds || []) {
      if (resourceId === id || resourceId.startsWith(`${id}.`) || resourceId.startsWith(`component.${id}.`)) set.add(id);
    }
  }

  function activeTraceSets(events, timeNs, componentIds) {
    const known = new Set(array(componentIds).map(asId).filter(Boolean));
    const nodeIds = new Set();
    const linkIds = new Set();
    const resourceIds = new Set();
    const activeEvents = [];
    for (const event of eventArray(events)) {
      if (!intervalActive(event, timeNs)) continue;
      activeEvents.push(event);
      const transfer = object(event.transfer);
      for (const id of [transfer.source_component, transfer.target_component, event.tensor?.component_id, event.rank?.component_id]) {
        if (id) nodeIds.add(String(id));
      }
      if (transfer.link_id) linkIds.add(String(transfer.link_id));
      for (const hop of array(transfer.hops)) {
        if (!intervalActive(hop, timeNs)) continue;
        if (hop.link_id) linkIds.add(String(hop.link_id));
        if (hop.resource_id) {
          linkIds.add(String(hop.resource_id));
          resourceIds.add(String(hop.resource_id));
        }
        if (hop.source_component) nodeIds.add(String(hop.source_component));
        if (hop.target_component) nodeIds.add(String(hop.target_component));
      }
      for (const resource of array(event.resources)) {
        if (!intervalActive(resource, timeNs)) continue;
        if (resource.resource_id) resourceIds.add(String(resource.resource_id));
        if (resource.resource_id) linkIds.add(String(resource.resource_id));
        addResourceNodeIds(nodeIds, resource, known);
      }
    }
    return {
      nodeIds,
      linkIds,
      resourceIds,
      activeEvents,
      activeNodeIds: nodeIds,
      activeLinkIds: linkIds,
      nodes: nodeIds,
      links: linkIds,
    };
  }

  function normalizePolyline(points) {
    const result = [];
    for (const value of array(points)) {
      const point = Array.isArray(value) ? { x: value[0], y: value[1] } : object(value);
      if (!hasFinite(point.x) || !hasFinite(point.y)) continue;
      const normalized = { x: Number(point.x), y: Number(point.y) };
      const previous = result[result.length - 1];
      if (!previous || previous.x !== normalized.x || previous.y !== normalized.y) result.push(normalized);
    }
    return result;
  }

  function polylineTotalLength(points) {
    const path = normalizePolyline(points);
    let length = 0;
    for (let index = 1; index < path.length; index += 1) {
      length += Math.abs(path[index].x - path[index - 1].x) + Math.abs(path[index].y - path[index - 1].y);
    }
    return length;
  }

  function pointAtDistance(points, distance) {
    const path = normalizePolyline(points);
    if (!path.length) return null;
    if (path.length === 1) return clone(path[0]);
    const total = polylineTotalLength(path);
    if (total <= 0) return clone(path[0]);
    const target = clamp(finite(distance, 0), 0, total);
    let traversed = 0;
    for (let index = 1; index < path.length; index += 1) {
      const left = path[index - 1];
      const right = path[index];
      const segment = Math.abs(right.x - left.x) + Math.abs(right.y - left.y);
      if (target <= traversed + segment || index === path.length - 1) {
        const ratio = segment > 0 ? clamp((target - traversed) / segment, 0, 1) : 0;
        return { x: left.x + (right.x - left.x) * ratio, y: left.y + (right.y - left.y) * ratio };
      }
      traversed += segment;
    }
    return clone(path[path.length - 1]);
  }

  function pointAtProgress(points, progress) {
    const value = progress && typeof progress === "object" ? progress.progress ?? progress.ratio ?? progress.t : progress;
    const total = polylineTotalLength(points);
    return pointAtDistance(points, total * clamp(finite(value, 0), 0, 1));
  }

  function loopProgress(value, phase = 0) {
    const raw = finite(value, 0) + finite(phase, 0);
    let progress = raw % 1;
    if (progress < 0) progress += 1;
    return Object.is(progress, -0) ? 0 : progress;
  }

  // RAF timestamps are relative to the page lifetime, while interval fallbacks
  // commonly pass Date.now().  Taking the modulo before dividing keeps both
  // clocks numerically stable and, importantly, wraps instead of being clamped
  // forever at the route endpoint.
  function particleLoopProgress(timestampMs, periodOrOptions = 1200, phaseArg = 0) {
    const options = periodOrOptions && typeof periodOrOptions === "object" ? periodOrOptions : {};
    const periodMs = Math.max(1, finite(
      periodOrOptions && typeof periodOrOptions === "object" ? options.periodMs ?? options.durationMs : periodOrOptions,
      1200,
    ));
    const phase = finite(options.phase ?? options.offset ?? phaseArg, 0);
    const timestamp = finite(timestampMs, 0);
    const withinPeriod = ((timestamp % periodMs) + periodMs) % periodMs;
    return loopProgress(withinPeriod / periodMs, phase);
  }

  function routeLookup(routes, key) {
    if (!routes || !key) return null;
    if (routes instanceof Map) return routes.get(String(key)) || routes.get(key) || null;
    return object(routes)[String(key)] || null;
  }

  function routePointsForTransfer(transfer, routes = {}) {
    const source = object(transfer);
    const direct = source.points || source.route_points || source.polyline || (Array.isArray(source.route) ? source.route : null);
    if (direct) return normalizePolyline(direct);
    if (source.route && typeof source.route === "object" && Array.isArray(source.route.points)) return normalizePolyline(source.route.points);
    const joined = [];
    for (const hop of array(source.hops)) {
      const route = routeLookup(routes, hop.link_id || hop.resource_id) || object(hop);
      const points = normalizePolyline(route.points || route.route_points || route.polyline);
      for (const point of points) {
        const previous = joined[joined.length - 1];
        if (!previous || previous.x !== point.x || previous.y !== point.y) joined.push(point);
      }
    }
    if (joined.length) return joined;
    const keyed = [source.transfer_id, source.event_id, source.link_id]
      .map((key) => routeLookup(routes, key))
      .find(Boolean);
    if (keyed) return normalizePolyline(keyed.points || keyed.route_points || keyed.polyline || keyed);
    if (source.source_point && source.target_point) return normalizePolyline([source.source_point, source.target_point]);
    return [];
  }

  function transferValue(item) {
    const source = object(item);
    return source.transfer && typeof source.transfer === "object"
      ? {
        ...source.transfer,
        event_id: source.event_id,
        event: source,
        start_ns: source.transfer.start_ns ?? source.start_ns,
        end_ns: source.transfer.end_ns ?? source.end_ns,
      }
      : source;
  }

  function transferProgress(transfer, options, index) {
    const source = object(transfer);
    const configured = options.progress;
    let value;
    if (typeof configured === "function") value = configured(source, index);
    else if (Array.isArray(configured)) value = configured[index];
    else if (configured && typeof configured === "object") value = configured[source.transfer_id || source.event_id || source.link_id] ?? configured[index];
    else value = configured;
    if (value == null) value = source.progress;
    if (value == null && hasFinite(options.timeNs) && hasFinite(source.start_ns) && hasFinite(source.end_ns)) {
      const duration = Number(source.end_ns) - Number(source.start_ns);
      value = duration > 0 ? (Number(options.timeNs) - Number(source.start_ns)) / duration : 1;
    }
    return clamp(finite(value, 0), 0, 1);
  }

  function buildTransferParticles(transfers, routes = {}, progressOrOptions) {
    let options = progressOrOptions === undefined ? {} : object(progressOrOptions);
    let routeMap = routes;
    if (typeof progressOrOptions === "number") options = { progress: progressOrOptions };
    if (routes && typeof routes === "object" && !Array.isArray(routes) && !(routes instanceof Map) && (routes.routes || routes.progress || routes.timeNs != null) && progressOrOptions === undefined) {
      options = routes;
      routeMap = options.routes || {};
    }
    const items = array(transfers).map(transferValue);
    const particles = [];
    items.forEach((transfer, index) => {
      const points = routePointsForTransfer(transfer, options.routes || routeMap);
      const baseProgress = transferProgress(transfer, options, index);
      const count = Math.max(1, Math.trunc(finite(transfer.particle_count ?? transfer.particleCount ?? options.particleCount, 1)));
      for (let particleIndex = 0; particleIndex < count; particleIndex += 1) {
        const progress = count > 1 ? (baseProgress + particleIndex / count) % 1 : baseProgress;
        particles.push({
          id: `${asId(transfer.transfer_id || transfer.event_id || transfer.link_id || `transfer-${index}`)}:${particleIndex}`,
          transfer_id: asId(transfer.transfer_id || transfer.event_id || transfer.link_id || `transfer-${index}`),
          event_id: transfer.event_id ? asId(transfer.event_id) : undefined,
          source_component: transfer.source_component,
          target_component: transfer.target_component,
          link_id: transfer.link_id,
          progress,
          points: clone(points),
          total_length: polylineTotalLength(points),
          point: pointAtProgress(points, progress),
          particle_index: particleIndex,
          particle_count: count,
          active: options.activeSets
            ? (options.activeSets.linkIds instanceof Set
              ? options.activeSets.linkIds.has(String(transfer.link_id || ""))
              : Boolean(options.activeSets.activeLinkIds?.[String(transfer.link_id || "")]))
            : true,
        });
      }
    });
    return particles;
  }

  function animationCapability(environment, options = {}) {
    const env = environment || root || {};
    const raf = typeof env.requestAnimationFrame === "function";
    const interval = typeof env.setInterval === "function";
    const reducedMotion = options.reduceMotion === true || options.prefersReducedMotion === true;
    const mode = reducedMotion ? "static" : raf ? "raf" : interval ? "interval" : "static";
    return {
      mode,
      scheduler: mode === "raf" ? "requestAnimationFrame" : mode === "interval" ? "setInterval" : "static",
      animated: mode !== "static",
      static: mode === "static",
      reducedMotion,
      hasRequestAnimationFrame: raf,
      hasSetInterval: interval,
      frameMs: Math.max(1, finite(options.frameMs, 16)),
      reason: reducedMotion ? "已启用减少动态效果，使用静态首帧" : mode === "raf" ? "requestAnimationFrame 可用" : mode === "interval" ? "requestAnimationFrame 不可用，使用 setInterval 降级" : "无可用动画调度器，使用静态首帧",
    };
  }

  // -----------------------------------------------------------------------
  // Pointer Events drag state (Chrome 108 compatible, DOM-free)
  // -----------------------------------------------------------------------

  function beginTracePointerDrag(event, componentId, origin) {
    const source = object(event);
    const id = asId(componentId);
    if (!id || source.pointerId == null) return null;
    const point = normalizedPoint(origin);
    return {
      pointerId: source.pointerId,
      componentId: id,
      startX: finite(source.clientX, 0),
      startY: finite(source.clientY, 0),
      originX: point.x,
      originY: point.y,
    };
  }

  function tracePointerDragPosition(drag, event, zoom = 1) {
    const state = object(drag);
    const source = object(event);
    if (state.pointerId == null || source.pointerId !== state.pointerId) return null;
    const scale = Math.max(0.000001, Math.abs(finite(zoom, 1)));
    return {
      x: finite(state.originX, 0) + (finite(source.clientX, state.startX) - finite(state.startX, 0)) / scale,
      y: finite(state.originY, 0) + (finite(source.clientY, state.startY) - finite(state.startY, 0)) / scale,
    };
  }

  function finishTracePointerDrag(drag, event) {
    if (!drag) return null;
    const state = object(drag);
    const source = object(event);
    if (state.pointerId == null || source.pointerId !== state.pointerId) return drag;
    const type = asId(source.type).toLowerCase();
    return type === "pointerup" || type === "pointercancel" || type === "lostpointercapture" ? null : drag;
  }

  // -----------------------------------------------------------------------
  // Fullscreen overlay state and node presentation models
  // -----------------------------------------------------------------------

  function normalizeFullscreenState(raw = {}) {
    if (typeof raw === "boolean") return { open: raw, mode: "overlay", escapeCloses: true, previousFocusId: null };
    const source = object(raw);
    return {
      open: source.open === true || source.active === true || source.fullscreen === true,
      mode: asId(source.mode || "overlay"),
      escapeCloses: source.escapeCloses !== false,
      previousFocusId: source.previousFocusId == null ? null : asId(source.previousFocusId),
    };
  }

  function setFullscreen(state, open, options = {}) {
    const next = normalizeFullscreenState(state);
    next.open = open === true;
    if (options.previousFocusId !== undefined) next.previousFocusId = options.previousFocusId == null ? null : asId(options.previousFocusId);
    return next;
  }

  function toggleFullscreen(state, force, options = {}) {
    const current = normalizeFullscreenState(state);
    const open = typeof force === "boolean" ? force : !current.open;
    return setFullscreen(current, open, options);
  }

  function closeFullscreen(state) {
    return setFullscreen(state, false);
  }

  function fullscreenEscapeAction(state, keyOrEvent) {
    const current = normalizeFullscreenState(state);
    const key = typeof keyOrEvent === "string" ? keyOrEvent : keyOrEvent?.key;
    if (current.open && current.escapeCloses && key === "Escape") return { state: closeFullscreen(current), handled: true, closed: true };
    return { state: current, handled: false, closed: false };
  }

  function nodePresentation(component) {
    const source = object(component);
    const componentId = asId(source.component_id || source.id);
    const type = componentKind(source);
    const typeText = type && type !== "component" ? type : "组件";
    return {
      component_id: componentId,
      text: componentId,
      label: componentId,
      kind: type,
      type,
      title: typeText,
      ariaLabel: `${componentId}${componentId ? "，" : ""}${typeText}`,
    };
  }

  // Event-stream filtering is intentionally separate from playback filtering.
  // These helpers only decide which rows are visible; they never change the
  // global playback sequence or its selected event.
  function traceEventTemporalState(event, timeNs) {
    const source = object(event);
    const start = finite(source.start_ns, 0);
    const end = Math.max(start, finite(source.end_ns, start));
    const time = finite(timeNs, start);
    if (time < start) return "future";
    if (start === end) return time === start ? "current" : "past";
    return time < end ? "current" : "past";
  }

  function traceEventSearchText(event) {
    const values = [];
    const visit = (value, depth = 0) => {
      if (value == null || depth > 3) return;
      if (["string", "number", "boolean"].includes(typeof value)) {
        values.push(String(value));
        return;
      }
      if (Array.isArray(value)) value.forEach((item) => visit(item, depth + 1));
      else if (typeof value === "object") Object.values(value).forEach((item) => visit(item, depth + 1));
    };
    visit(event);
    return values.join(" ").toLocaleLowerCase();
  }

  function traceEventMatchScore(event, query) {
    const normalized = String(query || "").trim().toLocaleLowerCase();
    if (!normalized) return 0;
    const tokens = normalized.split(/\s+/).filter(Boolean);
    const haystack = traceEventSearchText(event);
    if (!tokens.every((token) => haystack.includes(token))) return -1;
    const source = object(event);
    const primary = [source.event_id, source.task_id, source.name, source.operator_id, source.layer_id]
      .filter(Boolean).join(" ").toLocaleLowerCase();
    return tokens.reduce((score, token) => score + (primary === token ? 12 : primary.includes(token) ? 5 : 1), 0);
  }

  function filterSemanticTraceEvents(events, filters = {}, timeNs = 0) {
    const source = object(filters);
    const query = String(source.query || source.keyword || "").trim();
    const category = String(source.category || "");
    const phase = String(source.phase || "");
    const temporal = String(source.temporal || source.status || "");
    return array(events).map((event, index) => ({
      event,
      index,
      score: traceEventMatchScore(event, query),
      temporal: traceEventTemporalState(event, timeNs),
    })).filter((entry) => (
      entry.score >= 0
      && (!category || String(entry.event?.category || entry.event?.event_kind || "") === category)
      && (!phase || String(entry.event?.phase || "") === phase)
      && (!temporal || entry.temporal === temporal)
    )).sort((left, right) => (
      (query ? right.score - left.score : 0)
      || finite(left.event?.start_ns, 0) - finite(right.event?.start_ns, 0)
      || finite(left.event?.end_ns, 0) - finite(right.event?.end_ns, 0)
      || String(left.event?.event_id || "").localeCompare(String(right.event?.event_id || ""))
      || left.index - right.index
    )).map((entry) => entry.event);
  }

  function paginateSemanticTraceEvents(events, page = 0, pageSize = 50) {
    const size = Math.max(1, Math.min(50, Math.trunc(finite(pageSize, 50))));
    const total = array(events).length;
    const pageCount = Math.max(1, Math.ceil(total / size));
    const currentPage = Math.max(0, Math.min(pageCount - 1, Math.trunc(finite(page, 0))));
    const start = currentPage * size;
    return {
      events: array(events).slice(start, start + size),
      page: currentPage,
      pageSize: size,
      pageCount,
      total,
      start,
      end: Math.min(total, start + size),
    };
  }

  function buildTraceNodeModels(components, view, activity = {}) {
    const sourceView = normalizeTraceTopologyView(view, components);
    const nodeIds = activity.nodeIds || activity.activeNodeIds || new Set();
    const linkIds = activity.linkIds || activity.activeLinkIds || new Set();
    return componentList(components).map((component) => {
      const presentation = nodePresentation(component);
      const id = presentation.component_id;
      return {
        ...presentation,
        position: clone(sourceView.layout.positions[id] || { x: 0, y: 0 }),
        size: clone(sourceView.layout.node_sizes[id] || DEFAULT_NODE_SIZE),
        active: nodeIds instanceof Set ? nodeIds.has(id) : Boolean(nodeIds[id]),
        active_links: linkIds instanceof Set ? Array.from(linkIds).filter((value) => String(value).includes(id)) : [],
      };
    });
  }

  return Object.freeze({
    VIEW_VERSION,
    DEFAULT_VIEWPORT: clone(DEFAULT_VIEWPORT),
    DEFAULT_NODE_SIZE: clone(DEFAULT_NODE_SIZE),
    normalizeTraceTopologyView,
    normalizeNodeSizes,
    directedTraceTopologyLayout,
    defaultTraceTopologyLayout,
    arrangeTraceTopologyView,
    fitTraceViewport,
    computeViewBounds,
    dragTraceNode,
    panTraceViewport,
    zoomTraceViewport,
    setTraceViewport,
    applyTraceViewAction,
    normalizeMemoryLayout,
    mergeMemorySegments,
    buildMemoryComponentModels,
    intervalActive,
    activeTraceSets,
    normalizePolyline,
    polylineTotalLength,
    pointAtDistance,
    pointAtProgress,
    loopProgress,
    particleLoopProgress,
    routePointsForTransfer,
    buildTransferParticles,
    animationCapability,
    beginTracePointerDrag,
    tracePointerDragPosition,
    finishTracePointerDrag,
    normalizeFullscreenState,
    setFullscreen,
    toggleFullscreen,
    closeFullscreen,
    fullscreenEscapeAction,
    nodePresentation,
    buildTraceNodeModels,
    traceEventTemporalState,
    traceEventMatchScore,
    filterSemanticTraceEvents,
    paginateSemanticTraceEvents,
  });
}));
