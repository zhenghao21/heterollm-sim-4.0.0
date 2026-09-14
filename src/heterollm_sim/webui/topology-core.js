"use strict";

// Pure topology-view primitives.  This module deliberately has no DOM or network
// dependencies so the editor's geometry/state contracts can be exercised in Node.
(function topologyCoreFactory(root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.TopologyCore = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function buildTopologyCore() {
  const VIEW_VERSION = 1;
  const CLIPBOARD_TYPE = "heterollm-topology-clipboard";
  const DEFAULT_VIEWPORT = Object.freeze({ x: 0, y: 0, scale: 1 });

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function object(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  const PHYSICAL_COMPONENT_REFERENCE_FIELDS = Object.freeze(["controller_component_id", "memory_subsystem_id"]);

  function mappedComponentId(componentMap, value) {
    if (typeof value !== "string") return undefined;
    if (componentMap instanceof Map) return componentMap.get(value);
    const mappings = object(componentMap);
    return Object.hasOwn(mappings, value) ? mappings[value] : undefined;
  }

  function remapPhysicalCompositionComponentRefs(metadata, componentMap) {
    const physical = object(object(metadata).physical_composition);
    for (const field of PHYSICAL_COMPONENT_REFERENCE_FIELDS) {
      const nextId = mappedComponentId(componentMap, physical[field]);
      if (typeof nextId === "string" && nextId) physical[field] = nextId;
    }
  }

  function array(value) {
    return Array.isArray(value) ? value : [];
  }

  function finite(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function safeId(value, fallback = "item") {
    const text = String(value || fallback).trim().replace(/[^A-Za-z0-9_.:-]+/g, "-").replace(/^-+|-+$/g, "");
    return text || fallback;
  }

  function uniqueId(base, used) {
    const normalized = safeId(base);
    if (!used.has(normalized)) {
      used.add(normalized);
      return normalized;
    }
    let index = 2;
    while (used.has(`${normalized}-${index}`)) index += 1;
    const id = `${normalized}-${index}`;
    used.add(id);
    return id;
  }

  function incrementNumericSuffix(value) {
    const text = String(value || "").trim();
    const match = text.match(/^(.*?)(\d+)$/);
    if (!match) return `${text}1`;
    const suffix = BigInt(match[2]) + 1n;
    return `${match[1]}${suffix}`;
  }

  function nextNumericName(base, used, sanitize = false) {
    const raw = String(base || "").trim();
    const normalized = sanitize ? safeId(raw) : (raw || "group");
    let candidate = incrementNumericSuffix(normalized);
    while (used.has(candidate)) candidate = incrementNumericSuffix(candidate);
    used.add(candidate);
    return candidate;
  }

  function normalizedPoint(value, fallback = { x: 0, y: 0 }) {
    const source = object(value);
    return { x: finite(source.x, fallback.x), y: finite(source.y, fallback.y) };
  }

  function normalizeTopologyView(raw, componentIds = []) {
    const source = object(raw);
    const validIds = new Set(array(componentIds).map(String));
    const sourceLayout = object(source.layout);
    const sourcePositions = object(sourceLayout.positions ?? source.positions);
    const positions = {};
    for (const id of validIds) {
      const position = sourcePositions[id];
      if (position && Number.isFinite(Number(position.x)) && Number.isFinite(Number(position.y))) {
        positions[id] = normalizedPoint(position);
      }
    }

    const occupied = new Set();
    const groupIds = new Set();
    const groups = [];
    for (const candidate of array(source.groups)) {
      const group = object(candidate);
      const members = [];
      for (const member of array(group.members).map(String)) {
        if (validIds.has(member) && !occupied.has(member) && !members.includes(member)) members.push(member);
      }
      if (!members.length) continue;
      members.forEach((member) => occupied.add(member));
      const id = uniqueId(group.group_id || group.id || "group", groupIds);
      const requestedRoot = String(group.root ?? group.root_id ?? "");
      groups.push({
        group_id: id,
        label: String(group.label || id),
        members,
        root: members.includes(requestedRoot) ? requestedRoot : members[0],
        collapsed: group.collapsed === true,
      });
    }

    const viewportSource = object(source.viewport);
    return {
      version: VIEW_VERSION,
      groups,
      layout: {
        algorithm: String(sourceLayout.algorithm || "group-layered-v1"),
        positions,
        bounds: normalizeBounds(sourceLayout.bounds),
      },
      viewport: {
        x: finite(viewportSource.x, DEFAULT_VIEWPORT.x),
        y: finite(viewportSource.y, DEFAULT_VIEWPORT.y),
        scale: clamp(finite(viewportSource.scale, DEFAULT_VIEWPORT.scale), 0.25, 4),
      },
    };
  }

  function normalizeBounds(value) {
    const source = object(value);
    return {
      x: finite(source.x, 0),
      y: finite(source.y, 0),
      width: Math.max(0, finite(source.width, 0)),
      height: Math.max(0, finite(source.height, 0)),
    };
  }

  function validateGroups(groups, componentIds = []) {
    const valid = new Set(array(componentIds).map(String));
    const occupied = new Set();
    const errors = [];
    for (const group of array(groups)) {
      const members = array(group.members).map(String);
      if (!members.length) errors.push(`${group.group_id || "group"}：组内没有成员`);
      if (!members.includes(String(group.root))) errors.push(`${group.group_id || "group"}：根组件不在组成员中`);
      for (const member of members) {
        if (valid.size && !valid.has(member)) errors.push(`${group.group_id || "group"}：未知成员 ${member}`);
        if (occupied.has(member)) errors.push(`${member}：同时属于多个分组`);
        occupied.add(member);
      }
    }
    return errors;
  }

  function createGroup(view, members, rootId = null, label = "") {
    const next = clone(view);
    const selected = Array.from(new Set(array(members).map(String))).sort();
    if (!selected.length) throw new Error("建立分组至少需要选择一个组件。");
    const occupied = new Set(array(next.groups).flatMap((group) => array(group.members).map(String)));
    const conflict = selected.find((id) => occupied.has(id));
    if (conflict) throw new Error(`${conflict} 已经属于另一个分组。`);
    const used = new Set(array(next.groups).map((group) => String(group.group_id)));
    const id = uniqueId(label ? safeId(label) : "group", used);
    const root = rootId && selected.includes(String(rootId)) ? String(rootId) : selected[0];
    next.groups.push({ group_id: id, label: label || id, members: selected, root, collapsed: false });
    return { view: next, group: next.groups[next.groups.length - 1] };
  }

  function setGroupRoot(view, groupId, componentId) {
    const next = clone(view);
    const group = array(next.groups).find((item) => item.group_id === groupId);
    if (!group) throw new Error(`找不到分组：${groupId}`);
    if (!array(group.members).includes(componentId)) throw new Error("根组件必须是当前分组的成员。");
    group.root = componentId;
    return next;
  }

  function removeGroup(view, groupId) {
    const next = clone(view);
    next.groups = array(next.groups).filter((group) => group.group_id !== groupId);
    return next;
  }

  function setGroupCollapsed(view, groupId, collapsed) {
    const next = clone(view);
    const group = array(next.groups).find((item) => item.group_id === groupId);
    if (!group) throw new Error(`找不到分组：${groupId}`);
    group.collapsed = collapsed === true;
    return next;
  }

  function groupForComponent(groups, componentId) {
    return array(groups).find((group) => array(group.members).includes(componentId)) || null;
  }

  function screenToWorld(point, viewport) {
    const scale = clamp(finite(viewport?.scale, 1), 0.0001, 1000);
    return {
      x: (finite(point?.x) - finite(viewport?.x)) / scale,
      y: (finite(point?.y) - finite(viewport?.y)) / scale,
    };
  }

  function worldToScreen(point, viewport) {
    const scale = clamp(finite(viewport?.scale, 1), 0.0001, 1000);
    return {
      x: finite(point?.x) * scale + finite(viewport?.x),
      y: finite(point?.y) * scale + finite(viewport?.y),
    };
  }

  function normalizedRect(a, b) {
    const x1 = Math.min(finite(a?.x), finite(b?.x));
    const y1 = Math.min(finite(a?.y), finite(b?.y));
    const x2 = Math.max(finite(a?.x), finite(b?.x));
    const y2 = Math.max(finite(a?.y), finite(b?.y));
    return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
  }

  function rectsIntersect(a, b) {
    return a.x <= b.x + b.width && a.x + a.width >= b.x && a.y <= b.y + b.height && a.y + a.height >= b.y;
  }

  function marqueeSelection(startScreen, endScreen, viewport, nodeRects, mode = "replace", existing = []) {
    const start = screenToWorld(startScreen, viewport);
    const end = screenToWorld(endScreen, viewport);
    const marquee = normalizedRect(start, end);
    const hits = Object.keys(object(nodeRects)).filter((id) => rectsIntersect(marquee, object(nodeRects)[id])).sort();
    const selected = mode === "replace" ? new Set() : new Set(array(existing).map(String));
    for (const id of hits) {
      if (mode === "toggle" && selected.has(id)) selected.delete(id);
      else selected.add(id);
    }
    return Array.from(selected).sort();
  }

  function collisionFree(rects, obstacles, gap) {
    const expanded = obstacles.map((rect) => ({
      ...rect,
      x: rect.x - gap,
      y: rect.y - gap,
      width: rect.width + gap * 2,
      height: rect.height + gap * 2,
    }));
    return rects.every((rect) => expanded.every((obstacle) => !rectsIntersect(rect, obstacle)));
  }

  function spiralOffsets(step, maxRings) {
    const offsets = [{ x: 0, y: 0 }];
    for (let ring = 1; ring <= maxRings; ring += 1) {
      const candidates = [];
      for (let x = -ring; x <= ring; x += 1) {
        candidates.push({ x: x * step, y: -ring * step }, { x: x * step, y: ring * step });
      }
      for (let y = -ring + 1; y < ring; y += 1) {
        candidates.push({ x: -ring * step, y: y * step }, { x: ring * step, y: y * step });
      }
      candidates.sort((a, b) => (a.x * a.x + a.y * a.y) - (b.x * b.x + b.y * b.y) || a.y - b.y || a.x - b.x);
      offsets.push(...candidates);
    }
    return offsets;
  }

  function resolveCollisionPlacement(allPositions, sizes, movingPositions, options = {}) {
    const current = object(allPositions);
    const desired = object(movingPositions);
    const movingIds = Object.keys(desired).sort();
    if (!movingIds.length) return { positions: {}, adjusted: false, offset: { x: 0, y: 0 } };
    const movingSet = new Set(movingIds);
    const obstacles = Object.keys(current)
      .filter((id) => !movingSet.has(id))
      .map((id) => rectForNode(id, current, sizes));
    const baseRects = movingIds.map((id) => rectForNode(id, desired, sizes));
    for (let left = 0; left < baseRects.length; left += 1) for (let right = left + 1; right < baseRects.length; right += 1) {
      if (rectsIntersect(baseRects[left], baseRects[right])) throw new Error("所移动的组件之间发生了重叠。");
    }
    const gap = Math.max(0, finite(options.gap, 14));
    const step = Math.max(4, finite(options.step, 24));
    const maxRings = Math.max(1, Math.trunc(finite(options.maxRings, 80)));
    for (const offset of spiralOffsets(step, maxRings)) {
      const candidateRects = baseRects.map((rect) => ({ ...rect, x: rect.x + offset.x, y: rect.y + offset.y }));
      if (!collisionFree(candidateRects, obstacles, gap)) continue;
      const positions = {};
      movingIds.forEach((id) => {
        positions[id] = { x: finite(desired[id].x) + offset.x, y: finite(desired[id].y) + offset.y };
      });
      return { positions, adjusted: Boolean(offset.x || offset.y), offset };
    }
    throw new Error("在当前搜索范围内找不到无碰撞的放置位置。");
  }

  function collapseProjection(components, links, groups) {
    const componentIds = array(components).map((component) => String(component.component_id));
    const visible = new Set(componentIds);
    const proxyFor = new Map();
    for (const group of array(groups)) {
      if (!group.collapsed) continue;
      for (const member of array(group.members)) {
        proxyFor.set(member, group.root);
        if (member !== group.root) visible.delete(member);
      }
    }

    const projected = [];
    const aggregate = new Map();
    for (const rawLink of array(links)) {
      const link = object(rawLink);
      const source = proxyFor.get(link.source_component) || link.source_component;
      const target = proxyFor.get(link.target_component) || link.target_component;
      if (source === target) continue;
      const wasProjected = source !== link.source_component || target !== link.target_component;
      if (!wasProjected) {
        projected.push({ ...clone(link), display_id: String(link.link_id), original_link_ids: [String(link.link_id)], aggregate_count: 1, projected: false });
        continue;
      }
      const unordered = [String(source), String(target)].sort();
      const directional = link.bidirectional === false;
      const endpointKey = directional ? `${String(source)}\u0000${String(target)}` : `${unordered[0]}\u0000${unordered[1]}`;
      const projectionType = directional ? "directed" : "bidirectional";
      const key = `${projectionType}\u0000${endpointKey}\u0000${String(link.protocol || "")}`;
      let item = aggregate.get(key);
      if (!item) {
        const displayEndpoints = directional ? [String(source), String(target)] : unordered;
        item = {
          ...clone(link),
          source_component: source,
          target_component: target,
          display_id: `proxy:${projectionType}:${safeId(displayEndpoints[0])}:${safeId(displayEndpoints[1])}:${safeId(link.protocol || "link")}`,
          original_link_ids: [],
          aggregate_count: 0,
          bandwidth_gbps: 0,
          projected: true,
        };
        aggregate.set(key, item);
        projected.push(item);
      }
      item.original_link_ids.push(String(link.link_id));
      item.aggregate_count += 1;
      item.bandwidth_gbps += finite(link.bandwidth_gbps, 0);
    }
    return { visibleComponentIds: Array.from(visible), links: projected };
  }

  function componentSize(id, sizes) {
    const size = object(sizes)[id] || {};
    return { width: Math.max(40, finite(size.width, 130)), height: Math.max(30, finite(size.height, 66)) };
  }

  function buildUnits(components, groups, sizes, nodeGap) {
    const ids = array(components).map((item) => String(item.component_id)).sort();
    const claimed = new Set();
    const units = [];
    for (const group of array(groups).slice().sort((a, b) => String(a.group_id).localeCompare(String(b.group_id)))) {
      const members = array(group.members).filter((id) => ids.includes(id)).sort();
      if (!members.length) continue;
      members.forEach((id) => claimed.add(id));
      const columns = Math.max(1, Math.ceil(Math.sqrt(members.length)));
      const columnWidths = Array(columns).fill(0);
      const rows = Math.ceil(members.length / columns);
      const rowHeights = Array(rows).fill(0);
      members.forEach((id, index) => {
        const size = componentSize(id, sizes);
        columnWidths[index % columns] = Math.max(columnWidths[index % columns], size.width);
        rowHeights[Math.floor(index / columns)] = Math.max(rowHeights[Math.floor(index / columns)], size.height);
      });
      const local = {};
      const header = 34;
      const columnStarts = columnWidths.map((_value, column) => nodeGap + columnWidths.slice(0, column).reduce((sum, value) => sum + value + nodeGap, 0));
      const rowStarts = rowHeights.map((_value, row) => header + nodeGap + rowHeights.slice(0, row).reduce((sum, value) => sum + value + nodeGap, 0));
      members.forEach((id, index) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        const size = componentSize(id, sizes);
        local[id] = {
          x: columnStarts[column] + (columnWidths[column] - size.width) / 2,
          y: rowStarts[row] + (rowHeights[row] - size.height) / 2,
        };
      });
      units.push({
        id: `group:${group.group_id}`,
        groupId: group.group_id,
        members,
        local,
        width: columnWidths.reduce((sum, value) => sum + value, nodeGap * (columns + 1)),
        height: rowHeights.reduce((sum, value) => sum + value, header + nodeGap * (rows + 1)),
      });
    }
    for (const id of ids) {
      if (claimed.has(id)) continue;
      const size = componentSize(id, sizes);
      units.push({ id: `node:${id}`, groupId: null, members: [id], local: { [id]: { x: 0, y: 0 } }, width: size.width, height: size.height });
    }
    return units.sort((a, b) => a.id.localeCompare(b.id));
  }

  function layoutGraph(components, links, groups = [], sizes = {}, options = {}) {
    const margin = Math.max(16, finite(options.margin, 42));
    const nodeGap = Math.max(12, finite(options.nodeGap, 28));
    const layerGap = Math.max(40, finite(options.layerGap, 120));
    const rowGap = Math.max(28, finite(options.rowGap, 72));
    const viewportWidth = Math.max(0, finite(options.viewportWidth, 0));
    const viewportHeight = Math.max(0, finite(options.viewportHeight, 0));
    const fillViewport = options.fillViewport === true && (viewportWidth > 0 || viewportHeight > 0);
    const units = buildUnits(components, groups, sizes, nodeGap);
    const unitFor = new Map();
    units.forEach((unit) => unit.members.forEach((id) => unitFor.set(id, unit.id)));
    const adjacency = new Map(units.map((unit) => [unit.id, new Set()]));
    for (const link of array(links)) {
      const source = unitFor.get(link.source_component);
      const target = unitFor.get(link.target_component);
      if (!source || !target || source === target) continue;
      adjacency.get(source).add(target);
      adjacency.get(target).add(source);
    }
    const rank = new Map();
    const remaining = new Set(units.map((unit) => unit.id));
    let rankOffset = 0;
    while (remaining.size) {
      const seed = Array.from(remaining).sort((a, b) => {
        const degree = adjacency.get(b).size - adjacency.get(a).size;
        return degree || a.localeCompare(b);
      })[0];
      const queue = [seed];
      rank.set(seed, rankOffset);
      remaining.delete(seed);
      while (queue.length) {
        const current = queue.shift();
        const neighbors = Array.from(adjacency.get(current)).sort();
        for (const neighbor of neighbors) {
          if (!remaining.has(neighbor)) continue;
          rank.set(neighbor, rank.get(current) + 1);
          remaining.delete(neighbor);
          queue.push(neighbor);
        }
      }
      const componentRanks = Array.from(rank.values());
      rankOffset = Math.max(rankOffset + 1, ...componentRanks) + 1;
    }
    const columns = new Map();
    units.forEach((unit) => {
      const value = rank.get(unit.id) || 0;
      if (!columns.has(value)) columns.set(value, []);
      columns.get(value).push(unit);
    });
    const sortedRanks = Array.from(columns.keys()).sort((a, b) => a - b);
    sortedRanks.forEach((value) => columns.get(value).sort((a, b) => a.id.localeCompare(b.id)));
    // Deterministic two-way barycentric sweeps reduce crossings between adjacent
    // layers while retaining lexical tie-breaking for reproducible layouts.
    for (let sweep = 0; sweep < 4; sweep += 1) {
      const direction = sweep % 2 === 0 ? sortedRanks : sortedRanks.slice().reverse();
      const order = new Map();
      sortedRanks.forEach((value) => columns.get(value).forEach((unit, index) => order.set(unit.id, index)));
      for (const value of direction) {
        const candidates = columns.get(value);
        candidates.sort((a, b) => {
          const average = (unit) => {
            const neighbors = Array.from(adjacency.get(unit.id)).filter((id) => rank.get(id) !== value && order.has(id));
            return neighbors.length ? neighbors.reduce((sum, id) => sum + order.get(id), 0) / neighbors.length : order.get(unit.id);
          };
          return average(a) - average(b) || a.id.localeCompare(b.id);
        });
        candidates.forEach((unit, index) => order.set(unit.id, index));
      }
    }
    const positions = {};
    const groupBounds = {};
    const columnMetrics = sortedRanks.map((value) => {
      const column = columns.get(value);
      return {
        value,
        column,
        width: Math.max(...column.map((unit) => unit.width), 0),
        height: column.reduce((sum, unit) => sum + unit.height, 0) + Math.max(0, column.length - 1) * rowGap,
      };
    });
    const contentHeight = Math.max(0, ...columnMetrics.map((metric) => metric.height));
    const totalColumnWidth = columnMetrics.reduce((sum, metric) => sum + metric.width, 0);
    const naturalWidth = totalColumnWidth + Math.max(0, columnMetrics.length - 1) * layerGap + margin * 2;
    const canvasWidth = Math.max(viewportWidth || 640, Math.ceil(naturalWidth));
    const canvasHeight = Math.max(viewportHeight || 420, Math.ceil(contentHeight + margin * 2));
    const centerAxis = canvasHeight / 2;
    const responsiveLayerGap = fillViewport && columnMetrics.length > 1
      ? Math.max(layerGap, (canvasWidth - margin * 2 - totalColumnWidth) / (columnMetrics.length - 1))
      : layerGap;
    let x = columnMetrics.length === 1
      ? Math.max(margin, (canvasWidth - columnMetrics[0].width) / 2)
      : margin;
    for (const metric of columnMetrics) {
      const responsiveRowGap = fillViewport && metric.column.length > 1
        ? Math.max(rowGap, (canvasHeight - margin * 2 - metric.column.reduce((sum, unit) => sum + unit.height, 0)) / (metric.column.length - 1))
        : rowGap;
      let y = metric.column.length === 1
        ? centerAxis - metric.column[0].height / 2
        : Math.max(margin, centerAxis - (metric.column.reduce((sum, unit) => sum + unit.height, 0) + responsiveRowGap * (metric.column.length - 1)) / 2);
      for (const unit of metric.column) {
        const unitX = x + (metric.width - unit.width) / 2;
        for (const id of unit.members) {
          positions[id] = { x: Math.round(unitX + unit.local[id].x), y: Math.round(y + unit.local[id].y) };
        }
        if (unit.groupId) groupBounds[unit.groupId] = { x: Math.round(unitX), y: Math.round(y), width: Math.round(unit.width), height: Math.round(unit.height) };
        y += unit.height + responsiveRowGap;
      }
      x += metric.width + responsiveLayerGap;
    }
    return {
      positions,
      groupBounds,
      bounds: { x: 0, y: 0, width: canvasWidth, height: canvasHeight },
      algorithm: fillViewport ? "group-layered-viewport-v2" : "group-layered-v1",
    };
  }

  function rectForNode(id, positions, sizes, clearance = 0) {
    const position = normalizedPoint(object(positions)[id]);
    const size = componentSize(id, sizes);
    return {
      id,
      x: position.x - clearance,
      y: position.y - clearance,
      width: size.width + clearance * 2,
      height: size.height + clearance * 2,
    };
  }

  function center(rect) {
    return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
  }

  function anchors(source, target) {
    const a = center(source);
    const b = center(target);
    if (Math.abs(b.x - a.x) >= Math.abs(b.y - a.y)) {
      const direction = b.x >= a.x ? 1 : -1;
      return {
        start: { x: direction > 0 ? source.x + source.width : source.x, y: a.y },
        end: { x: direction > 0 ? target.x : target.x + target.width, y: b.y },
      };
    }
    const direction = b.y >= a.y ? 1 : -1;
    return {
      start: { x: a.x, y: direction > 0 ? source.y + source.height : source.y },
      end: { x: b.x, y: direction > 0 ? target.y : target.y + target.height },
    };
  }

  function portIdentity(componentId, portId) {
    return JSON.stringify([String(componentId ?? ""), String(portId ?? "")]);
  }

  function rectLookup(rects) {
    const lookup = new Map();
    if (Array.isArray(rects)) {
      rects.forEach((rect) => {
        if (rect?.id !== undefined && rect?.id !== null) lookup.set(String(rect.id), rect);
      });
    } else {
      Object.entries(object(rects)).forEach(([id, rect]) => {
        if (rect && typeof rect === "object") lookup.set(String(id), rect.id === undefined ? { ...rect, id } : rect);
      });
    }
    return lookup;
  }

  function endpointSide(ownerRect, counterpartRects, roleBalance = 0) {
    const owner = center(ownerRect);
    const counterparts = counterpartRects.length ? counterpartRects : [owner];
    const delta = counterparts.reduce((sum, rect) => {
      const other = center(rect);
      return { x: sum.x + other.x - owner.x, y: sum.y + other.y - owner.y };
    }, { x: 0, y: 0 });
    if (Math.abs(delta.x) >= Math.abs(delta.y)) {
      if (delta.x) return delta.x > 0 ? "right" : "left";
      return roleBalance >= 0 ? "right" : "left";
    }
    return delta.y > 0 ? "bottom" : "top";
  }

  function pointOnRectSide(rect, side, fraction) {
    const slot = clamp(finite(fraction, 0.5), 0, 1);
    if (side === "left" || side === "right") {
      return { x: side === "right" ? rect.x + rect.width : rect.x, y: rect.y + rect.height * slot };
    }
    return { x: rect.x + rect.width * slot, y: side === "bottom" ? rect.y + rect.height : rect.y };
  }

  // Assign one stable boundary point per real (component_id, port_id) identity.
  // All links using the same port share the same point; different ports on the
  // same side are placed at deterministic fractions rather than link-order slots.
  function assignPortEndpoints(links, rects, options = {}) {
    const lookup = rectLookup(rects);
    const usages = new Map();
    const input = array(links);
    const sideOverrides = object(options.portSides);
    const remember = (componentId, portId, counterpartId, role) => {
      const component = String(componentId ?? "");
      const port = String(portId ?? "");
      const key = portIdentity(component, port);
      if (!lookup.has(component)) throw new Error(`找不到组件 ${component || "<空>"} 的连线矩形。`);
      if (!lookup.has(String(counterpartId ?? ""))) throw new Error(`找不到组件 ${String(counterpartId ?? "") || "<空>"} 的连线矩形。`);
      if (!usages.has(key)) usages.set(key, { key, componentId: component, portId: port, counterpartIds: [], roleBalance: 0 });
      const usage = usages.get(key);
      usage.counterpartIds.push(String(counterpartId));
      usage.roleBalance += role === "source" ? 1 : -1;
      return key;
    };
    const records = input.map((link, index) => ({
      link,
      index,
      linkId: String(link?.display_id || link?.link_id || index),
      sourcePortKey: remember(link?.source_component, link?.source_port, link?.target_component, "source"),
      targetPortKey: remember(link?.target_component, link?.target_port, link?.source_component, "target"),
    }));
    const placements = new Map();
    usages.forEach((usage) => {
      const rect = lookup.get(usage.componentId);
      const side = ["left", "right", "top", "bottom"].includes(sideOverrides[usage.key])
        ? sideOverrides[usage.key]
        : endpointSide(rect, usage.counterpartIds.map((id) => lookup.get(id)), usage.roleBalance);
      placements.set(usage.key, { ...usage, side });
    });
    const sideGroups = new Map();
    placements.forEach((placement) => {
      const key = `${placement.componentId}\u001f${placement.side}`;
      if (!sideGroups.has(key)) sideGroups.set(key, []);
      sideGroups.get(key).push(placement);
    });
    sideGroups.forEach((ports) => {
      ports.sort((a, b) => a.portId.localeCompare(b.portId) || a.key.localeCompare(b.key));
      ports.forEach((placement, index) => {
        placement.fraction = (index + 1) / (ports.length + 1);
        placement.point = pointOnRectSide(lookup.get(placement.componentId), placement.side, placement.fraction);
      });
    });
    const byPort = {};
    Array.from(placements.keys()).sort().forEach((key) => {
      const placement = placements.get(key);
      byPort[key] = {
        componentId: placement.componentId,
        portId: placement.portId,
        side: placement.side,
        fraction: placement.fraction,
        point: { ...placement.point },
      };
    });
    return {
      byPort,
      links: records.map((record) => ({
        ...record,
        sourcePoint: { ...placements.get(record.sourcePortKey).point },
        targetPoint: { ...placements.get(record.targetPortKey).point },
        sourceSide: placements.get(record.sourcePortKey).side,
        targetSide: placements.get(record.targetPortKey).side,
      })),
    };
  }

  function pointInsideRect(point, rect) {
    return point.x > rect.x && point.x < rect.x + rect.width && point.y > rect.y && point.y < rect.y + rect.height;
  }

  function segmentHitsRect(a, b, rect) {
    if (a.x === b.x) {
      if (!(a.x > rect.x && a.x < rect.x + rect.width)) return false;
      return Math.max(Math.min(a.y, b.y), rect.y) < Math.min(Math.max(a.y, b.y), rect.y + rect.height);
    }
    if (a.y === b.y) {
      if (!(a.y > rect.y && a.y < rect.y + rect.height)) return false;
      return Math.max(Math.min(a.x, b.x), rect.x) < Math.min(Math.max(a.x, b.x), rect.x + rect.width);
    }
    return true;
  }

  function expandedRect(rectValue, padding = 0) {
    const rect = object(rectValue);
    const amount = Math.max(0, finite(padding, 0));
    return {
      ...rect,
      x: finite(rect.x) - amount,
      y: finite(rect.y) - amount,
      width: Math.max(0, finite(rect.width)) + amount * 2,
      height: Math.max(0, finite(rect.height)) + amount * 2,
    };
  }

  // General segment/rectangle clipping is used for sampled Bezier and rounded
  // geometry. The older segmentHitsRect helper intentionally keeps its strict
  // orthogonal boundary semantics for route-grid construction.
  function lineSegmentHitsRect(aValue, bValue, rectValue, padding = 0) {
    const a = normalizedPoint(aValue);
    const b = normalizedPoint(bValue);
    const rect = expandedRect(rectValue, padding);
    const minimumX = rect.x;
    const maximumX = rect.x + rect.width;
    const minimumY = rect.y;
    const maximumY = rect.y + rect.height;
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    let lower = 0;
    let upper = 1;
    for (const [p, q] of [[-dx, a.x - minimumX], [dx, maximumX - a.x], [-dy, a.y - minimumY], [dy, maximumY - a.y]]) {
      if (Math.abs(p) < 1e-9) {
        if (q < -1e-9) return false;
        continue;
      }
      const ratio = q / p;
      if (p < 0) lower = Math.max(lower, ratio);
      else upper = Math.min(upper, ratio);
      if (lower - upper > 1e-9) return false;
    }
    return upper >= -1e-9 && lower <= 1 + 1e-9;
  }

  function rectsIntersectWithGap(firstValue, secondValue, gap = 0) {
    return rectsIntersect(expandedRect(firstValue, Math.max(0, finite(gap, 0)) / 2), expandedRect(secondValue, Math.max(0, finite(gap, 0)) / 2));
  }

  function simplifyPath(points) {
    const clean = [];
    for (const point of points) {
      const last = clean[clean.length - 1];
      if (!last || last.x !== point.x || last.y !== point.y) clean.push(point);
    }
    return clean.filter((point, index) => {
      if (!index || index === clean.length - 1) return true;
      const before = clean[index - 1];
      const after = clean[index + 1];
      return !((before.x === point.x && point.x === after.x) || (before.y === point.y && point.y === after.y));
    });
  }

  function pathClear(points, obstacles) {
    for (let index = 1; index < points.length; index += 1) {
      if (obstacles.some((rect) => segmentHitsRect(points[index - 1], points[index], rect))) return false;
    }
    return true;
  }

  function pathCost(points) {
    let cost = 0;
    for (let index = 1; index < points.length; index += 1) {
      cost += Math.abs(points[index].x - points[index - 1].x) + Math.abs(points[index].y - points[index - 1].y);
    }
    return cost + Math.max(0, points.length - 2) * 12;
  }

  function routeSegments(points, metadata = {}) {
    const sourcePoint = points[0];
    const targetPoint = points[points.length - 1];
    const segments = [];
    for (let index = 1; index < points.length; index += 1) {
      const a = points[index - 1];
      const b = points[index];
      if (a.x === b.x && a.y === b.y) continue;
      const endpointPorts = [];
      if (index === 1 && metadata.sourcePortKey) endpointPorts.push({ key: metadata.sourcePortKey, point: { ...sourcePoint } });
      if (index === points.length - 1 && metadata.targetPortKey) endpointPorts.push({ key: metadata.targetPortKey, point: { ...targetPoint } });
      segments.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, endpointPorts, routeId: metadata.routeId });
    }
    return segments;
  }

  function normalizedOccupiedSegment(segment) {
    const a = segment?.a || segment?.start || { x: segment?.x1, y: segment?.y1 };
    const b = segment?.b || segment?.end || { x: segment?.x2, y: segment?.y2 };
    const x1 = Number(a?.x);
    const y1 = Number(a?.y);
    const x2 = Number(b?.x);
    const y2 = Number(b?.y);
    if (![x1, y1, x2, y2].every(Number.isFinite) || (x1 === x2 && y1 === y2)) return null;
    const endpointPorts = array(segment?.endpointPorts).map((entry) => ({
      key: String(entry?.key ?? ""),
      point: { x: Number(entry?.point?.x), y: Number(entry?.point?.y) },
    })).filter((entry) => entry.key && Number.isFinite(entry.point.x) && Number.isFinite(entry.point.y));
    if (segment?.portKey && segment?.endpoint) endpointPorts.push({
      key: String(segment.portKey),
      point: { x: Number(segment.endpoint.x), y: Number(segment.endpoint.y) },
    });
    return { x1, y1, x2, y2, endpointPorts, routeId: segment?.routeId };
  }

  function pointEquals(a, b) {
    return Math.abs(a.x - b.x) < 1e-7 && Math.abs(a.y - b.y) < 1e-7;
  }

  function pointOnSegment(point, segment) {
    const dx = segment.x2 - segment.x1;
    const dy = segment.y2 - segment.y1;
    const cross = (point.x - segment.x1) * dy - (point.y - segment.y1) * dx;
    if (Math.abs(cross) > 1e-7 * Math.max(1, Math.abs(dx) + Math.abs(dy))) return false;
    const dot = (point.x - segment.x1) * dx + (point.y - segment.y1) * dy;
    const squared = dx * dx + dy * dy;
    return dot >= -1e-7 && dot <= squared + 1e-7;
  }

  function collinearOverlap(a, b) {
    const adx = a.x2 - a.x1;
    const ady = a.y2 - a.y1;
    const bdx = b.x2 - b.x1;
    const bdy = b.y2 - b.y1;
    const crossDirections = adx * bdy - ady * bdx;
    const crossOrigins = adx * (b.y1 - a.y1) - ady * (b.x1 - a.x1);
    if (Math.abs(crossDirections) > 1e-7 || Math.abs(crossOrigins) > 1e-7) return null;
    const useX = Math.abs(adx) >= Math.abs(ady);
    const aStart = useX ? a.x1 : a.y1;
    const aEnd = useX ? a.x2 : a.y2;
    const bStart = useX ? b.x1 : b.y1;
    const bEnd = useX ? b.x2 : b.y2;
    const start = Math.max(Math.min(aStart, aEnd), Math.min(bStart, bEnd));
    const end = Math.min(Math.max(aStart, aEnd), Math.max(bStart, bEnd));
    if (end - start <= 1e-7) return null;
    const pointAt = (value) => {
      const denominator = aEnd - aStart;
      const ratio = Math.abs(denominator) < 1e-9 ? 0 : (value - aStart) / denominator;
      return { x: a.x1 + adx * ratio, y: a.y1 + ady * ratio };
    };
    const scale = Math.hypot(adx, ady) / Math.max(1e-9, Math.abs(aEnd - aStart));
    return { length: (end - start) * scale, start: pointAt(start), end: pointAt(end) };
  }

  function sharedEscapeAllowance(a, b, overlap, limit) {
    let allowance = 0;
    for (const first of array(a.endpointPorts)) for (const second of array(b.endpointPorts)) {
      if (first.key !== second.key || !pointEquals(first.point, second.point) || !pointOnSegment(first.point, a) || !pointOnSegment(second.point, b)) continue;
      if (!pointEquals(first.point, overlap.start) && !pointEquals(first.point, overlap.end)) continue;
      allowance = Math.max(allowance, Math.min(overlap.length, limit));
    }
    return allowance;
  }

  function perpendicularIntersection(a, b) {
    const ax = a.x2 - a.x1;
    const ay = a.y2 - a.y1;
    const bx = b.x2 - b.x1;
    const by = b.y2 - b.y1;
    const denominator = ax * by - ay * bx;
    if (Math.abs(denominator) < 1e-9) return null;
    const originX = b.x1 - a.x1;
    const originY = b.y1 - a.y1;
    const firstRatio = (originX * by - originY * bx) / denominator;
    const secondRatio = (originX * ay - originY * ax) / denominator;
    if (firstRatio < -1e-7 || firstRatio > 1 + 1e-7 || secondRatio < -1e-7 || secondRatio > 1 + 1e-7) return null;
    const point = { x: a.x1 + ax * firstRatio, y: a.y1 + ay * firstRatio };
    const firstEndpoint = firstRatio <= 1e-7 || firstRatio >= 1 - 1e-7;
    const secondEndpoint = secondRatio <= 1e-7 || secondRatio >= 1 - 1e-7;
    return firstEndpoint && secondEndpoint ? null : point;
  }

  function trafficCost(points, occupiedSegments, metadata, options = {}) {
    const sharedEscapeLength = Math.max(0, finite(options.sharedEscapeLength, finite(options.clearance, 14)));
    const crossingPenalty = Math.max(0, finite(options.crossingPenalty, 1800));
    const overlapPenalty = Math.max(0, finite(options.overlapPenalty, 100000));
    const forbidLongOverlap = options.forbidLongOverlap !== false;
    const current = routeSegments(points, metadata);
    let cost = 0;
    let crossings = 0;
    let overlapLength = 0;
    for (const segment of current) for (const occupied of occupiedSegments) {
      const overlap = collinearOverlap(segment, occupied);
      if (overlap) {
        const forbidden = Math.max(0, overlap.length - sharedEscapeAllowance(segment, occupied, overlap, sharedEscapeLength));
        if (forbidden > 1e-7 && forbidLongOverlap) return { cost: Infinity, crossings, overlapLength: overlapLength + forbidden, forbidden: true };
        overlapLength += forbidden;
        cost += forbidden * overlapPenalty;
        continue;
      }
      if (perpendicularIntersection(segment, occupied)) {
        crossings += 1;
        cost += crossingPenalty;
      }
    }
    return { cost, crossings, overlapLength, forbidden: false };
  }

  function visibilityRoute(start, end, obstacles, channelOffset, options = {}) {
    const xs = new Set([start.x, end.x]);
    const ys = new Set([start.y, end.y]);
    for (const rect of obstacles) {
      xs.add(rect.x - Math.abs(channelOffset));
      xs.add(rect.x + rect.width + Math.abs(channelOffset));
      ys.add(rect.y - Math.abs(channelOffset));
      ys.add(rect.y + rect.height + Math.abs(channelOffset));
    }
    const occupiedSegments = array(options.occupiedSegments).map(normalizedOccupiedSegment).filter(Boolean);
    const routeSpacing = Math.max(2, finite(options.routeSpacing, Math.max(8, finite(options.clearance, 14))));
    for (const segment of occupiedSegments) {
      xs.add(segment.x1);
      xs.add(segment.x2);
      ys.add(segment.y1);
      ys.add(segment.y2);
      if (segment.x1 === segment.x2) {
        xs.add(segment.x1 - routeSpacing);
        xs.add(segment.x1 + routeSpacing);
        ys.add(Math.min(segment.y1, segment.y2) - routeSpacing);
        ys.add(Math.max(segment.y1, segment.y2) + routeSpacing);
      } else if (segment.y1 === segment.y2) {
        ys.add(segment.y1 - routeSpacing);
        ys.add(segment.y1 + routeSpacing);
        xs.add(Math.min(segment.x1, segment.x2) - routeSpacing);
        xs.add(Math.max(segment.x1, segment.x2) + routeSpacing);
      } else {
        xs.add(Math.min(segment.x1, segment.x2) - routeSpacing);
        xs.add(Math.max(segment.x1, segment.x2) + routeSpacing);
        ys.add(Math.min(segment.y1, segment.y2) - routeSpacing);
        ys.add(Math.max(segment.y1, segment.y2) + routeSpacing);
      }
    }
    const xValues = Array.from(xs).sort((a, b) => a - b);
    const yValues = Array.from(ys).sort((a, b) => a - b);
    const byKey = new Map();
    for (const x of xValues) for (const y of yValues) {
      const point = { x, y };
      if (obstacles.some((rect) => pointInsideRect(point, rect))) continue;
      byKey.set(`${x},${y}`, point);
    }
    byKey.set(`${start.x},${start.y}`, start);
    byKey.set(`${end.x},${end.y}`, end);
    const nodes = Array.from(byKey.values());
    const neighbors = new Map(nodes.map((point) => [point, []]));
    for (const values of [xValues.map((x) => nodes.filter((p) => p.x === x).sort((a, b) => a.y - b.y)), yValues.map((y) => nodes.filter((p) => p.y === y).sort((a, b) => a.x - b.x))]) {
      for (const line of values) for (let index = 1; index < line.length; index += 1) {
        const a = line[index - 1];
        const b = line[index];
        if (obstacles.some((rect) => segmentHitsRect(a, b, rect))) continue;
        neighbors.get(a).push(b);
        neighbors.get(b).push(a);
      }
    }
    const distances = new Map([[start, 0]]);
    const previous = new Map();
    const open = new Set([start]);
    while (open.size) {
      const current = Array.from(open).sort((a, b) => (distances.get(a) - distances.get(b)) || a.x - b.x || a.y - b.y)[0];
      open.delete(current);
      if (current === end) break;
      const adjacent = (neighbors.get(current) || []).slice().sort((a, b) => a.x - b.x || a.y - b.y);
      for (const next of adjacent) {
        const conflict = trafficCost([current, next], occupiedSegments, {}, options);
        if (!Number.isFinite(conflict.cost)) continue;
        const distance = distances.get(current) + Math.abs(next.x - current.x) + Math.abs(next.y - current.y) + 1 + conflict.cost;
        if (distance >= (distances.get(next) ?? Infinity)) continue;
        distances.set(next, distance);
        previous.set(next, current);
        open.add(next);
      }
    }
    if (!previous.has(end)) return null;
    const path = [end];
    while (path[0] !== start) path.unshift(previous.get(path[0]));
    return simplifyPath(path);
  }

  function exactPoint(value, fallback) {
    return value && Number.isFinite(Number(value.x)) && Number.isFinite(Number(value.y))
      ? { x: Number(value.x), y: Number(value.y) }
      : { ...fallback };
  }

  function boundarySide(rect, point, fallbackPoint) {
    const distances = [
      { side: "left", value: Math.abs(point.x - rect.x) },
      { side: "right", value: Math.abs(point.x - (rect.x + rect.width)) },
      { side: "top", value: Math.abs(point.y - rect.y) },
      { side: "bottom", value: Math.abs(point.y - (rect.y + rect.height)) },
    ].sort((a, b) => a.value - b.value || a.side.localeCompare(b.side));
    if (distances[0].value < 1e-7) return distances[0].side;
    const dx = fallbackPoint.x - point.x;
    const dy = fallbackPoint.y - point.y;
    if (Math.abs(dx) >= Math.abs(dy)) return dx >= 0 ? "right" : "left";
    return dy >= 0 ? "bottom" : "top";
  }

  function escapePoint(rect, point, fallbackPoint, distance) {
    const side = boundarySide(rect, point, fallbackPoint);
    if (side === "left") return { x: point.x - distance, y: point.y };
    if (side === "right") return { x: point.x + distance, y: point.y };
    if (side === "top") return { x: point.x, y: point.y - distance };
    return { x: point.x, y: point.y + distance };
  }

  function routeSignature(points) {
    return points.map((point) => `${point.x},${point.y}`).join(";");
  }

  function routeOrthogonal(sourceRect, targetRect, obstacleRects = [], options = {}) {
    const clearance = Math.max(2, finite(options.clearance, 14));
    const channelOffset = finite(options.channelOffset, 0);
    const expanded = array(obstacleRects)
      .filter((rect) => {
        if (rect === sourceRect || rect === targetRect) return false;
        if (rect?.id === undefined || rect?.id === null) return true;
        return String(rect.id) !== String(sourceRect?.id ?? "") && String(rect.id) !== String(targetRect?.id ?? "");
      })
      .map((rect) => ({ x: rect.x - clearance, y: rect.y - clearance, width: rect.width + clearance * 2, height: rect.height + clearance * 2 }));
    const fallback = anchors(sourceRect, targetRect);
    const start = exactPoint(options.sourcePoint, fallback.start);
    const end = exactPoint(options.targetPoint, fallback.end);
    const sourceEscape = escapePoint(sourceRect, start, end, clearance);
    const targetEscape = escapePoint(targetRect, end, start, clearance);
    const bodyObstacles = [
      ...expanded,
      { x: sourceRect.x, y: sourceRect.y, width: sourceRect.width, height: sourceRect.height },
      { x: targetRect.x, y: targetRect.y, width: targetRect.width, height: targetRect.height },
    ];
    const occupiedSegments = array(options.occupiedSegments).map(normalizedOccupiedSegment).filter(Boolean);
    const metadata = { sourcePortKey: options.sourcePortKey, targetPortKey: options.targetPortKey, routeId: options.routeId };
    const complete = (body) => simplifyPath([start, sourceEscape, ...body.slice(1, -1), targetEscape, end]);
    const midX = (sourceEscape.x + targetEscape.x) / 2 + channelOffset;
    const midY = (sourceEscape.y + targetEscape.y) / 2 + channelOffset;
    let parallelCandidate = null;
    if (channelOffset) {
      if (Math.abs(targetEscape.x - sourceEscape.x) >= Math.abs(targetEscape.y - sourceEscape.y)) {
        parallelCandidate = complete([
          sourceEscape,
          { x: sourceEscape.x, y: sourceEscape.y + channelOffset },
          { x: targetEscape.x, y: targetEscape.y + channelOffset },
          targetEscape,
        ]);
      } else {
        parallelCandidate = complete([
          sourceEscape,
          { x: sourceEscape.x + channelOffset, y: sourceEscape.y },
          { x: targetEscape.x + channelOffset, y: targetEscape.y },
          targetEscape,
        ]);
      }
    }
    const candidates = [
      ...(parallelCandidate ? [parallelCandidate] : []),
      complete([sourceEscape, { x: targetEscape.x, y: sourceEscape.y }, targetEscape]),
      complete([sourceEscape, { x: sourceEscape.x, y: targetEscape.y }, targetEscape]),
      complete([sourceEscape, { x: midX, y: sourceEscape.y }, { x: midX, y: targetEscape.y }, targetEscape]),
      complete([sourceEscape, { x: sourceEscape.x, y: midY }, { x: targetEscape.x, y: midY }, targetEscape]),
    ];
    if (bodyObstacles.length) {
      const left = Math.min(...bodyObstacles.map((rect) => rect.x)) - clearance - Math.abs(channelOffset);
      const right = Math.max(...bodyObstacles.map((rect) => rect.x + rect.width)) + clearance + Math.abs(channelOffset);
      const top = Math.min(...bodyObstacles.map((rect) => rect.y)) - clearance - Math.abs(channelOffset);
      const bottom = Math.max(...bodyObstacles.map((rect) => rect.y + rect.height)) + clearance + Math.abs(channelOffset);
      candidates.push(
        complete([sourceEscape, { x: left, y: sourceEscape.y }, { x: left, y: targetEscape.y }, targetEscape]),
        complete([sourceEscape, { x: right, y: sourceEscape.y }, { x: right, y: targetEscape.y }, targetEscape]),
        complete([sourceEscape, { x: sourceEscape.x, y: top }, { x: targetEscape.x, y: top }, targetEscape]),
        complete([sourceEscape, { x: sourceEscape.x, y: bottom }, { x: targetEscape.x, y: bottom }, targetEscape]),
      );
    }
    const routeSpacing = Math.max(2, finite(options.routeSpacing, clearance));
    for (const occupied of occupiedSegments) {
      if (occupied.x1 === occupied.x2) {
        for (const x of [occupied.x1 - routeSpacing, occupied.x1 + routeSpacing]) {
          candidates.push(complete([sourceEscape, { x, y: sourceEscape.y }, { x, y: targetEscape.y }, targetEscape]));
        }
        for (const y of [Math.min(occupied.y1, occupied.y2) - routeSpacing, Math.max(occupied.y1, occupied.y2) + routeSpacing]) {
          candidates.push(complete([sourceEscape, { x: sourceEscape.x, y }, { x: targetEscape.x, y }, targetEscape]));
        }
      } else if (occupied.y1 === occupied.y2) {
        for (const y of [occupied.y1 - routeSpacing, occupied.y1 + routeSpacing]) {
          candidates.push(complete([sourceEscape, { x: sourceEscape.x, y }, { x: targetEscape.x, y }, targetEscape]));
        }
        for (const x of [Math.min(occupied.x1, occupied.x2) - routeSpacing, Math.max(occupied.x1, occupied.x2) + routeSpacing]) {
          candidates.push(complete([sourceEscape, { x, y: sourceEscape.y }, { x, y: targetEscape.y }, targetEscape]));
        }
      } else {
        for (const y of [
          Math.min(occupied.y1, occupied.y2) - routeSpacing,
          Math.max(occupied.y1, occupied.y2) + routeSpacing,
        ]) candidates.push(complete([sourceEscape, { x: sourceEscape.x, y }, { x: targetEscape.x, y }, targetEscape]));
        for (const x of [
          Math.min(occupied.x1, occupied.x2) - routeSpacing,
          Math.max(occupied.x1, occupied.x2) + routeSpacing,
        ]) candidates.push(complete([sourceEscape, { x, y: sourceEscape.y }, { x, y: targetEscape.y }, targetEscape]));
      }
    }
    const unique = new Map();
    candidates.map(simplifyPath).forEach((path) => unique.set(routeSignature(path), path));
    const cheapPaths = Array.from(unique.values()).sort((a, b) => pathCost(a) - pathCost(b) || routeSignature(a).localeCompare(routeSignature(b)));
    if (!occupiedSegments.length) {
      if (parallelCandidate && pathClear(parallelCandidate, bodyObstacles)) return parallelCandidate;
      const clear = cheapPaths.find((path) => pathClear(path, bodyObstacles));
      if (clear) return clear;
    }
    const valid = [];
    for (const path of cheapPaths) {
      if (!pathClear(path, bodyObstacles)) continue;
      const traffic = trafficCost(path, occupiedSegments, metadata, { ...options, clearance });
      const entry = { path, traffic, score: pathCost(path) + traffic.cost };
      if (!Number.isFinite(entry.score)) continue;
      valid.push(entry);
      // Paths are ordered by their traffic-free cost and penalties cannot be
      // negative, so the first conflict-free path is already globally best.
      if (traffic.cost === 0) return path;
    }
    const visibility = visibilityRoute(sourceEscape, targetEscape, bodyObstacles, channelOffset || clearance, { ...options, clearance, occupiedSegments });
    if (visibility) {
      const path = complete(visibility);
      if (pathClear(path, bodyObstacles) && !unique.has(routeSignature(path))) {
        const traffic = trafficCost(path, occupiedSegments, metadata, { ...options, clearance });
        const entry = { path, traffic, score: pathCost(path) + traffic.cost };
        if (Number.isFinite(entry.score)) valid.push(entry);
      }
    }
    valid.sort((a, b) => a.score - b.score || pathCost(a.path) - pathCost(b.path) || routeSignature(a.path).localeCompare(routeSignature(b.path)));
    if (valid.length) return valid[0].path;
    throw new Error("找不到避开组件的正交连线路径。");
  }

  function stableRouteKey(record) {
    const link = record.link || {};
    return [record.linkId, link.source_component, link.source_port, link.target_component, link.target_port].map((value) => String(value ?? "")).join("\u001f");
  }

  // Batch contract for app.js: port placement and occupied-segment routing are
  // both resolved without DOM state. Routes are emitted in stable routing order.
  function planOrthogonalRoutes(links, rects, options = {}) {
    const lookup = rectLookup(rects);
    const assignments = assignPortEndpoints(links, rects, options);
    const records = assignments.links.slice().sort((a, b) => stableRouteKey(a).localeCompare(stableRouteKey(b)) || a.index - b.index);
    const parallel = new Map();
    records.forEach((record) => {
      const link = record.link;
      const key = [String(link.source_component), String(link.target_component)].sort().join("\u001f");
      if (!parallel.has(key)) parallel.set(key, []);
      parallel.get(key).push(record);
    });
    const selectedRouteIds = options.routeIds === undefined || options.routeIds === null
      ? null
      : new Set(options.routeIds instanceof Set ? Array.from(options.routeIds, String) : array(options.routeIds).map(String));
    const routedRecords = selectedRouteIds ? records.filter((record) => selectedRouteIds.has(record.linkId)) : records;
    const occupiedSegments = array(options.occupiedSegments).map(normalizedOccupiedSegment).filter(Boolean);
    const routes = [];
    const errors = [];
    const parallelSpacing = Math.max(0, finite(options.parallelSpacing, 18));
    for (const record of routedRecords) {
      const link = record.link;
      const sourceRect = lookup.get(String(link.source_component));
      const targetRect = lookup.get(String(link.target_component));
      const peers = parallel.get([String(link.source_component), String(link.target_component)].sort().join("\u001f"));
      const channelOffset = options.parallelOffsets === false ? 0 : (peers.indexOf(record) - (peers.length - 1) / 2) * parallelSpacing;
      try {
        const points = routeOrthogonal(sourceRect, targetRect, Array.from(lookup.values()), {
          ...options,
          sourcePoint: record.sourcePoint,
          targetPoint: record.targetPoint,
          sourcePortKey: record.sourcePortKey,
          targetPortKey: record.targetPortKey,
          routeId: record.linkId,
          channelOffset,
          occupiedSegments,
        });
        const segments = routeSegments(points, {
          sourcePortKey: record.sourcePortKey,
          targetPortKey: record.targetPortKey,
          routeId: record.linkId,
        });
        occupiedSegments.push(...segments);
        routes.push({ ...record, channelOffset, points, segments });
      } catch (error) {
        errors.push({ ...record, error });
      }
    }
    const plan = { routes, errors, occupiedSegments, endpoints: assignments.byPort };
    return options.renderGeometry ? decorateRoutePlan(plan, Array.from(lookup.values()), options) : plan;
  }

  function sideVector(side) {
    if (side === "left") return { x: -1, y: 0 };
    if (side === "right") return { x: 1, y: 0 };
    if (side === "top") return { x: 0, y: -1 };
    return { x: 0, y: 1 };
  }

  function svgNumber(value) {
    return Math.round(finite(value) * 100) / 100;
  }

  function cubicBezierGeometry(sourceValue, targetValue, sourceSide = "right", targetSide = "left", options = {}) {
    const source = normalizedPoint(sourceValue);
    const target = normalizedPoint(targetValue);
    const distance = Math.hypot(target.x - source.x, target.y - source.y);
    const handle = Math.min(
      Math.max(18, distance * Math.max(0.2, finite(options.handleRatio, 0.34))),
      Math.max(18, finite(options.maxHandle, 88)),
    );
    const sourceVector = sideVector(sourceSide);
    const targetVector = sideVector(targetSide);
    return {
      source,
      first: { x: source.x + sourceVector.x * handle, y: source.y + sourceVector.y * handle },
      second: { x: target.x + targetVector.x * handle, y: target.y + targetVector.y * handle },
      target,
    };
  }

  function cubicBezierPath(geometryValue) {
    const geometry = object(geometryValue);
    return `M ${svgNumber(geometry.source.x)} ${svgNumber(geometry.source.y)} C ${svgNumber(geometry.first.x)} ${svgNumber(geometry.first.y)} ${svgNumber(geometry.second.x)} ${svgNumber(geometry.second.y)} ${svgNumber(geometry.target.x)} ${svgNumber(geometry.target.y)}`;
  }

  function sampleCubicBezier(geometryValue, stepsValue = 20) {
    const geometry = object(geometryValue);
    const steps = Math.max(8, Math.min(96, Math.trunc(finite(stepsValue, 20))));
    const points = [];
    for (let index = 0; index <= steps; index += 1) {
      const ratio = index / steps;
      const inverse = 1 - ratio;
      points.push({
        x: inverse ** 3 * geometry.source.x + 3 * inverse ** 2 * ratio * geometry.first.x + 3 * inverse * ratio ** 2 * geometry.second.x + ratio ** 3 * geometry.target.x,
        y: inverse ** 3 * geometry.source.y + 3 * inverse ** 2 * ratio * geometry.first.y + 3 * inverse * ratio ** 2 * geometry.second.y + ratio ** 3 * geometry.target.y,
      });
    }
    return points;
  }

  function roundedOrthogonalGeometry(pointsValue, radiusValue = 9, stepsValue = 4) {
    const points = simplifyPath(array(pointsValue).map(normalizedPoint));
    if (points.length < 2) return { path: pathToSvg(points), points };
    const radius = Math.max(0, finite(radiusValue, 9));
    const steps = Math.max(2, Math.min(12, Math.trunc(finite(stepsValue, 4))));
    const path = [`M ${svgNumber(points[0].x)} ${svgNumber(points[0].y)}`];
    const sampled = [{ ...points[0] }];
    const appendLine = (point) => {
      const last = sampled[sampled.length - 1];
      if (pointEquals(last, point)) return;
      path.push(`L ${svgNumber(point.x)} ${svgNumber(point.y)}`);
      sampled.push({ ...point });
    };
    for (let index = 1; index < points.length - 1; index += 1) {
      const before = points[index - 1];
      const corner = points[index];
      const after = points[index + 1];
      const incoming = Math.hypot(corner.x - before.x, corner.y - before.y);
      const outgoing = Math.hypot(after.x - corner.x, after.y - corner.y);
      const amount = Math.min(radius, incoming / 2, outgoing / 2);
      const enter = {
        x: corner.x + (before.x - corner.x) * (amount / Math.max(incoming, 1e-9)),
        y: corner.y + (before.y - corner.y) * (amount / Math.max(incoming, 1e-9)),
      };
      const leave = {
        x: corner.x + (after.x - corner.x) * (amount / Math.max(outgoing, 1e-9)),
        y: corner.y + (after.y - corner.y) * (amount / Math.max(outgoing, 1e-9)),
      };
      appendLine(enter);
      if (!amount) continue;
      path.push(`Q ${svgNumber(corner.x)} ${svgNumber(corner.y)} ${svgNumber(leave.x)} ${svgNumber(leave.y)}`);
      for (let sample = 1; sample <= steps; sample += 1) {
        const ratio = sample / steps;
        const inverse = 1 - ratio;
        sampled.push({
          x: inverse ** 2 * enter.x + 2 * inverse * ratio * corner.x + ratio ** 2 * leave.x,
          y: inverse ** 2 * enter.y + 2 * inverse * ratio * corner.y + ratio ** 2 * leave.y,
        });
      }
    }
    appendLine(points[points.length - 1]);
    return { path: path.join(" "), points: sampled };
  }

  function sampledPathClear(points, rects, padding = 0) {
    for (let index = 1; index < points.length; index += 1) {
      if (array(rects).some((rect) => lineSegmentHitsRect(points[index - 1], points[index], rect, padding))) return false;
    }
    return true;
  }

  function geometryTrafficClear(points, occupiedSegments, metadata, options = {}) {
    const traffic = trafficCost(points, array(occupiedSegments).map(normalizedOccupiedSegment).filter(Boolean), metadata, {
      ...options,
      crossingPenalty: 1,
      overlapPenalty: 1,
      forbidLongOverlap: true,
    });
    return Number.isFinite(traffic.cost) && traffic.crossings === 0 && traffic.overlapLength <= 1e-7;
  }

  // Rendering geometry is a proof-carrying decoration over the orthogonal plan.
  // Curves and rounded corners expose sampled points/segments so future drag
  // previews can route against what is actually drawn rather than an SVG C/Q.
  function decorateRoutePlan(planValue, rectsValue, options = {}) {
    const plan = object(planValue);
    const rects = array(rectsValue);
    const routes = array(plan.routes);
    const seedSegments = array(options.occupiedSegments).map(normalizedOccupiedSegment).filter(Boolean);
    const acceptedSegments = [];
    const decorated = [];
    const clearance = Math.max(2, finite(options.clearance, 14));
    const curveDistance = Math.max(40, finite(options.shortCurveDistance, 230));
    const cornerRadius = Math.max(0, finite(options.cornerRadius, 9));
    for (let routeIndex = 0; routeIndex < routes.length; routeIndex += 1) {
      const route = routes[routeIndex];
      const metadata = { sourcePortKey: route.sourcePortKey, targetPortKey: route.targetPortKey, routeId: route.linkId };
      const futureSegments = routes.slice(routeIndex + 1).flatMap((item) => item.segments);
      const proofSegments = [...seedSegments, ...acceptedSegments, ...futureSegments];
      const obstacles = rects.filter((rect) => String(rect?.id ?? "") !== String(route.link?.source_component ?? "") && String(rect?.id ?? "") !== String(route.link?.target_component ?? ""));
      const distance = Math.hypot(route.targetPoint.x - route.sourcePoint.x, route.targetPoint.y - route.sourcePoint.y);
      const curve = cubicBezierGeometry(route.sourcePoint, route.targetPoint, route.sourceSide, route.targetSide, options);
      const curvePoints = sampleCubicBezier(curve, Math.max(16, Math.ceil(distance / 10)));
      const curveLeavesEndpoints = curvePoints.slice(1, -1).every((point) => {
        const sourceRect = rects.find((rect) => String(rect?.id ?? "") === String(route.link?.source_component ?? ""));
        const targetRect = rects.find((rect) => String(rect?.id ?? "") === String(route.link?.target_component ?? ""));
        return (!sourceRect || !pointInsideRect(point, sourceRect)) && (!targetRect || !pointInsideRect(point, targetRect));
      });
      const directClear = !obstacles.some((rect) => lineSegmentHitsRect(route.sourcePoint, route.targetPoint, rect, clearance));
      const curveClear = sampledPathClear(curvePoints, obstacles, clearance)
        && geometryTrafficClear(curvePoints, proofSegments, metadata, { ...options, clearance });
      let geometry = null;
      let kind = "orthogonal";
      if (options.allowCurves !== false && distance <= curveDistance && directClear && curveLeavesEndpoints && curveClear) {
        geometry = { path: cubicBezierPath(curve), points: curvePoints };
        kind = "curve";
      }
      if (!geometry) {
        const rounded = roundedOrthogonalGeometry(route.points, cornerRadius, options.cornerSamples);
        const roundedClear = sampledPathClear(rounded.points, obstacles, clearance)
          && geometryTrafficClear(rounded.points, proofSegments, metadata, { ...options, clearance });
        geometry = roundedClear ? rounded : { path: pathToSvg(route.points), points: route.points.map((point) => ({ ...point })) };
      }
      const segments = routeSegments(geometry.points, metadata);
      acceptedSegments.push(...segments);
      decorated.push({
        ...route,
        kind,
        orthogonalPoints: route.points,
        points: geometry.points,
        segments,
        path: geometry.path,
        sampled: true,
      });
    }
    return { ...plan, routes: decorated, occupiedSegments: [...seedSegments, ...acceptedSegments] };
  }

  function estimateLabelSize(textValue, options = {}) {
    const text = String(textValue ?? "");
    const fontSize = Math.max(6, finite(options.fontSize, 9));
    const horizontalPadding = Math.max(0, finite(options.horizontalPadding, 5));
    const verticalPadding = Math.max(0, finite(options.verticalPadding, 3));
    let units = 0;
    for (const character of text) {
      if (/\s/u.test(character)) units += 0.36;
      else units += character.codePointAt(0) <= 0x7f ? 0.62 : 1;
    }
    return {
      width: Math.max(fontSize * 2, units * fontSize + horizontalPadding * 2),
      height: fontSize * 1.25 + verticalPadding * 2,
    };
  }

  function estimateLabelRect(textValue, centerValue, options = {}) {
    const size = estimateLabelSize(textValue, options);
    const centerPoint = normalizedPoint(centerValue);
    return { x: centerPoint.x - size.width / 2, y: centerPoint.y - size.height / 2, width: size.width, height: size.height };
  }

  function routeLabelCandidates(routeValue, labelSizeValue, options = {}) {
    const route = object(routeValue);
    const points = array(route.points).map(normalizedPoint);
    const labelSize = object(labelSizeValue);
    const routeGap = Math.max(2, finite(options.routeGap, 5));
    const labelGap = Math.max(2, finite(options.labelGap, 6));
    const bands = Math.max(1, Math.min(8, Math.trunc(finite(options.bands, 4))));
    const fractions = [0.5, 0.35, 0.65];
    const segments = points.slice(1).map((point, index) => {
      const start = points[index];
      return { start, end: point, index, length: Math.hypot(point.x - start.x, point.y - start.y) };
    }).filter((segment) => segment.length > 1e-7)
      .sort((a, b) => b.length - a.length || a.index - b.index);
    const candidates = [];
    for (const segment of segments) for (const fraction of fractions) {
      const normal = { x: -(segment.end.y - segment.start.y) / segment.length, y: (segment.end.x - segment.start.x) / segment.length };
      const centerPoint = {
        x: segment.start.x + (segment.end.x - segment.start.x) * fraction,
        y: segment.start.y + (segment.end.y - segment.start.y) * fraction,
      };
      for (let band = 0; band < bands; band += 1) for (const sign of [-1, 1]) {
        const distance = labelSize.height / 2 + routeGap + band * (labelSize.height + labelGap);
        candidates.push({ x: centerPoint.x + normal.x * distance * sign, y: centerPoint.y + normal.y * distance * sign, segmentIndex: segment.index, band, sign });
      }
    }
    return candidates;
  }

  function pointsBounds(pointsValue, padding = 0) {
    const points = array(pointsValue).map(normalizedPoint);
    if (!points.length) return null;
    const amount = Math.max(0, finite(padding, 0));
    const minimumX = Math.min(...points.map((point) => point.x)) - amount;
    const minimumY = Math.min(...points.map((point) => point.y)) - amount;
    const maximumX = Math.max(...points.map((point) => point.x)) + amount;
    const maximumY = Math.max(...points.map((point) => point.y)) + amount;
    return { x: minimumX, y: minimumY, width: maximumX - minimumX, height: maximumY - minimumY };
  }

  function placeRouteLabels(routesValue, nodeRectsValue, options = {}) {
    const routes = array(routesValue).slice().sort((a, b) => String(a.linkId ?? "").localeCompare(String(b.linkId ?? "")));
    const nodeRects = array(nodeRectsValue);
    const routeSegmentsValue = [
      ...array(options.routeSegments).map(normalizedOccupiedSegment).filter(Boolean),
      ...routes.flatMap((route) => array(route.segments).length ? route.segments : routeSegments(route.points, { routeId: route.linkId })),
    ];
    const placedRects = array(options.occupiedLabelRects).map((rect) => ({ ...rect }));
    const nodeGap = Math.max(0, finite(options.nodeGap, 8));
    const labelGap = Math.max(2, finite(options.labelGap, 6));
    const routeGap = Math.max(2, finite(options.routeGap, 5));
    const canvasMargin = Math.max(0, finite(options.canvasMargin, 12));
    const geometryRects = [
      ...nodeRects,
      ...routes.map((route) => pointsBounds(route.points)).filter(Boolean),
      ...placedRects,
    ];
    let fallbackBottom = geometryRects.length ? Math.max(...geometryRects.map((rect) => rect.y + rect.height)) + labelGap : canvasMargin;
    const placements = [];
    const clear = (rect) => rect.x >= canvasMargin && rect.y >= canvasMargin
      && !nodeRects.some((node) => rectsIntersectWithGap(rect, node, nodeGap))
      && !placedRects.some((placed) => rectsIntersectWithGap(rect, placed, labelGap))
      && !routeSegmentsValue.some((segment) => lineSegmentHitsRect({ x: segment.x1, y: segment.y1 }, { x: segment.x2, y: segment.y2 }, rect, routeGap));
    for (const route of routes) {
      const text = String(route.label ?? route.linkId ?? "");
      const size = estimateLabelSize(text, options);
      const candidates = routeLabelCandidates(route, size, options);
      let selected = candidates.find((candidate) => clear({ x: candidate.x - size.width / 2, y: candidate.y - size.height / 2, width: size.width, height: size.height }));
      let fallback = false;
      if (!selected) {
        fallback = true;
        const routeBounds = pointsBounds(route.points) || { x: canvasMargin, y: fallbackBottom, width: 0, height: 0 };
        const x = Math.max(canvasMargin + size.width / 2, routeBounds.x + routeBounds.width / 2);
        let y = fallbackBottom + size.height / 2;
        let rect = { x: x - size.width / 2, y: y - size.height / 2, width: size.width, height: size.height };
        while (!clear(rect)) {
          y += size.height + labelGap;
          rect = { ...rect, y: y - size.height / 2 };
        }
        selected = { x, y, segmentIndex: -1, band: -1, sign: 0 };
      }
      const rect = { x: selected.x - size.width / 2, y: selected.y - size.height / 2, width: size.width, height: size.height };
      placedRects.push(rect);
      fallbackBottom = Math.max(fallbackBottom, rect.y + rect.height + labelGap);
      placements.push({ linkId: String(route.linkId ?? ""), text, x: selected.x, y: selected.y, rect, fallback, candidate: selected });
    }
    return { placements, labelRects: placedRects, bounds: pointsBounds(placedRects.flatMap((rect) => [{ x: rect.x, y: rect.y }, { x: rect.x + rect.width, y: rect.y + rect.height }])) };
  }

  function pathToSvg(points) {
    return array(points).map((point, index) => `${index ? "L" : "M"} ${Math.round(point.x * 100) / 100} ${Math.round(point.y * 100) / 100}`).join(" ");
  }

  function computeWorldBounds(positions, sizes, groupBounds = {}, padding = 120) {
    const rects = Object.keys(object(positions)).map((id) => rectForNode(id, positions, sizes));
    rects.push(...Object.values(object(groupBounds)));
    if (!rects.length) return { x: 0, y: 0, width: padding * 2, height: padding * 2 };
    const minX = Math.min(...rects.map((rect) => rect.x)) - padding;
    const minY = Math.min(...rects.map((rect) => rect.y)) - padding;
    const maxX = Math.max(...rects.map((rect) => rect.x + rect.width)) + padding;
    const maxY = Math.max(...rects.map((rect) => rect.y + rect.height)) + padding;
    return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
  }

  function computeAutoFitScale(boundsValue, viewportValue, options = {}) {
    const bounds = normalizeBounds(boundsValue);
    const viewport = object(viewportValue);
    const padding = Math.max(0, finite(options.padding, 0));
    const minimum = clamp(finite(options.minScale, 0.8), 0.01, 100);
    const maximum = Math.max(minimum, finite(options.maxScale, 1));
    const availableWidth = Math.max(1, finite(viewport.width, 1) - padding * 2);
    const availableHeight = Math.max(1, finite(viewport.height, 1) - padding * 2);
    const requiredWidth = Math.max(1, bounds.width);
    const requiredHeight = Math.max(1, bounds.height);
    const ideal = Math.min(availableWidth / requiredWidth, availableHeight / requiredHeight);
    const scale = clamp(ideal, minimum, maximum);
    return {
      scale,
      idealScale: ideal,
      fits: requiredWidth * scale <= availableWidth + 1e-7 && requiredHeight * scale <= availableHeight + 1e-7,
      overflowX: Math.max(0, requiredWidth * scale - availableWidth),
      overflowY: Math.max(0, requiredHeight * scale - availableHeight),
    };
  }

  function normalizeWorldOrigin(positions, bounds, viewport) {
    const normalizedBounds = normalizeBounds(bounds);
    const shiftX = normalizedBounds.x < 0 ? -normalizedBounds.x : 0;
    const shiftY = normalizedBounds.y < 0 ? -normalizedBounds.y : 0;
    const nextPositions = {};
    for (const [id, position] of Object.entries(object(positions))) {
      const point = normalizedPoint(position);
      nextPositions[id] = { x: point.x + shiftX, y: point.y + shiftY };
    }
    const scale = clamp(finite(viewport?.scale, 1), 0.25, 4);
    return {
      positions: nextPositions,
      bounds: { ...normalizedBounds, x: normalizedBounds.x + shiftX, y: normalizedBounds.y + shiftY },
      viewport: {
        x: finite(viewport?.x) - shiftX * scale,
        y: finite(viewport?.y) - shiftY * scale,
        scale,
      },
      shift: { x: shiftX, y: shiftY },
    };
  }

  function internalLinks(links, ids) {
    const selected = new Set(ids);
    return array(links).filter((link) => selected.has(link.source_component) && selected.has(link.target_component));
  }

  function validateClipboardPayload(payload) {
    if (!payload || typeof payload !== "object" || Array.isArray(payload) || payload.type !== CLIPBOARD_TYPE || payload.version !== 1) {
      throw new Error("剪贴板中的拓扑数据格式或版本不受支持。");
    }
    if (!Array.isArray(payload.components) || !payload.components.length) throw new Error("剪贴板中的组件必须是非空数组。");
    if (!Array.isArray(payload.links) || !Array.isArray(payload.groups)) throw new Error("剪贴板中的链路和分组必须是数组。");
    const componentIds = new Set();
    const portsByComponent = new Map();
    for (const component of payload.components) {
      if (!component || typeof component !== "object" || Array.isArray(component)) throw new Error("剪贴板中的每个组件条目都必须是对象。");
      const id = String(component.component_id || "");
      if (!id || componentIds.has(id)) throw new Error(`组件 ID 缺失或重复：${id || "<空>"}`);
      componentIds.add(id);
      if (!Array.isArray(component.ports)) throw new Error(`${id} 的端口必须是数组。`);
      const localPorts = new Set();
      for (const port of component.ports) {
        if (!port || typeof port !== "object" || Array.isArray(port)) throw new Error(`${id} 的每个端口条目都必须是对象。`);
        const portId = String(port.port_id || "");
        if (!portId || localPorts.has(portId)) throw new Error(`${id} 中的端口 ID 缺失或重复：${portId || "<空>"}`);
        localPorts.add(portId);
      }
      portsByComponent.set(id, localPorts);
    }
    const linkIds = new Set();
    for (const link of payload.links) {
      if (!link || typeof link !== "object" || Array.isArray(link)) throw new Error("剪贴板中的每个链路条目都必须是对象。");
      const id = String(link.link_id || "");
      if (!id || linkIds.has(id)) throw new Error(`链路 ID 缺失或重复：${id || "<空>"}`);
      linkIds.add(id);
      const source = String(link.source_component || "");
      const target = String(link.target_component || "");
      if (!componentIds.has(source) || !componentIds.has(target)) throw new Error(`${id} 引用了剪贴板范围之外的组件。`);
      if (!portsByComponent.get(source).has(String(link.source_port || ""))) throw new Error(`${id} 引用了未知的源端口。`);
      if (!portsByComponent.get(target).has(String(link.target_port || ""))) throw new Error(`${id} 引用了未知的目标端口。`);
    }
    const groupIds = new Set();
    const groupedMembers = new Set();
    for (const group of payload.groups) {
      if (!group || typeof group !== "object" || Array.isArray(group)) throw new Error("剪贴板中的每个分组条目都必须是对象。");
      const id = String(group.group_id || "");
      if (!id || groupIds.has(id)) throw new Error(`分组 ID 缺失或重复：${id || "<空>"}`);
      groupIds.add(id);
      if (!Array.isArray(group.members) || !group.members.length) throw new Error(`${id} 的成员必须是非空数组。`);
      const members = new Set();
      for (const rawMember of group.members) {
        const member = String(rawMember || "");
        if (!componentIds.has(member) || members.has(member) || groupedMembers.has(member)) throw new Error(`${id} 包含无效或重复归组的成员：${member}`);
        members.add(member);
        groupedMembers.add(member);
      }
      if (!members.has(String(group.root || ""))) throw new Error(`${id} 的根组件必须是组成员。`);
    }
    const positions = object(payload.positions);
    for (const id of componentIds) {
      const position = positions[id];
      if (!position || !Number.isFinite(Number(position.x)) || !Number.isFinite(Number(position.y))) throw new Error(`${id} 的位置缺失或无效。`);
    }
    const extraPosition = Object.keys(positions).find((id) => !componentIds.has(id));
    if (extraPosition) throw new Error(`位置数据引用了未知组件：${extraPosition}`);
    if (!payload.origin || !Number.isFinite(Number(payload.origin.x)) || !Number.isFinite(Number(payload.origin.y))) throw new Error("剪贴板中的拓扑原点无效。");
    return true;
  }

  function copySelection(hardware, view, selectedIds = [], selectedGroupId = null) {
    const group = selectedGroupId ? array(view.groups).find((item) => item.group_id === selectedGroupId) : null;
    const ids = Array.from(new Set(group ? array(group.members) : array(selectedIds))).map(String).sort();
    if (!ids.length) return null;
    const idSet = new Set(ids);
    const components = array(hardware.components).filter((component) => idSet.has(component.component_id)).map(clone);
    const links = ids.length > 1 ? internalLinks(hardware.links, ids).map(clone) : [];
    const groups = array(view.groups).filter((item) => array(item.members).every((id) => idSet.has(id))).map(clone);
    const positions = {};
    ids.forEach((id) => { positions[id] = normalizedPoint(view.layout?.positions?.[id]); });
    const xs = Object.values(positions).map((point) => point.x);
    const ys = Object.values(positions).map((point) => point.y);
    return {
      type: CLIPBOARD_TYPE,
      version: 1,
      components,
      links,
      groups,
      positions,
      origin: { x: xs.length ? Math.min(...xs) : 0, y: ys.length ? Math.min(...ys) : 0 },
    };
  }

  function parseClipboardText(text) {
    try {
      const value = JSON.parse(String(text || ""));
      validateClipboardPayload(value);
      return value;
    } catch (_error) {
      return null;
    }
  }

  function pasteSelection(hardware, view, payload, targetPoint = null) {
    validateClipboardPayload(payload);
    const nextHardware = clone(hardware);
    const nextView = clone(view);
    const usedComponents = new Set(array(nextHardware.components).map((item) => String(item.component_id)));
    const usedLinks = new Set(array(nextHardware.links).map((item) => String(item.link_id)));
    const usedGroups = new Set(array(nextView.groups).map((item) => String(item.group_id)));
    const usedGroupLabels = new Set(array(nextView.groups).map((item) => String(item.label || "")).filter(Boolean));
    const usedPorts = new Set(array(nextHardware.components).flatMap((item) => array(item.ports).map((port) => String(port.port_id))));
    const componentMap = {};
    const portMap = {};
    const pastedIds = [];
    const pastedComponents = [];
    for (const original of array(payload.components)) {
      const component = clone(original);
      const oldId = String(component.component_id);
      const newId = nextNumericName(oldId, usedComponents, true);
      componentMap[oldId] = newId;
      pastedIds.push(newId);
      component.component_id = newId;
      portMap[oldId] = {};
      component.ports = array(component.ports).map((rawPort) => {
        const port = clone(rawPort);
        const oldPort = String(port.port_id);
        const newPort = nextNumericName(oldPort, usedPorts, true);
        portMap[oldId][oldPort] = newPort;
        port.port_id = newPort;
        return port;
      });
      nextHardware.components.push(component);
      pastedComponents.push(component);
    }
    pastedComponents.forEach((component) => remapPhysicalCompositionComponentRefs(component.metadata, componentMap));
    for (const original of array(payload.links)) {
      if (!componentMap[original.source_component] || !componentMap[original.target_component]) continue;
      const link = clone(original);
      link.link_id = nextNumericName(link.link_id, usedLinks, true);
      link.source_component = componentMap[original.source_component];
      link.target_component = componentMap[original.target_component];
      link.source_port = portMap[original.source_component]?.[original.source_port] || link.source_port;
      link.target_port = portMap[original.target_component]?.[original.target_port] || link.target_port;
      nextHardware.links.push(link);
    }
    const sourceOrigin = normalizedPoint(payload.origin);
    const target = targetPoint ? normalizedPoint(targetPoint) : { x: sourceOrigin.x + 48, y: sourceOrigin.y + 48 };
    nextView.layout ??= { positions: {} };
    nextView.layout.positions ??= {};
    for (const [oldId, newId] of Object.entries(componentMap)) {
      const oldPosition = normalizedPoint(payload.positions?.[oldId]);
      nextView.layout.positions[newId] = { x: target.x + oldPosition.x - sourceOrigin.x, y: target.y + oldPosition.y - sourceOrigin.y };
    }
    for (const original of array(payload.groups)) {
      const members = array(original.members).map((id) => componentMap[id]).filter(Boolean);
      if (!members.length) continue;
      const groupId = nextNumericName(original.group_id, usedGroups, true);
      const labelBase = original.label || original.group_id || groupId;
      const label = nextNumericName(labelBase, usedGroupLabels);
      nextView.groups.push({
        group_id: groupId,
        label,
        members,
        root: componentMap[original.root] || members[0],
        collapsed: false,
      });
    }
    return { hardware: nextHardware, topologyView: nextView, pastedIds, componentMap, portMap };
  }

  function motionDuration(reduceMotion, duration = 180) {
    return reduceMotion ? 0 : Math.max(0, finite(duration, 180));
  }

  return Object.freeze({
    VIEW_VERSION,
    CLIPBOARD_TYPE,
    normalizeTopologyView,
    validateGroups,
    createGroup,
    setGroupRoot,
    removeGroup,
    setGroupCollapsed,
    groupForComponent,
    screenToWorld,
    worldToScreen,
    normalizedRect,
    marqueeSelection,
    resolveCollisionPlacement,
    collapseProjection,
    layoutGraph,
    rectForNode,
    rectsIntersect,
    rectsIntersectWithGap,
    segmentHitsRect,
    lineSegmentHitsRect,
    portIdentity,
    assignPortEndpoints,
    routeOrthogonal,
    planOrthogonalRoutes,
    routeSegments,
    cubicBezierGeometry,
    cubicBezierPath,
    sampleCubicBezier,
    roundedOrthogonalGeometry,
    sampledPathClear,
    decorateRoutePlan,
    estimateLabelSize,
    estimateLabelRect,
    routeLabelCandidates,
    placeRouteLabels,
    pointsBounds,
    pathToSvg,
    computeWorldBounds,
    computeAutoFitScale,
    normalizeWorldOrigin,
    copySelection,
    validateClipboardPayload,
    parseClipboardText,
    pasteSelection,
    remapPhysicalCompositionComponentRefs,
    uniqueId,
    motionDuration,
  });
}));
