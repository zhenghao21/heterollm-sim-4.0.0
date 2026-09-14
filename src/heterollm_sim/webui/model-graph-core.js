"use strict";

// Pure model-graph authoring primitives.  The module intentionally has no DOM,
// storage, or network dependencies so every semantic edit can be verified in
// Node before the browser applies it to a scenario.
(function modelGraphCoreFactory(root, factory) {
  const api = factory();
  if (root) root.ModelGraphCore = api;
  if (typeof module === "object" && module.exports) module.exports = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function buildModelGraphCore() {
  const GRAPH_VERSION = 1;
  const TRANSFORM_KINDS = Object.freeze(["reshape", "transpose", "concat", "split", "cast", "broadcast"]);
  const DTYPE_ALIASES = Object.freeze({
    float16: "fp16", half: "fp16", fp16: "fp16",
    bfloat16: "bf16", bf16: "bf16",
    float32: "fp32", float: "fp32", fp32: "fp32",
    float64: "fp64", double: "fp64", fp64: "fp64",
    int8: "int8", uint8: "uint8", int16: "int16", int32: "int32", int64: "int64",
    bool: "bool", boolean: "bool",
  });

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

  function safeId(value, fallback = "item") {
    const text = String(value ?? fallback).trim()
      .replace(/[^A-Za-z0-9_.:-]+/g, "-")
      .replace(/^-+|-+$/g, "");
    return text || fallback;
  }

  function uniqueId(base, used, forceNew = false) {
    const normalized = safeId(base);
    if (!forceNew && !used.has(normalized)) {
      used.add(normalized);
      return normalized;
    }
    let index = 2;
    while (used.has(`${normalized}-${index}`)) index += 1;
    const result = `${normalized}-${index}`;
    used.add(result);
    return result;
  }

  function stableNodeId(kind, index = 0) {
    const base = safeId(kind || "operator", "operator");
    return index > 0 ? `${base}-${index + 1}` : base;
  }

  function stablePortId(direction, index = 0) {
    const prefixes = { input: "in", output: "out", weight: "weight" };
    return `${prefixes[direction] || "port"}${Math.max(0, Math.trunc(finite(index)))}`;
  }

  function stableTensorId(operatorId, portId) {
    return `${safeId(operatorId, "operator")}.${safeId(portId, "port")}`;
  }

  function stableStringify(value) {
    if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
    if (value && typeof value === "object") {
      return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`).join(",")}}`;
    }
    return JSON.stringify(value);
  }

  function normalizedDimension(value) {
    if (typeof value === "number") return Number.isFinite(value) ? value : String(value);
    const text = String(value ?? "").trim();
    if (/^-?\d+$/.test(text)) return Number(text);
    return text;
  }

  function normalizeShape(value) {
    if (Array.isArray(value)) return value.map(normalizedDimension);
    if (typeof value === "string") {
      return value.split(/[×x,\s]+/u).map((item) => item.trim()).filter(Boolean).map(normalizedDimension);
    }
    return [];
  }

  function canonicalDtype(value) {
    const text = String(value ?? "unknown").trim().toLowerCase().replace(/[\s-]+/g, "_") || "unknown";
    return DTYPE_ALIASES[text] || text;
  }

  function canonicalLayout(value) {
    return String(value ?? "logical").trim().toLowerCase().replace(/[\s-]+/g, "_") || "logical";
  }

  function normalizedContract(value, fallback = {}) {
    const source = object(value);
    const defaults = object(fallback);
    return {
      dtype: canonicalDtype(source.dtype ?? defaults.dtype ?? "unknown"),
      shape: normalizeShape(source.shape ?? defaults.shape ?? []),
      layout: canonicalLayout(source.layout ?? defaults.layout ?? "logical"),
    };
  }

  function isWildcard(value) {
    return value == null || value === "" || value === "*" || String(value).toLowerCase() === "unknown";
  }

  function isSymbol(value) {
    return typeof value === "string" && /^[A-Za-z_][A-Za-z0-9_]*$/.test(value) && !isWildcard(value);
  }

  function resolveBinding(value, bindings) {
    let current = value;
    const seen = new Set();
    while (isSymbol(current) && Object.hasOwn(bindings, current) && !seen.has(current)) {
      seen.add(current);
      current = bindings[current];
    }
    return current;
  }

  function bindDimension(expected, actual, bindings) {
    const left = resolveBinding(expected, bindings);
    const right = resolveBinding(actual, bindings);
    if (isWildcard(left) || isWildcard(right)) return { ok: true, value: isWildcard(left) ? right : left };
    if (left === right) return { ok: true, value: left };
    if (isSymbol(left)) {
      bindings[left] = right;
      return { ok: true, value: right };
    }
    if (isSymbol(right)) {
      bindings[right] = left;
      return { ok: true, value: left };
    }
    return { ok: false, expected: left, actual: right };
  }

  function shapeText(shape) {
    const dimensions = normalizeShape(shape);
    return dimensions.length ? dimensions.join(" × ") : "标量";
  }

  function contractText(contract) {
    const item = normalizedContract(contract);
    return `${item.dtype} [${shapeText(item.shape)}] / ${item.layout}`;
  }

  function unifyTensorContracts(expectedValue, actualValue, initialBindings = {}) {
    const expected = normalizedContract(expectedValue);
    const actual = normalizedContract(actualValue);
    const bindings = { ...object(initialBindings) };
    if (!isWildcard(expected.dtype) && !isWildcard(actual.dtype) && expected.dtype !== actual.dtype) {
      return {
        ok: false, code: "dtype_mismatch", expected, actual, bindings,
        message: `数据类型不匹配：期望 ${contractText(expected)}，实际 ${contractText(actual)}；请显式添加 cast 变换。`,
      };
    }
    if (!isWildcard(expected.layout) && !isWildcard(actual.layout) && expected.layout !== actual.layout) {
      return {
        ok: false, code: "layout_mismatch", expected, actual, bindings,
        message: `张量布局不匹配：期望 ${contractText(expected)}，实际 ${contractText(actual)}；请显式添加 transpose 变换。`,
      };
    }
    if (expected.shape.length !== actual.shape.length) {
      return {
        ok: false, code: "rank_mismatch", expected, actual, bindings,
        message: `张量阶数不匹配：期望 ${contractText(expected)}，实际 ${contractText(actual)}；请显式添加 reshape 变换。`,
      };
    }
    const shape = [];
    for (let index = 0; index < expected.shape.length; index += 1) {
      const result = bindDimension(expected.shape[index], actual.shape[index], bindings);
      if (!result.ok) {
        return {
          ok: false, code: "shape_mismatch", expected, actual, bindings, dimension: index,
          message: `张量第 ${index + 1} 维不匹配：期望 ${contractText(expected)}，实际 ${contractText(actual)}；请显式添加 reshape/transpose/broadcast 变换。`,
        };
      }
      shape.push(resolveBinding(result.value, bindings));
    }
    return {
      ok: true, code: "", expected, actual, bindings,
      contract: {
        dtype: isWildcard(actual.dtype) ? expected.dtype : actual.dtype,
        shape,
        layout: isWildcard(actual.layout) ? expected.layout : actual.layout,
      },
      message: `连接兼容：期望 ${contractText(expected)}，实际 ${contractText(actual)}。`,
    };
  }

  function rawTensorRefs(value) {
    return array(value).map((item) => {
      if (item && typeof item === "object") return String(item.tensor_id ?? item.id ?? "").trim();
      return String(item ?? "").trim();
    }).filter(Boolean);
  }

  function normalizeModelGraph(rawValue, modelValue = {}) {
    const rootValue = object(rawValue);
    const model = object(modelValue);
    const nested = object(object(rootValue.model).graph);
    const source = Object.keys(nested).length ? nested
      : (Object.keys(object(rootValue.graph)).length ? object(rootValue.graph) : rootValue);
    const rawOperators = array(source.operators ?? source.nodes);
    const usedOperators = new Set();
    const operatorIdMap = new Map();
    const operatorSeeds = rawOperators.map((raw, index) => {
      const item = object(raw);
      const oldId = String(item.operator_id ?? item.node_id ?? item.id ?? "").trim();
      const requested = oldId || stableNodeId(item.op_kind ?? item.kind ?? item.type ?? "operator", index);
      const operatorId = uniqueId(requested, usedOperators);
      if (oldId && !operatorIdMap.has(oldId)) operatorIdMap.set(oldId, operatorId);
      return { item, operatorId, index };
    });

    const usedTensors = new Set();
    const tensorIdMap = new Map();
    const tensors = [];
    const tensorById = new Map();
    for (const [index, raw] of array(source.tensors).entries()) {
      const item = object(raw);
      const oldId = String(item.tensor_id ?? item.id ?? "").trim();
      const tensorId = uniqueId(oldId || `tensor-${index + 1}`, usedTensors);
      if (oldId && !tensorIdMap.has(oldId)) tensorIdMap.set(oldId, tensorId);
      const tensor = {
        ...clone(item),
        tensor_id: tensorId,
        role: String(item.role || "activation"),
        logical_bytes: item.logical_bytes == null ? null : Math.max(0, Math.trunc(finite(item.logical_bytes))),
        producer_operator_id: item.producer_operator_id == null ? null : (operatorIdMap.get(String(item.producer_operator_id)) || safeId(item.producer_operator_id)),
        consumer_operator_ids: array(item.consumer_operator_ids).map((id) => operatorIdMap.get(String(id)) || safeId(id)),
        ...normalizedContract(item),
        attributes: clone(object(item.attributes)),
        provenance: clone(array(item.provenance)),
      };
      tensors.push(tensor);
      tensorById.set(tensorId, tensor);
    }

    const unknownTensorMap = new Map();
    function resolvedTensorId(value, fallback) {
      const raw = String(value ?? "").trim();
      if (raw && tensorIdMap.has(raw)) return tensorIdMap.get(raw);
      if (raw && unknownTensorMap.has(raw)) return unknownTensorMap.get(raw);
      const requested = raw || fallback;
      const id = uniqueId(requested, usedTensors);
      if (raw) unknownTensorMap.set(raw, id);
      return id;
    }

    function ensureTensor(tensorId, direction, contract) {
      if (tensorById.has(tensorId)) return tensorById.get(tensorId);
      const tensor = {
        tensor_id: tensorId,
        role: direction === "weight" ? "weight" : "activation",
        logical_bytes: null,
        producer_operator_id: null,
        consumer_operator_ids: [],
        ...normalizedContract(contract),
        attributes: {}, provenance: [],
      };
      tensors.push(tensor);
      tensorById.set(tensorId, tensor);
      return tensor;
    }

    const operators = operatorSeeds.map(({ item, operatorId, index }) => {
      const refs = {
        input: rawTensorRefs(item.input_tensor_ids ?? item.inputs),
        output: rawTensorRefs(item.output_tensor_ids ?? item.outputs),
        weight: rawTensorRefs(item.weight_tensor_ids ?? item.weights),
      };
      const usedPorts = new Set();
      let ports = array(item.ports).map((rawPort, portIndex) => {
        const port = object(rawPort);
        const direction = ["input", "output", "weight"].includes(String(port.direction)) ? String(port.direction) : "input";
        const directionIndex = array(item.ports).slice(0, portIndex).filter((entry) => String(object(entry).direction || "input") === direction).length;
        const fallbackRef = refs[direction][directionIndex];
        const portId = uniqueId(port.port_id ?? port.id ?? stablePortId(direction, directionIndex), usedPorts);
        const tensorId = resolvedTensorId(port.tensor_id ?? fallbackRef, stableTensorId(operatorId, portId));
        const tensor = ensureTensor(tensorId, direction, port);
        const explicit = normalizedContract(port, tensor);
        if (tensor.dtype === "unknown" && explicit.dtype !== "unknown") tensor.dtype = explicit.dtype;
        if (!tensor.shape.length && explicit.shape.length) tensor.shape = explicit.shape;
        if (tensor.layout === "logical" && Object.hasOwn(port, "layout")) tensor.layout = explicit.layout;
        return {
          ...clone(port), port_id: portId, direction, tensor_id: tensorId,
          dtype: tensor.dtype, shape: clone(tensor.shape), layout: tensor.layout,
          attributes: clone(object(port.attributes)),
        };
      });
      if (!ports.length) {
        ports = ["input", "output", "weight"].flatMap((direction) => refs[direction].map((tensorRef, portIndex) => {
          const portId = stablePortId(direction, portIndex);
          const tensorId = resolvedTensorId(tensorRef, stableTensorId(operatorId, portId));
          const tensor = ensureTensor(tensorId, direction, {});
          return { port_id: portId, direction, tensor_id: tensorId, dtype: tensor.dtype, shape: clone(tensor.shape), layout: tensor.layout, attributes: {} };
        }));
      }
      const byDirection = (direction) => ports.filter((port) => port.direction === direction).map((port) => port.tensor_id);
      const parent = String(object(item.attributes).parent_group_id || "");
      const attributes = clone(object(item.attributes));
      if (parent) attributes.parent_group_id = operatorIdMap.get(parent) || parent;
      return {
        ...clone(item), operator_id: operatorId,
        op_kind: String(item.op_kind ?? item.kind ?? item.type ?? "operator"),
        sequence_index: Number.isFinite(Number(item.sequence_index)) ? Number(item.sequence_index) : index,
        layer_id: item.layer_id == null ? null : String(item.layer_id),
        input_tensor_ids: byDirection("input"), output_tensor_ids: byDirection("output"), weight_tensor_ids: byDirection("weight"),
        ports, parameters: clone(object(item.parameters)), attributes, provenance: clone(array(item.provenance)),
      };
    });

    // Ports are the editing authority.  Rebuild producer/consumer indexes and
    // split accidental multi-producer tensors into deterministic tensor IDs.
    const producerFor = new Map();
    for (const operator of operators) {
      for (const port of operator.ports) {
        if (port.direction !== "output") continue;
        const owner = producerFor.get(port.tensor_id);
        if (owner && owner !== operator.operator_id) {
          const nextId = uniqueId(stableTensorId(operator.operator_id, port.port_id), usedTensors);
          const previous = tensorById.get(port.tensor_id);
          const next = { ...clone(previous), tensor_id: nextId, producer_operator_id: null, consumer_operator_ids: [] };
          tensors.push(next); tensorById.set(nextId, next); port.tensor_id = nextId;
        }
        producerFor.set(port.tensor_id, operator.operator_id);
      }
    }
    tensors.forEach((tensor) => { tensor.producer_operator_id = null; tensor.consumer_operator_ids = []; });
    for (const operator of operators) {
      const byDirection = { input: [], output: [], weight: [] };
      for (const port of operator.ports) {
        const tensor = ensureTensor(port.tensor_id, port.direction, port);
        port.dtype = tensor.dtype; port.shape = clone(tensor.shape); port.layout = tensor.layout;
        byDirection[port.direction].push(port.tensor_id);
        if (port.direction === "output") tensor.producer_operator_id = operator.operator_id;
        else if (!tensor.consumer_operator_ids.includes(operator.operator_id)) tensor.consumer_operator_ids.push(operator.operator_id);
      }
      operator.input_tensor_ids = byDirection.input;
      operator.output_tensor_ids = byDirection.output;
      operator.weight_tensor_ids = byDirection.weight;
    }

    const transforms = array(source.transforms).map((raw, index) => {
      const item = object(raw);
      const inputId = resolvedTensorId(item.input_tensor_id, `transform-${index + 1}.input`);
      const outputId = resolvedTensorId(item.output_tensor_id, `transform-${index + 1}.output`);
      ensureTensor(inputId, "input", {}); ensureTensor(outputId, "output", {});
      return {
        ...clone(item), transform_id: safeId(item.transform_id || `${item.kind || "transform"}-${index + 1}`),
        kind: String(item.kind || "reshape"), input_tensor_id: inputId, output_tensor_id: outputId,
        attributes: clone(object(item.attributes)),
      };
    });
    const usedTransforms = new Set();
    transforms.forEach((item) => { item.transform_id = uniqueId(item.transform_id, usedTransforms); });

    const attributes = clone(object(source.attributes));
    attributes.ui = clone(object(attributes.ui));
    const positions = object(attributes.ui.positions);
    attributes.ui.positions = Object.fromEntries(Object.entries(positions).map(([id, point]) => [operatorIdMap.get(id) || id, {
      x: finite(object(point).x), y: finite(object(point).y),
    }]));
    if (attributes.ui.viewport) attributes.ui.viewport = {
      x: finite(attributes.ui.viewport.x), y: finite(attributes.ui.viewport.y),
      scale: Math.min(4, Math.max(0.25, finite(attributes.ui.viewport.scale, 1))),
    };
    attributes.ui.collapsed_groups = array(attributes.ui.collapsed_groups).map((id) => operatorIdMap.get(String(id)) || String(id));

    operators.sort((a, b) => a.sequence_index - b.sequence_index || a.operator_id.localeCompare(b.operator_id));
    tensors.sort((a, b) => a.tensor_id.localeCompare(b.tensor_id));
    return {
      graph_id: String(source.graph_id || model.name || "model"),
      operators, tensors, transforms,
      executable: source.executable !== false && operators.length > 0,
      attributes, provenance: clone(array(source.provenance)),
    };
  }

  function modelGraphAuthoringSummary(graphValue) {
    const graph = normalizeModelGraph(graphValue);
    const attributes = object(graph.attributes);
    const architecture = String(attributes.architecture || "").trim();
    if (!architecture) throw new Error("model.graph.attributes.architecture 必须是非空字符串。");
    const symbols = object(attributes.symbols);
    const vocabularySize = Number(symbols.V ?? 0);
    const maxSequenceLength = Number(attributes.max_sequence_length ?? 0);
    for (const [label, value] of [["model.graph.attributes.symbols.V", vocabularySize], ["model.graph.attributes.max_sequence_length", maxSequenceLength]]) {
      if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} 必须是非负安全整数。`);
    }
    graph.operators.forEach((operator) => {
      const parameters = object(operator.parameters);
      if (["embedding", "lm_head", "mtp_aux_head"].includes(operator.op_kind)
          && Object.hasOwn(parameters, "vocabulary_size")
          && Number(parameters.vocabulary_size) !== vocabularySize) {
        throw new Error(`model.graph vocabulary_size 冲突：symbols.V=${vocabularySize}，${operator.operator_id}=${parameters.vocabulary_size}。`);
      }
      if (operator.op_kind === "model_input"
          && Object.hasOwn(parameters, "max_sequence_length")
          && Number(parameters.max_sequence_length) !== maxSequenceLength) {
        throw new Error(`model.graph max_sequence_length 冲突：attributes=${maxSequenceLength}，${operator.operator_id}=${parameters.max_sequence_length}。`);
      }
    });
    const embeddings = graph.operators.filter((operator) => operator.op_kind === "embedding");
    if (embeddings.length !== 1 || embeddings[0].weight_tensor_ids.length !== 1) {
      throw new Error("model.graph 必须包含一个且仅一个 embedding 组件，并声明一个权重张量。");
    }
    const embeddingTensorId = embeddings[0].weight_tensor_ids[0];
    const tensor = graph.tensors.find((item) => item.tensor_id === embeddingTensorId);
    if (!tensor) throw new Error(`model.graph embedding 引用了不存在的权重张量 ${embeddingTensorId}。`);
    let embeddingWeightBytes = tensor.logical_bytes;
    if (embeddingWeightBytes == null) {
      const dtypeKey = canonicalDtype(tensor.dtype).replaceAll("-", "").replaceAll("_", "");
      const dtypeBitWidths = { fp32: 32, float32: 32, fp16: 16, float16: 16, bf16: 16, bfloat16: 16, fp8: 8, float8: 8, int8: 8, uint8: 8, int4: 4, uint4: 4 };
      const bits = dtypeBitWidths[dtypeKey];
      if (!bits) throw new Error(`不支持的数据类型 ${tensor.dtype}，无法推导 embedding 权重字节数。`);
      let elements = 1;
      tensor.shape.forEach((rawDimension) => {
        const dimension = typeof rawDimension === "string" ? symbols[rawDimension] : rawDimension;
        if (!Number.isSafeInteger(dimension) || dimension < 0) {
          throw new Error("model.graph embedding 权重字节数无法从 shape 推导；请声明 tensor.logical_bytes。");
        }
        elements *= dimension;
        if (!Number.isSafeInteger(elements)) throw new Error("model.graph embedding 权重元素数超出安全整数范围。");
      });
      embeddingWeightBytes = Math.ceil((elements * bits) / 8);
    }
    if (!Number.isSafeInteger(embeddingWeightBytes) || embeddingWeightBytes < 0) {
      throw new Error("model.graph embedding logical_bytes 必须是非负安全整数。");
    }
    return Object.freeze({
      architecture,
      vocabulary_size: vocabularySize,
      max_sequence_length: maxSequenceLength,
      embedding_weight_bytes: embeddingWeightBytes,
    });
  }

  function layerTemplate(layerValue) {
    const layer = object(layerValue);
    return {
      kind: String(layer.kind || "dense"),
      hidden_size: Math.trunc(finite(layer.hidden_size)),
      intermediate_size: Math.trunc(finite(layer.intermediate_size)),
      attention_heads: Math.trunc(finite(layer.attention_heads)),
      kv_heads: Math.trunc(finite(layer.kv_heads)),
      attention_head_dim: Math.trunc(finite(layer.attention_head_dim)),
      sequence_mixer: String(layer.sequence_mixer || "full_attention"),
      linear_attention: layer.linear_attention == null ? null : clone(object(layer.linear_attention)),
      num_experts: Math.max(1, Math.trunc(finite(layer.num_experts, 1))),
      experts_per_token: Math.max(1, Math.trunc(finite(layer.experts_per_token, 1))),
      shared_expert_intermediate_size: Math.max(0, Math.trunc(finite(layer.shared_expert_intermediate_size))),
      shared_expert_gate: layer.shared_expert_gate === true,
      dtype: canonicalDtype(layer.dtype || "fp16"),
      quantization: layer.quantization == null || layer.quantization === "" ? null : String(layer.quantization),
      weight_bytes: Math.max(0, Math.trunc(finite(layer.weight_bytes))),
    };
  }

  function groupRepeatedLayers(layersValue) {
    const groups = [];
    let previousKey = null;
    array(layersValue).forEach((raw, index) => {
      const layer = clone(object(raw));
      const template = layerTemplate(layer);
      const key = stableStringify({ template, pattern_index: object(layer.metadata).pattern_index ?? null });
      if (!groups.length || key !== previousKey) {
        groups.push({ template, layers: [], start_index: index });
        previousKey = key;
      }
      groups.at(-1).layers.push(layer);
    });
    return groups;
  }

  function buildModelGraphFromLayerSpecs(layerSpecsValue, options = {}) {
    const model = object(options);
    const layers = array(layerSpecsValue).map((item) => clone(object(item)));
    if (!layers.length) throw new Error("无法从空 layer specs 生成模型组件图。");
    const first = layerTemplate(layers[0]);
    if (first.hidden_size < 1) throw new Error("layer specs 的 hidden_size 必须为正整数。");
    const graphId = String(options.graph_id || model.name || "model");
    const operators = [];
    const tensors = new Map();
    let sequenceIndex = 0;

    function ensureTensor(tensorId, role, dtype, shape, details = {}) {
      const contract = normalizedContract({ dtype, shape, layout: details.layout || "logical" });
      let tensor = tensors.get(tensorId);
      if (!tensor) {
        tensor = {
          tensor_id: tensorId, role, logical_bytes: details.logical_bytes ?? null,
          producer_operator_id: details.producer ?? null, consumer_operator_ids: [],
          ...contract, attributes: {}, provenance: [],
        };
        tensors.set(tensorId, tensor);
      }
      if (details.producer != null) tensor.producer_operator_id = details.producer;
      if (details.consumer != null && !tensor.consumer_operator_ids.includes(details.consumer)) tensor.consumer_operator_ids.push(details.consumer);
      return tensor;
    }

    function addOperator(operatorId, opKind, spec = {}) {
      const ports = [];
      const inputs = array(spec.inputs);
      const outputs = array(spec.outputs);
      const weights = array(spec.weights);
      inputs.forEach((tensorId, index) => {
        const tensor = tensors.get(tensorId);
        if (!tensor) throw new Error(`组件 ${operatorId} 引用了尚未声明的输入张量 ${tensorId}。`);
        ensureTensor(tensorId, tensor.role, tensor.dtype, tensor.shape, { consumer: operatorId, layout: tensor.layout });
        ports.push({ port_id: stablePortId("input", index), direction: "input", tensor_id: tensorId, dtype: tensor.dtype, shape: clone(tensor.shape), layout: tensor.layout, attributes: {} });
      });
      outputs.forEach((entry, index) => {
        const [tensorId, dtype, shape, role = "activation"] = entry;
        const tensor = ensureTensor(tensorId, role, dtype, shape, { producer: operatorId });
        ports.push({ port_id: stablePortId("output", index), direction: "output", tensor_id: tensorId, dtype: tensor.dtype, shape: clone(tensor.shape), layout: tensor.layout, attributes: {} });
      });
      weights.forEach((entry, index) => {
        const [tensorId, dtype, shape, logicalBytes = null] = entry;
        const tensor = ensureTensor(tensorId, "weight", dtype, shape, { consumer: operatorId, logical_bytes: logicalBytes });
        ports.push({ port_id: stablePortId("weight", index), direction: "weight", tensor_id: tensorId, dtype: tensor.dtype, shape: clone(tensor.shape), layout: tensor.layout, attributes: {} });
      });
      operators.push({
        operator_id: operatorId, op_kind: opKind, sequence_index: sequenceIndex,
        layer_id: spec.layer_id == null ? null : String(spec.layer_id),
        input_tensor_ids: inputs,
        output_tensor_ids: outputs.map((entry) => entry[0]),
        weight_tensor_ids: weights.map((entry) => entry[0]),
        ports, parameters: clone(object(spec.parameters)), attributes: clone(object(spec.attributes)), provenance: [],
      });
      sequenceIndex += 1;
    }

    const dtype = first.dtype;
    let hidden = first.hidden_size;
    addOperator("input", "model_input", {
      outputs: [["input.tokens", "int64", ["B", "T"], "input"]],
      parameters: { max_sequence_length: Math.max(0, Math.trunc(finite(model.max_sequence_length))) },
    });
    addOperator("embedding", "embedding", {
      inputs: ["input.tokens"], outputs: [["embedding.output", dtype, ["B", "T", hidden]]],
      weights: [["embedding_weights", dtype, ["V", hidden], model.embedding_weight_bytes == null ? null : Math.max(0, Math.trunc(finite(model.embedding_weight_bytes)))]],
      parameters: { vocabulary_size: Math.max(0, Math.trunc(finite(model.vocabulary_size))), hidden_size: hidden },
    });
    let previous = "embedding.output";
    for (const [groupIndex, grouped] of groupRepeatedLayers(layers).entries()) {
      const base = grouped.template;
      hidden = base.hidden_size;
      const groupId = `block-group-${String(groupIndex).padStart(3, "0")}`;
      const layerIds = grouped.layers.map((layer, offset) => String(layer.layer_id || `layer-${String(grouped.start_index + offset).padStart(3, "0")}`));
      const overrides = {};
      grouped.layers.forEach((layer, offset) => {
        overrides[layerIds[offset]] = {
          metadata: clone(object(layer.metadata)),
          schema_version: String(layer.schema_version || model.schema_version || "1.0"),
        };
      });
      addOperator(groupId, "layer_group", {
        parameters: { repeat: grouped.layers.length, layer_ids: layerIds, hidden_size: hidden, dtype: base.dtype, weight_bytes: base.weight_bytes, overrides },
        attributes: { collapsed: true, layer_ids: layerIds },
      });
      const parent = { parent_group_id: groupId, layer_ids: layerIds };
      const norm1 = `${groupId}.norm1.output`;
      addOperator(`${groupId}.norm1`, "rms_norm", { inputs: [previous], outputs: [[norm1, base.dtype, ["B", "T", hidden]]], attributes: parent });
      const mixerKind = base.sequence_mixer === "linear_attention" || base.linear_attention ? "linear_attention" : "attention";
      const mixer = `${groupId}.mixer.output`;
      addOperator(`${groupId}.${mixerKind}`, mixerKind, {
        inputs: [norm1], outputs: [[mixer, base.dtype, ["B", "T", hidden]]],
        weights: [[`${groupId}.attention_weights`, base.dtype, [hidden, hidden], null]],
        parameters: {
          sequence_mixer: base.sequence_mixer, attention_heads: base.attention_heads,
          kv_heads: base.kv_heads, attention_head_dim: base.attention_head_dim,
          linear_attention: clone(base.linear_attention),
        }, attributes: parent,
      });
      const residual1 = `${groupId}.residual1.output`;
      addOperator(`${groupId}.residual1`, "residual_add", { inputs: [previous, mixer], outputs: [[residual1, base.dtype, ["B", "T", hidden]]], attributes: parent });
      const norm2 = `${groupId}.norm2.output`;
      addOperator(`${groupId}.norm2`, "rms_norm", { inputs: [residual1], outputs: [[norm2, base.dtype, ["B", "T", hidden]]], attributes: parent });
      const ffParameters = {
        kind: base.kind, intermediate_size: base.intermediate_size,
        num_experts: base.num_experts, experts_per_token: base.experts_per_token,
        shared_expert_intermediate_size: base.shared_expert_intermediate_size,
        shared_expert_gate: base.shared_expert_gate, quantization: base.quantization,
      };
      const ffOutput = `${groupId}.ff.output`;
      const isMoe = base.kind.toLowerCase().includes("moe") || base.num_experts > 1;
      if (!isMoe) {
        addOperator(`${groupId}.mlp`, "dense_mlp", {
          inputs: [norm2], outputs: [[ffOutput, base.dtype, ["B", "T", hidden]]],
          weights: [[`${groupId}.mlp_weights`, base.dtype, [hidden, base.intermediate_size], base.weight_bytes]],
          parameters: ffParameters, attributes: parent,
        });
      } else {
        const route = `${groupId}.router.output`;
        addOperator(`${groupId}.router`, "moe_router", {
          inputs: [norm2], outputs: [[route, base.dtype, ["B", "T", base.experts_per_token]]],
          weights: [[`${groupId}.router_weights`, base.dtype, [hidden, base.num_experts], null]],
          parameters: ffParameters, attributes: parent,
        });
        const expert = `${groupId}.experts.output`;
        addOperator(`${groupId}.experts`, "moe_experts", {
          inputs: [norm2, route], outputs: [[expert, base.dtype, ["B", "T", hidden]]],
          weights: [[`${groupId}.expert_weights`, base.dtype, [base.num_experts, hidden, base.intermediate_size], base.weight_bytes]],
          parameters: ffParameters, attributes: parent,
        });
        const combineInputs = [expert];
        if (base.shared_expert_intermediate_size > 0) {
          const shared = `${groupId}.shared_expert.output`;
          addOperator(`${groupId}.shared_expert`, "shared_expert", {
            inputs: [norm2], outputs: [[shared, base.dtype, ["B", "T", hidden]]],
            weights: [[`${groupId}.shared_expert_weights`, base.dtype, [hidden, base.shared_expert_intermediate_size], null]],
            parameters: ffParameters, attributes: parent,
          });
          combineInputs.push(shared);
        }
        addOperator(`${groupId}.moe_combine`, "moe_combine", { inputs: combineInputs, outputs: [[ffOutput, base.dtype, ["B", "T", hidden]]], parameters: ffParameters, attributes: parent });
      }
      const output = `${groupId}.output`;
      addOperator(`${groupId}.residual2`, "residual_add", { inputs: [residual1, ffOutput], outputs: [[output, base.dtype, ["B", "T", hidden]]], attributes: parent });
      previous = output;
    }
    const mtpBranchSource = previous;
    const mtp = object(model.mtp);
    const mtpPredictionLayers = Math.max(0, Math.trunc(finite(mtp.prediction_layers)));
    const mtpPredictionWeightBytes = Math.max(0, Math.trunc(finite(mtp.prediction_layer_weight_bytes)));
    const mtpAuxHead = mtp.auxiliary_head === true;
    const mtpAuxHeadWeightBytes = Math.max(0, Math.trunc(finite(mtp.auxiliary_head_weight_bytes)));
    let mtpPrevious = mtpBranchSource;
    for (let index = 0; index < mtpPredictionLayers; index += 1) {
      const prefix = `mtp.prediction_layer.${String(index).padStart(3, "0")}`;
      const output = `${prefix}.output`;
      addOperator(prefix, "mtp_prediction_layer", {
        inputs: [mtpPrevious], outputs: [[output, dtype, ["B", "T", hidden]]],
        weights: [[`${prefix}.weights`, dtype, [hidden, hidden], mtpPredictionWeightBytes]],
        parameters: { hidden_size: hidden },
        attributes: { branch: "mtp", source_tensor_id: mtpBranchSource },
      });
      mtpPrevious = output;
    }
    if (mtpAuxHead) {
      addOperator("mtp.aux_head", "mtp_aux_head", {
        inputs: [mtpPrevious], outputs: [["mtp.proposal_logits", dtype, ["B", "T", "V"], "output"]],
        weights: [["mtp.aux_head.weights", dtype, [hidden, "V"], mtpAuxHeadWeightBytes]],
        parameters: { hidden_size: hidden, vocabulary_size: Math.max(0, Math.trunc(finite(model.vocabulary_size))) },
        attributes: { branch: "mtp", source_tensor_id: mtpBranchSource },
      });
    }
    addOperator("final_norm", "rms_norm", { inputs: [previous], outputs: [["final_norm.output", dtype, ["B", "T", hidden]]] });
    addOperator("lm_head", "lm_head", {
      inputs: ["final_norm.output"], outputs: [["logits", dtype, ["B", "T", "V"]]],
      weights: [["lm_head_weights", dtype, [hidden, "V"], null]],
      parameters: { vocabulary_size: Math.max(0, Math.trunc(finite(model.vocabulary_size))) },
    });
    addOperator("output", "model_output", { inputs: ["logits"], parameters: { kind: "logits" } });
    return normalizeModelGraph({
      graph_id: graphId, operators, tensors: Array.from(tensors.values()), transforms: [], executable: true,
      attributes: {
        authoritative: true, derivation: "layer_specs", architecture: String(model.architecture || "transformer"),
        symbols: { B: "batch", T: "sequence", V: Math.max(0, Math.trunc(finite(model.vocabulary_size))) },
        max_sequence_length: Math.max(0, Math.trunc(finite(model.max_sequence_length))),
        metadata: clone(object(model.metadata)),
        ui: { collapsed_groups: operators.filter((item) => item.op_kind === "layer_group").map((item) => item.operator_id), positions: {} },
      }, provenance: [],
    });
  }

  function graphToLayerSpecs(graphValue, options = {}) {
    const graph = normalizeModelGraph(graphValue);
    if (!graph.executable) return { ok: false, code: "not_executable", reason: "不可执行的模型图不能生成执行 layer specs。", layer_specs: [] };
    const groups = graph.operators.filter((item) => item.op_kind === "layer_group")
      .sort((a, b) => a.sequence_index - b.sequence_index || a.operator_id.localeCompare(b.operator_id));
    if (!groups.length) return { ok: false, code: "missing_layer_group", reason: "模型图缺少 layer_group，无法生成执行 layer specs。", layer_specs: [] };
    const result = [];
    for (const group of groups) {
      const params = object(group.parameters);
      const previous = result.length;
      let layerIds = array(params.layer_ids ?? group.attributes.layer_ids).map(String).filter(Boolean);
      const requestedRepeat = Math.trunc(finite(params.repeat, layerIds.length));
      const repeat = Math.max(1, requestedRepeat);
      if (!layerIds.length && object(params.layer_template).layer_id) {
        layerIds = Array.from({ length: repeat }, (_, index) => `${safeId(params.layer_template.layer_id)}-${index + 1}`);
      } else if (!layerIds.length && Object.keys(object(params.layer_template)).length) {
        layerIds = Array.from({ length: repeat }, (_, index) => `layer-${String(previous + index).padStart(3, "0")}`);
      }
      if (!layerIds.length) return { ok: false, code: "missing_layer_ids", reason: `组件组 ${group.operator_id} 缺少 layer_ids。`, layer_specs: [] };
      if (repeat !== layerIds.length) return { ok: false, code: "repeat_mismatch", reason: `组件组 ${group.operator_id} 的 repeat 与 layer_ids 数量不一致。`, layer_specs: [] };
      const children = graph.operators.filter((item) => String(item.attributes.parent_group_id || "") === group.operator_id);
      const mixer = children.find((item) => ["attention", "self_attention", "linear_attention"].includes(item.op_kind));
      const ff = children.find((item) => ["dense_mlp", "moe_router", "moe_experts", "experts"].includes(item.op_kind));
      const layerTemplateValue = object(params.layer_template);
      if ((!mixer || !ff) && !Object.keys(layerTemplateValue).length) {
        return { ok: false, code: "missing_components", reason: `组件组 ${group.operator_id} 缺少注意力或前馈组件，无法生成执行 layer specs。`, layer_specs: [] };
      }
      const mixerParams = object(mixer?.parameters);
      const ffParams = object(ff?.parameters);
      const overrides = object(params.overrides);
      layerIds.forEach((layerId, index) => {
        const override = object(overrides[layerId] ?? overrides[String(index)]);
        const base = Object.keys(layerTemplateValue).length ? clone(layerTemplateValue) : {
          kind: String(ffParams.kind || (ff?.op_kind === "dense_mlp" ? "dense" : "moe")),
          hidden_size: Math.trunc(finite(params.hidden_size)),
          intermediate_size: Math.trunc(finite(ffParams.intermediate_size)),
          attention_heads: Math.trunc(finite(mixerParams.attention_heads)),
          kv_heads: Math.trunc(finite(mixerParams.kv_heads)),
          attention_head_dim: Math.trunc(finite(mixerParams.attention_head_dim)),
          sequence_mixer: String(mixerParams.sequence_mixer || (mixer?.op_kind === "linear_attention" ? "linear_attention" : "full_attention")),
          linear_attention: mixerParams.linear_attention == null ? null : clone(object(mixerParams.linear_attention)),
          num_experts: Math.max(1, Math.trunc(finite(ffParams.num_experts, 1))),
          experts_per_token: Math.max(1, Math.trunc(finite(ffParams.experts_per_token, 1))),
          shared_expert_intermediate_size: Math.max(0, Math.trunc(finite(ffParams.shared_expert_intermediate_size))),
          shared_expert_gate: ffParams.shared_expert_gate === true,
          dtype: canonicalDtype(params.dtype || "fp16"),
          quantization: ffParams.quantization == null ? null : ffParams.quantization,
          weight_bytes: Math.max(0, Math.trunc(finite(params.weight_bytes))),
        };
        result.push({
          ...base, ...clone(override), layer_id: layerId,
          metadata: clone(object(override.metadata ?? base.metadata)),
          schema_version: String(override.schema_version || base.schema_version || options.schema_version || "1.0"),
        });
      });
    }
    return { ok: true, code: "", reason: "", layer_specs: result };
  }

  function setLayerGroupOverride(graphValue, groupId, layerRef, patchValue) {
    const graph = normalizeModelGraph(graphValue);
    const group = graph.operators.find((item) => item.operator_id === groupId && item.op_kind === "layer_group");
    if (!group) throw new Error(`找不到重复层组件组：${groupId}`);
    const layerIds = array(group.parameters.layer_ids ?? group.attributes.layer_ids).map(String);
    const layerId = typeof layerRef === "number" ? layerIds[layerRef] : String(layerRef);
    if (!layerId || !layerIds.includes(layerId)) throw new Error(`组件组 ${groupId} 中找不到层实例：${layerRef}`);
    const next = clone(graph);
    const target = next.operators.find((item) => item.operator_id === groupId);
    const overrides = clone(object(target.parameters.overrides));
    if (patchValue == null) delete overrides[layerId];
    else overrides[layerId] = { ...object(overrides[layerId]), ...clone(object(patchValue)) };
    target.parameters.overrides = overrides;
    return next;
  }

  function normalizeAxis(axisValue, rank) {
    let axis = Math.trunc(finite(axisValue));
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) throw new Error(`轴 ${axisValue} 超出 ${rank} 维张量范围。`);
    return axis;
  }

  function numericProduct(shape) {
    return shape.every((item) => Number.isInteger(item) && item >= 0) ? shape.reduce((total, item) => total * item, 1) : null;
  }

  function transformFailure(code, message, inputs = []) {
    return { ok: false, code, message, inputs: clone(inputs), outputs: [] };
  }

  function inferTransformContracts(kindValue, inputsValue, attributesValue = {}) {
    const kind = String(kindValue || "").toLowerCase();
    const inputs = array(inputsValue).map((item) => normalizedContract(item));
    const attributes = object(attributesValue);
    if (!TRANSFORM_KINDS.includes(kind)) return transformFailure("unknown_transform", `不支持的显式变换：${kindValue}`, inputs);
    if (!inputs.length) return transformFailure("missing_input", `${kind} 变换至少需要一个输入张量。`, inputs);
    const first = inputs[0];
    try {
      if (kind === "reshape") {
        const target = normalizeShape(attributes.shape ?? attributes.target_shape);
        if (!target.length) return transformFailure("missing_target_shape", "reshape 必须声明目标 shape。", inputs);
        const inferIndexes = target.map((item, index) => item === -1 ? index : -1).filter((index) => index >= 0);
        if (inferIndexes.length > 1) return transformFailure("multiple_inferred_dimensions", "reshape 最多只能包含一个 -1 推导维度。", inputs);
        const inputCount = numericProduct(first.shape);
        if (inferIndexes.length && inputCount != null) {
          const known = target.filter((_, index) => index !== inferIndexes[0]);
          const knownCount = numericProduct(known);
          if (!knownCount || inputCount % knownCount) return transformFailure("element_count_mismatch", `reshape 元素数量不匹配：期望可整除 ${knownCount}，实际 ${inputCount}。`, inputs);
          target[inferIndexes[0]] = inputCount / knownCount;
        }
        const targetCount = numericProduct(target);
        if (inputCount != null && targetCount != null && inputCount !== targetCount) {
          return transformFailure("element_count_mismatch", `reshape 元素数量不匹配：期望 ${targetCount}，实际 ${inputCount}。`, inputs);
        }
        return { ok: true, code: "", message: "reshape 维度规则有效。", inputs, outputs: [{ ...first, shape: target }] };
      }
      if (kind === "transpose") {
        const rank = first.shape.length;
        const permutation = array(attributes.permutation ?? attributes.perm).map((item) => Math.trunc(finite(item)));
        const perm = permutation.length ? permutation : Array.from({ length: rank }, (_, index) => rank - index - 1);
        if (perm.length !== rank || new Set(perm).size !== rank || perm.some((item) => item < 0 || item >= rank)) {
          return transformFailure("invalid_permutation", `transpose 的 permutation 必须是 0..${Math.max(0, rank - 1)} 的完整排列。`, inputs);
        }
        return { ok: true, code: "", message: "transpose 维度规则有效。", inputs, outputs: [{ ...first, shape: perm.map((index) => first.shape[index]), layout: canonicalLayout(attributes.layout || `${first.layout}_transposed`) }] };
      }
      if (kind === "concat") {
        if (inputs.length < 2) return transformFailure("missing_input", "concat 至少需要两个输入张量。", inputs);
        const rank = first.shape.length;
        const axis = normalizeAxis(attributes.axis, rank);
        const bindings = {};
        let axisTotal = 0;
        const symbolic = [];
        for (let inputIndex = 0; inputIndex < inputs.length; inputIndex += 1) {
          const input = inputs[inputIndex];
          if (input.shape.length !== rank) return transformFailure("rank_mismatch", `concat 输入 ${inputIndex + 1} 阶数不匹配：期望 ${rank}，实际 ${input.shape.length}。`, inputs);
          if (input.dtype !== first.dtype || input.layout !== first.layout) return transformFailure("contract_mismatch", `concat 输入 ${inputIndex + 1} 类型/布局不匹配：期望 ${contractText(first)}，实际 ${contractText(input)}。`, inputs);
          for (let index = 0; index < rank; index += 1) {
            if (index === axis) continue;
            const unified = bindDimension(first.shape[index], input.shape[index], bindings);
            if (!unified.ok) return transformFailure("shape_mismatch", `concat 第 ${index + 1} 维不匹配：期望 ${shapeText(first.shape)}，实际 ${shapeText(input.shape)}。`, inputs);
          }
          if (typeof input.shape[axis] === "number") axisTotal += input.shape[axis];
          else symbolic.push(String(input.shape[axis]));
        }
        const outputShape = first.shape.map((item) => resolveBinding(item, bindings));
        outputShape[axis] = symbolic.length ? [...(axisTotal ? [String(axisTotal)] : []), ...symbolic].join("+") : axisTotal;
        return { ok: true, code: "", message: "concat 维度规则有效。", inputs, outputs: [{ ...first, shape: outputShape }], bindings };
      }
      if (kind === "split") {
        const rank = first.shape.length;
        const axis = normalizeAxis(attributes.axis, rank);
        let sizes = array(attributes.sizes ?? attributes.split_sizes).map(normalizedDimension);
        const sections = Math.max(0, Math.trunc(finite(attributes.sections)));
        if (!sizes.length && sections > 0) {
          const dimension = first.shape[axis];
          if (typeof dimension === "number" && dimension % sections) return transformFailure("split_size_mismatch", `split 轴长度 ${dimension} 不能被 ${sections} 等分。`, inputs);
          sizes = Array.from({ length: sections }, () => typeof dimension === "number" ? dimension / sections : `${dimension}/${sections}`);
        }
        if (!sizes.length) return transformFailure("missing_split_sizes", "split 必须声明 sizes 或 sections。", inputs);
        const dimension = first.shape[axis];
        if (typeof dimension === "number" && sizes.every((item) => typeof item === "number") && sizes.reduce((sum, item) => sum + item, 0) !== dimension) {
          return transformFailure("split_size_mismatch", `split 分段总长度期望 ${dimension}，实际 ${sizes.reduce((sum, item) => sum + item, 0)}。`, inputs);
        }
        return { ok: true, code: "", message: "split 维度规则有效。", inputs, outputs: sizes.map((size) => ({ ...first, shape: first.shape.map((item, index) => index === axis ? size : item) })) };
      }
      if (kind === "cast") {
        const dtype = canonicalDtype(attributes.dtype ?? attributes.to_dtype);
        if (isWildcard(dtype)) return transformFailure("missing_dtype", "cast 必须声明目标 dtype。", inputs);
        return { ok: true, code: "", message: "cast 类型规则有效。", inputs, outputs: [{ ...first, dtype }] };
      }
      const target = normalizeShape(attributes.shape ?? attributes.target_shape);
      if (!target.length || target.length < first.shape.length) return transformFailure("invalid_broadcast_shape", `broadcast 目标维度必须不少于输入：期望至少 ${first.shape.length} 维，实际 ${target.length} 维。`, inputs);
      const padded = Array(target.length - first.shape.length).fill(1).concat(first.shape);
      const bindings = {};
      for (let index = 0; index < target.length; index += 1) {
        if (padded[index] === 1) continue;
        const unified = bindDimension(target[index], padded[index], bindings);
        if (!unified.ok) return transformFailure("broadcast_mismatch", `broadcast 第 ${index + 1} 维不匹配：期望 ${shapeText(target)}，实际 ${shapeText(first.shape)}。`, inputs);
      }
      return { ok: true, code: "", message: "broadcast 维度规则有效。", inputs, outputs: [{ ...first, shape: target.map((item) => resolveBinding(item, bindings)) }], bindings };
    } catch (error) {
      return transformFailure("invalid_transform", String(error.message || error), inputs);
    }
  }

  function modelGraphEdges(graphValue) {
    const graph = normalizeModelGraph(graphValue);
    const ids = new Set(graph.operators.map((item) => item.operator_id));
    return graph.tensors.flatMap((tensor) => {
      if (!tensor.producer_operator_id || !ids.has(tensor.producer_operator_id)) return [];
      return tensor.consumer_operator_ids.filter((id) => ids.has(id)).map((targetId) => ({
        edge_id: `${tensor.tensor_id}:${tensor.producer_operator_id}:${targetId}`,
        tensor_id: tensor.tensor_id, source_operator_id: tensor.producer_operator_id,
        target_operator_id: targetId, dtype: tensor.dtype, shape: clone(tensor.shape), layout: tensor.layout,
      }));
    });
  }

  function overviewGroupDescriptor(graphValue, groupValue) {
    const graph = normalizeModelGraph(graphValue);
    const groupId = String(object(groupValue).operator_id || "");
    const group = graph.operators.find((item) => item.operator_id === groupId && item.op_kind === "layer_group");
    if (!group) return null;
    const children = graph.operators
      .filter((item) => String(object(item.attributes).parent_group_id || "") === groupId)
      .sort((left, right) => left.sequence_index - right.sequence_index || left.operator_id.localeCompare(right.operator_id));
    const mixer = children.find((item) => ["attention", "self_attention", "full_attention", "linear_attention"].includes(item.op_kind));
    const mixerKind = mixer?.op_kind === "linear_attention" || String(object(mixer?.parameters).sequence_mixer || "").includes("linear")
      ? "linear_attention"
      : (mixer ? "full_attention" : "unknown_attention");
    const router = children.find((item) => item.op_kind === "moe_router");
    const experts = children.find((item) => ["experts", "moe_experts"].includes(item.op_kind));
    const dense = children.find((item) => ["dense_mlp", "mlp", "feed_forward"].includes(item.op_kind));
    const ffnKind = router || experts ? "moe" : (dense ? "dense" : "unknown");
    const parameters = object(router?.parameters || experts?.parameters || dense?.parameters);
    const repeat = Math.max(1, Math.trunc(finite(object(group.parameters).repeat, 1)));
    const descriptor = {
      group_id: groupId,
      representative_group_id: groupId,
      repeat,
      layer_ids: array(object(group.parameters).layer_ids || object(group.attributes).layer_ids).map(String),
      mixer_kind: mixerKind,
      ffn_kind: ffnKind,
      num_experts: Math.max(1, Math.trunc(finite(parameters.num_experts, 1))),
      experts_per_token: Math.max(1, Math.trunc(finite(parameters.experts_per_token, 1))),
      shared_expert: children.some((item) => item.op_kind === "shared_expert"),
      operator_ids: children.map((item) => item.operator_id),
      residual_operator_ids: children.filter((item) => item.op_kind === "residual_add").map((item) => item.operator_id),
    };
    descriptor.signature = stableStringify({
      mixer_kind: descriptor.mixer_kind,
      ffn_kind: descriptor.ffn_kind,
      repeat: descriptor.repeat,
      num_experts: descriptor.num_experts,
      experts_per_token: descriptor.experts_per_token,
      shared_expert: descriptor.shared_expert,
    });
    return descriptor;
  }

  function overviewPatternPeriod(descriptorsValue) {
    const descriptors = array(descriptorsValue);
    if (!descriptors.length) return { length: 0, repetitions: 0, pattern: [] };
    for (let length = 1; length <= descriptors.length; length += 1) {
      if (descriptors.length % length) continue;
      const matches = descriptors.every((item, index) => item.signature === descriptors[index % length].signature);
      if (matches) return { length, repetitions: descriptors.length / length, pattern: descriptors.slice(0, length) };
    }
    return { length: descriptors.length, repetitions: 1, pattern: descriptors.slice() };
  }

  function overviewBlockLabel(descriptor, onlyPattern) {
    const item = object(descriptor);
    const attention = item.mixer_kind === "linear_attention" ? "Linear Attention" : item.mixer_kind === "full_attention" ? "Full Attention" : "Decoder";
    if (item.ffn_kind === "moe") return `${attention} MoE Block`;
    if (onlyPattern && item.mixer_kind === "full_attention") return "Decoder Block";
    return `${attention} Block`;
  }

  function attentionLoweringProjection(operatorValue, contextValue = {}) {
    const operator = object(operatorValue);
    const parameters = object(operator.parameters);
    const context = object(contextValue);
    const attentionHeads = Math.max(1, Math.trunc(finite(parameters.attention_heads, context.attention_heads || 1)));
    const kvHeads = Math.max(1, Math.min(attentionHeads, Math.trunc(finite(parameters.kv_heads, context.kv_heads || attentionHeads))));
    const hiddenSize = Math.max(1, Math.trunc(finite(context.hidden_size)));
    const headDim = Math.max(1, Math.trunc(finite(parameters.attention_head_dim, context.attention_head_dim || Math.floor(hiddenSize / attentionHeads) || 1)));
    const attentionMode = kvHeads === attentionHeads ? "MHA" : kvHeads === 1 ? "MQA" : "GQA";
    const sourceOperatorId = String(operator.operator_id || "attention");
    const nodes = [
      ["q-projection", "Q Projection", ["B", "T", attentionHeads, headDim]],
      ["k-projection", "K Projection", ["B", "T", kvHeads, headDim]],
      ["v-projection", "V Projection", ["B", "T", kvHeads, headDim]],
      ["q-heads", `Q Heads ×${attentionHeads}`, ["B", attentionHeads, "T", headDim]],
      ["k-heads", `K Heads ×${kvHeads}`, ["B", kvHeads, "T", headDim]],
      ["v-heads", `V Heads ×${kvHeads}`, ["B", kvHeads, "T", headDim]],
      ["qk-scores", "QKᵀ Scores", ["B", attentionHeads, "T", "T"]],
      ["softmax", "Softmax", ["B", attentionHeads, "T", "T"]],
      ["pv", "P × V", ["B", "T", attentionHeads, headDim]],
      ["output", "Output Projection", ["B", "T", hiddenSize || "H"]],
    ].map(([suffix, label, shape]) => ({
      derived_id: `${sourceOperatorId}:derived:${suffix}`,
      label,
      shape,
      authoritative: false,
    }));
    const edge = (source, target, label) => ({
      source_id: `${sourceOperatorId}:derived:${source}`,
      target_id: `${sourceOperatorId}:derived:${target}`,
      label,
      authoritative: false,
    });
    return {
      kind: "attention_lowering_projection",
      source_operator_id: sourceOperatorId,
      authoritative: false,
      derived_only: true,
      disclaimer: "派生视图 / 不新增执行 IR",
      attention_mode: attentionMode,
      attention_heads: attentionHeads,
      kv_heads: kvHeads,
      head_dim: headDim,
      nodes,
      edges: [
        edge("q-projection", "q-heads", "split Q heads"),
        edge("k-projection", "k-heads", "split K heads"),
        edge("v-projection", "v-heads", "split V heads"),
        edge("q-heads", "qk-scores", `Q · ${attentionHeads} heads`),
        edge("k-heads", "qk-scores", `K · ${kvHeads} KV heads`),
        edge("qk-scores", "softmax", "scale + mask"),
        edge("softmax", "pv", "attention probabilities"),
        edge("v-heads", "pv", `V · ${kvHeads} KV heads`),
        edge("pv", "output", "concat heads"),
      ],
    };
  }

  function authoritativeOperatorProjection(graphValue, operatorIdsValue, optionsValue = {}) {
    const graph = normalizeModelGraph(graphValue);
    const ids = new Set(array(operatorIdsValue).map(String));
    const options = object(optionsValue);
    const operators = graph.operators
      .filter((operator) => ids.has(operator.operator_id))
      .sort((left, right) => left.sequence_index - right.sequence_index || left.operator_id.localeCompare(right.operator_id))
      .map((operator) => ({
        operator_id: operator.operator_id,
        op_kind: operator.op_kind,
        sequence_index: operator.sequence_index,
        ports: clone(array(operator.ports)),
        parameters: clone(object(operator.parameters)),
        attributes: clone(object(operator.attributes)),
        authoritative: true,
      }));
    const operatorById = new Map(graph.operators.map((operator) => [operator.operator_id, operator]));
    const edges = modelGraphEdges(graph)
      .filter((edge) => ids.has(edge.source_operator_id) && ids.has(edge.target_operator_id))
      .map((edge) => ({ ...clone(edge), authoritative: true }));
    const boundaryPorts = operators.flatMap((operator) => operator.ports.filter((port) => {
      if (!["input", "output"].includes(port.direction)) return false;
      const tensor = graph.tensors.find((item) => item.tensor_id === port.tensor_id);
      return port.direction === "input"
        ? !ids.has(String(tensor?.producer_operator_id || ""))
        : !array(tensor?.consumer_operator_ids).length || array(tensor?.consumer_operator_ids).some((consumerId) => !ids.has(String(consumerId)));
    }).map((port) => ({ ...clone(port), operator_id: operator.operator_id })));
    const attentionOperators = operators.filter((operator) => ["attention", "self_attention", "full_attention"].includes(operator.op_kind));
    const moeOperatorIds = operators.filter((operator) => ["moe_router", "moe_experts", "experts", "shared_expert", "moe_combine"].includes(operator.op_kind)).map((operator) => operator.operator_id);
    const residualEdges = edges.filter((edge) => operatorById.get(edge.target_operator_id)?.op_kind === "residual_add");
    return {
      authoritative: true,
      operators,
      edges,
      boundary_ports: boundaryPorts,
      attention_operator_ids: attentionOperators.map((operator) => operator.operator_id),
      attention_projections: Object.fromEntries(attentionOperators
        .filter((operator) => array(options.expanded_components).includes(operator.operator_id))
        .map((operator) => [operator.operator_id, attentionLoweringProjection(operator, options)])),
      moe_operator_ids: moeOperatorIds,
      moe_expanded: !moeOperatorIds.length || array(options.expanded_components).includes(`${String(options.group_id || "")}.moe`),
      residual_edges: residualEdges,
    };
  }

  function overviewResidualRoute(widthValue, heightValue, sideValue = "left") {
    const width = Math.max(96, finite(widthValue, 320));
    const height = Math.max(72, finite(heightValue, 180));
    const side = sideValue === "right" ? "right" : "left";
    const nodeX = side === "left" ? 34 : width - 34;
    const railX = side === "left" ? 12 : width - 12;
    const direction = side === "left" ? -1 : 1;
    const radius = 8;
    const top = 24;
    const bottom = height - 24;
    return `M ${nodeX} ${top} H ${railX - direction * radius} Q ${railX} ${top} ${railX} ${top + radius} V ${bottom - radius} Q ${railX} ${bottom} ${railX - direction * radius} ${bottom} H ${nodeX}`;
  }

  function overviewBoundaryPortCounts(nodeValue) {
    const ports = array(object(nodeValue).boundary_ports);
    const inputs = ports.filter((port) => object(port).direction === "input").length;
    const outputs = ports.filter((port) => object(port).direction === "output").length;
    return { inputs, outputs, rows: Math.max(inputs, outputs) };
  }

  function overviewNodeSize(nodeValue, scaleValue = 1) {
    const node = object(nodeValue);
    const scale = Math.max(0.1, finite(scaleValue, 1));
    const boundaryRows = overviewBoundaryPortCounts(node).rows;
    // Each visible boundary-port row must contribute to the measured box.  A
    // previous UI-local formula capped this allowance at three rows, allowing
    // later ports to overflow without invalidating the overview layout.
    const portAllowance = boundaryRows ? 24 + Math.max(0, boundaryRows - 1) * 16 : 0;
    // The overview exposes only each semantic component's name. Compact port
    // dots still need distinct rows, so only their count can grow the box.
    const width = node.kind === "decoder_stack" ? 216 : node.kind === "mtp_proposer" ? 204 : 176;
    return { width: Math.round(width * scale), height: Math.round((48 + portAllowance) * scale) };
  }

  function overviewNodeSizes(projectionValue, scaleValue = 1) {
    return Object.fromEntries(array(object(projectionValue).nodes).map((node) => [
      String(object(node).display_id || ""),
      overviewNodeSize(node, scaleValue),
    ]).filter(([displayId]) => displayId));
  }

  function overviewLayoutKey(projectionValue, scaleValue = 1) {
    const projection = object(projectionValue);
    const scale = Math.round(Math.max(0.1, finite(scaleValue, 1)) * 1e6) / 1e6;
    return stableStringify({
      graph_id: String(projection.graph_id || ""),
      scale,
      nodes: array(projection.nodes).map((nodeValue) => {
        const node = object(nodeValue);
        const boundaryPorts = array(node.boundary_ports).map((portValue) => {
          const port = object(portValue);
          return `${String(port.direction || "")}:${String(port.operator_id || "")}.${String(port.port_id || "")}`;
        }).sort();
        return {
          display_id: String(node.display_id || ""),
          kind: String(node.kind || ""),
          op_kind: String(node.op_kind || ""),
          label: String(node.label || ""),
          instance_count: Math.max(1, Math.trunc(finite(node.instance_count, 1))),
          summary: String(node.summary || ""),
          pattern_length: array(node.pattern).length,
          expanded: node.expanded === true,
          expanded_components: array(node.pattern).filter((item) => object(item).expanded).map((item) => ({
            group_id: String(object(item).representative_group_id || ""),
            moe_expanded: object(object(item).inline_graph).moe_expanded === true,
            attention: Object.keys(object(object(item).inline_graph).attention_projections).sort(),
          })),
          boundary_ports: boundaryPorts,
          size: overviewNodeSize(node, scale),
        };
      }),
    });
  }

  function overviewResponsiveLayout(projectionValue, canvasWidthValue, scaleValue = 1, optionsValue = {}) {
    const projection = object(projectionValue);
    const nodes = array(projection.nodes);
    const options = object(optionsValue);
    const scale = Math.max(0.1, finite(scaleValue, 1));
    const canvasWidth = Math.max(1, finite(canvasWidthValue, Math.round(900 * scale)));
    const viewportPadding = Math.max(0, finite(options.viewportPadding, 16));
    const availableWidth = Math.max(1, canvasWidth - viewportPadding * 2);
    const suppliedSizes = object(options.nodeSizes);
    const nodeSizes = Object.fromEntries(nodes.map((nodeValue) => {
      const node = object(nodeValue);
      const supplied = object(suppliedSizes[node.display_id]);
      const measured = overviewNodeSize(node, scale);
      return [String(node.display_id || ""), {
        width: Math.max(1, finite(supplied.width, measured.width)),
        height: Math.max(1, finite(supplied.height, measured.height)),
      }];
    }).filter(([displayId]) => displayId));
    const ids = nodes.map((node) => String(object(node).display_id || "")).filter(Boolean);
    const nodeFor = new Map(nodes.map((node, index) => [String(object(node).display_id || ""), { node: object(node), index }]));
    const compare = (left, right) => {
      const a = nodeFor.get(left); const b = nodeFor.get(right);
      return finite(a?.node.sequence_index) - finite(b?.node.sequence_index)
        || finite(a?.index) - finite(b?.index)
        || left.localeCompare(right);
    };
    const adjacency = new Map(ids.map((id) => [id, new Set()]));
    const predecessors = new Map(ids.map((id) => [id, new Set()]));
    array(projection.edges).forEach((edgeValue) => {
      const edge = object(edgeValue);
      const sourceId = String(edge.source_id || "");
      const targetId = String(edge.target_id || "");
      if (!adjacency.has(sourceId) || !adjacency.has(targetId) || sourceId === targetId) return;
      adjacency.get(sourceId).add(targetId);
      predecessors.get(targetId).add(sourceId);
    });
    const indegree = new Map(ids.map((id) => [id, predecessors.get(id).size]));
    const rank = new Map(ids.map((id) => [id, 0]));
    const queue = ids.filter((id) => indegree.get(id) === 0).sort(compare);
    const visited = [];
    while (queue.length) {
      const current = queue.shift();
      visited.push(current);
      Array.from(adjacency.get(current)).sort(compare).forEach((target) => {
        rank.set(target, Math.max(rank.get(target), rank.get(current) + 1));
        indegree.set(target, indegree.get(target) - 1);
        if (indegree.get(target) === 0) { queue.push(target); queue.sort(compare); }
      });
    }
    // Deduplicating a repeated A/B decoder pattern can intentionally turn the
    // derived view into A -> B -> A. Keep that direction in the edges, while
    // assigning its remaining representatives deterministic sequence ranks.
    const cycleNodeIds = ids.filter((id) => !visited.includes(id)).sort(compare);
    let nextRank = visited.length ? Math.max(...visited.map((id) => rank.get(id))) + 1 : 0;
    cycleNodeIds.forEach((id) => { rank.set(id, nextRank); nextRank += 1; });
    const layersMap = new Map();
    ids.forEach((id) => {
      const value = rank.get(id);
      if (!layersMap.has(value)) layersMap.set(value, []);
      layersMap.get(value).push(id);
    });
    const ranks = Array.from(layersMap.keys()).sort((left, right) => left - right);
    ranks.forEach((value) => layersMap.get(value).sort(compare));

    const horizontalGap = Math.max(8, finite(options.horizontalGap, 18) * scale);
    const verticalGap = Math.max(8, finite(options.verticalGap, 16) * scale);
    const layerGap = Math.max(16, finite(options.layerGap, 28) * scale);
    const worldMargin = Math.max(0, finite(options.worldMargin, 16) * scale);
    const widestNode = Math.max(0, ...ids.map((id) => nodeSizes[id]?.width || 0));
    const contentWidth = Math.max(widestNode, availableWidth - worldMargin * 2);
    const positions = {};
    let cursorY = worldMargin;
    let usedWidth = 0;
    ranks.forEach((value) => {
      const rows = [];
      let row = [];
      let rowWidth = 0;
      layersMap.get(value).forEach((id) => {
        const width = nodeSizes[id].width;
        const proposed = row.length ? rowWidth + horizontalGap + width : width;
        if (row.length && proposed > contentWidth) {
          rows.push(row);
          row = [];
          rowWidth = 0;
        }
        row.push(id);
        rowWidth += (row.length > 1 ? horizontalGap : 0) + width;
      });
      if (row.length) rows.push(row);
      rows.forEach((rowIds, rowIndex) => {
        const width = rowIds.reduce((total, id) => total + nodeSizes[id].width, 0) + horizontalGap * Math.max(0, rowIds.length - 1);
        const height = Math.max(0, ...rowIds.map((id) => nodeSizes[id].height));
        let cursorX = worldMargin + Math.max(0, (contentWidth - width) / 2);
        rowIds.forEach((id) => {
          positions[id] = { x: Math.round(cursorX), y: Math.round(cursorY) };
          cursorX += nodeSizes[id].width + horizontalGap;
        });
        usedWidth = Math.max(usedWidth, width);
        cursorY += height + (rowIndex < rows.length - 1 ? verticalGap : 0);
      });
      cursorY += layerGap;
    });
    const bounds = {
      width: Math.ceil(Math.max(widestNode, usedWidth, contentWidth) + worldMargin * 2),
      height: Math.ceil(Math.max(0, cursorY - layerGap) + worldMargin),
    };
    const scrollRequired = bounds.width > canvasWidth;
    return {
      positions,
      nodeSizes,
      bounds,
      layers: ranks.map((value) => ({ rank: value, node_ids: clone(layersMap.get(value)) })),
      hasCycle: cycleNodeIds.length > 0,
      cycleNodeIds,
      scroll_required: scrollRequired,
      viewport: {
        x: scrollRequired ? viewportPadding : Math.max(viewportPadding, (canvasWidth - bounds.width) / 2),
        y: viewportPadding,
        scale: 1,
      },
    };
  }

  function svgCoordinate(value) {
    const rounded = Math.round(finite(value) * 1000) / 1000;
    return Object.is(rounded, -0) ? 0 : rounded;
  }

  function overviewPortBezierPath(sourceValue, targetValue, optionsValue = {}) {
    const source = object(sourceValue);
    const target = object(targetValue);
    const options = object(optionsValue);
    const x1 = svgCoordinate(source.x);
    const y1 = svgCoordinate(source.y);
    const x2 = svgCoordinate(target.x);
    const y2 = svgCoordinate(target.y);
    const curvature = Math.max(0, finite(options.curvature, 0.5));
    const minimumHandle = Math.max(0, finite(options.minHandle, 36));
    const maximumHandle = Math.max(minimumHandle, finite(options.maxHandle, 240));
    const handle = Math.min(maximumHandle, Math.max(minimumHandle, Math.abs(x2 - x1) * curvature));
    const sourceSign = options.sourceSide === "left" ? -1 : 1;
    const targetSign = options.targetSide === "right" ? 1 : -1;
    return `M ${x1} ${y1} C ${svgCoordinate(x1 + sourceSign * handle)} ${y1}, ${svgCoordinate(x2 + targetSign * handle)} ${y2}, ${x2} ${y2}`;
  }

  function modelGraphRect(value) {
    const rect = object(value);
    return {
      id: String(rect.id || ""),
      x: finite(rect.x),
      y: finite(rect.y),
      width: Math.max(0, finite(rect.width)),
      height: Math.max(0, finite(rect.height)),
    };
  }

  function modelGraphRectCenter(value) {
    const rect = modelGraphRect(value);
    return { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 };
  }

  function modelGraphAnchor(value, sideValue = "right", fractionValue = 0.5) {
    const rect = modelGraphRect(value);
    const side = ["top", "right", "bottom", "left"].includes(String(sideValue)) ? String(sideValue) : "right";
    const fraction = Math.max(0.08, Math.min(0.92, finite(fractionValue, 0.5)));
    if (side === "top") return { x: rect.x + rect.width * fraction, y: rect.y, side, fraction };
    if (side === "bottom") return { x: rect.x + rect.width * fraction, y: rect.y + rect.height, side, fraction };
    if (side === "left") return { x: rect.x, y: rect.y + rect.height * fraction, side, fraction };
    return { x: rect.x + rect.width, y: rect.y + rect.height * fraction, side, fraction };
  }

  function modelGraphPreferredSide(fromValue, toValue) {
    const from = modelGraphRectCenter(fromValue);
    const to = modelGraphRectCenter(toValue);
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    if (Math.abs(dx) > Math.abs(dy)) return dx >= 0 ? "right" : "left";
    return dy >= 0 ? "bottom" : "top";
  }

  function modelGraphPointInsideRect(pointValue, rectValue) {
    const point = object(pointValue);
    const rect = modelGraphRect(rectValue);
    return finite(point.x) > rect.x && finite(point.x) < rect.x + rect.width
      && finite(point.y) > rect.y && finite(point.y) < rect.y + rect.height;
  }

  // Liang-Barsky clipping keeps the short-curve decision honest as well as
  // the axis-aligned router. Touching an obstacle boundary is allowed; its
  // routing clearance has already been folded into the rectangle.
  function modelGraphSegmentHitsRect(startValue, endValue, rectValue) {
    const start = { x: finite(object(startValue).x), y: finite(object(startValue).y) };
    const end = { x: finite(object(endValue).x), y: finite(object(endValue).y) };
    const rect = modelGraphRect(rectValue);
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const p = [-dx, dx, -dy, dy];
    const q = [start.x - rect.x, rect.x + rect.width - start.x, start.y - rect.y, rect.y + rect.height - start.y];
    let low = 0;
    let high = 1;
    for (let index = 0; index < 4; index += 1) {
      if (Math.abs(p[index]) < 1e-9) {
        if (q[index] < 0) return false;
        continue;
      }
      const ratio = q[index] / p[index];
      if (p[index] < 0) low = Math.max(low, ratio);
      else high = Math.min(high, ratio);
      if (low > high) return false;
    }
    return high > 1e-6 && low < 1 - 1e-6 && high - low > 1e-6;
  }

  function modelGraphSimplifyPoints(pointsValue) {
    const points = array(pointsValue).map((point) => ({ x: svgCoordinate(object(point).x), y: svgCoordinate(object(point).y) }));
    const unique = points.filter((point, index) => !index || point.x !== points[index - 1].x || point.y !== points[index - 1].y);
    return unique.filter((point, index) => {
      if (!index || index === unique.length - 1) return true;
      const before = unique[index - 1];
      const after = unique[index + 1];
      return !((before.x === point.x && point.x === after.x) || (before.y === point.y && point.y === after.y));
    });
  }

  function modelGraphPathClear(pointsValue, obstaclesValue) {
    const points = array(pointsValue);
    const obstacles = array(obstaclesValue);
    for (let index = 1; index < points.length; index += 1) {
      if (obstacles.some((rect) => modelGraphSegmentHitsRect(points[index - 1], points[index], rect))) return false;
    }
    return true;
  }

  function modelGraphPathCost(pointsValue) {
    const points = array(pointsValue);
    let length = 0;
    for (let index = 1; index < points.length; index += 1) {
      length += Math.abs(finite(points[index].x) - finite(points[index - 1].x))
        + Math.abs(finite(points[index].y) - finite(points[index - 1].y));
    }
    return length + Math.max(0, points.length - 2) * 10;
  }

  function modelGraphOccupiedSegments(value) {
    const result = [];
    const append = (startValue, endValue, metadataValue = {}) => {
      const start = { x: svgCoordinate(object(startValue).x), y: svgCoordinate(object(startValue).y) };
      const end = { x: svgCoordinate(object(endValue).x), y: svgCoordinate(object(endValue).y) };
      if (start.x === end.x && start.y === end.y) return;
      const metadata = object(metadataValue);
      result.push({
        start,
        end,
        padding: Math.max(0, finite(metadata.padding ?? metadata.channel_padding)),
        weight: Math.max(0, finite(metadata.weight, 1)),
      });
    };
    const appendEntry = (rawValue) => {
      if (Array.isArray(rawValue)) {
        for (let index = 1; index < rawValue.length; index += 1) append(rawValue[index - 1], rawValue[index]);
        return;
      }
      const raw = object(rawValue);
      if (array(raw.points).length > 1) {
        for (let index = 1; index < raw.points.length; index += 1) append(raw.points[index - 1], raw.points[index], raw);
        return;
      }
      const start = raw.start ?? raw.from;
      const end = raw.end ?? raw.to;
      if (start && end) {
        append(start, end, raw);
        return;
      }
      if ([raw.x1, raw.y1, raw.x2, raw.y2].every((item) => item != null && item !== "" && Number.isFinite(Number(item)))) {
        append({ x: raw.x1, y: raw.y1 }, { x: raw.x2, y: raw.y2 }, raw);
      }
    };
    const values = array(value);
    const isPoint = (item) => Number.isFinite(Number(object(item).x)) && Number.isFinite(Number(object(item).y));
    if (values.length > 1 && values.every(isPoint)) appendEntry(values);
    else values.forEach(appendEntry);
    return result;
  }

  function modelGraphRoutingContext(optionsValue = {}) {
    const options = object(optionsValue);
    const occupiedSegments = modelGraphOccupiedSegments(options.occupiedSegments);
    return {
      occupiedSegments,
      overlapPenalty: Math.max(0, finite(options.overlapPenalty, occupiedSegments.length ? 180 : 0)),
      crossingPenalty: Math.max(0, finite(options.crossingPenalty, occupiedSegments.length ? 72 : 0)),
      channelPadding: Math.max(0, finite(options.channelPadding, occupiedSegments.length ? 4 : 0)),
    };
  }

  function modelGraphSamePoint(leftValue, rightValue) {
    const left = object(leftValue);
    const right = object(rightValue);
    return Math.abs(finite(left.x) - finite(right.x)) <= 1e-6
      && Math.abs(finite(left.y) - finite(right.y)) <= 1e-6;
  }

  function modelGraphSegmentRelation(startValue, endValue, occupiedValue, paddingValue = 0) {
    const start = { x: finite(object(startValue).x), y: finite(object(startValue).y) };
    const end = { x: finite(object(endValue).x), y: finite(object(endValue).y) };
    const occupied = object(occupiedValue);
    const before = { x: finite(object(occupied.start).x), y: finite(object(occupied.start).y) };
    const after = { x: finite(object(occupied.end).x), y: finite(object(occupied.end).y) };
    const padding = Math.max(0, finite(paddingValue));
    const epsilon = 1e-6;
    const candidateHorizontal = Math.abs(end.y - start.y) <= epsilon;
    const candidateVertical = Math.abs(end.x - start.x) <= epsilon;
    const occupiedHorizontal = Math.abs(after.y - before.y) <= epsilon;
    const occupiedVertical = Math.abs(after.x - before.x) <= epsilon;
    const intervalOverlap = (leftA, rightA, leftB, rightB) => (
      Math.min(Math.max(leftA, rightA), Math.max(leftB, rightB))
      - Math.max(Math.min(leftA, rightA), Math.min(leftB, rightB))
    );
    if (candidateHorizontal && occupiedHorizontal && Math.abs(start.y - before.y) <= padding + epsilon) {
      const overlap = intervalOverlap(start.x, end.x, before.x, after.x);
      if (overlap > epsilon) return { kind: "overlap", length: overlap };
      return null;
    }
    if (candidateVertical && occupiedVertical && Math.abs(start.x - before.x) <= padding + epsilon) {
      const overlap = intervalOverlap(start.y, end.y, before.y, after.y);
      if (overlap > epsilon) return { kind: "overlap", length: overlap };
      return null;
    }
    const orientation = (a, b, c) => (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    const first = orientation(start, end, before);
    const second = orientation(start, end, after);
    const third = orientation(before, after, start);
    const fourth = orientation(before, after, end);
    const collinear = Math.abs(first) <= epsilon && Math.abs(second) <= epsilon
      && Math.abs(third) <= epsilon && Math.abs(fourth) <= epsilon;
    if (collinear) {
      const useX = Math.abs(end.x - start.x) >= Math.abs(end.y - start.y);
      const overlap = useX
        ? intervalOverlap(start.x, end.x, before.x, after.x)
        : intervalOverlap(start.y, end.y, before.y, after.y);
      if (overlap > epsilon) return { kind: "overlap", length: overlap };
      return null;
    }
    const intersects = ((first <= epsilon && second >= -epsilon) || (first >= -epsilon && second <= epsilon))
      && ((third <= epsilon && fourth >= -epsilon) || (third >= -epsilon && fourth <= epsilon));
    if (!intersects) return null;
    const sharedEndpoint = [start, end].some((candidate) => [before, after].some((point) => modelGraphSamePoint(candidate, point)));
    return sharedEndpoint ? null : { kind: "crossing", length: 0 };
  }

  function modelGraphSegmentRoutingPenalty(startValue, endValue, contextValue = {}) {
    const context = object(contextValue);
    let cost = 0;
    let overlaps = 0;
    let crossings = 0;
    array(context.occupiedSegments).forEach((occupied) => {
      const relation = modelGraphSegmentRelation(
        startValue,
        endValue,
        occupied,
        Math.max(finite(context.channelPadding), finite(object(occupied).padding)),
      );
      if (!relation) return;
      const weight = Math.max(0, finite(object(occupied).weight, 1));
      if (relation.kind === "overlap") {
        overlaps += 1;
        cost += finite(context.overlapPenalty) * weight * (1 + Math.min(4, relation.length / 80));
      } else {
        crossings += 1;
        cost += finite(context.crossingPenalty) * weight;
      }
    });
    return { cost, overlaps, crossings };
  }

  function modelGraphPathRoutingPenalty(pointsValue, contextValue = {}) {
    const points = array(pointsValue);
    const result = { cost: 0, overlaps: 0, crossings: 0 };
    for (let index = 1; index < points.length; index += 1) {
      const penalty = modelGraphSegmentRoutingPenalty(points[index - 1], points[index], contextValue);
      result.cost += penalty.cost;
      result.overlaps += penalty.overlaps;
      result.crossings += penalty.crossings;
    }
    return result;
  }

  function modelGraphVisibilityRoute(startValue, endValue, obstaclesValue, marginValue = 18, routingContextValue = {}) {
    const start = { x: svgCoordinate(object(startValue).x), y: svgCoordinate(object(startValue).y) };
    const end = { x: svgCoordinate(object(endValue).x), y: svgCoordinate(object(endValue).y) };
    const obstacles = array(obstaclesValue).map(modelGraphRect);
    const routingContext = object(routingContextValue);
    const occupiedSegments = array(routingContext.occupiedSegments);
    const margin = Math.max(4, finite(marginValue, 18));
    const xs = new Set([start.x, end.x]);
    const ys = new Set([start.y, end.y]);
    obstacles.forEach((rect) => {
      xs.add(rect.x); xs.add(rect.x + rect.width);
      ys.add(rect.y); ys.add(rect.y + rect.height);
    });
    const channelOffset = Math.max(2, finite(routingContext.channelPadding) + 2);
    const horizontalFlow = Math.abs(end.x - start.x) >= Math.abs(end.y - start.y);
    occupiedSegments.forEach((segment) => {
      const before = object(segment.start);
      const after = object(segment.end);
      const segmentOffset = Math.max(channelOffset, finite(object(segment).padding) + 2);
      if (horizontalFlow) {
        ys.add(finite(before.y) - segmentOffset); ys.add(finite(before.y) + segmentOffset);
        ys.add(finite(after.y) - segmentOffset); ys.add(finite(after.y) + segmentOffset);
      } else {
        xs.add(finite(before.x) - segmentOffset); xs.add(finite(before.x) + segmentOffset);
        xs.add(finite(after.x) - segmentOffset); xs.add(finite(after.x) + segmentOffset);
      }
    });
    const allRects = obstacles.length ? obstacles : [
      { x: Math.min(start.x, end.x), y: Math.min(start.y, end.y), width: Math.abs(end.x - start.x), height: Math.abs(end.y - start.y) },
    ];
    xs.add(Math.min(start.x, end.x, ...allRects.map((rect) => rect.x)) - margin);
    xs.add(Math.max(start.x, end.x, ...allRects.map((rect) => rect.x + rect.width)) + margin);
    ys.add(Math.min(start.y, end.y, ...allRects.map((rect) => rect.y)) - margin);
    ys.add(Math.max(start.y, end.y, ...allRects.map((rect) => rect.y + rect.height)) + margin);
    const xValues = Array.from(xs).sort((left, right) => left - right);
    const yValues = Array.from(ys).sort((left, right) => left - right);
    const points = new Map();
    xValues.forEach((x) => yValues.forEach((y) => {
      const point = { x, y };
      if (!obstacles.some((rect) => modelGraphPointInsideRect(point, rect))) points.set(`${x},${y}`, point);
    }));
    points.set(`${start.x},${start.y}`, start);
    points.set(`${end.x},${end.y}`, end);
    const nodes = Array.from(points.values());
    const neighbors = new Map(nodes.map((point) => [point, []]));
    const connectLine = (line) => {
      for (let index = 1; index < line.length; index += 1) {
        const before = line[index - 1];
        const after = line[index];
        if (obstacles.some((rect) => modelGraphSegmentHitsRect(before, after, rect))) continue;
        neighbors.get(before)?.push(after);
        neighbors.get(after)?.push(before);
      }
    };
    xValues.forEach((x) => connectLine(nodes.filter((point) => point.x === x).sort((left, right) => left.y - right.y)));
    yValues.forEach((y) => connectLine(nodes.filter((point) => point.y === y).sort((left, right) => left.x - right.x)));
    const distance = new Map([[start, 0]]);
    const bends = new Map([[start, 0]]);
    const previous = new Map();
    const open = new Set([start]);
    while (open.size) {
      const current = Array.from(open).sort((left, right) => (distance.get(left) - distance.get(right)) || left.x - right.x || left.y - right.y)[0];
      open.delete(current);
      if (current === end) break;
      (neighbors.get(current) || []).forEach((next) => {
        const prior = previous.get(current);
        const turn = prior && (prior.x === current.x) !== (current.x === next.x) ? 9 : 0;
        const occupiedCost = modelGraphSegmentRoutingPenalty(current, next, routingContext).cost;
        const nextDistance = distance.get(current) + Math.abs(next.x - current.x) + Math.abs(next.y - current.y) + turn + occupiedCost;
        const nextBends = (bends.get(current) || 0) + (turn ? 1 : 0);
        if (nextDistance > (distance.get(next) ?? Infinity)) return;
        if (nextDistance === distance.get(next) && nextBends >= (bends.get(next) ?? Infinity)) return;
        distance.set(next, nextDistance);
        bends.set(next, nextBends);
        previous.set(next, current);
        open.add(next);
      });
    }
    if (start !== end && !previous.has(end)) return null;
    const path = [end];
    while (path[0] !== start) path.unshift(previous.get(path[0]));
    return modelGraphSimplifyPoints(path);
  }

  function modelGraphRoundedPath(pointsValue, radiusValue = 9) {
    const points = modelGraphSimplifyPoints(pointsValue);
    if (!points.length) return "";
    if (points.length === 1) return `M ${points[0].x} ${points[0].y}`;
    const radius = Math.max(0, finite(radiusValue, 9));
    const parts = [`M ${points[0].x} ${points[0].y}`];
    for (let index = 1; index < points.length - 1; index += 1) {
      const before = points[index - 1];
      const corner = points[index];
      const after = points[index + 1];
      const incoming = Math.abs(corner.x - before.x) + Math.abs(corner.y - before.y);
      const outgoing = Math.abs(after.x - corner.x) + Math.abs(after.y - corner.y);
      const amount = Math.min(radius, incoming / 2, outgoing / 2);
      const enter = { x: corner.x + Math.sign(before.x - corner.x) * amount, y: corner.y + Math.sign(before.y - corner.y) * amount };
      const leave = { x: corner.x + Math.sign(after.x - corner.x) * amount, y: corner.y + Math.sign(after.y - corner.y) * amount };
      parts.push(`L ${svgCoordinate(enter.x)} ${svgCoordinate(enter.y)}`);
      if (amount) parts.push(`Q ${corner.x} ${corner.y} ${svgCoordinate(leave.x)} ${svgCoordinate(leave.y)}`);
    }
    const last = points[points.length - 1];
    parts.push(`L ${last.x} ${last.y}`);
    return parts.join(" ");
  }

  function modelGraphCurveGeometry(sourceValue, targetValue, optionsValue = {}) {
    const source = object(sourceValue);
    const target = object(targetValue);
    const options = object(optionsValue);
    const distance = Math.hypot(finite(target.x) - finite(source.x), finite(target.y) - finite(source.y));
    const handle = Math.min(Math.max(28, distance * 0.38), Math.max(28, finite(options.maxHandle, 96)));
    const vector = (side) => side === "left" ? [-1, 0] : side === "right" ? [1, 0] : side === "top" ? [0, -1] : [0, 1];
    const sourceVector = vector(source.side || "right");
    const targetVector = vector(target.side || "left");
    return {
      source: { x: svgCoordinate(source.x), y: svgCoordinate(source.y) },
      first: {
        x: svgCoordinate(finite(source.x) + sourceVector[0] * handle),
        y: svgCoordinate(finite(source.y) + sourceVector[1] * handle),
      },
      second: {
        x: svgCoordinate(finite(target.x) + targetVector[0] * handle),
        y: svgCoordinate(finite(target.y) + targetVector[1] * handle),
      },
      target: { x: svgCoordinate(target.x), y: svgCoordinate(target.y) },
    };
  }

  function modelGraphCurvePath(sourceValue, targetValue, optionsValue = {}) {
    const curve = modelGraphCurveGeometry(sourceValue, targetValue, optionsValue);
    return `M ${curve.source.x} ${curve.source.y} C ${curve.first.x} ${curve.first.y}, ${curve.second.x} ${curve.second.y}, ${curve.target.x} ${curve.target.y}`;
  }

  function modelGraphCurvePoints(sourceValue, targetValue, optionsValue = {}, stepsValue = 16) {
    const curve = modelGraphCurveGeometry(sourceValue, targetValue, optionsValue);
    const steps = Math.max(4, Math.trunc(finite(stepsValue, 16)));
    const points = [];
    for (let index = 0; index <= steps; index += 1) {
      const t = index / steps;
      const inverse = 1 - t;
      points.push({
        x: svgCoordinate(
          inverse ** 3 * curve.source.x
          + 3 * inverse ** 2 * t * curve.first.x
          + 3 * inverse * t ** 2 * curve.second.x
          + t ** 3 * curve.target.x
        ),
        y: svgCoordinate(
          inverse ** 3 * curve.source.y
          + 3 * inverse ** 2 * t * curve.first.y
          + 3 * inverse * t ** 2 * curve.second.y
          + t ** 3 * curve.target.y
        ),
      });
    }
    return points;
  }

  function modelGraphEscape(pointValue, distanceValue) {
    const point = object(pointValue);
    const distance = Math.max(2, finite(distanceValue, 14));
    if (point.side === "left") return { x: finite(point.x) - distance, y: finite(point.y) };
    if (point.side === "right") return { x: finite(point.x) + distance, y: finite(point.y) };
    if (point.side === "top") return { x: finite(point.x), y: finite(point.y) - distance };
    return { x: finite(point.x), y: finite(point.y) + distance };
  }

  function modelGraphRouteEdge(sourceRectValue, targetRectValue, obstacleRectsValue = [], optionsValue = {}) {
    const sourceRect = modelGraphRect(sourceRectValue);
    const targetRect = modelGraphRect(targetRectValue);
    const options = object(optionsValue);
    const clearance = Math.max(4, finite(options.clearance, 12));
    const cornerRadius = Math.max(0, finite(options.cornerRadius, 9));
    const routingContext = modelGraphRoutingContext(options);
    const sourceSides = options.sourceSide ? [String(options.sourceSide)] : [modelGraphPreferredSide(sourceRect, targetRect), "right", "bottom", "left", "top"];
    const targetSides = options.targetSide ? [String(options.targetSide)] : [modelGraphPreferredSide(targetRect, sourceRect), "left", "top", "right", "bottom"];
    const uniqueSides = (values) => Array.from(new Set(values.filter((side) => ["top", "right", "bottom", "left"].includes(side))));
    const obstacles = array(obstacleRectsValue)
      .map(modelGraphRect)
      .filter((rect) => !(rect.id && (rect.id === sourceRect.id || rect.id === targetRect.id)))
      .map((rect) => ({ ...rect, x: rect.x - clearance, y: rect.y - clearance, width: rect.width + clearance * 2, height: rect.height + clearance * 2 }));
    const exactSource = options.sourcePoint ? { ...object(options.sourcePoint), side: String(object(options.sourcePoint).side || options.sourceSide || "right") } : null;
    const exactTarget = options.targetPoint ? { ...object(options.targetPoint), side: String(object(options.targetPoint).side || options.targetSide || "left") } : null;
    const sourceSideChoices = exactSource ? [exactSource.side] : sourceSides;
    const targetSideChoices = exactTarget ? [exactTarget.side] : targetSides;
    const pairs = [];
    uniqueSides(sourceSideChoices).forEach((sourceSide) => uniqueSides(targetSideChoices).forEach((targetSide) => {
      const source = exactSource || modelGraphAnchor(sourceRect, sourceSide, options.sourceFraction);
      const target = exactTarget || modelGraphAnchor(targetRect, targetSide, options.targetFraction);
      const sourceEscape = modelGraphEscape(source, clearance);
      const targetEscape = modelGraphEscape(target, clearance);
      const inner = modelGraphVisibilityRoute(sourceEscape, targetEscape, obstacles, clearance * 2, routingContext);
      if (!inner) return;
      const points = modelGraphSimplifyPoints([source, sourceEscape, ...inner, targetEscape, target]);
      if (!modelGraphPathClear(points, obstacles)) return;
      const preferredPenalty = (sourceSide === sourceSides[0] ? 0 : 24) + (targetSide === targetSides[0] ? 0 : 24);
      const routingPenalty = modelGraphPathRoutingPenalty(points, routingContext);
      pairs.push({
        source: { ...source, side: sourceSide },
        target: { ...target, side: targetSide },
        points,
        routingPenalty,
        order: pairs.length,
        cost: modelGraphPathCost(points) + routingPenalty.cost + preferredPenalty,
      });
    }));
    let selected = pairs.sort((left, right) => left.cost - right.cost || left.order - right.order)[0] || null;
    if (options.preferOuter && pairs.length) {
      const allRects = [sourceRect, targetRect, ...obstacles];
      const rails = {
        left: Math.min(...allRects.map((rect) => rect.x)) - clearance * 2,
        right: Math.max(...allRects.map((rect) => rect.x + rect.width)) + clearance * 2,
        top: Math.min(...allRects.map((rect) => rect.y)) - clearance * 2,
        bottom: Math.max(...allRects.map((rect) => rect.y + rect.height)) + clearance * 2,
      };
      const outer = [];
      const sourceCenter = modelGraphRectCenter(sourceRect);
      const targetCenter = modelGraphRectCenter(targetRect);
      const verticalFlow = Math.abs(targetCenter.y - sourceCenter.y) >= Math.abs(targetCenter.x - sourceCenter.x);
      pairs.forEach((pair) => {
        const sourceEscape = modelGraphEscape(pair.source, clearance);
        const targetEscape = modelGraphEscape(pair.target, clearance);
        const railCandidates = [];
        const sourceAllowsLeft = ["left", "top", "bottom"].includes(pair.source.side);
        const targetAllowsLeft = ["left", "top", "bottom"].includes(pair.target.side);
        const sourceAllowsRight = ["right", "top", "bottom"].includes(pair.source.side);
        const targetAllowsRight = ["right", "top", "bottom"].includes(pair.target.side);
        const sourceAllowsTop = ["top", "left", "right"].includes(pair.source.side);
        const targetAllowsTop = ["top", "left", "right"].includes(pair.target.side);
        const sourceAllowsBottom = ["bottom", "left", "right"].includes(pair.source.side);
        const targetAllowsBottom = ["bottom", "left", "right"].includes(pair.target.side);
        if (verticalFlow && sourceAllowsLeft && targetAllowsLeft) railCandidates.push([{ x: rails.left, y: sourceEscape.y }, { x: rails.left, y: targetEscape.y }]);
        if (verticalFlow && sourceAllowsRight && targetAllowsRight) railCandidates.push([{ x: rails.right, y: sourceEscape.y }, { x: rails.right, y: targetEscape.y }]);
        if (!verticalFlow && sourceAllowsTop && targetAllowsTop) railCandidates.push([{ x: sourceEscape.x, y: rails.top }, { x: targetEscape.x, y: rails.top }]);
        if (!verticalFlow && sourceAllowsBottom && targetAllowsBottom) railCandidates.push([{ x: sourceEscape.x, y: rails.bottom }, { x: targetEscape.x, y: rails.bottom }]);
        railCandidates.forEach((rail) => {
          const points = modelGraphSimplifyPoints([pair.source, sourceEscape, ...rail, targetEscape, pair.target]);
          if (modelGraphPathClear(points, obstacles)) {
            const routingPenalty = modelGraphPathRoutingPenalty(points, routingContext);
            outer.push({ ...pair, points, routingPenalty, cost: modelGraphPathCost(points) + routingPenalty.cost });
          }
        });
      });
      if (outer.length) selected = outer.sort((left, right) => left.cost - right.cost || left.order - right.order)[0];
    }
    if (!selected) {
      const source = exactSource || modelGraphAnchor(sourceRect, modelGraphPreferredSide(sourceRect, targetRect), options.sourceFraction);
      const target = exactTarget || modelGraphAnchor(targetRect, modelGraphPreferredSide(targetRect, sourceRect), options.targetFraction);
      const points = [source, target];
      const routingPenalty = modelGraphPathRoutingPenalty(points, routingContext);
      return {
        kind: "fallback",
        source,
        target,
        points,
        path: modelGraphCurvePath(source, target),
        obstacle_free: false,
        routing_penalty: routingPenalty.cost,
        overlaps: routingPenalty.overlaps,
        crossings: routingPenalty.crossings,
        label: { x: (finite(source.x) + finite(target.x)) / 2, y: (finite(source.y) + finite(target.y)) / 2 },
      };
    }
    const curvePoints = modelGraphCurvePoints(selected.source, selected.target, options);
    const straightClear = !obstacles.some((rect) => modelGraphSegmentHitsRect(selected.source, selected.target, rect));
    const curveClear = modelGraphPathClear(curvePoints, obstacles);
    const directRoutingPenalty = modelGraphPathRoutingPenalty([selected.source, selected.target], routingContext);
    const distance = Math.hypot(selected.target.x - selected.source.x, selected.target.y - selected.source.y);
    const curveDistance = Object.hasOwn(options, "shortCurveDistance")
      ? Math.max(40, finite(options.shortCurveDistance, 190))
      : Infinity;
    const occupancyCanImprove = selected.routingPenalty.cost + 1e-6 < directRoutingPenalty.cost;
    const canCurve = !options.preferOuter && straightClear && curveClear && !occupancyCanImprove && distance <= curveDistance;
    const points = canCurve ? [selected.source, selected.target] : selected.points;
    const longest = points.slice(1).map((point, index) => ({
      start: points[index], end: point,
      length: Math.abs(point.x - points[index].x) + Math.abs(point.y - points[index].y),
    })).sort((left, right) => right.length - left.length)[0];
    return {
      kind: canCurve ? "smooth" : "orthogonal",
      source: selected.source,
      target: selected.target,
      points,
      path: canCurve ? modelGraphCurvePath(selected.source, selected.target, options) : modelGraphRoundedPath(points, cornerRadius),
      obstacle_free: true,
      routing_penalty: canCurve ? directRoutingPenalty.cost : selected.routingPenalty.cost,
      overlaps: canCurve ? directRoutingPenalty.overlaps : selected.routingPenalty.overlaps,
      crossings: canCurve ? directRoutingPenalty.crossings : selected.routingPenalty.crossings,
      label: longest ? { x: (longest.start.x + longest.end.x) / 2, y: (longest.start.y + longest.end.y) / 2 } : { x: selected.source.x, y: selected.source.y },
    };
  }

  function isMtpOperator(operatorValue) {
    const kind = String(object(operatorValue).op_kind || "").toLowerCase();
    return kind === "mtp_prediction_layer" || kind === "mtp_aux_head";
  }

  function mtpOverviewDescriptor(graphValue) {
    const graph = normalizeModelGraph(graphValue);
    const operators = graph.operators.filter(isMtpOperator)
      .sort((left, right) => left.sequence_index - right.sequence_index || left.operator_id.localeCompare(right.operator_id));
    if (!operators.length) return null;
    const predictionOperators = operators.filter((item) => item.op_kind === "mtp_prediction_layer");
    const auxiliaryOperators = operators.filter((item) => item.op_kind === "mtp_aux_head");
    const predictionLayers = predictionOperators.reduce(
      (total, item) => total + Math.max(1, Math.trunc(finite(object(item.parameters).repeat, 1))),
      0,
    );
    const operatorIds = new Set(operators.map((item) => item.operator_id));
    const proposalTensors = graph.tensors.filter((tensor) => (
      operatorIds.has(String(tensor.producer_operator_id || ""))
      && !tensor.consumer_operator_ids.some((consumerId) => operatorIds.has(String(consumerId)))
    ));
    return {
      display_id: "overview:mtp-proposer",
      kind: "mtp_proposer",
      op_kind: "mtp_proposer",
      label: "MTP Proposer",
      sequence_index: Math.min(...operators.map((item) => item.sequence_index)),
      operator_ids: operators.map((item) => item.operator_id),
      representative_operator_id: predictionOperators[0]?.operator_id || auxiliaryOperators[0]?.operator_id,
      prediction_operator_ids: predictionOperators.map((item) => item.operator_id),
      auxiliary_operator_ids: auxiliaryOperators.map((item) => item.operator_id),
      prediction_layers: predictionLayers,
      has_auxiliary_head: auxiliaryOperators.length > 0,
      proposal_tensor_ids: proposalTensors.map((item) => item.tensor_id),
      proposal_contracts: proposalTensors.map((item) => ({
        tensor_id: item.tensor_id, dtype: item.dtype, shape: clone(item.shape), layout: item.layout,
      })),
      summary: [
        predictionLayers ? `Prediction Layer ×${predictionLayers}` : "Decoder Hidden",
        auxiliaryOperators.length ? "Auxiliary Head / Proposal Logits" : "Proposal Output",
      ].join(" · "),
      detail: {
        operators: operators.map((item) => ({
          operator_id: item.operator_id, op_kind: item.op_kind,
          parameters: clone(object(item.parameters)), ports: clone(array(item.ports)),
        })),
      },
    };
  }

  function overviewProjectionAttributes(attributesValue) {
    const attributes = clone(object(attributesValue));
    // These fields identify an authoritative instance, not its executable
    // contract.  Keeping them in a signature would prevent structurally
    // identical templates from being represented by one overview node.
    delete attributes.parent_group_id;
    delete attributes.layer_ids;
    delete attributes.source_tensor_id;
    return attributes;
  }

  function overviewProjectionParameters(parametersValue) {
    const parameters = clone(object(parametersValue));
    // repeat is represented explicitly as instance_count on the derived node.
    // It must not make two otherwise identical templates look different.
    delete parameters.repeat;
    delete parameters.layer_ids;
    delete parameters.overrides;
    return parameters;
  }

  function overviewPortContractSignature(portsValue) {
    return array(portsValue).map((portValue) => {
      const port = object(portValue);
      return {
        port_id: String(port.port_id || ""),
        direction: String(port.direction || ""),
        dtype: canonicalDtype(port.dtype),
        shape: normalizeShape(port.shape),
        layout: canonicalLayout(port.layout),
        attributes: clone(object(port.attributes)),
      };
    });
  }

  function overviewOperatorSignature(operatorValue) {
    const operator = object(operatorValue);
    return stableStringify({
      op_kind: String(operator.op_kind || "operator"),
      parameters: overviewProjectionParameters(operator.parameters),
      attributes: overviewProjectionAttributes(operator.attributes),
      ports: overviewPortContractSignature(operator.ports),
    });
  }

  function overviewGroupProjectionInfo(graphValue, groupValue) {
    const graph = object(graphValue);
    const group = object(groupValue);
    const groupId = String(group.operator_id || "");
    const children = array(graph.operators)
      .filter((operator) => String(object(operator.attributes).parent_group_id || "") === groupId)
      .sort((left, right) => left.sequence_index - right.sequence_index || left.operator_id.localeCompare(right.operator_id));
    const kindCounts = new Map();
    const roles = children.map((operator) => {
      const kind = String(operator.op_kind || "operator");
      const occurrence = kindCounts.get(kind) || 0;
      kindCounts.set(kind, occurrence + 1);
      return { operator, role: `${kind}:${occurrence}` };
    });
    const roleFor = new Map(roles.map((item) => [item.operator.operator_id, item.role]));
    const childIds = new Set(roleFor.keys());
    const descriptor = overviewGroupDescriptor(graph, group) || {};
    const internalEdges = modelGraphEdges(graph)
      .filter((edge) => childIds.has(edge.source_operator_id) && childIds.has(edge.target_operator_id))
      .map((edge) => ({
        source_role: roleFor.get(edge.source_operator_id),
        target_role: roleFor.get(edge.target_operator_id),
        dtype: edge.dtype,
        shape: clone(edge.shape),
        layout: edge.layout,
      }))
      .sort((left, right) => stableStringify(left).localeCompare(stableStringify(right)));
    const signature = stableStringify({
      mixer_kind: descriptor.mixer_kind,
      ffn_kind: descriptor.ffn_kind,
      num_experts: descriptor.num_experts,
      experts_per_token: descriptor.experts_per_token,
      shared_expert: descriptor.shared_expert,
      parameters: overviewProjectionParameters(group.parameters),
      attributes: overviewProjectionAttributes(group.attributes),
      operators: roles.map((item) => ({ role: item.role, signature: overviewOperatorSignature(item.operator) })),
      internal_edges: internalEdges,
    });
    return {
      group_id: groupId,
      repeat: Math.max(1, Math.trunc(finite(object(group.parameters).repeat, 1))),
      descriptor,
      roles,
      signature,
    };
  }

  function overviewLeafLabel(operatorValue) {
    return String(object(operatorValue).op_kind || "operator").replaceAll("_", " ");
  }

  function buildLeafOverviewProjection(graphValue) {
    const graph = normalizeModelGraph(graphValue);
    const groups = graph.operators
      .filter((operator) => operator.op_kind === "layer_group")
      .sort((left, right) => left.sequence_index - right.sequence_index || left.operator_id.localeCompare(right.operator_id));
    const groupInfo = groups.map((group) => overviewGroupProjectionInfo(graph, group));
    const groupInfoFor = new Map(groupInfo.map((item) => [item.group_id, item]));
    const displayForOperator = new Map();
    const nodeForKey = new Map();
    const nodes = [];

    function addLeaf(operator, key, multiplicity, role, info = null) {
      let node = nodeForKey.get(key);
      if (!node) {
        node = {
          display_id: `overview:${operator.operator_id}`,
          kind: "operator",
          op_kind: String(operator.op_kind || "operator"),
          label: overviewLeafLabel(operator),
          structural_role: role,
          operator_ids: [],
          representative_operator_id: operator.operator_id,
          instance_count: 0,
          sequence_index: operator.sequence_index,
          group_ids: [],
          detail: {
            parameters: clone(object(operator.parameters)),
            ports: clone(array(operator.ports)),
            attributes: clone(object(operator.attributes)),
          },
        };
        if (info) node.structure = {
          mixer_kind: info.descriptor.mixer_kind,
          ffn_kind: info.descriptor.ffn_kind,
          num_experts: info.descriptor.num_experts,
          experts_per_token: info.descriptor.experts_per_token,
          shared_expert: info.descriptor.shared_expert,
        };
        nodes.push(node);
        nodeForKey.set(key, node);
      }
      if (!node.operator_ids.includes(operator.operator_id)) node.operator_ids.push(operator.operator_id);
      if (info && !node.group_ids.includes(info.group_id)) node.group_ids.push(info.group_id);
      node.instance_count += multiplicity;
      node.sequence_index = Math.min(node.sequence_index, operator.sequence_index);
      displayForOperator.set(operator.operator_id, node.display_id);
    }

    groupInfo.forEach((info) => info.roles.forEach(({ operator, role }) => {
      const key = `group:${info.signature}:${role}:${overviewOperatorSignature(operator)}`;
      addLeaf(operator, key, info.repeat, role, info);
    }));

    graph.operators
      .filter((operator) => {
        if (["layer_group", "model_input", "model_output"].includes(operator.op_kind)) return false;
        return !groupInfoFor.has(String(object(operator.attributes).parent_group_id || ""));
      })
      .forEach((operator) => {
        const repeat = Math.max(1, Math.trunc(finite(object(operator.parameters).repeat, 1)));
        // MTP prediction layers form one semantic role even when the authority
        // stores several explicit chain members. Other top-level operators keep
        // their positional identity so equal kinds in different locations are
        // never collapsed accidentally.
        const key = isMtpOperator(operator)
          ? `mtp:${overviewOperatorSignature(operator)}`
          : `operator:${operator.operator_id}`;
        addLeaf(operator, key, repeat, isMtpOperator(operator) ? operator.op_kind : `top_level:${operator.operator_id}`);
      });

    const operatorById = new Map(graph.operators.map((item) => [item.operator_id, item]));
    const tensorById = new Map(graph.tensors.map((item) => [item.tensor_id, item]));
    nodes.forEach((node) => {
      const memberIds = new Set(node.operator_ids || []);
      const boundaryPorts = new Map();
      memberIds.forEach((operatorId) => {
        const operator = operatorById.get(operatorId);
        array(operator?.ports).forEach((port) => {
          if (!["input", "output"].includes(port.direction)) return;
          const tensor = tensorById.get(port.tensor_id);
          const producerDisplay = tensor?.producer_operator_id
            ? displayForOperator.get(tensor.producer_operator_id)
            : null;
          const consumerDisplays = array(tensor?.consumer_operator_ids)
            .map((consumerId) => displayForOperator.get(consumerId))
            .filter(Boolean);
          const crossesBoundary = port.direction === 'input'
            ? !tensor?.producer_operator_id || producerDisplay !== node.display_id
            : !array(tensor?.consumer_operator_ids).length
              || consumerDisplays.some((displayId) => displayId !== node.display_id)
              || array(tensor?.consumer_operator_ids).some((consumerId) => !displayForOperator.has(consumerId));
          if (!crossesBoundary) return;
          const projectedPort = {
            operator_id: operator.operator_id,
            port_id: port.port_id,
            direction: port.direction,
            tensor_id: port.tensor_id,
            dtype: port.dtype,
            shape: clone(port.shape),
            layout: port.layout,
          };
          const key = stableStringify({
            port_id: projectedPort.port_id,
            direction: projectedPort.direction,
            dtype: projectedPort.dtype,
            shape: projectedPort.shape,
            layout: projectedPort.layout,
          });
          const previous = boundaryPorts.get(key);
          if (!previous || operator.operator_id === node.representative_operator_id) boundaryPorts.set(key, projectedPort);
        });
      });
      node.boundary_ports = Array.from(boundaryPorts.values()).sort((left, right) => (
        left.direction.localeCompare(right.direction)
        || left.operator_id.localeCompare(right.operator_id)
        || left.port_id.localeCompare(right.port_id)
      ));
    });

    const nodeByDisplayId = new Map(nodes.map((node) => [node.display_id, node]));
    function representativeBoundaryPort(displayId, operatorId, tensorId, direction) {
      const node = nodeByDisplayId.get(displayId);
      const operator = operatorById.get(operatorId);
      const actual = array(operator?.ports).find((port) => port.direction === direction && port.tensor_id === tensorId);
      if (!node || !actual) return null;
      const ports = array(node.boundary_ports).filter((port) => port.direction === direction);
      const exact = ports.find((port) => port.tensor_id === tensorId);
      const equivalent = ports.find((port) => (
        port.port_id === actual.port_id
        && stableStringify(normalizedContract(port)) === stableStringify(normalizedContract(actual))
      ));
      const projected = exact || equivalent || ports[0];
      return projected ? {
        operator_id: projected.operator_id,
        port_id: projected.port_id,
        tensor_id: projected.tensor_id,
      } : null;
    }

    const edgeMap = new Map();
    for (const edge of modelGraphEdges(graph)) {
      const sourceId = displayForOperator.get(edge.source_operator_id);
      const targetId = displayForOperator.get(edge.target_operator_id);
      if (!sourceId || !targetId || sourceId === targetId) continue;
      const key = `${sourceId}->${targetId}`;
      if (!edgeMap.has(key)) {
        const source = representativeBoundaryPort(sourceId, edge.source_operator_id, edge.tensor_id, "output");
        const target = representativeBoundaryPort(targetId, edge.target_operator_id, edge.tensor_id, "input");
        edgeMap.set(key, {
          source_id: sourceId, target_id: targetId, semantic_edge_count: 0,
          tensor_ids: [], contracts: [], visual_only: false,
          representative_edge: source && target ? { source, target } : null,
        });
      }
      const projected = edgeMap.get(key);
      projected.semantic_edge_count += 1;
      if (!projected.tensor_ids.includes(edge.tensor_id)) projected.tensor_ids.push(edge.tensor_id);
      const contract = { tensor_id: edge.tensor_id, dtype: edge.dtype, shape: clone(edge.shape), layout: edge.layout };
      if (!projected.contracts.some((item) => stableStringify(item) === stableStringify(contract))) projected.contracts.push(contract);
    }
    nodes.sort((left, right) => left.sequence_index - right.sequence_index || left.display_id.localeCompare(right.display_id));
    return {
      mode: "overview",
      graph_id: graph.graph_id,
      direction: "TB",
      operator_first: true,
      nodes,
      edges: Array.from(edgeMap.values()),
      semantic_counts: {
        operators: graph.operators.length,
        tensors: graph.tensors.length,
        layer_groups: groups.length,
        projected_leaf_nodes: nodes.length,
        represented_leaf_instances: nodes.reduce((total, node) => total + node.instance_count, 0),
      },
      attributes: {
        visual_projection_only: true,
        leaf_operator_projection: true,
        deduplicated: true,
        authoritative_graph_id: graph.graph_id,
      },
    };
  }

  function buildOverviewProjection(graphValue, optionsValue = {}) {
    return buildRepeatGroupOverviewProjection(graphValue, optionsValue);
  }

  function overviewSemanticOverrides(groupValue) {
    const group = object(groupValue);
    const overrides = object(object(group.parameters).overrides);
    const semantic = {};
    Object.keys(overrides).sort().forEach((key) => {
      const value = clone(object(overrides[key]));
      delete value.metadata;
      delete value.schema_version;
      if (Object.keys(value).length) semantic[key] = value;
    });
    return semantic;
  }

  function overviewGroupBoundary(graph, memberIdsValue) {
    const memberIds = new Set(array(memberIdsValue).map(String));
    const operatorById = new Map(graph.operators.map((item) => [item.operator_id, item]));
    const sequence = (operatorId) => finite(operatorById.get(operatorId)?.sequence_index);
    const inputs = [];
    const outputs = [];
    graph.tensors.forEach((tensor) => {
      const insideConsumers = tensor.consumer_operator_ids.filter((id) => memberIds.has(id));
      const outsideConsumers = tensor.consumer_operator_ids.filter((id) => !memberIds.has(id));
      const producerInside = memberIds.has(String(tensor.producer_operator_id || ""));
      if (insideConsumers.length && !producerInside && tensor.role !== "weight") {
        const mappedPorts = insideConsumers.flatMap((operatorId) => {
          const operator = operatorById.get(operatorId);
          return array(operator?.ports)
            .filter((port) => port.direction === "input" && port.tensor_id === tensor.tensor_id)
            .map((port) => ({ operator_id: operatorId, port_id: port.port_id }));
        }).sort((left, right) => sequence(left.operator_id) - sequence(right.operator_id)
          || left.operator_id.localeCompare(right.operator_id) || left.port_id.localeCompare(right.port_id));
        const endpoint = mappedPorts[0];
        const port = endpoint && operatorById.get(endpoint.operator_id)?.ports.find((item) => item.port_id === endpoint.port_id);
        if (port) inputs.push({
          ...clone(port), operator_id: endpoint.operator_id,
          proxy_port_id: `boundary:input:${tensor.tensor_id}`,
          mapped_ports: mappedPorts,
          fan_out: mappedPorts.length,
        });
      }
      if (producerInside && (!tensor.consumer_operator_ids.length || outsideConsumers.length)) {
        const operator = operatorById.get(tensor.producer_operator_id);
        const port = array(operator?.ports).find((item) => item.direction === "output" && item.tensor_id === tensor.tensor_id);
        if (port) outputs.push({
          ...clone(port), operator_id: tensor.producer_operator_id,
          proxy_port_id: `boundary:output:${tensor.tensor_id}`,
          mapped_ports: [{ operator_id: tensor.producer_operator_id, port_id: port.port_id }],
          fan_out: Math.max(1, outsideConsumers.length),
        });
      }
    });
    const sortPorts = (left, right) => sequence(left.operator_id) - sequence(right.operator_id)
      || left.tensor_id.localeCompare(right.tensor_id) || left.port_id.localeCompare(right.port_id);
    return { inputs: inputs.sort(sortPorts), outputs: outputs.sort(sortPorts), ports: [...inputs, ...outputs] };
  }

  function overviewPatternBoundaryIsSimple(graph, infosValue) {
    const infos = array(infosValue);
    if (!infos.length) return false;
    const membersFor = (info) => new Set(array(info.roles).map((item) => item.operator.operator_id));
    const allMembers = new Set(infos.flatMap((info) => Array.from(membersFor(info))));
    const boundary = overviewGroupBoundary(graph, Array.from(allMembers));
    if (boundary.inputs.length !== 1 || boundary.outputs.length !== 1) return false;
    const firstMembers = membersFor(infos[0]);
    const lastMembers = membersFor(infos.at(-1));
    if (!array(boundary.inputs[0].mapped_ports).every((port) => firstMembers.has(port.operator_id))) return false;
    if (!lastMembers.has(boundary.outputs[0].operator_id)) return false;
    for (let index = 1; index < infos.length; index += 1) {
      const previous = membersFor(infos[index - 1]);
      const current = membersFor(infos[index]);
      const directlyConnected = graph.tensors.some((tensor) => (
        previous.has(String(tensor.producer_operator_id || ""))
        && tensor.consumer_operator_ids.some((consumerId) => current.has(consumerId))
      ));
      if (!directlyConnected) return false;
    }
    // A producer/consumer that crosses the interval anywhere except the one
    // entry and one exit is a bypass. Such a candidate remains uncompressed.
    return graph.tensors.every((tensor) => {
      const producerInside = allMembers.has(String(tensor.producer_operator_id || ""));
      const insideConsumers = tensor.consumer_operator_ids.filter((id) => allMembers.has(id));
      const outsideConsumers = tensor.consumer_operator_ids.filter((id) => !allMembers.has(id));
      if (producerInside && outsideConsumers.length) return tensor.tensor_id === boundary.outputs[0].tensor_id;
      if (!producerInside && insideConsumers.length && tensor.role !== "weight") return tensor.tensor_id === boundary.inputs[0].tensor_id;
      return true;
    });
  }

  function overviewPatternRanges(graphValue, groupInfosValue) {
    const graph = normalizeModelGraph(graphValue);
    const infos = array(groupInfosValue);
    const ranges = [];
    let start = 0;
    while (start < infos.length) {
      let best = null;
      const remaining = infos.length - start;
      for (let period = 1; period <= Math.floor(remaining / 2); period += 1) {
        let repetitions = 1;
        while (start + (repetitions + 1) * period <= infos.length) {
          const matches = Array.from({ length: period }, (_, offset) => (
            infos[start + repetitions * period + offset].signature === infos[start + offset].signature
          )).every(Boolean);
          if (!matches) break;
          repetitions += 1;
        }
        if (repetitions < 2) continue;
        const candidate = infos.slice(start, start + period * repetitions);
        if (candidate.some((info) => Object.keys(info.semantic_overrides || {}).length)) continue;
        if (!overviewPatternBoundaryIsSimple(graph, candidate)) continue;
        const covered = period * repetitions;
        if (!best || covered > best.covered || (covered === best.covered && period < best.period)) {
          best = { start, end: start + covered, period, repetitions, covered };
        }
      }
      if (best) {
        ranges.push(best);
        start = best.end;
      } else {
        ranges.push({ start, end: start + 1, period: 1, repetitions: 1, covered: 1 });
        start += 1;
      }
    }
    return ranges;
  }

  function buildRepeatGroupOverviewProjection(graphValue, optionsValue = {}) {
    const graph = normalizeModelGraph(graphValue);
    const options = object(optionsValue);
    const operatorById = new Map(graph.operators.map((item) => [item.operator_id, item]));
    const groups = graph.operators.filter((item) => item.op_kind === "layer_group")
      .sort((left, right) => left.sequence_index - right.sequence_index || left.operator_id.localeCompare(right.operator_id));
    const collapsed = new Set(Array.isArray(options.collapsed_groups)
      ? options.collapsed_groups.map(String)
      : groups.map((item) => item.operator_id));
    const infoById = new Map(groups.map((group) => {
      const info = overviewGroupProjectionInfo(graph, group);
      info.group = group;
      info.semantic_overrides = overviewSemanticOverrides(group);
      // Contract, topology, semantic parameters and meaningful per-instance
      // overrides all split patterns deterministically. Repeat is part of the
      // segment identity so A×2 cannot match A×1, while [A×2,B×1] can repeat.
      info.signature = stableStringify({ template: info.signature, repeat: info.repeat, semantic_overrides: info.semantic_overrides });
      return [info.group_id, info];
    }));
    const childGroupId = new Map();
    infoById.forEach((info) => info.roles.forEach(({ operator }) => childGroupId.set(operator.operator_id, info.group_id)));
    const topLevel = graph.operators.filter((operator) => (
      !childGroupId.has(operator.operator_id) && !["model_input", "model_output"].includes(operator.op_kind)
    )).sort((left, right) => left.sequence_index - right.sequence_index || left.operator_id.localeCompare(right.operator_id));
    const consecutiveRuns = [];
    let run = [];
    topLevel.forEach((operator) => {
      if (operator.op_kind === "layer_group") run.push(infoById.get(operator.operator_id));
      else {
        if (run.length) consecutiveRuns.push(run);
        run = [];
      }
    });
    if (run.length) consecutiveRuns.push(run);
    const rangeForGroup = new Map();
    consecutiveRuns.forEach((infos) => overviewPatternRanges(graph, infos).forEach((range) => {
      const slice = infos.slice(range.start, range.end);
      slice.forEach((info) => rangeForGroup.set(info.group_id, { ...range, infos: slice }));
    }));

    const nodes = [];
    const displayForOperator = new Map();
    const consumedGroups = new Set();
    topLevel.forEach((operator) => {
      if (operator.op_kind !== "layer_group") {
        const memberIds = [operator.operator_id];
        const boundary = overviewGroupBoundary(graph, memberIds);
        const repeat = Math.max(1, Math.trunc(finite(object(operator.parameters).repeat, 1)));
        const node = {
          display_id: `overview:${operator.operator_id}`,
          kind: "operator", op_kind: operator.op_kind, label: overviewLeafLabel(operator),
          operator_ids: memberIds, representative_operator_id: operator.operator_id,
          sequence_index: operator.sequence_index, instance_count: repeat,
          boundary_ports: boundary.ports,
          detail: { parameters: clone(operator.parameters), ports: clone(operator.ports), attributes: clone(operator.attributes) },
        };
        nodes.push(node);
        displayForOperator.set(operator.operator_id, node.display_id);
        return;
      }
      if (consumedGroups.has(operator.operator_id)) return;
      const range = rangeForGroup.get(operator.operator_id) || { infos: [infoById.get(operator.operator_id)], period: 1, repetitions: 1 };
      const infos = range.infos;
      infos.forEach((info) => consumedGroups.add(info.group_id));
      const templateInfos = infos.slice(0, range.period);
      const allMemberIds = infos.flatMap((info) => info.roles.map(({ operator: child }) => child.operator_id));
      const templateMemberIds = templateInfos.flatMap((info) => info.roles.map(({ operator: child }) => child.operator_id));
      const groupIds = infos.map((info) => info.group_id);
      const repeatCount = infos.reduce((total, info) => total + info.repeat, 0);
      const patternKinds = templateInfos.map((info) => overviewBlockLabel(info.descriptor, templateInfos.length === 1));
      const displayId = range.repetitions > 1
        ? `overview:pattern:${groupIds[0]}:${range.period}:${range.repetitions}`
        : `overview:group:${groupIds[0]}`;
      const boundary = overviewGroupBoundary(graph, allMemberIds);
      const expanded = groupIds.some((id) => !collapsed.has(id));
      const node = {
        display_id: displayId,
        kind: "repeat_group", op_kind: "layer_group",
        label: patternKinds.length > 1 ? `${patternKinds.join(" / ")} Pattern` : patternKinds[0],
        group_id: groupIds[0], group_ids: groupIds,
        operator_ids: allMemberIds, template_operator_ids: templateMemberIds,
        representative_operator_id: groupIds[0], sequence_index: operator.sequence_index,
        instance_count: repeatCount, repeat_count: repeatCount,
        pattern_repetitions: range.repetitions, pattern_period: range.period,
        pattern: templateInfos.map((info) => ({
          group_id: info.group_id, signature: info.signature,
          repeat: info.repeat,
          mixer_kind: info.descriptor.mixer_kind, ffn_kind: info.descriptor.ffn_kind,
        })),
        expanded,
        boundary_ports: boundary.ports,
        boundary: {
          input_count: boundary.inputs.length, output_count: boundary.outputs.length,
          input_tensor_ids: boundary.inputs.map((port) => port.tensor_id),
          output_tensor_ids: boundary.outputs.map((port) => port.tensor_id),
        },
        overrides: Object.fromEntries(infos.map((info) => [info.group_id, clone(info.semantic_overrides)])),
        has_semantic_overrides: infos.some((info) => Object.keys(info.semantic_overrides).length),
        inline_graph: authoritativeOperatorProjection(graph, templateMemberIds, {
          group_id: groupIds[0],
          expanded_components: [...templateMemberIds, ...templateInfos.map((info) => `${info.group_id}.moe`)],
        }),
        visual_repeat: repeatCount > 1,
      };
      nodes.push(node);
      allMemberIds.forEach((id) => displayForOperator.set(id, displayId));
    });

    const nodeById = new Map(nodes.map((node) => [node.display_id, node]));
    const edgeMap = new Map();
    modelGraphEdges(graph).forEach((edge) => {
      const sourceId = displayForOperator.get(edge.source_operator_id);
      const targetId = displayForOperator.get(edge.target_operator_id);
      if (!sourceId || !targetId || sourceId === targetId) return;
      const key = `${sourceId}->${targetId}`;
      if (!edgeMap.has(key)) edgeMap.set(key, {
        source_id: sourceId, target_id: targetId, semantic_edge_count: 0,
        tensor_ids: [], contracts: [], visual_only: false, representative_edge: null,
      });
      const projected = edgeMap.get(key);
      projected.semantic_edge_count += 1;
      if (!projected.tensor_ids.includes(edge.tensor_id)) projected.tensor_ids.push(edge.tensor_id);
      const contract = { tensor_id: edge.tensor_id, dtype: edge.dtype, shape: clone(edge.shape), layout: edge.layout };
      if (!projected.contracts.some((item) => stableStringify(item) === stableStringify(contract))) projected.contracts.push(contract);
    });
    edgeMap.forEach((edge) => {
      const source = array(nodeById.get(edge.source_id)?.boundary_ports)
        .find((port) => port.direction === "output" && edge.tensor_ids.includes(port.tensor_id));
      const target = array(nodeById.get(edge.target_id)?.boundary_ports)
        .find((port) => port.direction === "input" && edge.tensor_ids.includes(port.tensor_id));
      if (source && target) edge.representative_edge = {
        source: { operator_id: source.operator_id, port_id: source.port_id, tensor_id: source.tensor_id },
        target: { operator_id: target.operator_id, port_id: target.port_id, tensor_id: target.tensor_id },
      };
    });
    nodes.filter((node) => node.visual_repeat && node.boundary_ports.some((port) => port.direction === "input")
      && node.boundary_ports.some((port) => port.direction === "output")).forEach((node) => {
      const source = node.boundary_ports.find((port) => port.direction === "output");
      const target = node.boundary_ports.find((port) => port.direction === "input");
      edgeMap.set(`${node.display_id}->${node.display_id}:visual-repeat`, {
        source_id: node.display_id, target_id: node.display_id,
        semantic_edge_count: 0, tensor_ids: [], contracts: [], visual_only: true,
        repeat_count: node.repeat_count, label: `×${node.repeat_count}`,
        representative_edge: {
          source: { operator_id: source.operator_id, port_id: source.port_id, tensor_id: source.tensor_id },
          target: { operator_id: target.operator_id, port_id: target.port_id, tensor_id: target.tensor_id },
        },
      });
    });
    nodes.sort((left, right) => left.sequence_index - right.sequence_index || left.display_id.localeCompare(right.display_id));
    return {
      mode: "overview", graph_id: graph.graph_id, direction: "TB", operator_first: true,
      nodes, edges: Array.from(edgeMap.values()),
      semantic_counts: {
        operators: graph.operators.length, tensors: graph.tensors.length, layer_groups: groups.length,
        repeat_group_nodes: nodes.filter((node) => node.kind === "repeat_group").length,
        represented_instances: nodes.reduce((total, node) => total + Math.max(1, finite(node.instance_count, 1)), 0),
      },
      attributes: {
        visual_projection_only: true, repeat_group_projection: true,
        leaf_operator_projection: false, deduplicated: false,
        repeat_edges_authoritative: false, authoritative_graph_id: graph.graph_id,
      },
    };
  }

  function endpoint(graph, value) {
    const item = object(value);
    const operatorId = String(item.operator_id ?? item.node_id ?? item.operatorId ?? "");
    const portId = String(item.port_id ?? item.portId ?? "");
    const operator = graph.operators.find((candidate) => candidate.operator_id === operatorId);
    const port = operator?.ports.find((candidate) => candidate.port_id === portId);
    return { operatorId, portId, operator, port };
  }

  function pathExists(graph, startId, goalId) {
    const adjacency = new Map(graph.operators.map((item) => [item.operator_id, []]));
    modelGraphEdges(graph).forEach((edge) => adjacency.get(edge.source_operator_id)?.push(edge.target_operator_id));
    const queue = [startId];
    const seen = new Set();
    while (queue.length) {
      const current = queue.shift();
      if (current === goalId) return true;
      if (seen.has(current)) continue;
      seen.add(current);
      queue.push(...(adjacency.get(current) || []));
    }
    return false;
  }

  function validateConnection(graphValue, sourceValue, targetValue) {
    const graph = normalizeModelGraph(graphValue);
    const source = endpoint(graph, sourceValue);
    const target = endpoint(graph, targetValue);
    if (!source.operator || !source.port) return { compatible: false, ok: false, code: "missing_source", message: `找不到源组件端口：${source.operatorId}.${source.portId}` };
    if (!target.operator || !target.port) return { compatible: false, ok: false, code: "missing_target", message: `找不到目标组件端口：${target.operatorId}.${target.portId}` };
    if (source.port.direction !== "output" || target.port.direction !== "input") return { compatible: false, ok: false, code: "direction_mismatch", message: `端口方向不匹配：期望 output → input，实际 ${source.port.direction} → ${target.port.direction}。` };
    if (source.operatorId === target.operatorId || pathExists(graph, target.operatorId, source.operatorId)) return { compatible: false, ok: false, code: "cycle", message: `连接 ${source.operatorId} → ${target.operatorId} 会形成有向环，已拒绝。` };
    const unified = unifyTensorContracts(target.port, source.port);
    return { ...unified, compatible: unified.ok, source: { operator_id: source.operatorId, port_id: source.portId }, target: { operator_id: target.operatorId, port_id: target.portId } };
  }

  function previewEndpoint(value) {
    const item = object(value);
    const operatorId = String(item.operator_id ?? item.node_id ?? item.operatorId ?? "");
    const portId = String(item.port_id ?? item.portId ?? "");
    return operatorId && portId ? { operator_id: operatorId, port_id: portId } : null;
  }

  function previewPoint(value) {
    if (!value || typeof value !== "object") return null;
    return { x: finite(value.x), y: finite(value.y) };
  }

  function inactiveConnectionPreview(status = "idle", diagnostic = null) {
    return {
      status,
      active: false,
      source: null,
      target: null,
      pointer: null,
      diagnostic: diagnostic ? clone(diagnostic) : null,
    };
  }

  function createConnectionPreview(graphValue, sourceValue, pointerValue = null) {
    const graph = normalizeModelGraph(graphValue);
    const requested = previewEndpoint(sourceValue);
    const source = endpoint(graph, requested || {});
    if (!source.operator || !source.port) {
      const diagnostic = {
        compatible: false, ok: false, code: "missing_source",
        message: `找不到源组件端口：${source.operatorId}.${source.portId}`,
      };
      return inactiveConnectionPreview("invalid", diagnostic);
    }
    if (source.port.direction !== "output") {
      const diagnostic = {
        compatible: false, ok: false, code: "direction_mismatch",
        message: `端口方向不匹配：连接必须从 output 端口开始，实际为 ${source.port.direction}。`,
      };
      return inactiveConnectionPreview("invalid", diagnostic);
    }
    return {
      status: "preview",
      active: true,
      source: { operator_id: source.operatorId, port_id: source.portId },
      target: null,
      pointer: previewPoint(pointerValue),
      diagnostic: null,
    };
  }

  function connectionPreviewCompatibility(graphValue, previewValue, targetValue) {
    const preview = object(previewValue);
    const source = previewEndpoint(preview.source || previewValue);
    if (preview.active === false || !source) return {
      compatible: false, ok: false, code: "preview_inactive",
      message: "连接预览未激活；请先选择一个输出端口。",
    };
    return validateConnection(graphValue, source, targetValue);
  }

  function updateConnectionPreview(graphValue, previewValue, pointerValue = null, targetValue = undefined) {
    const current = object(previewValue);
    let pointer = pointerValue;
    let target = targetValue;
    if (targetValue === undefined && pointerValue && typeof pointerValue === "object"
      && (Object.hasOwn(pointerValue, "pointer") || Object.hasOwn(pointerValue, "target"))) {
      pointer = pointerValue.pointer;
      target = pointerValue.target;
    }
    if (!current.active) return clone(current);
    const restarted = createConnectionPreview(graphValue, current.source, pointer ?? current.pointer);
    if (!restarted.active || target == null) return restarted;
    const diagnostic = connectionPreviewCompatibility(graphValue, restarted, target);
    return {
      ...restarted,
      status: diagnostic.ok ? "compatible" : "incompatible",
      target: previewEndpoint(target),
      diagnostic,
    };
  }

  function cancelConnectionPreview(previewValue, reasonValue = "cancelled") {
    const reason = String(reasonValue || "cancelled");
    const label = reason === "escape" || reason === "esc" ? "已按 Esc 取消端口连接预览。" : "已取消端口连接预览。";
    return inactiveConnectionPreview("cancelled", {
      compatible: false, ok: false, code: "cancelled", reason, message: label,
    });
  }

  function commitConnectionPreview(graphValue, previewValue, targetValue = undefined) {
    const preview = object(previewValue);
    const target = targetValue === undefined ? preview.target : targetValue;
    if (!preview.active) {
      const diagnostic = {
        compatible: false, ok: false, code: "preview_inactive",
        message: "连接预览未激活；graph 未改变。",
      };
      return { connected: false, graph: graphValue, preview: clone(previewValue), diagnostic };
    }
    const diagnostic = connectionPreviewCompatibility(graphValue, preview, target);
    if (!diagnostic.ok) {
      return {
        connected: false,
        graph: graphValue,
        preview: updateConnectionPreview(graphValue, preview, preview.pointer, target),
        diagnostic,
      };
    }
    // Keep one authoritative semantic edit path.  connect() intentionally
    // re-runs direction, contract, and DAG checks immediately before cloning
    // and changing the graph.
    const graph = connect(graphValue, diagnostic.source, diagnostic.target);
    return {
      connected: true,
      graph,
      preview: inactiveConnectionPreview("committed", diagnostic),
      diagnostic,
    };
  }

  function pruneOrphanTensors(graph) {
    const transformRefs = new Set(graph.transforms.flatMap((item) => [item.input_tensor_id, item.output_tensor_id]));
    graph.tensors = graph.tensors.filter((tensor) => tensor.producer_operator_id || tensor.consumer_operator_ids.length || transformRefs.has(tensor.tensor_id) || ["input", "output"].includes(tensor.role));
    return graph;
  }

  function connect(graphValue, sourceValue, targetValue) {
    const diagnostic = validateConnection(graphValue, sourceValue, targetValue);
    if (!diagnostic.ok) {
      const error = new Error(diagnostic.message);
      error.code = diagnostic.code; error.diagnostic = diagnostic;
      throw error;
    }
    const next = normalizeModelGraph(graphValue);
    const source = endpoint(next, sourceValue);
    const target = endpoint(next, targetValue);
    const sourceTensor = next.tensors.find((item) => item.tensor_id === source.port.tensor_id);
    const oldTensor = next.tensors.find((item) => item.tensor_id === target.port.tensor_id);
    if (oldTensor) oldTensor.consumer_operator_ids = oldTensor.consumer_operator_ids.filter((id) => id !== target.operatorId);
    const contract = diagnostic.contract;
    sourceTensor.dtype = contract.dtype; sourceTensor.shape = clone(contract.shape); sourceTensor.layout = contract.layout;
    next.operators.forEach((operator) => operator.ports.forEach((port) => {
      if (port.tensor_id !== sourceTensor.tensor_id) return;
      port.dtype = contract.dtype; port.shape = clone(contract.shape); port.layout = contract.layout;
    }));
    target.port.tensor_id = sourceTensor.tensor_id;
    target.port.dtype = contract.dtype; target.port.shape = clone(contract.shape); target.port.layout = contract.layout;
    if (!sourceTensor.consumer_operator_ids.includes(target.operatorId)) sourceTensor.consumer_operator_ids.push(target.operatorId);
    return pruneOrphanTensors(normalizeModelGraph(next));
  }

  function disconnect(graphValue, targetValue) {
    const next = normalizeModelGraph(graphValue);
    const target = endpoint(next, targetValue);
    if (!target.operator || !target.port) throw new Error(`找不到要断开的目标端口：${target.operatorId}.${target.portId}`);
    if (target.port.direction !== "input") throw new Error("只能从输入端口断开张量连接。");
    const old = next.tensors.find((item) => item.tensor_id === target.port.tensor_id);
    if (old) old.consumer_operator_ids = old.consumer_operator_ids.filter((id) => id !== target.operatorId);
    const used = new Set(next.tensors.map((item) => item.tensor_id));
    const tensorId = uniqueId(`${target.operatorId}.${target.portId}.input`, used);
    target.port.tensor_id = tensorId;
    next.tensors.push({
      tensor_id: tensorId, role: "activation", logical_bytes: null,
      producer_operator_id: null, consumer_operator_ids: [target.operatorId],
      dtype: target.port.dtype, shape: clone(target.port.shape), layout: target.port.layout,
      attributes: {}, provenance: [],
    });
    return pruneOrphanTensors(normalizeModelGraph(next));
  }

  function addNode(graphValue, nodeValue, options = {}) {
    const graph = normalizeModelGraph(graphValue);
    const used = new Set(graph.operators.map((item) => item.operator_id));
    const node = clone(object(nodeValue));
    node.operator_id = uniqueId(node.operator_id || node.id || node.op_kind || "operator", used);
    node.sequence_index = Number.isFinite(Number(node.sequence_index)) ? Number(node.sequence_index) : graph.operators.reduce((maximum, item) => Math.max(maximum, item.sequence_index), -1) + 1;
    graph.operators.push(node);
    const next = normalizeModelGraph(graph);
    if (options.position) {
      next.attributes.ui ??= {};
      next.attributes.ui.positions ??= {};
      next.attributes.ui.positions[node.operator_id] = { x: finite(options.position.x), y: finite(options.position.y) };
    }
    return next;
  }

  function updateNode(graphValue, operatorId, patchValue) {
    const graph = normalizeModelGraph(graphValue);
    const node = graph.operators.find((item) => item.operator_id === operatorId);
    if (!node) throw new Error(`找不到模型组件：${operatorId}`);
    const patch = clone(object(patchValue));
    if (patch.operator_id && patch.operator_id !== operatorId && graph.operators.some((item) => item.operator_id === patch.operator_id)) throw new Error(`模型组件 ID 已存在：${patch.operator_id}`);
    const oldId = node.operator_id;
    const nextId = patch.operator_id ? safeId(patch.operator_id) : oldId;
    Object.assign(node, patch);
    node.operator_id = nextId;
    if (patch.parameters) node.parameters = { ...object(node.parameters), ...object(patch.parameters) };
    if (patch.attributes) node.attributes = { ...object(node.attributes), ...object(patch.attributes) };
    if (nextId !== oldId) {
      graph.tensors.forEach((tensor) => {
        if (tensor.producer_operator_id === oldId) tensor.producer_operator_id = nextId;
        tensor.consumer_operator_ids = tensor.consumer_operator_ids.map((id) => id === oldId ? nextId : id);
      });
      graph.operators.forEach((item) => { if (item.attributes.parent_group_id === oldId) item.attributes.parent_group_id = nextId; });
      const positions = object(graph.attributes.ui?.positions);
      if (Object.hasOwn(positions, oldId)) { positions[nextId] = positions[oldId]; delete positions[oldId]; }
    }
    return normalizeModelGraph(graph);
  }

  function deleteNode(graphValue, operatorId, options = {}) {
    const graph = normalizeModelGraph(graphValue);
    const existing = graph.operators.find((item) => item.operator_id === operatorId);
    if (!existing) return graph;
    const deleted = new Set([operatorId]);
    if (options.cascade !== false && existing.op_kind === "layer_group") graph.operators.forEach((item) => {
      if (item.attributes.parent_group_id === operatorId) deleted.add(item.operator_id);
    });
    graph.operators = graph.operators.filter((item) => !deleted.has(item.operator_id));
    graph.tensors.forEach((tensor) => {
      if (deleted.has(tensor.producer_operator_id)) tensor.producer_operator_id = null;
      tensor.consumer_operator_ids = tensor.consumer_operator_ids.filter((id) => !deleted.has(id));
    });
    graph.transforms = graph.transforms.filter((item) => graph.tensors.some((tensor) => tensor.tensor_id === item.input_tensor_id) && graph.tensors.some((tensor) => tensor.tensor_id === item.output_tensor_id));
    const positions = object(graph.attributes.ui?.positions);
    deleted.forEach((id) => delete positions[id]);
    graph.attributes.ui.collapsed_groups = array(graph.attributes.ui?.collapsed_groups).filter((id) => !deleted.has(id));
    return pruneOrphanTensors(normalizeModelGraph(graph));
  }

  function duplicateNode(graphValue, operatorId, options = {}) {
    const graph = normalizeModelGraph(graphValue);
    const source = graph.operators.find((item) => item.operator_id === operatorId);
    if (!source) throw new Error(`找不到要复制的模型组件：${operatorId}`);
    const usedNodes = new Set(graph.operators.map((item) => item.operator_id));
    const duplicate = clone(source);
    duplicate.operator_id = options.operator_id ? uniqueId(options.operator_id, usedNodes) : uniqueId(operatorId, usedNodes, true);
    duplicate.sequence_index = graph.operators.reduce((maximum, item) => Math.max(maximum, item.sequence_index), -1) + 1;
    const usedTensors = new Set(graph.tensors.map((item) => item.tensor_id));
    duplicate.ports.forEach((port) => {
      const tensorId = uniqueId(stableTensorId(duplicate.operator_id, port.port_id), usedTensors);
      port.tensor_id = tensorId;
      graph.tensors.push({
        tensor_id: tensorId, role: port.direction === "weight" ? "weight" : "activation", logical_bytes: null,
        producer_operator_id: port.direction === "output" ? duplicate.operator_id : null,
        consumer_operator_ids: port.direction === "output" ? [] : [duplicate.operator_id],
        dtype: port.dtype, shape: clone(port.shape), layout: port.layout, attributes: {}, provenance: [],
      });
    });
    duplicate.input_tensor_ids = duplicate.ports.filter((item) => item.direction === "input").map((item) => item.tensor_id);
    duplicate.output_tensor_ids = duplicate.ports.filter((item) => item.direction === "output").map((item) => item.tensor_id);
    duplicate.weight_tensor_ids = duplicate.ports.filter((item) => item.direction === "weight").map((item) => item.tensor_id);
    graph.operators.push(duplicate);
    const position = object(graph.attributes.ui?.positions)[operatorId];
    if (position) graph.attributes.ui.positions[duplicate.operator_id] = { x: finite(position.x) + 48, y: finite(position.y) + 48 };
    return normalizeModelGraph(graph);
  }

  function layoutDag(graphValue, sizesValue = {}, options = {}) {
    const graph = normalizeModelGraph(graphValue);
    const ids = graph.operators.map((item) => item.operator_id);
    const order = new Map(graph.operators.map((item, index) => [item.operator_id, [item.sequence_index, index]]));
    const compare = (left, right) => {
      const a = order.get(left) || [0, 0]; const b = order.get(right) || [0, 0];
      return a[0] - b[0] || a[1] - b[1] || left.localeCompare(right);
    };
    const adjacency = new Map(ids.map((id) => [id, new Set()]));
    const predecessors = new Map(ids.map((id) => [id, new Set()]));
    modelGraphEdges(graph).forEach((edge) => {
      if (edge.source_operator_id === edge.target_operator_id) return;
      adjacency.get(edge.source_operator_id)?.add(edge.target_operator_id);
      predecessors.get(edge.target_operator_id)?.add(edge.source_operator_id);
    });
    const indegree = new Map(ids.map((id) => [id, predecessors.get(id).size]));
    const rank = new Map(ids.map((id) => [id, 0]));
    const queue = ids.filter((id) => indegree.get(id) === 0).sort(compare);
    const visited = [];
    while (queue.length) {
      const current = queue.shift();
      visited.push(current);
      Array.from(adjacency.get(current)).sort(compare).forEach((target) => {
        rank.set(target, Math.max(rank.get(target), rank.get(current) + 1));
        indegree.set(target, indegree.get(target) - 1);
        if (indegree.get(target) === 0) { queue.push(target); queue.sort(compare); }
      });
    }
    const cycleNodeIds = ids.filter((id) => !visited.includes(id)).sort(compare);
    const cycleRank = visited.length ? Math.max(...visited.map((id) => rank.get(id))) + 1 : 0;
    cycleNodeIds.forEach((id, index) => rank.set(id, cycleRank + index));
    const layersMap = new Map();
    ids.forEach((id) => {
      const value = rank.get(id);
      if (!layersMap.has(value)) layersMap.set(value, []);
      layersMap.get(value).push(id);
    });
    const ranks = Array.from(layersMap.keys()).sort((a, b) => a - b);
    ranks.forEach((value) => layersMap.get(value).sort((left, right) => {
      const average = (id) => {
        const parents = Array.from(predecessors.get(id)).filter((parent) => rank.get(parent) < value);
        if (!parents.length) return Infinity;
        return parents.reduce((sum, parent) => sum + (layersMap.get(rank.get(parent))?.indexOf(parent) ?? 0), 0) / parents.length;
      };
      return average(left) - average(right) || compare(left, right);
    }));
    const sizes = object(sizesValue);
    const nodeSize = (id) => ({
      width: Math.max(80, finite(object(sizes[id]).width, finite(options.nodeWidth, 220))),
      height: Math.max(48, finite(object(sizes[id]).height, finite(options.nodeHeight, 112))),
    });
    const margin = Math.max(0, finite(options.margin, 40));
    const layerGap = Math.max(24, finite(options.layerGap, 100));
    const rowGap = Math.max(16, finite(options.rowGap, 52));
    const direction = String(options.direction || "LR").toUpperCase();
    const positions = {};
    let primary = margin;
    let maximumSecondary = 0;
    for (const value of ranks) {
      const nodes = layersMap.get(value);
      const primarySize = Math.max(0, ...nodes.map((id) => direction === "TB" ? nodeSize(id).height : nodeSize(id).width));
      let secondary = margin;
      for (const id of nodes) {
        const size = nodeSize(id);
        positions[id] = direction === "TB" ? { x: secondary, y: primary } : { x: primary, y: secondary };
        secondary += (direction === "TB" ? size.width : size.height) + rowGap;
      }
      maximumSecondary = Math.max(maximumSecondary, secondary - rowGap + margin);
      primary += primarySize + layerGap;
    }
    const bounds = direction === "TB"
      ? { x: 0, y: 0, width: maximumSecondary, height: Math.max(0, primary - layerGap + margin) }
      : { x: 0, y: 0, width: Math.max(0, primary - layerGap + margin), height: maximumSecondary };
    return {
      positions,
      layers: ranks.map((value) => ({ rank: value, operator_ids: clone(layersMap.get(value)) })),
      bounds, hasCycle: cycleNodeIds.length > 0, cycleNodeIds,
    };
  }

  function semanticProjection(graphValue) {
    const graph = normalizeModelGraph(graphValue);
    const semantic = clone(graph);
    if (semantic.attributes) delete semantic.attributes.ui;
    semantic.operators.forEach((item) => { if (item.attributes) delete item.attributes.ui; });
    semantic.tensors.forEach((item) => { if (item.attributes) delete item.attributes.ui; });
    return semantic;
  }

  function uiProjection(graphValue) {
    const graph = normalizeModelGraph(graphValue);
    return clone(object(graph.attributes.ui));
  }

  function withUiProjection(graphValue, uiValue) {
    const graph = normalizeModelGraph(graphValue);
    graph.attributes = clone(object(graph.attributes));
    graph.attributes.ui = clone(object(uiValue));
    return normalizeModelGraph(graph);
  }

  function validateModelGraph(graphValue) {
    const graph = normalizeModelGraph(graphValue);
    const errors = [];
    const operators = new Set(graph.operators.map((item) => item.operator_id));
    const tensors = new Map(graph.tensors.map((item) => [item.tensor_id, item]));
    graph.operators.forEach((operator) => operator.ports.forEach((port) => {
      const tensor = tensors.get(port.tensor_id);
      if (!tensor) errors.push(`组件 ${operator.operator_id} 的端口 ${port.port_id} 引用了不存在的张量 ${port.tensor_id}。`);
      else if (stableStringify(normalizedContract(port)) !== stableStringify(normalizedContract(tensor))) errors.push(`组件 ${operator.operator_id} 端口 ${port.port_id} 维度/类型不匹配：期望 ${contractText(port)}，实际 ${contractText(tensor)}。`);
    }));
    graph.tensors.forEach((tensor) => {
      if (tensor.producer_operator_id && !operators.has(tensor.producer_operator_id)) errors.push(`张量 ${tensor.tensor_id} 的生产组件不存在。`);
      tensor.consumer_operator_ids.filter((id) => !operators.has(id)).forEach((id) => errors.push(`张量 ${tensor.tensor_id} 的消费组件不存在：${id}。`));
    });
    graph.transforms.forEach((item) => {
      if (!tensors.has(item.input_tensor_id) || !tensors.has(item.output_tensor_id)) errors.push(`显式变换 ${item.transform_id} 引用了不存在的张量。`);
      if (!TRANSFORM_KINDS.includes(item.kind)) errors.push(`显式变换 ${item.transform_id} 使用了不支持的类型 ${item.kind}。`);
    });
    const layout = layoutDag(graph);
    if (layout.hasCycle) errors.push(`模型组件图必须是 DAG；检测到环：${layout.cycleNodeIds.join("、")}。`);
    return { valid: errors.length === 0, errors };
  }

  function connectPorts(graph, sourceOperatorId, sourcePortId, targetOperatorId, targetPortId) {
    return connect(graph, { operator_id: sourceOperatorId, port_id: sourcePortId }, { operator_id: targetOperatorId, port_id: targetPortId });
  }

  function disconnectPort(graph, targetOperatorId, targetPortId) {
    return disconnect(graph, { operator_id: targetOperatorId, port_id: targetPortId });
  }

  function modelPortCompatibility(outputPort, inputPort) {
    const result = unifyTensorContracts(inputPort, outputPort);
    return { ...result, compatible: result.ok };
  }

  function updatePortContract(graphValue, operatorIdValue, portIdValue, patchValue = {}) {
    const graph = normalizeModelGraph(graphValue);
    const operatorId = String(operatorIdValue || "");
    const portId = String(portIdValue || "");
    const operator = graph.operators.find((item) => item.operator_id === operatorId);
    const port = operator?.ports.find((item) => item.port_id === portId);
    if (!operator || !port) throw new Error(`找不到要编辑的模型端口：${operatorId}.${portId}`);
    const patch = object(patchValue);
    const contract = {
      dtype: Object.hasOwn(patch, "dtype") ? canonicalDtype(patch.dtype) : port.dtype,
      shape: Object.hasOwn(patch, "shape") ? normalizeShape(patch.shape) : clone(port.shape),
      layout: Object.hasOwn(patch, "layout") ? canonicalLayout(patch.layout) : port.layout,
    };
    if (!contract.dtype || isWildcard(contract.dtype)) throw new Error("端口 DType 必须明确声明。");
    if (!contract.shape.length) throw new Error("端口 Shape 必须至少包含一个维度。");
    const tensor = graph.tensors.find((item) => item.tensor_id === port.tensor_id);
    if (!tensor) throw new Error(`端口 ${operatorId}.${portId} 引用了不存在的张量 ${port.tensor_id}。`);
    tensor.dtype = contract.dtype;
    tensor.shape = clone(contract.shape);
    tensor.layout = contract.layout;
    graph.operators.forEach((item) => item.ports.forEach((candidate) => {
      if (candidate.tensor_id !== tensor.tensor_id) return;
      candidate.dtype = contract.dtype;
      candidate.shape = clone(contract.shape);
      candidate.layout = contract.layout;
    }));
    const validation = validateModelGraph(graph);
    if (!validation.valid) throw new Error(validation.errors.join("\n"));
    return graph;
  }

  return Object.freeze({
    GRAPH_VERSION,
    TRANSFORM_KINDS,
    safeId,
    uniqueId,
    stableNodeId,
    stablePortId,
    stableTensorId,
    normalizeShape,
    canonicalDtype,
    canonicalLayout,
    normalizedContract,
    shapeText,
    contractText,
    unifyTensorContracts,
    modelPortCompatibility,
    updatePortContract,
    normalizeModelGraph,
    modelGraphAuthoringSummary,
    layerTemplate,
    groupRepeatedLayers,
    buildModelGraphFromLayerSpecs,
    graphToLayerSpecs,
    setLayerGroupOverride,
    inferTransformContracts,
    modelGraphEdges,
    overviewGroupDescriptor,
    overviewPatternPeriod,
    overviewResidualRoute,
    attentionLoweringProjection,
    authoritativeOperatorProjection,
    overviewBoundaryPortCounts,
    overviewNodeSize,
    overviewNodeSizes,
    overviewLayoutKey,
    overviewResponsiveLayout,
    overviewPortBezierPath,
    modelGraphRect,
    modelGraphAnchor,
    modelGraphPreferredSide,
    modelGraphSegmentHitsRect,
    modelGraphRoundedPath,
    modelGraphRouteEdge,
    mtpOverviewDescriptor,
    buildLeafOverviewProjection,
    buildRepeatGroupOverviewProjection,
    buildOverviewProjection,
    overviewPatternRanges,
    validateConnection,
    createConnectionPreview,
    updateConnectionPreview,
    connectionPreviewCompatibility,
    cancelConnectionPreview,
    commitConnectionPreview,
    connect,
    connectPorts,
    disconnect,
    disconnectPort,
    addNode,
    updateNode,
    deleteNode,
    duplicateNode,
    layoutDag,
    semanticProjection,
    uiProjection,
    withUiProjection,
    validateModelGraph,
  });
}));
