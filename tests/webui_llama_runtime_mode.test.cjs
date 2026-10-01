"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
function helpers() {
  const context = vm.createContext({ console, document: { addEventListener() {} }, Intl,
    localStorage: { getItem() { return null; } }, setTimeout, clearTimeout,
    ModelGraphCore: require(path.join(webui, "model-graph-core.js")),
    TopologyCore: require(path.join(webui, "topology-core.js")),
    TraceViewCore: require(path.join(webui, "trace-view-core.js")),
  });
  vm.runInContext(fs.readFileSync(path.join(webui, "app.js"), "utf8") +
    ";globalThis.h = { state, llamaRuntimeDefaults, setLlamaRuntimeMode, updateLlamaRuntimeField, llamaRuntimeControlsMarkup, clearLlamaRuntimeExposure, mappingImpactView };", context);
  return context.h;
}
function scenario() {
  return { profiles: { runtime: { keep: true } }, workload: {
    requests: [{ prompt_tokens: 8192, output_tokens: 1024 }],
    scheduler: { max_num_seqs: 4, max_num_batched_tokens: 2048, max_num_ubatch_tokens: 512 },
    metadata: { llama_cpp_runtime: { old: true }, llama_cpp_slot_order_contract: { source: true } },
  }, placement: { metadata: { llama_cpp_kv_layer_components: { old: "hbm" }, control_plane: {
    policy: { options: {} }, decision: { old: true }, evidence: { old: true },
  } } } };
}
test("llama mode serializes workload-derived runtime knobs and direct HBM/HBF tiering", () => {
  const h = helpers(), s = scenario();
  h.setLlamaRuntimeMode("llama_cpp", s);
  assert.deepEqual(JSON.parse(JSON.stringify(s.profiles.llama_cpp)), { policy: "llama_cpp", gpu_layers: -1, batch: 2048, ubatch: 512, context: 9216, parallel: 4, offload_kqv: true, device_memory_tiering: true });
  assert.equal(s.profiles.runtime.keep, true);
  assert.equal(s.workload.metadata.llama_cpp_runtime, undefined);
  assert.equal(s.placement.metadata.control_plane.decision, undefined);
  assert.equal(s.workload.metadata.llama_cpp_slot_order_contract.source, true);
  assert.equal(h.mappingImpactView(s).profiles.llama_cpp.device_memory_tiering, true);
  h.state.scenario = s;
  assert.match(h.llamaRuntimeControlsMarkup(), /HBM\/HBF统一显存/);
  assert.match(h.llamaRuntimeControlsMarkup(), /data-llama-runtime-field="context" value="9216"/);
  h.setLlamaRuntimeMode("auto", s);
  assert.equal(s.profiles.llama_cpp, undefined);
  assert.doesNotMatch(h.llamaRuntimeControlsMarkup(), /data-llama-runtime-field=/);
});
test("runtime fields reject invalid limits and batch reductions reconcile ubatch", () => {
  const h = helpers(), s = scenario(); h.setLlamaRuntimeMode("llama_cpp", s);
  assert.equal(h.updateLlamaRuntimeField("batch", 128, s), true);
  assert.equal(s.profiles.llama_cpp.ubatch, 128);
  for (const [field, value] of [["ubatch", 129], ["gpu_layers", -2], ["parallel", 1.5], ["context", 0], ["context", 4096], ["device_memory_tiering", "false"]]) assert.equal(h.updateLlamaRuntimeField(field, value, s), false);
  assert.equal(h.updateLlamaRuntimeField("offload_kqv", false, s), true);
  assert.equal(s.profiles.llama_cpp.offload_kqv, false);
});

test("RTX5080 llama mode requests analytical kernels without enabling other GPUs", () => {
  const h=helpers(), s=scenario();
  s.hardware={components:[{metadata:{component_preset_id:"nvidia-rtx-5080"}}]};
  h.setLlamaRuntimeMode("llama_cpp",s);
  assert.equal(s.workload.metadata.llama_cpp_kernel_model_preset,"blackwell_analytical_v1");
  h.setLlamaRuntimeMode("auto",s);
  assert.equal(s.workload.metadata.llama_cpp_kernel_model_preset,undefined);
});
