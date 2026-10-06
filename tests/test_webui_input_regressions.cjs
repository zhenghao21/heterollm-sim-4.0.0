const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const core = require(path.join(root, "src/heterollm_sim/webui/model-graph-core.js"));

function loadApp() {
  const source = fs.readFileSync(path.join(root, "src/heterollm_sim/webui/app.js"), "utf8");
  const context = {
    console,
    document: { addEventListener() {} },
    TopologyCore: {},
    ModelGraphCore: core,
    TraceViewCore: {},
    localStorage: { getItem() { return null; }, setItem() {}, removeItem() {} },
    window: { addEventListener() {}, setTimeout, clearTimeout },
    setTimeout,
    clearTimeout,
    URL,
    JSON,
    Math,
    Number,
    String,
    Object,
    Array,
    Map,
    Set,
    Date,
    RegExp,
    Promise,
    Intl,
    navigator: {},
  };
  context.globalThis = context;
  const suffix = `
    globalThis.__webuiTest = {
      state,
      dom,
      addRequest,
      updateRequestField,
      renderWorkload,
      setStubs(changed, notices) {
        markWorkloadChanged = changed;
        toast = (...args) => notices.push(args);
        renderRequestTable = () => {};
        hydrateConceptHelp = () => {};
      },
    };
  `;
  vm.createContext(context);
  vm.runInContext(source + suffix, context, { timeout: 30_000 });
  return context.__webuiTest;
}

function workload(prompt_tokens = 512, output_tokens = 128) {
  return {
    name: "test",
    requests: [],
    request_count: 1,
    prompt_tokens,
    output_tokens,
    arrival_rate_rps: 0,
    random_seed: 0,
    scheduler: {
      mode: "continuous",
      max_num_seqs: 1,
      max_num_batched_tokens: 512,
      prefill_chunk_tokens: 512,
      preemption_enabled: false,
    },
    mtp: null,
    metadata: {},
  };
}

function graphWithPorts(firstContract, secondContract) {
  return {
    graph_id: "contract-test",
    tensors: [{
      tensor_id: "x",
      role: "activation",
      producer_operator_id: "source",
      consumer_operator_ids: ["sink"],
    }],
    operators: [
      {
        operator_id: "source",
        op_kind: "producer",
        ports: [{ port_id: "out", direction: "output", tensor_id: "x", ...firstContract }],
      },
      {
        operator_id: "sink",
        op_kind: "consumer",
        ports: [{ port_id: "in", direction: "input", tensor_id: "x", ...secondContract }],
      },
    ],
  };
}

test("model graph rejects conflicting port contracts when tensor contract is absent", () => {
  for (const [field, first, second] of [
    ["dtype", "fp16", "int64"],
    ["shape", ["B", "T", "H"], ["B", "T"]],
    ["layout", "logical", "row_major"],
  ]) {
    const graph = graphWithPorts({ [field]: first }, { [field]: second });
    const result = core.validateModelGraph(graph);
    assert.equal(result.valid, false, `${field} conflict should fail closed`);
    assert.match(result.errors.join("\n"), new RegExp(`契约 ${field}`));
  }
});

test("model graph still accepts matching port contracts and fills the omitted tensor contract", () => {
  const graph = graphWithPorts(
    { dtype: "fp16", shape: ["B", "T", "H"], layout: "logical" },
    { dtype: "fp16", shape: ["B", "T", "H"], layout: "logical" },
  );
  assert.equal(core.validateModelGraph(graph).valid, true);
  const normalized = core.normalizeModelGraph(graph);
  assert.deepEqual(normalized.tensors[0].dtype, "fp16");
  assert.deepEqual(normalized.tensors[0].shape, ["B", "T", "H"]);
});

test("new explicit request inherits current synthetic token defaults", () => {
  const api = loadApp();
  api.state.scenario = { workload: workload(2048, 37) };
  let changed = 0;
  api.setStubs(() => { changed += 1; }, []);
  api.addRequest();
  assert.equal(api.state.scenario.workload.requests[0].prompt_tokens, 2048);
  assert.equal(api.state.scenario.workload.requests[0].output_tokens, 37);
  assert.equal(changed, 1);
});

test("blank workload number is rejected and arrival rate is exposed", () => {
  const api = loadApp();
  api.state.scenario = { workload: workload() };
  const notices = [];
  let changed = 0;
  const form = {
    innerHTML: "",
    querySelector(selector) {
      if (selector === "#workloadPresetSelect") return { addEventListener() {} };
      if (selector === "#mtpEnabledInput") return { addEventListener() {} };
      return null;
    },
    querySelectorAll(selector) {
      return selector === "[data-workload-field]" ? [this.promptControl, this.arrivalControl] : [];
    },
    promptControl: {
      type: "number",
      value: "",
      dataset: { workloadField: "prompt_tokens" },
      addEventListener(type, handler) { this[type] = handler; },
    },
    arrivalControl: {
      type: "number",
      value: "2.5",
      dataset: { workloadField: "arrival_rate_rps" },
      addEventListener(type, handler) { this[type] = handler; },
    },
  };
  api.dom.workloadMetaForm = form;
  api.setStubs(() => { changed += 1; }, notices);
  api.renderWorkload();
  assert.match(form.innerHTML, /arrival_rate_rps/);
  form.arrivalControl.change();
  assert.equal(api.state.scenario.workload.arrival_rate_rps, 2.5);
  assert.equal(changed, 1);
  form.promptControl.change();
  assert.equal(api.state.scenario.workload.prompt_tokens, 512);
  assert.equal(changed, 1);
  assert.equal(notices.length, 1);
});

test("empty deadline remains nullable while invalid request numbers are rejected", () => {
  const api = loadApp();
  api.state.scenario = { workload: workload() };
  api.state.scenario.workload.requests.push({ request_id: "r0", arrival_ns: 0, prompt_tokens: 8, output_tokens: 2, priority: 0, deadline_ns: 10 });
  const notices = [];
  let changed = 0;
  api.dom.requestTableBody = { innerHTML: "" };
  api.setStubs(() => { changed += 1; }, notices);
  api.updateRequestField(0, { type: "number", value: "", dataset: { requestField: "deadline_ns" } });
  assert.equal(api.state.scenario.workload.requests[0].deadline_ns, null);
  assert.equal(changed, 1);
  api.updateRequestField(0, { type: "number", value: "", dataset: { requestField: "prompt_tokens" } });
  assert.equal(api.state.scenario.workload.requests[0].prompt_tokens, 8);
  assert.equal(changed, 1);
  assert.equal(notices.length, 1);
});
