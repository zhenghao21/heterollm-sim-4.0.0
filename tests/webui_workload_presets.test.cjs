const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const app = fs.readFileSync(
  path.join(__dirname, "..", "src", "heterollm_sim", "webui", "app.js"),
  "utf8",
);
const reference = fs.readFileSync(
  path.join(__dirname, "..", "src", "heterollm_sim", "reference.py"),
  "utf8",
);
const web = fs.readFileSync(
  path.join(__dirname, "..", "src", "heterollm_sim", "web.py"),
  "utf8",
);

test("workload presets include the llama baseline and core AI application shapes", () => {
  for (const id of [
    "llama_cpp_default",
    "personal_assistant",
    "enterprise_office",
    "customer_service",
    "education",
    "content_creation",
    "software_development",
    "search_platform",
    "conversational_commerce",
    "edge_personal_assistant",
  ]) {
    assert.match(app, new RegExp(`id: "${id}"`));
  }
  assert.match(app, /WORKLOAD_PRESET_SOURCE/);
  assert.match(app, /function applyWorkloadPreset/);
  assert.match(app, /workload_preset_source_scenario/);
  assert.match(app, /promptTokens: 24576/);
  assert.match(app, /source: "llama\.cpp runtime config defaults \+ explicit smoke shape"/);
  assert.match(app, /apiRequest\("\/workload-presets"/);
  assert.match(app, /function loadWorkloadPresetCatalog\(\)/);
  assert.match(web, /path == "\/api\/workload-presets"/);
  assert.match(web, /path\.startswith\("\/api\/workload-presets\/"\)/);
});

test("minimal imported workloads fall back to the llama-aligned request defaults", () => {
  assert.match(app, /scenario\.workload\.prompt_tokens \?\?= 512/);
  assert.match(app, /scenario\.workload\.output_tokens \?\?= 128/);
  assert.match(app, /scheduler\.mode \?\?= "continuous"/);
  assert.match(app, /scheduler\.max_num_batched_tokens \?\?= 512/);
  assert.match(app, /scheduler\.max_num_ubatch_tokens \?\?= 512/);
  assert.match(app, /scheduler\.preemption_enabled \?\?= false/);
});

test("preset application resets scheduler and runtime metadata contracts", () => {
  assert.match(app, /mixed_phase_batching: false/);
  assert.match(app, /phase_candidate_order: "least_recently_served"/);
  assert.match(app, /prefill_stop_offsets: \[\]/);
  assert.match(app, /arrival_rate_rps: 0/);
  assert.match(app, /"llama_cpp_effective_scheduler_policy"/);
  assert.match(app, /preset\.source \|\| WORKLOAD_PRESET_SOURCE/);
  assert.match(app, /delete metadata\[field\]/);
});

test("manual workload edits clear the selected preset marker", () => {
  assert.match(app, /function markWorkloadChanged\(message = ""\)/);
  assert.match(app, /delete metadata\[field\]/);
  assert.match(app, /workload_preset_source_scenario/);
});

test("the API reference endpoint serves the llama-aligned default scenario", () => {
  assert.match(reference, /def build_llama_default_scenario\(\)/);
  assert.match(web, /scenario_to_payload\(build_llama_default_scenario\(\)\)/);
});
