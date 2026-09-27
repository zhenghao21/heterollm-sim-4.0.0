"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.join(__dirname, "../src/heterollm_sim/webui");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const app = fs.readFileSync(path.join(root, "app.js"), "utf8");

test("hardware component preset editor exposes CRUD controls and unit-bearing fields", () => {
  for (const id of [
    "createComponentPresetButton", "componentPresetEditorDialog", "componentPresetEditorForm",
    "componentPresetEditorCapacity", "componentPresetEditorReadBandwidth", "componentPresetEditorWriteBandwidth",
    "componentPresetEditorPeakOps", "componentPresetEditorReadLatency", "componentPresetEditorWriteLatency",
    "componentPresetEditorPorts", "componentPresetEditorCostProfile",
  ]) assert.match(html, new RegExp(`id="${id}"`), id);
  assert.match(app, /data-edit-component-preset/);
  assert.match(app, /data-delete-component-preset/);
  assert.match(app, /method:\s*state\.editingComponentPresetId\s*\?\s*"PUT"\s*:\s*"POST"/);
  assert.match(app, /method:\s*"DELETE"/);
  assert.match(app, /costProfile\.read_bandwidth_gb_s\s*=\s*readBandwidth\s*\/\s*8/);
  assert.match(app, /gpuExternalMemory\s*=\s*normalizedComponentKind\(spec\.kind\)\s*===\s*"gpu"/);
  assert.match(app, /writeStatus\s*===\s*"not_published"/);
  assert.match(app, /writeLatencyStatus\s*===\s*"not_published"/);
  assert.match(html, /componentPresetEditorReadLatency[\s\S]*field-unit[^>]*>ns</u);
});

