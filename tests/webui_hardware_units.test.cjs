"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const appPath = path.join(__dirname, "..", "src", "heterollm_sim", "webui", "app.js");
const app = fs.readFileSync(appPath, "utf8");
const css = fs.readFileSync(path.join(__dirname, "..", "src", "heterollm_sim", "webui", "styles.css"), "utf8");

function helpers() {
  const context = vm.createContext({
    AbortController,
    CSS: { escape: String },
    Intl,
    ModelGraphCore: {},
    Promise,
    TopologyCore: {},
    TraceViewCore: {},
    URL,
    clearTimeout,
    console,
    document: { addEventListener() {} },
    localStorage: { getItem() { return null; }, removeItem() {}, setItem() {} },
    setTimeout,
  });
  vm.runInContext(`${app}\n;globalThis.__hardwareUnits = {
    formatUnit,
    formatRatioPercent,
    formatBytes,
    formatResultBytes,
    parseBytes,
    parseOps,
    parseBandwidthToGbps,
    componentPresetListText,
    componentPresetFacts,
    costProfileFieldUnit,
    costProfileNumberField,
    metadataField,
  };`, context);
  return context.__hardwareUnits;
}

test("hardware unit helper renders scaled values with an explicit suffix", () => {
  const ui = helpers();
  assert.equal(ui.formatUnit(1.25, "GHz"), "1.25 GHz");
  assert.equal(ui.formatRatioPercent(0.875), "87.5 %");
  assert.equal(ui.formatUnit(Number.NaN, "W"), "—");
  assert.equal(ui.formatBytes(1024), "1.024 KB");
  assert.equal(ui.formatResultBytes(1024).text, "1.024 KB");
  assert.equal(ui.parseBytes("1.024 KB"), 1024);
  assert.equal(ui.parseBytes("1.024 TB"), 1_024_000_000_000);
  assert.equal(ui.parseBandwidthToGbps("12 GB/s"), 96);
  assert.equal(ui.parseOps("2.25 POPS"), 2.25e15);
  assert.equal(ui.costProfileFieldUnit("tensor_core.frequency_ghz"), "GHz");
  assert.equal(ui.costProfileFieldUnit("occupancy"), "0–1");
  assert.match(ui.costProfileNumberField("频率（GHz）", "gpu", "pipeline.frequency_ghz", 3.2), /field-unit[^>]*>GHz/u);
});

test("component preset facts keep units for derived numeric metadata", () => {
  const ui = helpers();
  assert.equal(ui.componentPresetListText(1024, "", "capacity_bytes"), "1.024 KB");
  assert.equal(ui.componentPresetListText(2.5e12, "", "peak_ops_per_s"), "2.5 TOPS");
  assert.equal(ui.componentPresetListText(3.2, "", "frequency_ghz"), "3.2 GHz");
  assert.equal(ui.componentPresetListText(0.92, "", "efficiency"), "0.92 (0–1)");
  assert.equal(ui.componentPresetListText(0.8, "", "read_latency_ns"), "0.8 ns");
});

test("hardware inspector labels distinguish raw Gb/s and ratio percent fields", () => {
  const ui = helpers();
  assert.match(ui.metadataField("DMA 带宽（DMA Bandwidth）", "dma_bandwidth_gbps", 128, { unit: "Gb/s" }), /field-unit[^>]*>Gb\/s/u);
  assert.match(app, /DMA 带宽（DMA Bandwidth, Gb\/s）/u);
  assert.match(app, /命令队列深度（depth，2 的幂）/u);
  assert.match(app, /最大并发请求（Max Outstanding Requests, depth）/u);
  assert.match(app, /cycles_per_mma（Derived, cycles\/MMA）/u);
  assert.match(app, /物理 Plane 数（count，可选）/u);
  assert.match(app, /频率比例（%）/u);
  assert.match(app, /Memory bandwidth scale \(%\)/u);
  assert.match(css, /\.field-input-with-unit\s*\{[\s\S]*display:\s*flex/u);
  assert.match(css, /\.field-unit\s*\{[\s\S]*white-space:\s*nowrap/u);
});

console.log("hardware unit display contract: ok");
