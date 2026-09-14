"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const css = fs.readFileSync(
  path.join(__dirname, "..", "src", "heterollm_sim", "webui", "styles.css"),
  "utf8",
);

function cssBodiesContaining(selectorFragment) {
  const bodies = [];
  for (const match of css.matchAll(/([^{}]+)\{([^{}]*)\}/gu)) {
    if (match[1].includes(selectorFragment)) bodies.push(match[2]);
  }
  return bodies;
}

function assertCssContract(selectorFragment, patterns) {
  const bodies = cssBodiesContaining(selectorFragment);
  assert.ok(bodies.length, `missing CSS selector containing ${selectorFragment}`);
  for (const pattern of patterns) {
    assert.ok(
      bodies.some((body) => pattern.test(body)),
      `no ${selectorFragment} rule matched ${pattern}`,
    );
  }
}

test("extreme settings keep header and footer fixed around the scrolling content row", () => {
  assertCssContract(':root[data-font-band="extreme"] .settings-shell', [
    /display:\s*grid/u,
    /grid-template-rows:\s*auto\s+minmax\(0,\s*1fr\)\s+auto/u,
    /overflow:\s*hidden/u,
  ]);
  assertCssContract(':root[data-font-band="extreme"] .settings-content', [
    /min-height:\s*0/u,
    /overflow-y:\s*auto/u,
    /overscroll-behavior:\s*contain/u,
  ]);
  assertCssContract(':root[data-font-band="extreme"] .settings-shell > .dialog-head', [
    /grid-template-columns:\s*minmax\(0,\s*1fr\)\s+auto/u,
  ]);
});

test("desktop settings groups opt out of implicit grid-row stretching", () => {
  assertCssContract(".settings-content", [/align-content:\s*start/u, /align-items:\s*start/u]);
  assertCssContract(".settings-group", [/align-self:\s*start/u]);
});

test("short desktop results compact the pre-chart stack without clipping metrics", () => {
  assert.match(css, /@media\s*\(min-width:\s*1000px\)\s*and\s*\(max-height:\s*760px\)/u);
  assertCssContract(".results-view .run-manifest-bar", [
    /min-height:\s*calc\(32px\s*\*\s*var\(--layout-scale\)\)/u,
    /margin-bottom:\s*calc\(6px\s*\*\s*var\(--layout-scale\)\)/u,
  ]);
  assertCssContract(".results-view .metric-cell", [/min-height:\s*0/u, /padding:\s*6px\s+8px/u]);
  assertCssContract(".results-view .component-timeseries-panel > .section-heading", [
    /min-height:\s*calc\(38px\s*\*\s*var\(--layout-scale\)\)/u,
  ]);
  assertCssContract(".results-view .timeseries-chart-controls", [
    /grid-template-columns:\s*minmax\(0,\s*\.9fr\)[^;]*minmax\(86px,\s*\.5fr\)\s+auto/u,
    /padding:\s*calc\(5px\s*\*\s*var\(--layout-scale\)\)/u,
  ]);
  assertCssContract(".results-view .timeseries-curve-control", [/grid-column:\s*auto/u]);

  const metricRules = cssBodiesContaining(".results-view .metric-cell").join("\n");
  assert.doesNotMatch(metricRules, /overflow:\s*hidden|white-space:\s*nowrap|text-overflow:\s*ellipsis/u);
});

test("metric labels use readable sentence case and a compact percentile subscript", () => {
  assertCssContract(".metric-cell .label", [
    /font-family:\s*var\(--body\)/u,
    /letter-spacing:\s*\.015em/u,
    /text-transform:\s*none/u,
  ]);
  assertCssContract(".metric-cell .label sub", [
    /font-size:\s*\.72em/u,
    /letter-spacing:\s*0/u,
    /line-height:\s*0/u,
  ]);
});

test("trace filter, page, and narrative bands are compact without hiding controls", () => {
  assertCssContract(".trace-filter-bar", [/gap:\s*calc\(6px/u, /padding:\s*calc\(6px/u]);
  assertCssContract(".playback-view .trace-page-bar", [/gap:\s*calc\(6px/u, /padding:\s*calc\(5px/u]);
  assertCssContract(".trace-narrative", [/padding:\s*calc\(8px/u, /border-color:\s*var\(--line\)/u]);
  assertCssContract(".trace-narrative-primary", [/margin-top:\s*calc\(4px/u, /line-height:\s*1\.4/u]);

  for (const selector of [".trace-filter-bar", ".playback-view .trace-page-bar", ".trace-narrative"]) {
    const rules = cssBodiesContaining(selector).join("\n");
    assert.doesNotMatch(rules, /display:\s*none|visibility:\s*hidden/u, `${selector} must remain visible`);
  }
});

test("desktop top-bar action words do not fracture under grid pressure", () => {
  assertCssContract(".top-actions .button", [
    /white-space:\s*nowrap/u,
    /overflow-wrap:\s*normal/u,
  ]);
  assertCssContract(':root[data-font-band="large"] .top-actions .button', [
    /white-space:\s*normal/u,
  ]);
});
