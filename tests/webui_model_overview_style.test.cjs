"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const webui = path.join(__dirname, "..", "src", "heterollm_sim", "webui");
const html = fs.readFileSync(path.join(webui, "index.html"), "utf8");
const css = fs.readFileSync(path.join(webui, "styles.css"), "utf8");

function ruleBody(selector) {
  const match = css.match(new RegExp(`${selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*\\{([^}]*)\\}`, "s"));
  assert.ok(match, `missing CSS rule: ${selector}`);
  return match[1];
}

test("model overview toolbar keeps badge, status, and right tools in responsive source order", () => {
  const badge = html.indexOf('class="model-graph-mode-badge"');
  const status = html.indexOf('class="model-graph-status"');
  const tools = html.indexOf('class="model-graph-toolbar-group model-graph-toolbar-tools"');
  assert.ok(badge >= 0 && badge < status && status < tools);
  assert.match(html, /class="model-graph-status"[^>]*aria-live="polite"[^>]*title="No component selected/);
  assert.match(html, /data-i18n-title-zh="未选择组件 · 展开重复组查看完整权威模板/);

  assert.match(ruleBody(".model-graph-mode-switch"), /border-radius:\s*999px/);
  assert.match(ruleBody(".model-graph-mode-badge"), /min-height:\s*calc\(24px \* var\(--layout-scale\)\)/);
  assert.match(ruleBody(".model-graph-status"), /min-width:\s*0/);
  assert.match(ruleBody(".model-graph-status"), /white-space:\s*nowrap/);
  assert.match(ruleBody(".model-graph-status"), /text-overflow:\s*ellipsis/);
  assert.doesNotMatch(ruleBody(".model-graph-status"), /flex:\s*1 1 100%/);
  assert.match(ruleBody(".model-graph-toolbar-tools"), /flex:\s*0 0 auto/);
  assert.match(ruleBody(".model-graph-toolbar-tools"), /margin-inline-start:\s*auto/);
  assert.match(ruleBody(".model-graph-toolbar"), /flex-wrap:\s*nowrap/);
  assert.match(css, /@media \(max-width: 900px\) \{[\s\S]*?\.model-graph-toolbar \{[^}]*flex-wrap:\s*wrap/);
  assert.match(css, /data-font-band="extreme"\] \.model-graph-toolbar[^}]*flex-direction:\s*column/);
  assert.match(css, /data-font-band="extreme"\] \.model-graph-toolbar \{[^}]*flex-wrap:\s*nowrap/);
  assert.match(css, /@media \(max-width: 900px\)[\s\S]*?\.model-graph-status \{[^}]*white-space:\s*normal/);
  assert.match(css, /data-font-band="extreme"\] \.model-graph-status \{[^}]*white-space:\s*normal/);
});

test("model graph large font band stacks chrome and prevents horizontal overflow at 200% zoom", () => {
  assert.match(css, /html\[data-font-band="large"\] \.model-graph-shell\s*\{[^}]*grid-template-columns:\s*minmax\(0,1fr\)/s);
  assert.match(css, /html\[data-font-band="large"\] \.model-graph-toolbar\s*\{[^}]*align-items:\s*stretch[^}]*flex-direction:\s*column[^}]*flex-wrap:\s*nowrap/s);
  assert.match(css, /html\[data-font-band="large"\] \.model-graph-toolbar-group\s*\{[^}]*width:\s*100%/s);
  assert.match(css, /html\[data-font-band="large"\] \.model-graph-toolbar \.tool-button\s*\{[^}]*min-height:\s*40px[^}]*white-space:\s*normal/s);
  assert.match(css, /html\[data-font-band="large"\] \.model-graph-status\s*\{[^}]*width:\s*100%[^}]*overflow:\s*visible[^}]*text-overflow:\s*clip[^}]*white-space:\s*normal/s);
  assert.match(css, /html\[data-font-band="large"\] \.model-graph-status\s*\{[^}]*flex:\s*0 1 auto/s);
  assert.match(css, /html\[data-font-band="large"\] \.model-graph-inspector\s*\{[^}]*max-height:\s*none[^}]*border-top:\s*1px solid var\(--line\)[^}]*border-left:\s*0/s);
});

test("overview endpoints retain a transparent 14px hit target around only the 5px visible dot", () => {
  assert.match(ruleBody(".model-overview-port"), /width:\s*calc\(14px \* var\(--layout-scale\)\)/);
  assert.match(ruleBody(".model-overview-port"), /height:\s*calc\(14px \* var\(--layout-scale\)\)/);
  assert.match(ruleBody(".model-overview-port::before"), /inset:\s*0/);
  assert.match(ruleBody(".model-overview-port::before"), /border:\s*0/);
  assert.match(ruleBody(".model-overview-port::before"), /background:\s*transparent/);
  assert.match(ruleBody(".model-overview-port i"), /width:\s*calc\(5px \* var\(--layout-scale\)\)/);
  assert.match(ruleBody(".model-overview-port i"), /height:\s*calc\(5px \* var\(--layout-scale\)\)/);
});

test("selected overview one-hop edges distinguish source-to-target direction and respect reduced motion", () => {
  assert.match(css, /:is\(\.model-graph-edge\.is-overview,\.model-inline-edge\)\.is-one-hop\.is-incoming\s*\{[^}]*stroke:\s*#e19a5f/s);
  assert.match(css, /:is\(\.model-graph-edge\.is-overview,\.model-inline-edge\)\.is-one-hop\.is-outgoing\s*\{[^}]*stroke:\s*#62cbd4/s);
  assert.match(css, /@keyframes model-overview-edge-flow\s*\{[^}]*stroke-dashoffset:\s*-32/s);
  assert.match(css, /prefers-reduced-motion:\s*reduce[\s\S]*?:is\(\.model-graph-edge\.is-overview,\.model-inline-edge\)\.is-one-hop\s*\{[^}]*stroke-dasharray:\s*none/s);
  assert.match(css, /data-reduce-motion="true"\] :is\(\.model-graph-edge\.is-overview,\.model-inline-edge\)\.is-one-hop\s*\{[^}]*stroke-dasharray:\s*none/s);
});

test("repeat group controls and dashed repeat rails do not cover selectable structure", () => {
  assert.match(ruleBody(".model-graph-edge-layer"), /z-index:\s*1/);
  assert.match(ruleBody(".model-graph-edge-layer"), /pointer-events:\s*none/);
  assert.match(css, /\.model-graph-node-layer\s*\{[^}]*z-index:\s*3/s);
  assert.match(ruleBody(".model-overview-ports"), /z-index:\s*7/);
  assert.match(ruleBody(".model-overview-ports"), /pointer-events:\s*none/);
  assert.match(ruleBody(".model-overview-port"), /pointer-events:\s*auto/);

  const toggle = ruleBody(".model-overview-group-toggle");
  assert.match(toggle, /width:\s*calc\(24px \* var\(--layout-scale\)\)/);
  assert.match(toggle, /flex:\s*0 0 calc\(24px \* var\(--layout-scale\)\)/);
  assert.doesNotMatch(toggle, /position:\s*absolute/);

  const repeatRail = ruleBody(".model-graph-edge.is-visual-repeat");
  assert.match(repeatRail, /stroke-dasharray:\s*7 5/);
  assert.match(repeatRail, /pointer-events:\s*none/);
});

test("leaf nodes are rounded and all semantic kind families use theme-derived accents", () => {
  for (const selector of [".model-overview-node", ".model-inline-operator", ".model-graph-node"]) {
    assert.match(ruleBody(selector), /border-radius:\s*calc\(6px \* var\(--layout-scale\)\)/);
    assert.match(ruleBody(selector), /border:\s*1px solid color-mix\(in srgb, var\(--model-kind-accent\) 82%, var\(--line\)\)/);
  }

  for (const kind of [
    "kind-embedding", "kind-lm_head", "kind-rms_norm", "kind-attention",
    "kind-linear_attention", "kind-dense_mlp", "kind-linear", "kind-moe_router",
    "kind-residual_add", "kind-mtp_prediction_layer", "kind-transform", "kind-cast",
  ]) assert.ok(css.includes(`.${kind}`), `missing semantic family member ${kind}`);

  assert.match(css, /\.is-selected \{ border-color: var\(--cyan-bright\)/);
  assert.match(css, /:hover:not\(\.is-selected\)/);
});
