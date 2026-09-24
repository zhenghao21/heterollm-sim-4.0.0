"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const css = fs.readFileSync(
  path.join(__dirname, "..", "src", "heterollm_sim", "webui", "styles.css"),
  "utf8",
);
const marker = "/* Runtime policy and contract controls";
const runtimeStart = css.lastIndexOf(marker);
assert.ok(runtimeStart >= 0, "runtime policy styles must have a final cascade block");
const runtime = css.slice(runtimeStart);

test("runtime policy styles are in the final cascade and preserve the dark UI contract", () => {
  assert.equal(runtimeStart, css.indexOf(marker), "there must be one final runtime policy block");
  assert.match(runtime, /details:is\([^)]*control-plane-policy-section[^)]*\)/u);
  assert.match(runtime, /background:\s*var\(--surface-raised\)/u);
  assert.match(runtime, /border:\s*1px\s+solid\s+var\(--line-soft\)/u);
});

test("contract details expose an explicit arrow and capped summary typography", () => {
  assert.match(runtime, /> summary \{[\s\S]*?font-size:\s*max\(\.625rem,\s*12px\)/u);
  assert.match(runtime, /> summary::-webkit-details-marker \{\s*display:\s*none;/u);
  assert.match(runtime, /> summary::before \{[\s\S]*?content:\s*"▸";/u);
  assert.match(runtime, /\[open\] > summary::before \{\s*content:\s*"▾";/u);
  assert.match(runtime, /> summary :is\(strong, \.control-section-title\) \{[\s\S]*?font-size:\s*inherit;/u);
});

test("forms keep usable dimensions and checkbox copy stays as one labelled group", () => {
  assert.match(runtime, /\.field input,[\s\S]*?min-height:\s*calc\(36px \* var\(--layout-scale\)\);/u);
  assert.match(runtime, /min-height:\s*max\(calc\(36px \* var\(--layout-scale\)\),\s*36px\)/u);
  assert.match(runtime, /\.field > span,[\s\S]*?font-size:\s*max\(\.625rem,\s*12px\)/u);
  assert.match(runtime, /\.checkbox-field > span \{[\s\S]*?display:\s*grid;[\s\S]*?overflow-wrap:\s*anywhere;/u);
  assert.match(runtime, /\.checkbox-field > span > strong,[\s\S]*?display:\s*block;/u);
  assert.match(runtime, /\.checkbox-field > span > small \{[\s\S]*?display:\s*block;/u);
});

test("narrow Inspectors collapse the two-field grid and keep target rows bounded", () => {
  assert.match(
    runtime,
    /\.field-grid-2 \{[\s\S]*?grid-template-columns:\s*repeat\(auto-fit,\s*minmax\(min\(100%,\s*calc\(220px \* var\(--layout-scale\)\)\),\s*1fr\)\)/u,
  );
  assert.match(runtime, /\.placement-target-row \{[\s\S]*?grid-template-columns:\s*minmax\(0,\s*\.9fr\)\s+minmax\(0,\s*1\.1fr\)\s+auto;/u);
  assert.match(runtime, /\.placement-target-row \.field > :is\(input, select\) \{[\s\S]*?width:\s*100%;/u);
  assert.match(runtime, /\.placement-target-delete \{[\s\S]*?min-height:\s*max\(calc\(36px \* var\(--layout-scale\)\),\s*36px\)/u);
});

test("360px and 560px layouts move the fixed rail into a horizontal step navigator", () => {
  const narrow = runtime.slice(runtime.indexOf("@media (max-width: 680px)"));
  assert.match(narrow, /\.workbench \{[\s\S]*?grid-template-columns:\s*minmax\(0,\s*1fr\);[\s\S]*?grid-template-rows:\s*auto\s+minmax\(0,\s*1fr\)/u);
  assert.match(narrow, /\.step-rail \{[\s\S]*?flex-direction:\s*row;[\s\S]*?overflow-x:\s*auto;[\s\S]*?border-right:\s*0;/u);
  assert.match(narrow, /\.step-list \{[\s\S]*?display:\s*flex;[\s\S]*?min-width:\s*max-content;/u);
  assert.match(narrow, /\.step-button \{[\s\S]*?width:\s*auto;[\s\S]*?border-left:\s*0;/u);
  assert.match(runtime, /@media \(max-width: 560px\)/u);
  assert.match(narrow, /\.main-stage,[\s\S]*?\.view \{[\s\S]*?width:\s*100%;[\s\S]*?min-width:\s*0;/u);
});

test("large font bands keep the evidence and dirty badges on the header row", () => {
  assert.match(runtime, /:root:is\(\[data-font-band="large"\],\s*\[data-font-band="extreme"\]\) \.topbar \{[\s\S]*?grid-template-columns:\s*minmax\(0,\s*1fr\)\s+auto;/u);
  assert.match(runtime, /\.brand-block \{[\s\S]*?flex-wrap:\s*nowrap;/u);
  assert.match(runtime, /\.evidence-badge,[\s\S]*?\.dirty-mark \{[\s\S]*?flex:\s*0 0 auto;[\s\S]*?white-space:\s*nowrap;/u);
  assert.match(runtime, /\.top-actions \{[\s\S]*?grid-column:\s*1 \/ -1;/u);
});
