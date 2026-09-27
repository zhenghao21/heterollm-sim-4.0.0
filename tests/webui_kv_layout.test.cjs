const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const app = fs.readFileSync(path.join(__dirname, '..', 'src', 'heterollm_sim', 'webui', 'app.js'), 'utf8');
const css = fs.readFileSync(path.join(__dirname, '..', 'src', 'heterollm_sim', 'webui', 'styles.css'), 'utf8');

test('KV residency modes preserve legacy fields and expose experimental pool explicitly', () => {
  assert.match(app, /kvPolicy\.layout_mode \?\?= "auto"/);
  assert.match(app, /\["auto", uiText\("跟随 llama\.cpp layer placement"/);
  assert.match(app, /\["fixed", uiText\("固定单组件"/);
  assert.match(app, /\["manual", uiText\("手动按 layer 指定"/);
  assert.match(app, /\["paged_pool", uiText\("动态 KV Pool（实验性）"/);
  assert.match(app, /data-placement-field="pool_components"/);
  assert.match(app, /不会把多个组件自动合并成统一容量池/);
});

test('runtime KV analysis reads component and layer accounting fields', () => {
  assert.match(app, /kv_capacity_bytes_by_component/);
  assert.match(app, /kv_used_bytes_by_component/);
  assert.match(app, /kv_component_owner_layers/);
  assert.match(app, /kv_layer_components \|\| kvRoot\.kv_layer_owner/);
  assert.match(app, /kv_batch_retry_count/);
  assert.match(app, /data-runtime-section="kv-analysis"/);
  assert.match(app, /运行事件：批次重试 \{retry\}；上下文偏移 \{shift\}；空闲槽位清理 \{idle\}/);
});

test('runtime KV analysis gets a full-width row so its tables do not collapse into a metric column', () => {
  assert.match(css, /\.runtime-kv-analysis\s*\{[\s\S]*?grid-column:\s*1\s*\/\s*-1;/);
  assert.match(css, /\.runtime-kv-analysis \.data-table\s*\{[\s\S]*?min-width:\s*620px;/);
});

test('runtime summary separates overview cards from detail cards before the KV tables', () => {
  assert.match(app, /const runtimeOverviewMarkup = \[/);
  assert.match(app, /const runtimeDetailsMarkup = \[/);
  assert.match(app, /runtime-card-grid runtime-grid runtime-overview-grid/);
  assert.match(app, /runtime-card-grid runtime-grid runtime-details-grid/);
  assert.match(css, /#runtimeSummary\s*>\s*\.runtime-card-grid\.runtime-overview-grid\s*\{[\s\S]*?repeat\(4,/);
  assert.match(css, /#runtimeSummary\s*>\s*\.runtime-card-grid\.runtime-details-grid\s*\{[\s\S]*?repeat\(3,/);
});

test('KV data rows do not inherit table header styling', () => {
  assert.match(app, /return `<tr><td>\$\{escapeHtml\(component\)\}<\/td><td>/);
  assert.match(app, /kvLayerOwner\)\.map\(\(layer\) => `<tr><td>/);
});

test('automatic concept help preserves table-cell layout without requiring a table-specific fix', () => {
  const sharedRule = css.match(/th\.concept-help-title,\s*td\.concept-help-title\s*\{([^}]+)\}/);
  assert.ok(sharedRule, 'help-enabled cells need a shared table layout rule');
  assert.match(sharedRule[1], /display:\s*table-cell;/);
  assert.match(sharedRule[1], /max-width:\s*none;/);
});

test('parallel and KV controls use stable columns and keep manual content full width', () => {
  assert.match(css, /#placementControls \.parallel-grid\s*\{[\s\S]*?grid-template-columns:\s*repeat\(3,\s*minmax\(0,\s*1fr\)\)/);
  assert.match(css, /#placementControls \.kv-grid\s*\{[\s\S]*?grid-template-columns:\s*minmax\(0,\s*1\.35fr\)\s+minmax\(0,\s*1fr\)\s+minmax\(0,\s*\.9fr\)/);
  assert.match(css, /#placementControls \.kv-grid > \.span-all,[\s\S]*?grid-column:\s*1\s*\/\s*-1;/);
});

test('workload form has stable desktop tracks and responsive collapse rules', () => {
  assert.match(css, /#view-workload \.workload-config-grid\s*\{[\s\S]*?grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/);
  assert.match(css, /data-workload-section="continuous-batching"[^}]+?\.workload-field-grid\s*\{[\s\S]*?grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/);
  assert.match(css, /@media \(max-width:\s*560px\)[\s\S]*?#placementControls \.parallel-grid,[\s\S]*?grid-template-columns:\s*minmax\(0,\s*1fr\)/);
  assert.match(css, /:root\[data-font-band="extreme"\] #view-workload \.workload-config-grid,[\s\S]*?grid-template-columns:\s*minmax\(0,\s*1fr\)/);
});
