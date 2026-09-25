const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const app = fs.readFileSync(path.join(__dirname, '..', 'src', 'heterollm_sim', 'webui', 'app.js'), 'utf8');

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
});
