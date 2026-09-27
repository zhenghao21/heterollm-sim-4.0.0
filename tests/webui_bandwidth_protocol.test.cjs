const assert = require('assert');
const fs = require('fs');

const app = fs.readFileSync('src/heterollm_sim/webui/app.js', 'utf8');
const css = fs.readFileSync('src/heterollm_sim/webui/styles.css', 'utf8');
const html = fs.readFileSync('src/heterollm_sim/webui/index.html', 'utf8');

assert.match(app, /COMPONENT_INSPECTOR_KINDS\s*=\s*Object\.freeze\(COMPONENT_KINDS\.filter\(\(kind\) => kind !== "hbm_stack"\)\)/);
assert.match(app, /function componentSharedBandwidthGbps\(component\)/);
assert.match(app, /function storageTransportParameters\(component, scenario = state\.scenario\)/);
assert.match(app, /Number\(component\?\.bandwidth_gbps\)/);
assert.match(app, /Number\(port\?\.bandwidth_gbps\)/);
assert.match(app, /if \(compact === "ddr" \|\| \/\^ddr\[345\]\/\.test\(compact\)\) return "ddr";/);
assert.match(css, /\.protocol-ddr\s*\{/);
assert.strictEqual((html.match(/protocol-legend-line protocol-ddr/g) || []).length, 2);

console.log('unified memory bandwidth and DDR protocol UI contract: ok');
