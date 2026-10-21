"""Remote workbench transport keeps endpoints and credentials separate."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from test_current_report_callers import browser_check


MODULE = Path(__file__).parents[1] / "src/heterollm_sim/webui/api-connection.js"


def check_js(assertions):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for browser transport checks")
    script = f"""
const assert = require('node:assert/strict');
const {{createConnection, normalizeBackendUrl}} = require({json.dumps(str(MODULE))});
function storage() {{
  const values = new Map();
  return {{values, getItem: key => values.get(key), setItem: (key, value) => values.set(key, value), removeItem: key => values.delete(key)}};
}}
function environment(hostname = 'zh.gripe') {{
  const origin = hostname === 'localhost' ? 'http://localhost:8000' : 'https://' + hostname;
  const calls = [];
  return {{location: {{origin, hostname, href: origin + '/simulator/'}}, localStorage: storage(), sessionStorage: storage(), calls,
    fetch: async (url, options) => {{ calls.push({{url, options}}); return response({{backend_url: 'https://backend.example'}}); }},
    setTimeout, clearTimeout}};
}}
function response(payload, status = 200) {{
  return {{ok: status < 400, status, headers: new Headers({{'Content-Type': 'application/json'}}), json: async () => payload, text: async () => JSON.stringify(payload)}};
}}
(async () => {{
{assertions}
}})().catch(error => {{ console.error(error); process.exitCode = 1; }});
"""
    result = subprocess.run([node], input=script, text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stderr


def test_remote_urls_require_https_and_keep_credentials_out_of_url():
    check_js("""
assert.equal(normalizeBackendUrl(' https://backend.example/ '), 'https://backend.example');
assert.equal(normalizeBackendUrl('http://localhost:8901/'), 'http://localhost:8901');
assert.equal(normalizeBackendUrl('http://127.0.0.1:8901'), 'http://127.0.0.1:8901');
assert.equal(normalizeBackendUrl('https://backend.example/service/'), 'https://backend.example/service');
for (const url of ['http://backend.example', 'ftp://backend.example', 'http://localhost.evil',
  'https://name:secret@backend.example', 'https://backend.example?token=secret', 'https://backend.example#secret',
  'https://backend.example?', 'https://backend.example#', '/api', 'javascript:alert(1)']) {
  assert.throws(() => normalizeBackendUrl(url));
}
""")


def test_cloud_bootstrap_only_reads_public_config_and_waits_for_connection():
    check_js("""
const env = environment(); const connection = createConnection(env);
const initial = await connection.initialize();
assert.equal(initial.active, false);
assert.equal(initial.backendUrl, 'https://backend.example');
assert.deepEqual(env.calls.map(call => call.url), ['https://zh.gripe/simulator/backend-config.json']);
await assert.rejects(connection.request('/health'), error => error.code === 'connection_required');
assert.equal(env.calls.length, 1);
assert.throws(() => connection.configure({url: '', accessToken: 'secret'}));
assert.throws(() => connection.configure({url: 'https://backend.example'}));
env.fetch = async () => { throw new TypeError('offline'); };
const empty = await createConnection(env).initialize();
assert.equal(empty.active, false);
assert.equal(empty.backendUrl, '');
""")


def test_bearer_transport_uses_one_root_and_never_persists_token_locally():
    check_js("""
const env = environment(); const connection = createConnection(env);
connection.configure({url: 'https://backend.example/', accessToken: 'private-test-value', remember: true});
const reply = await connection.request('/validate', {method: 'POST', headers: {'X-Client': 'workbench'}, body: '{}'});
await reply.json();
const {url, options} = env.calls[0];
assert.equal(url, 'https://backend.example/api/validate');
assert.equal(options.headers.get('Authorization'), 'Bearer private-test-value');
assert.equal(options.headers.get('Content-Type'), 'application/json');
assert.equal(options.headers.get('X-Client'), 'workbench');
assert.equal(options.credentials, 'omit');
assert.equal(options.redirect, 'error');
assert.equal(env.localStorage.values.size, 1);
assert.deepEqual([...env.localStorage.values.values()], ['https://backend.example']);
assert.equal(env.sessionStorage.values.size, 1);
assert.equal(connection.savedToken('https://backend.example'), 'private-test-value');
assert.equal(JSON.stringify(connection.snapshot()).includes('private-test-value'), false);
const secondTab = createConnection(env);
await secondTab.initialize();
assert.equal(secondTab.snapshot().active, false);
connection.disconnect();
assert.equal(env.sessionStorage.values.size, 0);
await assert.rejects(connection.request('/health'), error => error.code === 'connection_required');
""")


def test_switching_backends_aborts_pending_requests_and_stale_body_results():
    check_js("""
const env = environment(); const connection = createConnection(env);
connection.configure({url: 'https://one.example', accessToken: 'one', remember: true});
let finishBody;
env.fetch = async (url, options) => {
  env.calls.push({url, options});
  return {...response({}), json: () => new Promise(resolve => { finishBody = resolve; })};
};
const reply = await connection.request('/run-jobs/old');
const oldBody = reply.json();
connection.configure({url: 'https://two.example', accessToken: 'two', remember: true});
assert.equal(env.calls[0].options.signal.aborted, true);
finishBody({job_id: 'old', report: {}});
await assert.rejects(oldBody, error => error.name === 'AbortError');
assert.equal(connection.savedToken('https://one.example'), '');
assert.equal(connection.savedToken('https://two.example'), 'two');
let finishFetch;
env.fetch = () => new Promise(resolve => { finishFetch = resolve; });
const pendingFetch = connection.request('/health');
connection.disconnect();
finishFetch(response({ok: true}));
await assert.rejects(pendingFetch, error => error.name === 'AbortError');
""")


def test_loopback_keeps_local_same_origin_and_caller_cancellation():
    check_js("""
const env = environment('localhost'); const connection = createConnection(env);
await connection.initialize();
assert.equal(connection.snapshot().active, true);
assert.equal(env.calls.length, 0);
const cancellation = new AbortController();
const reply = await connection.request('/health', {signal: cancellation.signal});
assert.equal(env.calls[0].url, '/api/health');
assert.equal(env.calls[0].options.headers.has('Authorization'), false);
cancellation.abort();
assert.equal(env.calls[0].options.signal.aborted, true);
await reply.json();
connection.configure({url: 'http://127.0.0.1:8901'});
const remote = await connection.request('/health'); await remote.json();
assert.equal(env.calls[1].url, 'http://127.0.0.1:8901/api/health');
""")


def test_authorization_failures_never_appear_as_connected():
    browser_check(("apiRequest",), None, """
(async () => {
  for (const status of [401, 403]) {
    responseStatus = status;
    await assert.rejects(apiRequest('/health'), error => error.status === status);
    assert.equal(connectionStatus, 'offline');
    assert.match(connectionMessage.zh, /鉴权失败/);
    assert.equal(panelFailed, true);
    assert.match(panelMessage, /令牌/);
  }
  assert.equal(disconnected, 2);
})().catch(error => { console.error(error); process.exitCode = 1; });
""", """
let disconnected = 0, responseStatus = 401, connectionStatus, connectionMessage, panelMessage, panelFailed;
const backendConnection = {disconnect: () => { disconnected += 1; },
  snapshot: () => ({generation: 1}), assertGeneration: () => {},
  request: async () => ({ok: false, get status() {return responseStatus;},
    headers: {get: () => 'application/json'}, json: async () => ({error: {code: 'unauthorized'}})})};
const backendChineseMessage = (_envelope, fallback) => fallback;
const setConnection = (status, message) => { connectionStatus = status; connectionMessage = message; };
const setBackendConnectionMessage = (message, failed) => { panelMessage = message; panelFailed = failed; };
const runtimeConnectionLabel = () => { throw new Error('An authorization failure must not become online'); };
""")


def test_old_recovery_probe_cannot_change_new_backend_connection_status():
    browser_check(("apiRequest", "apiReachableAfterRequestFailure"), None, """
(async () => {
  const oldRequest = apiRequest('/validate');
  await probeStarted;
  generation = 2;
  connectionStatus = 'online';
  connectionLabel = 'new backend';
  failProbe(new DOMException('Old backend disconnected', 'AbortError'));
  await assert.rejects(oldRequest, error => error.name === 'AbortError');
  assert.equal(connectionStatus, 'online');
  assert.equal(connectionLabel, 'new backend');
  assert.equal(connectionUpdates, 0);
})().catch(error => { console.error(error); process.exitCode = 1; });
""", """
let generation = 1, connectionStatus, connectionLabel, connectionUpdates = 0, failProbe, beginProbe;
const probeStarted = new Promise(resolve => {beginProbe = resolve;});
const backendConnection = {
  snapshot: () => ({generation}),
  assertGeneration: expected => {if (expected !== generation) throw new DOMException('Stale connection', 'AbortError');},
  request: async path => {
    if (path === '/validate') throw new TypeError('Request transport failed');
    assert.equal(path, '/health');
    beginProbe();
    return new Promise((_resolve, reject) => {failProbe = reject;});
  },
};
const CONNECTION_RECOVERY_PROBE_TIMEOUT_MS = 1500;
const CONNECTION_LABELS = {unreachable: {zh: 'old backend offline'}};
const setConnection = (status, label) => {connectionStatus = status; connectionLabel = label; connectionUpdates += 1;};
const runtimeConnectionLabel = () => ({zh: 'old backend', en: 'old backend'});
""")


def test_backend_switch_clears_job_trace_reports_and_remote_catalogs():
    browser_check(("clearBackendDerivedState",), None, """
clearBackendDerivedState();
assert.equal(runJobPollTimer, null);
assert.equal(timerCleared, true);
assert.equal(catalogAborted, true);
assert.equal(state.runJob, null);
assert.equal(state.runJobSubmitting, false);
assert.equal(state.report, null);
assert.equal(state.runtimeHealth, null);
assert.equal(state.modelPresetsLoaded, false);
assert.deepEqual(state.architecturePresets, []);
assert.equal(state.componentPresetDetails.size, 0);
assert.deepEqual(workloadPresetCatalog, []);
assert.equal(state.scenario.model.metadata.artifact_id, undefined);
assert.equal(traceCleared, true);
assert.equal(particlesStopped, true);
assert.equal(state.scenarioGeneration, 2);
assert.equal(tracePageLoadSerial, 2);
""", """
let runJobPollTimer = 7, timerCleared = false, catalogAborted = false, traceCleared = false, particlesStopped = false;
const window = {clearTimeout: () => {timerCleared = true;}};
const presetCatalogController = {abort: () => {catalogAborted = true;}};
let presetCatalogGeneration = 1, validationRequestGeneration = 1, runEstimateRequestGeneration = 1, tracePageLoadSerial = 1;
let workloadPresetCatalog = [{id:'old'}], workloadPresetCatalogError = 'old';
const state = {runJob: {job_id: 'old'}, runJobSubmitting: true, scenarioGeneration: 1, mappingGeneration: 1,
  scenario: {model: {metadata: {artifact_id: 'old-backend-artifact'}}}, report: {old: true}, runtimeHealth: {old: true},
  modelPresetsLoaded: true, componentPresetDetails: new Map([['old', {}]])};
const dom = {};
const resetTracePlaybackState = () => {traceCleared = true;};
const stopTraceParticles = () => {particlesStopped = true;};
const setBusy = () => {}, renderAll = () => {};
""")
