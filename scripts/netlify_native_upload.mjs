/** Upload one fixed native-comparison snapshot using the existing CLI login. */
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const TARGET = Object.freeze({
  name: 'b200-scan',
  siteID: '078ff573-023f-4672-ab6b-14e0d5a94dc5',
  key: 'native-comparison/latest.json',
});
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const CLI_ROOT = 'C:/Users/A/AppData/Local/npm-cache/_npx/da5c1b6ea715e8b4/node_modules/netlify-cli';
const SCHEMA = 'exploring-hbf-native-dashboard/v1';
const SCAN_ID = 'exploring_hbf_native_20261019';

function fail(code) {
  const error = new Error();
  error.name = 'NativePublishError';
  error.code = code;
  throw error;
}

export function validateSnapshot(snapshot) {
  if (!snapshot || typeof snapshot !== 'object' || Array.isArray(snapshot)
      || snapshot.schema !== SCHEMA || snapshot.scan_id !== SCAN_ID
      || typeof snapshot.observed_at !== 'string' || !Number.isFinite(Date.parse(snapshot.observed_at))) {
    fail('INVALID_SNAPSHOT');
  }
  return snapshot;
}

export function safeError(error) {
  const status = Number.isInteger(error?.status) && error.status >= 100 && error.status <= 599
    ? error.status : null;
  // Never serialize messages, stacks, response bodies, URLs, headers or credentials.
  const code = typeof error?.code === 'string' && /^[A-Z_][A-Z0-9_]{0,31}$/.test(error.code)
    ? error.code : null;
  const name = typeof error?.name === 'string' && /^[A-Za-z][A-Za-z0-9]{0,47}$/.test(error.name)
    ? error.name : 'Error';
  return { ok: false, status, code, name };
}

export async function uploadSnapshot(snapshot, { cliRoot = CLI_ROOT, tokenProvider, storeFactory,
  fetchImplementation = globalThis.fetch } = {}) {
  validateSnapshot(snapshot);
  if (!tokenProvider) {
    const helpers = await import(pathToFileURL(path.join(cliRoot, 'dist/utils/command-helpers.js')).href);
    tokenProvider = helpers.getToken;
  }
  if (!storeFactory) {
    const require = createRequire(path.join(ROOT, 'tmp/b200-scan-netlify/package.json'));
    storeFactory = require('@netlify/blobs').getStore;
  }
  const [token] = await tokenProvider();
  if (typeof token !== 'string' || !token) fail('CLI_LOGIN_UNAVAILABLE');
  let lastStatus = null;
  const store = storeFactory({ name: TARGET.name, siteID: TARGET.siteID, token,
    fetch: async (url, init) => {
      const headers = new Headers(init?.headers);
      headers.set('User-Agent', 'netlify-cli/27.12.1');
      const response = await fetchImplementation(url, { ...init, headers });
      lastStatus = response.status;
      return response;
    } });
  await store.setJSON(TARGET.key, snapshot);
  const stored = await store.get(TARGET.key, { type: 'json', consistency: 'strong' });
  validateSnapshot(stored);
  if (stored.observed_at !== snapshot.observed_at) fail('READBACK_MISMATCH');
  return { ok: true, verified: true, observed_at: snapshot.observed_at,
    http_status: lastStatus, key: TARGET.key };
}

async function main() {
  const args = process.argv.slice(2);
  let cliRoot = CLI_ROOT;
  if (args.length) {
    if (args.length !== 2 || args[0] !== '--cli-root') fail('INVALID_ARGUMENTS');
    cliRoot = args[1];
  }
  let input = '';
  process.stdin.setEncoding('utf8');
  for await (const chunk of process.stdin) {
    input += chunk;
    if (Buffer.byteLength(input, 'utf8') > 2 * 1024 * 1024) fail('SNAPSHOT_TOO_LARGE');
  }
  const snapshot = JSON.parse(input);
  const result = await uploadSnapshot(snapshot, { cliRoot });
  process.stdout.write(JSON.stringify(result) + '\n');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    process.stdout.write(JSON.stringify(safeError(error)) + '\n');
    process.exitCode = 2;
  });
}
