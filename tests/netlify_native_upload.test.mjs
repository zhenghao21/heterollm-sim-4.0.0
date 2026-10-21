import test from 'node:test';
import assert from 'node:assert/strict';
import { TARGET, uploadSnapshot, safeError } from '../scripts/netlify_native_upload.mjs';

const snapshot = () => ({ schema: 'exploring-hbf-native-dashboard/v1',
  scan_id: 'exploring_hbf_native_20261019', observed_at: '2026-10-20T08:00:00+08:00' });

test('fixed store/site/key, current CLI credential in memory, strong readback and application User-Agent', async () => {
  let stored;
  const calls = [];
  const result = await uploadSnapshot(snapshot(), {
    tokenProvider: async () => ['test-memory-value', 'config'],
    fetchImplementation: async (_url, init) => {
      assert.equal(init.headers.get('User-Agent'), 'netlify-cli/27.12.1');
      return { status: 200 };
    },
    storeFactory: options => {
      assert.equal(options.name, 'b200-scan');
      assert.equal(options.siteID, '078ff573-023f-4672-ab6b-14e0d5a94dc5');
      assert.equal(options.token, 'test-memory-value');
      return {
        setJSON: async (key, value) => {
          calls.push(['setJSON', key]); stored = value;
          await options.fetch('https://example.invalid', {});
        },
        get: async (key, options) => {
          calls.push(['get', key, options]); return stored;
        },
      };
    },
  });
  assert.deepEqual(calls, [['setJSON', 'native-comparison/latest.json'],
    ['get', 'native-comparison/latest.json', { type: 'json', consistency: 'strong' }]]);
  assert.equal(result.verified, true);
  assert.equal(result.http_status, 200);
  assert.equal(result.observed_at, snapshot().observed_at);
  assert.equal(JSON.stringify(result).includes('test-memory-value'), false);
  assert.equal(TARGET.key, 'native-comparison/latest.json');
});

test('wrong experiment/schema is rejected before credential lookup or store access', async () => {
  let accesses = 0;
  for (const invalid of [{ ...snapshot(), scan_id: 'old-scan' }, { ...snapshot(), schema: 'old' }]) {
    await assert.rejects(uploadSnapshot(invalid, {
      tokenProvider: async () => { accesses++; return ['test']; },
      storeFactory: () => { accesses++; throw new Error(); },
    }), { code: 'INVALID_SNAPSHOT' });
  }
  assert.equal(accesses, 0);
});

test('readback mismatch is an error and sanitized failures never expose response data', async () => {
  await assert.rejects(uploadSnapshot(snapshot(), {
    tokenProvider: async () => ['test'],
    storeFactory: () => ({ setJSON: async () => {},
      get: async () => ({ ...snapshot(), observed_at: '2026-10-20T07:00:00+08:00' }) }),
  }), { code: 'READBACK_MISMATCH' });
  const error = Object.assign(new Error('private body and credential'), {
    name: 'FetchError', code: 'NETWORK_ERROR', status: 403, headers: { authorization: 'private' },
    body: 'private-response' });
  assert.deepEqual(safeError(error), { ok: false, name: 'FetchError', code: 'NETWORK_ERROR', status: 403 });
  assert.equal(JSON.stringify(safeError(error)).includes('private'), false);
});
