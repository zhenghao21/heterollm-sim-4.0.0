"""Exact-token preparation checks; these never launch a model or server."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from tools import prepare_native_grid_prompts as prep


def test_tokenizer_output_requires_real_ids_and_matching_count():
    assert prep.parse_tokenizer_output('[1, 19, 8]\nTotal number of tokens: 3\n') == [1, 19, 8]
    for output in ('Total number of tokens: 3\n', '[1, 2]\nTotal number of tokens: 3\n',
                   '[-1]\nTotal number of tokens: 1\n', '[1]\n[2]\nTotal number of tokens: 1\n'):
        with pytest.raises(RuntimeError):
            prep.parse_tokenizer_output(output)
    with pytest.raises(RuntimeError):
        prep.parse_tokenizer_output('[1]\nTotal number of tokens: 1', returncode=1)


def test_exact_prefixes_include_native_bos_once():
    body = list(range(100, 2100))
    prompts, bos = prep.token_prefixes([1] + body, body)
    assert bos == {'policy': 'model_native', 'automatically_added': True, 'token_id': 1, 'tokens_in_budget': 1}
    for n in (128, 512, 1536):
        assert len(prompts[str(n)]['ids']) == n
        assert prompts[str(n)]['ids'] == [1] + body[:n-1]


def test_no_bos_models_do_not_receive_invented_bos():
    body = list(range(100, 2100))
    prompts, bos = prep.token_prefixes(body, body)
    assert not bos['automatically_added']
    assert prompts['128']['ids'] == body[:128]
    assert prompts['128']['bos_tokens_in_budget'] == 0


def test_short_corpus_and_unexplained_special_tokens_fail_closed():
    with pytest.raises(ValueError, match='actual tokens'):
        prep.token_prefixes([1, 2, 3], [2, 3])
    with pytest.raises(ValueError, match='special-token difference'):
        prep.token_prefixes([1, 2, 3, 4], [2, 3])
    with pytest.raises(ValueError, match='positive integers'):
        prep.token_prefixes([1, 2], [1, 2], (True,))


def fixture_manifest(tmp_path, tokenizer_fn=None):
    runtime = tmp_path / 'llama-server.exe'; runtime.write_bytes(b'server')
    tokenizer = tmp_path / 'llama-tokenize.exe'; tokenizer.write_bytes(b'tokenizer')
    (tmp_path / 'llama.dll').write_bytes(b'vocab-library')
    model = tmp_path / 'model.gguf'; model.write_bytes(b'model')
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps({'jobs': [{'id': 'toy_short_p1', 'model': str(model)}]}))
    def fake(exe, path, text, no_bos=False):
        assert exe == tokenizer and path == model
        ids = list(range(100, 1800))
        if not no_bos:
            ids = [1] + ids
        return {'ids': ids, 'count': len(ids), 'argv': [str(exe)], 'stdin_sha256': 'fake'}
    return prep.prepare_manifest(protocol, runtime, corpus_text='a natural text', tokenizer_fn=tokenizer_fn or fake, capture_contract=False)


def test_manifest_freezes_model_runtime_corpus_and_verified_tokens(tmp_path):
    manifest = fixture_manifest(tmp_path)
    prep.validate_manifest(manifest)
    assert len(manifest['models']) == 1
    assert manifest['models'][0]['verification']['repeat_ids_equal']
    assert len(manifest['models'][0]['verification']['runs']) == 3
    assert manifest['tokenizer']['sha256'] == prep.file_identity(tmp_path/'llama-tokenize.exe')['sha256']
    assert len(manifest['runtime']['files']) == 3
    corrupted = deepcopy(manifest)
    corrupted['models'][0]['prompts']['128']['ids'][1] += 1
    with pytest.raises(ValueError, match='checksum'):
        prep.validate_manifest(corrupted)
    corrupted['manifest_sha256'] = prep.stable_hash({k:v for k,v in corrupted.items() if k != 'manifest_sha256'})
    with pytest.raises(ValueError, match='exact source token prefix'):
        prep.validate_manifest(corrupted)


def test_tokenizer_determinism_failure_is_rejected(tmp_path):
    state = [0]
    def unstable(*args, **kwargs):
        state[0] += 1
        return {'ids': list(range(state[0], 1800 + state[0]))}
    with pytest.raises(RuntimeError, match='repeat verification'):
        fixture_manifest(tmp_path, unstable)


def test_missing_adjacent_tokenizer_never_falls_back(tmp_path):
    runtime = tmp_path/'llama-server.exe'; runtime.write_bytes(b'server')
    with pytest.raises(FileNotFoundError):
        prep.prepare_manifest(tmp_path/'protocol.json', runtime, capture_contract=False)


def test_native_server_source_preserves_integer_arrays():
    contract = prep.server_contract()
    assert contract['prompt_type'] == 'array_of_integer_token_ids'
    assert contract['retokenizes'] is False
    assert contract['adds_bos_to_integer_array'] is False
    assert len(contract['evidence']) == 6
