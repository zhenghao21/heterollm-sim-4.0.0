"""Source-schedule arithmetic behind the independent Attention evaluator."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('attention_eval', Path(__file__).parents[1] / 'tools/evaluate_attention_streamk_holdout.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('context,expected', [(1024,64), (1536,96), (2048,128), (4096,168)])
def test_streamk_grid_matches_observed_causal_specialization(context, expected):
    assert module.stream_k_blocks(context, kv_tile_tokens=64, output_tiles=4, max_active_blocks=168) == expected


def test_streamk_rounding_obeys_source_integer_five_percent_gate():
    # A small rounding loss uses uniform partitions; a larger loss keeps raw.
    assert module.stream_k_blocks(4096, kv_tile_tokens=64, output_tiles=8, max_active_blocks=171) == 168
    assert module.stream_k_blocks(4096, kv_tile_tokens=64, output_tiles=8, max_active_blocks=20) == 20
    assert module.stream_k_blocks(4096, kv_tile_tokens=64, output_tiles=32, max_active_blocks=20) == 20
    with pytest.raises(ValueError):
        module.stream_k_blocks(0, kv_tile_tokens=64, output_tiles=4, max_active_blocks=168)
