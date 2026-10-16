"""Cached physical contracts retain validation and metadata ownership."""
from dataclasses import asdict

import pytest

from heterollm_sim.physical_contract import parse_execution_physical_config
from heterollm_sim.memory_types import make_hbm_config, make_hbf_config


@pytest.mark.parametrize("factory", [make_hbm_config, make_hbf_config])
def test_parsed_config_metadata_cannot_mutate_later_parse(factory):
    raw = asdict(factory(metadata={"custom": {"value": 1}}))
    first = parse_execution_physical_config(raw)
    expected = asdict(first)
    second = parse_execution_physical_config(raw)
    second.metadata["custom"]["value"] = 8
    assert asdict(parse_execution_physical_config(raw)) == expected
    raw["metadata"]["custom"]["value"] = 9
    assert parse_execution_physical_config(raw).metadata["custom"]["value"] == 9


@pytest.mark.parametrize("factory", [make_hbm_config, make_hbf_config])
def test_invalid_bool_cannot_reuse_valid_integer_contract(factory):
    raw = asdict(factory(channels=1))
    parse_execution_physical_config(raw)
    raw["channels"] = True
    with pytest.raises(ValueError):
        parse_execution_physical_config(raw)


def test_cache_does_not_fill_missing_authored_fields():
    raw = asdict(make_hbm_config())
    parse_execution_physical_config(raw)
    del raw["burst_bytes"]
    with pytest.raises(ValueError, match="missing explicit fields"):
        parse_execution_physical_config(raw)
