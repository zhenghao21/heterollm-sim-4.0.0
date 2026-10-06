import pytest

from heterollm_sim.runtime import _validated_capacity_bytes
from heterollm_sim.runtime_state import RuntimeState


@pytest.mark.parametrize("value", (None, 0, [], "capacity"))
def test_capacity_bytes_requires_mapping_even_when_falsey(value):
    with pytest.raises(TypeError, match="context.capacity_bytes"):
        _validated_capacity_bytes(value, label="context.capacity_bytes")


@pytest.mark.parametrize("entry", (("gpu0", 1.5), ("gpu0", True), ("", 1)))
def test_capacity_bytes_validates_existing_and_new_entries(entry):
    with pytest.raises(ValueError, match="context.capacity_bytes"):
        _validated_capacity_bytes({entry[0]: entry[1]}, label="context.capacity_bytes")


def test_capacity_bytes_accepts_zero_capacity_as_explicit_value():
    assert _validated_capacity_bytes(
        {"gpu0": 0}, label="context.capacity_bytes"
    ) == {"gpu0": 0}


@pytest.mark.parametrize("field", ("weight_cache", "page_cache", "completed_requests"))
def test_runtime_state_rejects_bare_string_collections(field):
    with pytest.raises(ValueError, match=field):
        RuntimeState(**{field: "request-1"})
