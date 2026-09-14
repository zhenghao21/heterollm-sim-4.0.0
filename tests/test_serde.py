import hashlib
import json
import unittest
from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum
from pathlib import Path

from heterollm_sim.serde import canonical_json, stable_hash, to_primitive


class _Count(IntEnum):
    ONE = 1


class _Kind(Enum):
    FAST = "fast"


class _Label(StrEnum):
    FAST = "fast"


@dataclass(frozen=True)
class _Payload:
    label: str
    count: int


class SerdeTests(unittest.TestCase):
    def test_exact_builtin_leaves_are_returned_unchanged(self):
        for value in (None, "text", 7, True, 1.25):
            with self.subTest(value=value):
                self.assertIs(to_primitive(value), value)

    def test_enum_subclasses_are_not_captured_by_builtin_fast_path(self):
        self.assertEqual(to_primitive(_Count.ONE), 1)
        self.assertIs(type(to_primitive(_Count.ONE)), int)
        self.assertEqual(to_primitive(_Kind.FAST), "fast")
        self.assertEqual(to_primitive(_Label.FAST), "fast")
        self.assertIs(type(to_primitive(_Label.FAST)), str)

    def test_dataclass_path_and_containers_keep_existing_semantics(self):
        value = {
            _Label.FAST: (_Payload("text", _Count.ONE), Path("models/base.bin")),
        }
        self.assertEqual(
            to_primitive(value),
            {"fast": [{"label": "text", "count": 1}, str(Path("models/base.bin"))]},
        )

    def test_non_finite_floats_are_rejected_by_to_primitive_and_json(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "有限值"):
                    to_primitive(value)
                with self.assertRaisesRegex(ValueError, "有限值"):
                    canonical_json({"value": value})

    def test_canonical_json_and_stable_hash_remain_deterministic(self):
        value = {"b": 2, "a": [None, False, 1.25, "中文"]}
        expected_json = (
            '{\n  "a": [\n    null,\n    false,\n    1.25,\n    "中文"\n  ],\n  "b": 2\n}'
        )
        rendered = canonical_json(value)
        self.assertEqual(rendered, expected_json)
        self.assertEqual(
            stable_hash(value),
            hashlib.sha256(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    indent=None,
                    sort_keys=True,
                    separators=(",", ": "),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
