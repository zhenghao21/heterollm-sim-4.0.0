import unittest

from heterollm_sim.precision import (
    canonical_dtype,
    layer_precision_bits,
    weight_storage_bits,
)


class PrecisionParsingTests(unittest.TestCase):
    def test_canonical_dtype_unifies_aliases_but_not_representations(self):
        self.assertEqual(canonical_dtype("f32"), "fp32")
        self.assertEqual(canonical_dtype("float32"), "fp32")
        self.assertEqual(canonical_dtype("f16"), "fp16")
        self.assertNotEqual(canonical_dtype("f16"), canonical_dtype("bf16"))

    def test_quantization_separators_preserve_weight_and_activation_widths(self):
        for value in ("w8a8", "W8-A8", "w8_a8", "w8 a8"):
            with self.subTest(value=value):
                self.assertEqual(
                    layer_precision_bits(
                        "bf16", value, unsupported_dtype_message="bad dtype"
                    ),
                    (8, 8),
                )

    def test_weight_only_quantization_keeps_activation_dtype(self):
        self.assertEqual(
            layer_precision_bits(
                "fp16", "w4", unsupported_dtype_message="bad dtype"
            ),
            (16, 4),
        )
        self.assertEqual(
            weight_storage_bits(
                "fp16",
                "w4",
                unsupported_dtype_message="bad dtype",
                unsupported_quantization_message="bad quantization",
            ),
            4,
        )

    def test_nonempty_unrecognized_quantization_fails_closed(self):
        for value in ("mystery", "w0a8", "w8a0", "prefix-w8a8"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "bad quantization"):
                    layer_precision_bits(
                        "bf16",
                        value,
                        unsupported_dtype_message="bad dtype",
                        unsupported_quantization_message="bad quantization",
                    )
                with self.assertRaisesRegex(ValueError, "bad quantization"):
                    weight_storage_bits(
                        "bf16",
                        value,
                        unsupported_dtype_message="bad dtype",
                        unsupported_quantization_message="bad quantization",
                    )

    def test_absent_quantization_uses_dtype(self):
        for value in (None, ""):
            with self.subTest(value=value):
                self.assertEqual(
                    layer_precision_bits(
                        "bf16", value, unsupported_dtype_message="bad dtype"
                    ),
                    (16, 16),
                )


if __name__ == "__main__":
    unittest.main()
