"""Regression tests for the reporting-v2 source-evidence schema correction."""
from __future__ import annotations
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("summarize_ablation.py")
SPEC = importlib.util.spec_from_file_location("r21_reporting_v2", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class SourceEvidenceReferenceTests(unittest.TestCase):
    def test_path_and_sha_only_source_reference_records_observed_size(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "locked-source.cpp"
            payload = b"source-only schema omission\n"
            path.write_bytes(payload)
            ref = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}

            with self.assertRaises(REPORT.EvidenceError):
                REPORT.normalized_ref(ref)

            observed = REPORT.verify_source_evidence_reference(ref)
            self.assertEqual(observed["path"], str(path.resolve()))
            self.assertEqual(observed["sha256"], ref["sha256"])
            self.assertEqual(observed["observed_bytes"], len(payload))
            self.assertIsNone(observed["declared_bytes"])

    def test_source_closure_merges_size_omission_with_same_path_and_sha(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "locked-source.cpp"
            payload = b"shared source evidence"
            path.write_bytes(payload)
            sha = hashlib.sha256(payload).hexdigest()
            freeze = {
                "recurrent_batching": {
                    "evidence_refs": [
                        {"path": str(path), "sha256": sha},
                        {"path": str(path), "sha256": sha, "size_bytes": len(payload)},
                    ]
                }
            }
            closure = REPORT.evidence_closure(freeze)
            self.assertEqual(len(closure), 1)
            self.assertEqual(closure[0]["declared_bytes"], len(payload))

    def test_cross_variant_merge_accepts_only_size_omission(self):
        refs = {}
        first = {"path": "C:/fixed/source.cpp", "sha256": "a" * 64}
        second = {**first, "declared_bytes": 17}
        REPORT.merge_source_evidence_ref(refs, first)
        merged = REPORT.merge_source_evidence_ref(refs, second)
        self.assertEqual(merged["declared_bytes"], 17)
        with self.assertRaises(REPORT.EvidenceError):
            REPORT.merge_source_evidence_ref(refs, {**first, "declared_bytes": 18})

    def test_source_reference_never_accepts_missing_sha_or_changed_content(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "locked-source.cpp"
            path.write_bytes(b"initial")
            with self.assertRaises(REPORT.EvidenceError):
                REPORT.verify_source_evidence_reference({"path": str(path)})

            ref = {"path": str(path), "sha256": hashlib.sha256(b"initial").hexdigest()}
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(REPORT.EvidenceError, "changed source evidence"):
                REPORT.verify_source_evidence_reference(ref)


if __name__ == "__main__":
    unittest.main()
