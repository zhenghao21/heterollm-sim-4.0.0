from pathlib import Path

from tools.source_identity import source_hashes, verify_source_hashes


def test_source_hashes_use_git_commit_content():
    path = "tools/command_provenance.py"
    record = source_hashes("HEAD", [path], cwd=Path.cwd())[path]
    assert record["basis"] == "git_blob_content"
    assert record["commit"] == "HEAD"
    assert len(record["sha256"]) == 64


def test_source_hash_verification_rejects_hardcoded_mismatch():
    errors = verify_source_hashes(
        "HEAD",
        {"tools/command_provenance.py": {"sha256": "0" * 64}},
        cwd=Path.cwd(),
    )
    assert errors and "source hash mismatch" in errors[0]
