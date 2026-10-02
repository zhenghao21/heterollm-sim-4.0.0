import hashlib
import json
import subprocess
import sys
import base64

from tools.command_provenance import run_command, write_record


def test_run_command_captures_success_and_failure_output(tmp_path):
    success = run_command(
        [sys.executable, "-c", "print('ok')"], cwd=tmp_path
    )
    failure = run_command(
        [sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(7)"],
        cwd=tmp_path,
    )

    assert success["returncode"] == 0
    assert success["timed_out"] is False
    assert success["stdout"].replace("\r\n", "\n") == "ok\n"
    assert success["stdout_sha256"] == hashlib.sha256(
        success["stdout"].encode("utf-8")
    ).hexdigest()
    assert base64.b64decode(success["stdout_base64"]) == success["stdout"].encode(
        "utf-8"
    )
    assert failure["returncode"] == 7
    assert failure["stderr"].replace("\r\n", "\n") == "bad\n"
    assert failure["stderr_sha256"] == hashlib.sha256(
        failure["stderr"].encode("utf-8")
    ).hexdigest()


def test_write_record_is_replayable_json(tmp_path):
    record = run_command([sys.executable, "-c", "print('bound')"])
    output = tmp_path / "provenance.json"
    write_record(record, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["schema"] == "heterollm.command_provenance/v1"
    assert loaded["returncode"] == 0
    assert loaded["stdout_sha256"] == hashlib.sha256(
        loaded["stdout"].encode("utf-8")
    ).hexdigest()
    assert output.with_name(output.name + ".sha256").exists()
