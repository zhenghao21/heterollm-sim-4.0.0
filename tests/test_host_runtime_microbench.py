from pathlib import Path
from types import SimpleNamespace

from tools.host_runtime_microbench import benchmark_json, benchmark_loopback_sse, benchmark_tokenizer


def test_json_microbench_reports_units_and_samples():
    result = benchmark_json(3, items=2)
    assert result["payload_bytes"] > 0
    assert result["encode"]["unit"] == "ms"
    assert result["encode"]["samples"] == 3
    assert result["decode"]["samples"] == 3


def test_loopback_sse_separates_done_tail():
    result = benchmark_loopback_sse(2, token_count=3)
    assert result["ttft"]["unit"] == "ms"
    assert result["last_token_e2e"]["samples"] == 2
    assert result["request_end_including_done"]["samples"] == 2
    assert result["done_tail_ms_median"] >= 0


def test_tokenizer_microbench_uses_cli_without_model_timing(monkeypatch, tmp_path: Path):
    tokenizer = tmp_path / "llama-tokenize.exe"
    model = tmp_path / "model.gguf"
    tokenizer.write_bytes(b"stub")
    model.write_bytes(b"stub")

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="Total number of tokens: 2\n", stderr="")

    monkeypatch.setattr("tools.host_runtime_microbench.subprocess.run", fake_run)
    result = benchmark_tokenizer(tokenizer, model, ["Hi."], samples=2)
    assert result["samples_per_prompt"] == 2
    assert result["cases"][0]["token_count"] == 2
    assert result["cases"][0]["latency"]["unit"] == "ms"
