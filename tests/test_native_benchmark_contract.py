"""A benchmark must not accept a truncated stream or a mismatched runtime."""
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path

import pytest


_SPEC = spec_from_file_location("native_benchmark", Path(__file__).parents[1] / "tools/native_benchmark.py")
assert _SPEC is not None and _SPEC.loader is not None
benchmark = module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


@pytest.mark.parametrize("props,logs", [
    ({}, []),
    ({"default_generation_settings": {"n_ctx": 640}}, []),
    ({"default_generation_settings": {"n_ctx": 768}}, ["n_ctx_slot = 1024"]),
])
def test_paired_native_rejects_missing_or_conflicting_actual_context(props, logs):
    with pytest.raises(RuntimeError, match="effective context"):
        benchmark._require_effective_context(props, logs, 768, 768)


def test_paired_native_accepts_confirmed_actual_context():
    benchmark._require_effective_context(
        {"default_generation_settings": {"n_ctx": 768}}, ["n_ctx_slot = 768"], 768, 768,
    )


@pytest.mark.parametrize("has_final", [False, True])
def test_native_stream_requires_terminal_event(monkeypatch, has_final):
    data = b'data: {"content":"x","tokens":[1],"stop":false}\n\n'
    if has_final:
        data += b'data: {"content":"","stop":true,"tokens_predicted":1}\n\n'

    class Response(BytesIO):
        status = 200

    class Connection:
        def __init__(self, *args, **kwargs):
            pass

        def request(self, *args, **kwargs):
            pass

        def getresponse(self):
            return Response(data)

        def close(self):
            pass

    monkeypatch.setattr(benchmark.http.client, "HTTPConnection", Connection)
    if has_final:
        result, _, _ = benchmark._stream_completion("http://127.0.0.1:1234", {"n_predict": 1}, 1)
        assert result["final"]["stop"] is True
    else:
        with pytest.raises(RuntimeError, match="without a final stop event"):
            benchmark._stream_completion("http://127.0.0.1:1234", {"n_predict": 1}, 1)


@pytest.mark.parametrize("disable,inherited", [(True, None), (True, "0"), (False, None), (False, "0")])
def test_cuda_graph_flag_is_child_only_and_actual_environment_is_recorded(monkeypatch, tmp_path, disable, inherited):
    key = "GGML_CUDA_DISABLE_GRAPHS"
    if inherited is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, inherited)
    argv = ["native_benchmark.py", "--server", str(tmp_path / "server.exe"),
            "--model", str(tmp_path / "model.gguf"), "--output", str(tmp_path / "result.json"),
            "--flash-attn", "off"]
    if disable:
        argv.append("--disable-cuda-graphs")
    monkeypatch.setattr(benchmark.sys, "argv", argv)
    args = benchmark._parse_args()
    monkeypatch.setattr(benchmark, "_validate_args", lambda args: None)
    received = {}
    class Process:
        stderr = BytesIO()
        def poll(self):
            return 0
    def popen(command, **kwargs):
        received.update(kwargs)
        return Process()
    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    def stop(*args):
        raise RuntimeError("stop after checking process arguments")
    monkeypatch.setattr(benchmark, "_wait_ready", stop)
    result = benchmark.run(args)
    expected = "1" if disable else inherited
    assert received["env"].get(key) == expected
    assert received["env"] is not benchmark.os.environ
    assert benchmark.os.environ.get(key) == inherited
    assert result["configuration"]["effective_env"] == {key: expected}
    assert result["configuration"]["cuda_graphs_disabled"] is (expected is not None)
    assert result["configuration"]["cuda_graphs_disable_requested"] is disable
