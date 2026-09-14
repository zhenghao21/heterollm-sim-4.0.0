import json

from tools.build_stage_holdout_report import build


def _profile(path, *, qkv_total, qkv_instances, launch_total, launch_calls):
    payload = {
        "stats": {"api": {"rows": [
            {"Name": "cudaLaunchKernel", "Total Time (ns)": str(launch_total), "Num Calls": str(launch_calls)},
        ]}, "kernel": {"rows": [{"x": 1}]}},
        "kernel_stage_mapping": {"aggregate": {
            "attention_qkv": {"total_ns": qkv_total, "instances": qkv_instances, "share": 0.5},
            "ffn": {"total_ns": 0, "instances": 0, "share": 0.0},
            "kv": {"total_ns": 0, "instances": 0, "share": 0.0},
            "lm_head": {"total_ns": 0, "instances": 0, "share": 0.0},
            "unknown": {"share": 0.5},
        }},
        "gguf": {"gguf": {"sha256": "x"}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stage_holdout_report_calibrates_known_stage_and_blocks_missing_kv(tmp_path):
    train, holdout = tmp_path / "train.json", tmp_path / "holdout.json"
    _profile(train, qkv_total=100, qkv_instances=10, launch_total=1000, launch_calls=10)
    _profile(holdout, qkv_total=200, qkv_instances=20, launch_total=1800, launch_calls=20)
    result = build(train, holdout)
    assert result["stages"]["attention_qkv"]["holdout_relative_error_pct"] == 0.0
    assert result["stages"]["kv"]["status"] == "blocked_no_semantic_evidence"
    assert result["api"]["launch"]["holdout_relative_error_pct"] == 11.11111111111111
