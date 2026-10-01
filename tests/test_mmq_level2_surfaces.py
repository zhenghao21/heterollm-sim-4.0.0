from dataclasses import replace
import json
from pathlib import Path

import pytest

from heterollm_sim.mmq_level2_surfaces import (
    MMQ_REQUIRED_PREFILL_M,
    MMQStageSample,
    evaluate_mmq_stage_holdout,
    mmq_stage_dispatch_signature,
    mmq_stage_source_binding,
    uncalibrated_mmq_stage_surface,
)
from heterollm_sim.mmq_work import derive_mmq_work


def source(m=512, n=3072, k=5120):
    return derive_mmq_work(m=m, n=n, k=k, weight_format="Q4_K", sm_count=84,
                           shared_memory_per_block=101376,
                           runtime_binary_sha256="a" * 64)


def sample(stage, shape, split="train", signature=None):
    item = source(*shape)
    return MMQStageSample(stage, "q4_k", *shape, 100.0, 100.0, 1.0, 3,
                          "synthetic-stage-evidence", signature or mmq_stage_dispatch_signature(stage, item),
                          "a" * 64, split=split)


def test_stage_binding_and_dispatch_are_distinct_from_main_kernel():
    item = source()
    conversion = mmq_stage_dispatch_signature("activation_repack", item)
    fixup = mmq_stage_dispatch_signature("stream_k_fixup", item)
    assert conversion != fixup
    assert conversion.startswith("mmq_repack:")
    assert fixup.startswith("mmq_fixup:")
    binding = mmq_stage_source_binding("activation_repack", item)
    assert binding["runtime_binary_sha256"] == "a" * 64
    assert binding["timing_surface_bound"] is False


def test_missing_stage_evidence_fails_closed_even_with_runtime_binding():
    item = source()
    surface = uncalibrated_mmq_stage_surface(
        "stream_k_fixup", "Q4_K", runtime_binary_sha256="a" * 64,
        dispatch_signature=mmq_stage_dispatch_signature("stream_k_fixup", item),
    )
    prediction = surface.predict((512, 3072, 5120), runtime_binary_sha256="a" * 64,
                                 dispatch_signature=surface.dispatch_signature)
    assert prediction["accepted"] is False
    assert prediction["reason"] == "no_independent_stage_holdout_surface"


def test_stage_holdout_rejects_unmeasured_m_sweep():
    item = source()
    sig = mmq_stage_dispatch_signature("activation_repack", item)
    train = [sample("activation_repack", (512, 2048, 2048), signature=sig)]
    holdout = [sample("activation_repack", (64, 2048, 2048), split="holdout", signature=sig)]
    result = evaluate_mmq_stage_holdout(stage="activation_repack", weight_format="Q4_K",
                                        training=train, holdout=holdout,
                                        runtime_binary_sha256="a" * 64,
                                        dispatch_signature=sig)
    assert result["accepted"] is False
    assert result["domain_rejected_m"] == (64,)
    assert tuple(result["required_prefill_m"]) == MMQ_REQUIRED_PREFILL_M


def test_stage_artifact_records_fail_closed_m_sweep():
    root = Path(__file__).resolve().parents[1] / "artifacts/development/level2_mmq_stage_20260930"
    report = json.loads((root / "holdout_evaluation.json").read_text(encoding="utf-8"))
    assert report["required_prefill_m"] == [64, 128, 256, 512, 1024]
    if set(report["required_prefill_m"]) <= set(report["measured_prefill_m"]):
        assert all(
            item.get("production_accepted") in {True, False}
            for stage in report["stage_evaluation"].values()
            for fmt in stage.values()
            for item in fmt.values()
        )
    else:
        assert report["accepted_stages"] == []
        assert "m_sweep_missing_64_128_256_1024" in report["rejection_reasons"]
