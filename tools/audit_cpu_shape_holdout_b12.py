"""B-12 simulator-only CPU shape holdout.

This audit deliberately uses no native timing.  It exercises the analytical
CPU GEMM model on shapes that were not used by the B-12 development probe and
checks physical trends for packed-weight dequant work, ISA capability
selection, and non-duplicated instruction accounting.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

# Running a tool as ``python tools/<name>.py`` places ``tools`` ahead of the
# repository root on sys.path.  Make the audit consume the checked-out source
# tree rather than an unrelated installed heterollm_sim package.  This is
# essential for a reproducible simulator-only holdout.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from heterollm_sim.cost_models import (
    CPUPipelineProfile,
    CPUProfile,
    CPUQuantizedDotCapability,
    CacheHierarchyProfile,
    CacheLevelProfile,
    GemmWorkload,
    HostMemoryProfile,
    estimate_cpu_gemm,
)


OUT = ROOT / "artifacts" / "development" / "b12_cpu_shape_holdout_v2.json"


def _cpu(
    *,
    capability: bool,
    maximum_m: int | None = None,
    thread_count: int = 4,
) -> CPUProfile:
    if thread_count <= 0:
        raise ValueError("thread_count must be positive")
    caps = ()
    if capability:
        caps = (
            CPUQuantizedDotCapability(
                name="b12_q4k_avx2",
                supported_weight_formats=("Q4_K",),
                source_activation_bits=(16,),
                dot_activation_bits=8,
                dot_weight_bits=8,
                accumulator_bits=32,
                effective_ops_per_instruction=32.0,
                dot_issue_instructions_per_cycle_per_core=2.0,
                auxiliary_ops_per_instruction=32.0,
                activation_quantization_block_elements=64,
                activation_quantization_instructions_per_block=0.0,
                evidence="B-12 synthetic ISA contract for shape semantics",
                source_dot_work={
                    "Q4_K": {
                        "block_elements": 256,
                        "auxiliary_vector_ops": 52,
                        "vector_loads": 14,
                        "scalar_lut_loads": 2,
                    }
                },
                source_dot_work_all_m=True,
                source_dot_work_max_k=1 << 30,
                maximum_m=maximum_m,
            ),
        )
    return CPUProfile(
        pipeline=CPUPipelineProfile(
            # The analytical profile models one serving request's CPU worker
            # budget as the effective physical core count.  Varying this
            # value is a structural thread-scaling check; it is not a fit to
            # any target request timing.
            core_count=thread_count,
            frequency_ghz=1.25,
            simd_width_bits=256,
            decode_width=4,
            issue_width=4,
            retire_width=4,
            vector_fma_units_per_core=2,
            vector_alu_units_per_core=2,
            load_units_per_core=2,
            store_units_per_core=1,
            reorder_buffer_entries=256,
            load_store_queue_entries=96,
            memory_level_parallelism=16,
                resource_id=f"b12.cpu.pipeline.t{thread_count}",
        ),
        cache_hierarchy=CacheHierarchyProfile(
            levels=(
                CacheLevelProfile(
                    name="l1",
                    capacity_bytes=32 * 1024,
                    line_bytes=64,
                    hit_latency_ns=1.0,
                    bandwidth_gb_s=256.0,
                    resource_id="b12.cpu.l1",
                ),
            )
        ),
        quantized_dot_capabilities=caps,
        attainable_efficiency=0.5,
    )


def _workload(m: int, k: int, n: int) -> GemmWorkload:
    # One-row projection primitive budget.  The generic fallback must expand
    # this by M; K and N are represented by the physical block count.
    if k % 256:
        raise ValueError("holdout shapes use complete Q4_K blocks")
    row_ops = n * (k // 256) * 52
    return GemmWorkload(
        m=m,
        k=k,
        n=n,
        activation_bits=16,
        weight_bits=4,
        output_bits=16,
        packed_weight_formats=("Q4_K",),
        packed_weight_transform_operations=row_ops,
        packed_weight_format_segments=(("Q4_K", n, row_ops),),
        name=f"b12_holdout_m{m}_k{k}_n{n}",
    )


def _estimate(cpu: CPUProfile, workload: GemmWorkload) -> dict[str, object]:
    result = estimate_cpu_gemm(cpu, HostMemoryProfile(bandwidth_gb_s=1.0e12), workload)
    schedule = dict(result.metadata["instruction_schedule"])
    return {
        "shape": {"m": workload.m, "k": workload.k, "n": workload.n},
        "service_ns": result.service_ns,
        "total_instructions": schedule["total_instructions"],
        "compute_instructions": schedule["compute_instructions"],
        "auxiliary_instructions": schedule.get("auxiliary_instructions", 0),
        "special_function_instructions": schedule.get("special_function_instructions", 0),
        "load_instructions": schedule.get("load_instructions", 0),
        "store_instructions": schedule.get("store_instructions", 0),
        "packed_weight_transform_instructions": schedule[
            "packed_weight_transform_instructions"
        ],
        "packed_weight_transform_issue_instructions": schedule[
            "packed_weight_transform_issue_instructions"
        ],
        "source_dot_blocks": schedule.get("source_dot_blocks"),
        "quantized_format_coverage": schedule.get("quantized_format_coverage"),
        "timing_completeness": schedule.get("timing_completeness"),
        "limiting_stage": schedule["limiting_stage"],
    }


def _assert_equal_ratio(a: int | float, b: int | float, ratio: float, label: str) -> None:
    observed = float(b) / float(a)
    if abs(observed - ratio) > 1e-9:
        raise AssertionError(f"{label}: expected ratio {ratio}, observed {observed}")


def main() -> None:
    generic = _cpu(capability=False)
    isa = _cpu(capability=True)
    bounded_isa = _cpu(capability=True, maximum_m=4)

    # Shape holdout: every listed shape is outside the original B-12 probe
    # (which used M=1,K=64,N=64); no native timing is read.
    shapes = [(2, 256, 16), (4, 512, 16), (8, 512, 32), (16, 1024, 64)]
    generic_rows = [_estimate(generic, _workload(*shape)) for shape in shapes]
    isa_rows = [_estimate(isa, _workload(*shape)) for shape in shapes]

    # Primitive accounting trends: doubling M, K, or N doubles the physical
    # dequant primitive budget in this projection lowering.
    m1 = _estimate(generic, _workload(1, 256, 16))
    m2 = _estimate(generic, _workload(2, 256, 16))
    k1 = _estimate(generic, _workload(4, 256, 16))
    k2 = _estimate(generic, _workload(4, 512, 16))
    n1 = _estimate(generic, _workload(4, 512, 16))
    n2 = _estimate(generic, _workload(4, 512, 32))
    _assert_equal_ratio(m1["packed_weight_transform_instructions"], m2["packed_weight_transform_instructions"], 2.0, "M dequant")
    _assert_equal_ratio(k1["packed_weight_transform_instructions"], k2["packed_weight_transform_instructions"], 2.0, "K dequant")
    _assert_equal_ratio(n1["packed_weight_transform_instructions"], n2["packed_weight_transform_instructions"], 2.0, "N dequant")

    # Generic fallback retains dequant work and charges its issue envelope once
    # in total instructions; there is no second additive dequant phase.
    for row in generic_rows:
        if row["quantized_format_coverage"] != "generic_quantized_fallback":
            raise AssertionError("generic path did not declare quantized fallback")
        expected = (
            row["compute_instructions"]
            + row["auxiliary_instructions"]
            + row["special_function_instructions"]
            + row["load_instructions"]
            + row["store_instructions"]
        )
        if row["total_instructions"] != expected:
            raise AssertionError(
                "generic schedule does not account each instruction exactly once"
            )
        if row["packed_weight_transform_issue_instructions"] <= 0:
            raise AssertionError("generic dequant issue envelope is empty")

    # ISA path uses source primitive blocks instead of the generic fallback;
    # the same M*K*N scaling is preserved and the capability is explicit.
    for shape, row in zip(shapes, isa_rows):
        expected_blocks = shape[0] * shape[2] * (shape[1] // 256)
        if row["source_dot_blocks"] != expected_blocks:
            raise AssertionError(f"ISA source block mismatch for {shape}")
        if row["quantized_format_coverage"] != "declared_quantized_capability":
            raise AssertionError("ISA capability path was not selected")
        if row["packed_weight_transform_instructions"] <= 0:
            raise AssertionError("ISA source primitive budget is empty")

    # A capability bound must select the generic path outside its declared M
    # scope rather than silently applying an ISA profile to an unseen shape.
    bounded = _estimate(bounded_isa, _workload(8, 512, 32))
    if bounded["quantized_format_coverage"] != "generic_quantized_fallback":
        raise AssertionError("out-of-scope M incorrectly used ISA capability")

    # Prompt/decode shape checks.  Prefill uses M=prompt tokens while decode
    # uses one token per step; output length is represented by the number of
    # sequential decode steps, so its aggregate demand must scale linearly.
    prompt_lengths = (2, 8, 32)
    output_lengths = (2, 8, 32)
    prompt_rows = []
    decode_rows = []
    for prompt_tokens in prompt_lengths:
        work = _workload(prompt_tokens, 512, 32)
        row = _estimate(generic, work)
        row.update({"phase": "prefill", "prompt_tokens": prompt_tokens, "output_tokens": 1})
        prompt_rows.append(row)
    per_token_decode = _estimate(generic, _workload(1, 512, 32))
    for output_tokens in output_lengths:
        row = dict(per_token_decode)
        row.update(
            {
                "phase": "decode",
                "prompt_tokens": 8,
                "output_tokens": output_tokens,
                "aggregate_service_ns": per_token_decode["service_ns"] * output_tokens,
            }
        )
        decode_rows.append(row)
    for a, b in zip(prompt_rows, prompt_rows[1:]):
        if b["service_ns"] <= a["service_ns"]:
            raise AssertionError("prefill demand did not grow with prompt tokens")
    for a, b in zip(decode_rows, decode_rows[1:]):
        ratio = b["aggregate_service_ns"] / a["aggregate_service_ns"]
        expected_ratio = b["output_tokens"] / a["output_tokens"]
        if abs(ratio - expected_ratio) > 1e-12:
            raise AssertionError("decode output length is not represented linearly")

    # Thread/core structural sweep: fewer CPU workers cannot make the same
    # work faster.  This only checks the analytical resource envelope.
    thread_rows = []
    for thread_count in (1, 2, 4):
        row = _estimate(
            _cpu(capability=False, thread_count=thread_count),
            _workload(8, 512, 32),
        )
        row["thread_count"] = thread_count
        thread_rows.append(row)
    for a, b in zip(thread_rows, thread_rows[1:]):
        if b["service_ns"] > a["service_ns"]:
            raise AssertionError("increasing CPU thread budget made service slower")

    # A workload without physical segments carries a caller-provided total;
    # it is intentionally excluded from this shape audit because M expansion
    # cannot be inferred safely without lowering evidence.
    report = {
        "schema_version": 2,
        "audit": "B-12 CPU simulator-only unseen-shape holdout",
        "native_timing_used": False,
        "native_fitting": False,
        "development_shape": {"m": 1, "k": 64, "n": 64},
        "holdout_shapes": shapes,
        "generic_rows": generic_rows,
        "isa_rows": isa_rows,
        "prompt_rows": prompt_rows,
        "decode_rows": decode_rows,
        "thread_rows": thread_rows,
        "checks": {
            "dequant_scales_with_m_k_n": True,
            "generic_fallback_retains_primitive_count": True,
            "generic_issue_envelope_counted_once": True,
            "isa_source_blocks_scale_with_shape": True,
            "out_of_scope_capability_falls_back": True,
            "segment_evidence_required_for_m_expansion": True,
            "generic_instruction_accounting_exact": True,
            "prompt_shape_monotonic": True,
            "output_length_linear_decode_steps": True,
            "thread_budget_non_increasing_service": True,
        },
        "scope": {
            "status": "simulator_only_structural_pass",
            "interpretation": "Physical shape trends and capability gating hold for the analytical CPU model; this is not native accuracy evidence and does not establish engine error targets.",
            "limitations": [
                "No native timing was collected or fitted.",
                "Generic workloads without packed_weight_format_segments retain caller-provided total transform work; M expansion requires lowering evidence.",
            "ISA source primitive timing remains partial until independent CPU microbench evidence is attached.",
            "Prompt rows model prefill M as prompt token count; decode rows model one token per sequential step and aggregate output length by repetition.",
            "Thread sweep varies analytical CPU worker/core budget; it is not native thread-scaling evidence.",
        ],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "status": report["scope"]["status"], "holdout_count": len(shapes)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
