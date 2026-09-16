"""MMVQ adapter checks: static facts only; no native/GPU/LLM execution."""
import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from tools import predict_stable_native_dataset as adapter
from tests.test_predict_stable_native_dataset import fixture, document, seal

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "artifacts/development/native_long_grid_135_20260915/optimization_loop"


@pytest.fixture
def facts(tmp_path, monkeypatch):
    from heterollm_sim import mmvq_issue_bound, mmvq_work, conversion_work
    from tests.test_mmvq_mechanism import _qualified_mmvq_case
    scenario = _qualified_mmvq_case()
    selection_path, selection, row, calls = fixture(tmp_path, monkeypatch)
    pdf = tmp_path / "official-fixture.pdf"
    pdf.write_bytes(b"%PDF-1.7 synthetic unit-test bytes; not actual NVIDIA evidence")
    monkeypatch.setattr(mmvq_issue_bound, "HARDWARE_DOCUMENT_SHA256", hashlib.sha256(pdf.read_bytes()).hexdigest())
    # Synthetic source identities are isolated to these unit tests.
    monkeypatch.setattr(mmvq_work, "SOURCE_SHA256", {})
    monkeypatch.setattr(conversion_work, "SOURCE_SHA256", {})
    row["static_hardware"]["frozen_hardware"]["gpu"]["compute_capability"] = "12.0"
    hardware_ref = document(tmp_path / "hardware.json", row["static_hardware"]["frozen_hardware"])
    probe_ref = document(tmp_path / "device.json", {
        "schema": "cuda-driver-static-device-properties/v1", "gpu_uuid": "GPU-test",
        "native_configuration_modified": False, "attributes": {
            "CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT": 84,
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR": 12,
            "CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR": 0,
            "CU_DEVICE_ATTRIBUTE_WARP_SIZE": 32}})
    cuda = next(ref for ref in row["native_runtime_refs"] if ref["path"].endswith("ggml-cuda.dll"))
    device = {"available": True, "source_ref": probe_ref, "selected_hardware_ref": hardware_ref,
        "gpu_uuid": "GPU-test", "sm_count": 84, "warp_size": 32, "cuda_compute_capability": 1200}
    contract = {"cuda_compute_capability": 1200, "mmq_device_evidence": device,
        "source_refs": [], "runtime_modules": {"ggml-cuda.dll": cuda}}
    gpu = {"contract": contract, "mmq_source_costs_requested": True,
        "conversion_cta_costs_requested": True, "cuda_compute_capability": 1200,
        "device_evidence": {"gpu_uuid": "GPU-test"}}
    binding = {"cells": {row["cell_id"]: gpu}, "mmq_source_costs_requested": True,
        "conversion_cta_costs_requested": True, "evidence_refs": []}
    document(selection_path, seal(selection))
    issue = adapter.verified_mmvq_issue_binding([row], binding, tmp_path / "evidence-freeze", document_path=pdf)
    inputs = adapter.static_inputs(row, selection, tmp_path, gpu_invocation=binding, mmvq_issue=issue)
    components = []
    for component in scenario.hardware.components:
        if str(component.kind).lower() == "gpu":
            conversion = {**component.metadata["llama_cpp_conversion_source_contract"],
                "source_hashes": {}, "runtime_binary_sha256": cuda["sha256"]}
            component = replace(component, metadata={**component.metadata, "cuda_compute_capability": 1200,
                "llama_cpp_conversion_source_contract": conversion})
        components.append(component)
    scenario = replace(scenario, hardware=replace(scenario.hardware, components=tuple(components)))
    return dict(row=row, selection=selection, selection_path=selection_path, binding=binding,
        issue=issue, inputs=inputs, scenario=scenario, pdf=pdf, calls=calls)


def test_document_verifies_bytes_and_detects_mutation(facts):
    doc = facts["issue"]["hardware_document"]
    adapter.verify_mmvq_hardware_document(doc)
    assert doc["size_bytes"] == len(facts["pdf"].read_bytes())
    Path(doc["ref"]["path"]).write_bytes(b"%PDF-forged")
    with pytest.raises(ValueError, match="content SHA256"):
        adapter.verify_mmvq_hardware_document(doc)
    with pytest.raises(ValueError, match="bounded PDF"):
        adapter.checked_mmvq_document_bytes(b"not a PDF")


def test_unavailable_document_never_installs_switch(facts, tmp_path):
    issue = adapter.verified_mmvq_issue_binding([facts["row"]], facts["binding"], tmp_path / "missing",
        document_path=tmp_path / "not-present.pdf")
    proof = issue["cells"][facts["row"]["cell_id"]]
    assert proof["status"] == "uncovered" and proof["contract"] is None
    inputs = {**facts["inputs"], "mmvq_issue_evidence": proof, "mmvq_issue_contract": None}
    result = adapter.apply_mmvq_issue_static_contract(facts["scenario"], inputs)
    assert "llama_cpp_mmvq_vector_issue_bound" not in result.workload.metadata
    assert result.workload.metadata["llama_cpp_mmvq_vector_issue_qualification"]["status"] == "uncovered"


def test_worker_installs_only_rederived_conditional_contract(facts):
    inputs = facts["inputs"]
    result = adapter.apply_mmvq_issue_static_contract(facts["scenario"], inputs)
    assert result.workload.metadata["llama_cpp_mmvq_vector_issue_bound"] is True
    proof = result.workload.metadata["llama_cpp_mmvq_vector_issue_qualification"]
    assert proof["status"] == "conditional" and proof["native_instruction_mapping_proven"] is False
    assert proof["declared_clock"] == adapter.gpu_clock(inputs)
    gpu = next(c for c in result.hardware.components if str(c.kind).lower() == "gpu")
    assert gpu.metadata["llama_cpp_mmvq_vector_issue_contract"] == inputs["mmvq_issue_contract"]
    changed = {**inputs, "mmvq_vector_issue_bound": False}
    with pytest.raises(ValueError, match="switch"):
        adapter.apply_mmvq_issue_static_contract(facts["scenario"], changed)
    changed = copy.deepcopy(inputs)
    changed["mmvq_issue_contract"]["sm_count"] = 80
    with pytest.raises(ValueError, match="re-derived"):
        adapter.apply_mmvq_issue_static_contract(facts["scenario"], changed)


@pytest.mark.parametrize("change", ["sm", "uuid", "module", "sources", "source_bytes"])
def test_static_evidence_rejects_mismatches(facts, monkeypatch, tmp_path, change):
    from heterollm_sim import mmvq_work
    binding = copy.deepcopy(facts["binding"])
    cell = binding["cells"][facts["row"]["cell_id"]]
    if change == "sm": cell["contract"]["mmq_device_evidence"]["sm_count"] = 80
    if change == "uuid": cell["device_evidence"]["gpu_uuid"] = "GPU-other"
    if change == "module": cell["contract"]["runtime_modules"]["ggml-cuda.dll"]["sha256"] = "b" * 64
    if change in {"sources", "source_bytes"}:
        source = tmp_path / "ggml-cuda/test.cu"
        source.parent.mkdir()
        source.write_text("static source")
        ref = adapter.grid.file_ref(source)
        monkeypatch.setattr(mmvq_work, "SOURCE_SHA256", {"ggml-cuda/test.cu": ref["sha256"]})
        if change == "source_bytes":
            cell["contract"]["source_refs"] = [ref]
            source.write_text("changed source")
    with pytest.raises(ValueError):
        adapter.verified_mmvq_issue_binding([facts["row"]], binding, tmp_path / "bad", document_path=facts["pdf"])


def test_worker_rejects_profile_and_conversion_mismatch(facts):
    scene = facts["scenario"]
    profiles = {**scene.component_profiles, "gpu": {key: replace(profile,
        tensor_core=replace(profile.tensor_core, sm_count=80)) for key, profile in scene.component_profiles["gpu"].items()}}
    with pytest.raises(ValueError, match="runtime/profile"):
        adapter.apply_mmvq_issue_static_contract(replace(scene, component_profiles=profiles), facts["inputs"])
    components = tuple(replace(c, metadata={key: value for key, value in c.metadata.items()
        if key != "llama_cpp_conversion_source_contract"}) for c in scene.hardware.components)
    with pytest.raises(ValueError, match="installed conversion"):
        adapter.apply_mmvq_issue_static_contract(replace(scene, hardware=replace(scene.hardware, components=components)), facts["inputs"])


def test_default_off_is_identity_and_does_not_read_document(tmp_path, monkeypatch):
    selection_path, selection, row, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "freeze_mmvq_hardware_document", lambda *a, **k: pytest.fail("default off read PDF"))
    baseline = adapter.static_inputs(row, selection, tmp_path)
    frozen = adapter.freeze_selection(selection_path, tmp_path / "off", data_root=tmp_path, mmvq_vector_issue_bound=False)
    assert frozen["cells"][0]["static_inputs"] == baseline
    assert "mmvq_issue_bound" not in frozen and "mmvq_vector_issue_bound" not in baseline
    sentinel = object()
    assert adapter.apply_mmvq_issue_static_contract(sentinel, {}) is sentinel
    assert adapter.apply_mmvq_issue_static_contract(sentinel, {"mmvq_vector_issue_bound": False}) is sentinel


def test_freeze_persists_per_cell_and_detects_switch_tamper(facts, tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "verified_gpu_invocation_contract", lambda *a, **k: copy.deepcopy(facts["binding"]))
    frozen = adapter.freeze_selection(facts["selection_path"], tmp_path / "on", data_root=tmp_path,
        gpu_invocation_contract_path=tmp_path / "template.json", gpu_mmq_source_costs=True,
        gpu_conversion_cta_costs=True, mmvq_vector_issue_bound=True,
        mmvq_issue_hardware_document_path=facts["pdf"])
    adapter.verify_freeze_references(frozen)
    inputs = frozen["cells"][0]["static_inputs"]
    assert frozen["mmvq_vector_issue_bound"] is True and inputs["mmvq_vector_issue_bound"] is True
    assert inputs["mmvq_issue_contract"]["sm_count"] == 84
    assert "native_latency_ms" not in json.dumps(inputs) and "123456" not in json.dumps(inputs)
    inputs["mmvq_vector_issue_bound"] = False
    with pytest.raises(ValueError, match="cell switch"):
        adapter.verify_freeze_references(frozen)


@pytest.mark.parametrize("option", ["--mmvq-vector-issue-bound", "--no-mmvq-vector-issue-bound", "--mmvq-issue-hardware-document"])
def test_resume_cannot_change_switch_or_document(tmp_path, option):
    args = ["--output", str(tmp_path), "--resume", option]
    if option.endswith("document"): args.append("changed.pdf")
    with pytest.raises(SystemExit) as exc:
        adapter.main(args)
    assert exc.value.code == 2


def test_initial_freeze_cli_accepts_switch(tmp_path, monkeypatch):
    calls = []
    real_freeze_selection = adapter.freeze_selection
    monkeypatch.setattr(adapter, "freeze_selection", lambda *a, **kw: calls.append(kw))
    adapter.main(["--selection", "selection.json", "--output", str(tmp_path), "--freeze-only",
        "--gpu-invocation-contract", "contract.json", "--gpu-mmq-source-costs",
        "--gpu-conversion-cta-costs", "--mmvq-vector-issue-bound"])
    assert calls[0]["mmvq_vector_issue_bound"] is True
    with pytest.raises(ValueError, match="requires"):
        real_freeze_selection("unused.json", tmp_path / "invalid", mmvq_vector_issue_bound=True)


def test_real_131_static_cells_prepare_with_actual_pdf(tmp_path):
    """Explicit local PDF opt-in; reads static freeze fields only, never predictions."""
    path = os.environ.get("MMVQ_TEST_HARDWARE_DOCUMENT")
    freeze_path = ROUND / "round_021/physical/freeze.json"
    if not path or not freeze_path.is_file():
        pytest.skip("set MMVQ_TEST_HARDWARE_DOCUMENT to a local PDF path or download for real 131-cell static preparation")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert len(frozen["cells"]) == 131
    rows = [{"cell_id": cell["cell_id"],
        "static_hardware": {"frozen_hardware": cell["static_inputs"]["hardware_snapshot"]},
        "native_runtime_refs": cell["static_inputs"]["runtime_module_refs"]}
        for cell in frozen["cells"]]
    # R21 physical is intentionally CTA-off. Build a test-only static binding
    # for the new opt-in freeze contract; do not mutate the frozen evidence.
    binding_cells = {}
    for cell in frozen["cells"]:
        evidence = json.loads(json.dumps(cell["static_inputs"]["gpu_invocation_evidence"]))
        evidence["conversion_cta_costs_requested"] = True
        binding_cells[cell["cell_id"]] = evidence
    binding = {"cells": binding_cells,
        "runtime_source_binding_ref": frozen["gpu_invocation"]["runtime_source_binding_ref"],
        "mmq_source_costs_requested": True, "conversion_cta_costs_requested": True}
    result = adapter.verified_mmvq_issue_binding(rows, binding, tmp_path / "real", document_path=None if path == "download" else path)
    assert result["conditional_cell_count"] == 131 and result["uncovered_cell_count"] == 0
    assert result["hardware_document"]["content_bytes_verified"] is True
    sources = result["mmvq_source_binding"]
    assert sources["status"] == "conditional" and sources["snapshot_identity_verified"] is True
    assert sources["original_compile_header_bytes_proven"] is False
    assert sources["header_snapshot_ref"]["sha256"] == "9400f25f32d530685c0b5cc5c5bb104aef6dff96bdce2f8ae348f74406019686"
    assert any(ref["path"].endswith("vecdotq.cuh") for ref in sources["source_refs"])
    assert len(result["cells"]) == 131 and result["native_latency_used"] is False
    for cell in frozen["cells"]:
        proof = result["cells"][cell["cell_id"]]
        assert proof["contract"]["sm_count"] == 84
        assert proof["contract"]["runtime_binary_sha256"] == cell["static_inputs"]["gpu_invocation_contract"]["runtime_modules"]["ggml-cuda.dll"]["sha256"]

    # Exercise installation with a tiny synthetic scenario and the actual static
    # proofs. No model/engine run, GGUF read, or target answer is involved.
    from tests.test_mmvq_mechanism import _qualified_mmvq_case
    from heterollm_sim.mmvq_work import SOURCE_SHA256
    first = frozen["cells"][0]
    inputs = copy.deepcopy(first["static_inputs"])
    proof = result["cells"][first["cell_id"]]
    inputs.update(gpu_conversion_cta_costs=True, mmvq_vector_issue_bound=True,
        gpu_invocation_evidence=binding_cells[first["cell_id"]],
        mmvq_issue_contract=proof["contract"], mmvq_issue_evidence=proof)
    scene = _qualified_mmvq_case()
    components = tuple(replace(c, metadata={**c.metadata, "cuda_compute_capability": 1200,
        "llama_cpp_conversion_source_contract": {
            **c.metadata["llama_cpp_conversion_source_contract"],
            "runtime_binary_sha256": proof["contract"]["runtime_binary_sha256"]}})
        if str(c.kind).lower() == "gpu" else c for c in scene.hardware.components)
    scene = replace(scene, hardware=replace(scene.hardware, components=components),
        workload=replace(scene.workload, metadata={**scene.workload.metadata,
            "llama_cpp_gpu_native_invocations": {"applied": True,
                "source_refs": inputs["gpu_invocation_contract"]["source_refs"]}}))
    installed = adapter.apply_mmvq_issue_static_contract(scene, inputs)
    refs = installed.workload.metadata["llama_cpp_gpu_native_invocations"]["source_refs"]
    for relative, expected in SOURCE_SHA256.items():
        matches = [ref for ref in refs if ref["path"].replace(chr(92), "/").endswith("/" + relative)]
        assert len(matches) == 1 and matches[0]["sha256"] == expected
    assert installed.workload.metadata["llama_cpp_mmvq_vector_issue_bound"] is True
    assert installed.workload.metadata["llama_cpp_mmvq_vector_issue_qualification"]["mmvq_source_binding"]["original_compile_header_bytes_proven"] is False
    changed = copy.deepcopy(inputs)
    changed["mmvq_issue_evidence"]["mmvq_source_binding"]["original_compile_header_bytes_proven"] = True
    with pytest.raises(ValueError, match="recorded build/header history"):
        adapter.apply_mmvq_issue_static_contract(scene, changed)
