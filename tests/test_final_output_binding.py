"""Bounded source/static/production binding tests, never a native or LLM run."""
import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from heterollm_sim import planner
from heterollm_sim.gguf_parity import GGUFMetadata, GGUFTensor, build_model_from_gguf
from heterollm_sim.final_layer_output_selection import SOURCE_KEY, model_declaration, source_declaration
from tools import native_final_output_binding as binding
from tools import predict_stable_native_dataset as adapter
from tools.native_sampling_contract import _profile_id
from tests.test_final_layer_output_selection_planner import _scenario, _cohort


def document(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return binding.ref(path)


def tiny_gguf(arch="qwen2"):
    tensors = []
    h = 256 if arch == "qwen35" else 32
    def add(name, shape):
        elements = 1
        for dim in shape:
            elements *= dim
        tensors.append(GGUFTensor(name, shape, 12 if arch == "qwen35" else 0,
                                   "Q4_K" if arch == "qwen35" else "F32",
                                   256 if arch == "qwen35" else 1,
                                   elements * 144 // 256 if arch == "qwen35" else elements * 4, 0))
    for il in range(2):
        for name, shape in (("attn_q", (h, h * 2 if arch == "qwen35" else h)),
                            ("attn_k", (h, h)), ("attn_v", (h, h)), ("attn_output", (h, h)),
                            ("ffn_gate", (h, h * 2)), ("ffn_up", (h, h * 2)), ("ffn_down", (h * 2, h))):
            add(f"blk.{il}.{name}.weight", shape)
    add("token_embd.weight", (h, 64)); add("output.weight", (h, 64))
    return GGUFMetadata("synthetic-no-weight-read.gguf", "1" * 64, 3, len(tensors), 0,
                        arch, 2, h, 1, 1, 64, 4096, "ALL_F32", 0, {}, tuple(tensors))


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """Existing native build verifier is a seam; source/binding logic is real."""
    root = tmp_path / "native"
    paths = {a: root / "src/models" / (a + ".cpp") for a in binding.GRAPH_ARCHITECTURES}
    paths.update(graph=root / "src/llama-graph.cpp", model=root / "src/llama-model.cpp",
                 context=root / "overlay/llama-context.cpp", server_context=root / "server-context.cpp",
                 server_task_header=root / "tools/server/server-task.h", speculative=root / "common/speculative.cpp")
    reviewed = {}
    for role, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("independently reviewed synthetic source " + role, encoding="utf-8")
        reviewed[role] = binding.ref(path)["sha256"]
    monkeypatch.setattr(binding, "REVIEWED_SOURCES", reviewed)
    collector = tmp_path / "collector.py"
    collector.write_text("raise RuntimeError('never execute a collector')", encoding="utf-8")
    collector_ref = binding.ref(collector)
    monkeypatch.setattr(binding, "REVIEWED_COMPLETION_COLLECTORS", frozenset({collector_ref["sha256"]}))
    module_refs = [document(tmp_path / "bin" / name, {"fake_binary": name})
                   for name in ("llama-server.exe", "llama.dll", "ggml-cuda.dll")]
    module_map = {Path(r["path"]).name: r for r in module_refs}
    runtime = {"contract": {"runtime_modules": module_map},
               "contract_ref": document(tmp_path / "runtime.json", {"runtime_modules": module_map}),
               "runtime_build_audit_ref": document(tmp_path / "audit.json", {"static": True}),
               "base_build_receipt_ref": document(tmp_path / "base.json", {"static": True})}
    linkage = {"source_compilation": {role: {"source": str(paths[role]), "source_sha256": reviewed[role],
               "module": module_map["llama.dll"], "recorded_object_sha256": "2" * 64}
               for role in ("graph", "model", "context")}, "evidence_refs": []}
    effective = {"backend_sampling": False, "speculative.types": "none,none", "lora": []}
    typed = {"mode": "greedy", "temperature": 0., "implementation": "llama_cpp_cpu_chain", "top_k": 1, "top_p": .95, "min_p": .05, "min_keep": 0}
    selection_ref = document(tmp_path / "selection.json", {"static_only": True})
    sampling_cell = {"backend_sampling": False, "effective_settings": effective, "typed_policy": typed,
        "provenance": {"collector_source_ref": collector_ref, "selection_ref": selection_ref,
                       "native_lineage": {"verified_elsewhere": True}, "resolved_request_settings_checked": {"measured": 2}}}
    sampling_doc = {"source_snapshots": {"server_context": {"original": binding.ref(paths["server_context"])}},
        "selection_ref": selection_ref, "effective_profiles": {_profile_id(effective): effective},
        "cells": [{"cell_id": "unit", "sampling_profile_id": _profile_id(effective),
                   "typed_policy_candidate": typed, "backend_sampling": False}]}
    sampling = {"contract_ref": document(tmp_path / "sampling.json", sampling_doc), "cells": {"unit": sampling_cell}}
    source = binding.make_source_contract(runtime, sampling, linkage)
    monkeypatch.setattr(binding, "rederive_source", lambda _source: binding.make_source_contract(runtime, sampling, linkage))
    gguf = tiny_gguf()
    cfg = {"batch": 64, "ubatch": 64, "parallel": 1, "ctx": 2048, "gpu_layers": -1,
           "op_offload": True, "flash_attn": False, "seed": 42, "output": 2, "expected_prompt_tokens": 64,
           "prompt_token_ids": list(range(64)), "model": gguf.path, "model_sha256": gguf.sha256,
           "threads": 16, "threads_batch": 16, "gpu_sm_clock_mhz": 2100}
    model_ref = {"path": gguf.path, "sha256": gguf.sha256}
    inputs = {"cell_id": "unit", "config": cfg, "prediction_model_ref": model_ref, "native_model_ref": model_ref,
              "runtime_ref": module_refs[0], "runtime_module_refs": module_refs[1:], "sampling_binding": sampling_cell,
              "hardware_snapshot": None, binding.FLAG: True}
    def refresh(architecture="qwen2"):
        inputs[binding.INPUT_KEY] = binding.derive_cell(source, cell_id="unit", model_ref=model_ref,
            model_scope={"architecture": architecture}, config=cfg, native_refs=module_refs, sampling=sampling_cell)
    refresh()
    return SimpleNamespace(inputs=inputs, gguf=gguf, source=source, sampling=sampling, runtime=runtime,
                           linkage=linkage, refresh=refresh, paths=paths, module_refs=module_refs)


def production_scene(gguf):
    from tools.native_llama_compare import build_matching_scenario
    scene = build_matching_scenario(64, 2, ctx=2048, parallel=1, batch=64, ubatch=64, threads=16,
        gpu_layers=-1, model=build_model_from_gguf(gguf), runtime_binary="unselected-fixture-runtime.exe")
    # Existing independently tested storage adapter is outside this fixture.
    return replace(scene, workload=replace(scene.workload,
        metadata={**scene.workload.metadata, "llama_cpp_f32_hidden_storage": True}))


@pytest.mark.parametrize("arch", ["qwen2", "llama", "qwen35"])
def test_real_gguf_builder_binding_survives_final_replan(frozen, arch):
    gguf = tiny_gguf(arch); frozen.refresh(arch)
    scene = production_scene(gguf)
    assert model_declaration(scene.model) is None
    bound = binding.apply_binding(scene, frozen.inputs, gguf=gguf)
    final, audit = adapter.replan_final_static_scenario(bound)
    assert audit["normal_validation_passed"]
    assert model_declaration(final.model) == source_declaration()
    assert final.model.graph.attributes["metadata"].get(SOURCE_KEY) is None
    assert final.workload.metadata[binding.AUDIT_KEY]["status"] == "conditional"
    assert not final.workload.metadata[binding.AUDIT_KEY]["native_dispatch_proven"]
    assert final.model.architecture == scene.model.architecture


@pytest.mark.parametrize("arch", ["qwen2", "llama", "qwen35"])
@pytest.mark.parametrize("tokens,logits", [(64, 0), (64, 1), (4, 4)])
def test_production_architecture_rows_dependencies_and_no_duplicate_head(frozen, arch, tokens, logits):
    gguf = tiny_gguf(arch); frozen.refresh(arch)
    base = production_scene(gguf)
    bound, _ = adapter.replan_final_static_scenario(binding.apply_binding(base, frozen.inputs, gguf=gguf))
    cohort = _cohort(tokens, logits)
    legacy = planner.compile_serving_cohort_schedule(base, cohort)
    selected = planner.compile_serving_cohort_schedule(bound, cohort)
    before = {t.name: t for t in legacy.tasks}
    h = gguf.n_embd
    tail = tokens if arch == "qwen35" else logits
    ffns = [t for t in selected.tasks if '.layer-001.' in t.name and t.metadata.get('phase') == 'gpu_gemm'
            and t.metadata.get('projection_id') in {'mlp.up_gate', 'mlp.down'}]
    assert len(ffns) == (2 if tail else 0)
    for t in ffns:
        width = h if t.metadata['projection_id'] == 'mlp.up_gate' else 2 * h
        assert t.metadata['cost_model']['activation_bytes'] == tail * width * 4
    norms = [t for t in selected.tasks if t.metadata.get('event_kind') == 'final_norm_apply'
             and t.metadata.get('phase') == 'gpu_elementwise']
    assert [t.metadata['cost_model']['write_bytes'] for t in norms] == ([tail * h * 4] if tail else [])
    heads = [t for t in selected.tasks if t.metadata.get('event_kind') == 'lm_head_projection'
             and t.metadata.get('phase') == 'gpu_gemm']
    assert len(heads) == (1 if logits else 0)
    for t in heads:
        assert t.metadata['cost_model']['activation_bytes'] == logits * h * 4
    gathers = [t for t in selected.tasks if t.metadata.get('event_kind') == 'output_row_selection'
               and t.metadata.get('phase') != 'kernel_launch']
    assert len(gathers) == ((1 if arch == 'qwen35' else 2) if logits else 0)
    assert len([t for t in selected.tasks if t.metadata.get('event_kind') == 'output_row_indices']) == 1
    ids = {t.task_id for t in selected.tasks}
    assert all(d in ids for t in selected.tasks for d in t.dependencies)
    for t in selected.tasks:
        if t.metadata.get('projection_id', '').startswith('attention.') or '.layer-000.' in t.name:
            assert t.demands == before[t.name].demands
    # Repeat application must not add another output/head/input chain.
    again = binding.apply_binding(bound, frozen.inputs, gguf=gguf)
    assert planner.compile_serving_cohort_schedule(again, cohort) == selected


def test_default_off_is_identity_and_nested_declaration_cannot_silently_enable(frozen):
    scene = production_scene(frozen.gguf)
    assert binding.apply_binding(scene, {}, gguf=frozen.gguf) is scene
    hidden = replace(scene.model, metadata={**scene.model.metadata, 'metadata': {SOURCE_KEY: source_declaration()}})
    with pytest.raises(ValueError, match='noncanonical'):
        model_declaration(hidden)
    conflict = replace(hidden, metadata={**hidden.metadata, SOURCE_KEY: {**source_declaration(), 'embeddings': True}})
    with pytest.raises(ValueError, match='conflicting'):
        model_declaration(conflict)


@pytest.mark.parametrize('change', ['switch', 'digest', 'model', 'config', 'runtime', 'source_upgrade', 'collector', 'sampling'])
def test_corrupt_static_inputs_fail_closed_even_after_resealing(frozen, change):
    inputs = copy.deepcopy(frozen.inputs)
    if change == 'switch': inputs[binding.FLAG] = False
    if change == 'digest': inputs[binding.INPUT_KEY]['content_sha256'] = '0' * 64
    if change == 'model': inputs['prediction_model_ref']['sha256'] = '0' * 64
    if change == 'config': inputs['config']['parallel'] = 4
    if change == 'runtime': inputs['runtime_module_refs'][0]['sha256'] = '0' * 64
    if change == 'collector': inputs['sampling_binding']['provenance']['collector_source_ref']['sha256'] = '0' * 64
    if change == 'sampling': inputs['sampling_binding']['effective_settings']['speculative.types'] = 'draft-mtp'
    if change == 'source_upgrade':
        source = {k: v for k, v in inputs[binding.INPUT_KEY]['source_contract'].items() if k != 'content_sha256'}
        source['historical_include_content_proven'] = True
        inputs[binding.INPUT_KEY]['source_contract'] = binding.seal(source)
        inputs[binding.INPUT_KEY] = binding.seal({k: v for k, v in inputs[binding.INPUT_KEY].items() if k != 'content_sha256'})
    with pytest.raises(ValueError): binding.verify_cell(inputs)


def test_changed_source_and_foreign_runtime_mapping_are_not_accepted_by_digest_only(frozen):
    inputs = copy.deepcopy(frozen.inputs)
    source = {k: v for k, v in inputs[binding.INPUT_KEY]['source_contract'].items() if k != 'content_sha256'}
    source['runtime_modules']['llama.dll'] = '3' * 64
    inputs[binding.INPUT_KEY]['source_contract'] = binding.seal(source)
    inputs[binding.INPUT_KEY] = binding.seal({k: v for k, v in inputs[binding.INPUT_KEY].items() if k != 'content_sha256'})
    with pytest.raises(ValueError, match='re-derived'): binding.verify_cell(inputs)
    frozen.paths['qwen2'].write_text('changed source', encoding='utf-8')
    with pytest.raises(ValueError): binding.verify_cell(frozen.inputs)


def test_missing_completion_or_storage_evidence_stays_uncovered(frozen):
    frozen.inputs['sampling_binding']['provenance']['resolved_request_settings_checked'] = {}
    frozen.refresh()
    assert frozen.inputs[binding.INPUT_KEY]['status'] == 'uncovered'
    scene = production_scene(frozen.gguf)
    result = binding.apply_binding(scene, frozen.inputs, gguf=frozen.gguf)
    assert model_declaration(result.model) is None
    assert result.workload.metadata[binding.AUDIT_KEY]['status'] == 'uncovered'
    frozen.inputs['sampling_binding']['provenance']['resolved_request_settings_checked'] = {'measured': 1}
    frozen.refresh()
    scene = replace(scene, workload=replace(scene.workload, metadata={}))
    result = binding.apply_binding(scene, frozen.inputs, gguf=frozen.gguf)
    assert model_declaration(result.model) is None
    assert 'source_qualified_f32_hidden_storage_missing' in result.workload.metadata[binding.AUDIT_KEY]['reasons']


def test_campaign_must_match_each_static_cell_and_default_off_has_no_new_inputs(frozen):
    campaign = {'requested': True, 'source_contract': frozen.source, 'cells': {'unit': frozen.inputs[binding.INPUT_KEY]},
                'evidence_refs': frozen.source['evidence_refs']}
    freeze = {binding.FLAG: True, binding.INPUT_KEY: campaign, 'cells': [{'cell_id': 'unit', 'static_inputs': frozen.inputs}]}
    binding.verify_freeze(freeze)
    binding.verify_freeze({'cells': [{'static_inputs': {}}]})
    bad = copy.deepcopy(freeze); bad[binding.INPUT_KEY]['cells']['unit']['status'] = 'verified'
    with pytest.raises(ValueError): binding.verify_freeze(bad)
    with pytest.raises(ValueError): binding.verify_freeze({'cells': freeze['cells']})


@pytest.mark.parametrize("strict_identity", [True, False])
def test_predict_worker_applies_binding_before_real_final_replan_without_simulation(frozen, monkeypatch, strict_identity):
    if not strict_identity:
        frozen.paths["server_context"].write_text("later source revision", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(adapter.grid, 'read_gguf_metadata', lambda path: frozen.gguf)
    real_builder = adapter.grid.build_matching_scenario
    def builder(*args, **kwargs):
        kwargs['hardware_snapshot'] = None
        scene = real_builder(*args, **kwargs)
        return replace(scene, workload=replace(scene.workload,
            metadata={**scene.workload.metadata, 'llama_cpp_f32_hidden_storage': True}))
    monkeypatch.setattr(adapter.grid, 'build_matching_scenario', builder)
    class ReachedSimulationBoundary(Exception): pass
    def capture(scene, **kwargs):
        captured['scenario'] = scene
        raise ReachedSimulationBoundary()
    monkeypatch.setattr(adapter.grid.reporting, 'run_scenario', capture)
    with pytest.raises(ReachedSimulationBoundary):
        adapter.predict_cell(frozen.inputs, strict_identity=strict_identity)
    scene = captured['scenario']
    assert model_declaration(scene.model) == source_declaration()
    from heterollm_sim.control_plane_state import mapping_fingerprint_status
    assert mapping_fingerprint_status(scene)['mapping_stale'] is False


def test_relaxed_final_output_binding_keeps_semantics_but_skips_file_rechecks(frozen, monkeypatch):
    scene = production_scene(frozen.gguf)
    strict = binding.apply_binding(scene, frozen.inputs, gguf=frozen.gguf)
    def changed_reference(_value):
        raise ValueError("historical evidence changed")
    monkeypatch.setattr(binding, "verify_ref", changed_reference)
    with pytest.raises(ValueError, match="historical evidence changed"):
        binding.apply_binding(scene, frozen.inputs, gguf=frozen.gguf)
    relaxed = binding.apply_binding(scene, frozen.inputs, gguf=frozen.gguf, strict_identity=False)
    assert model_declaration(relaxed.model) == model_declaration(strict.model)
    assert relaxed.workload.metadata[binding.AUDIT_KEY] == strict.workload.metadata[binding.AUDIT_KEY]
    changed = copy.deepcopy(frozen.inputs)
    changed["config"]["batch"] += 1
    with pytest.raises(ValueError, match="cell proof differs"):
        binding.apply_binding(scene, changed, gguf=frozen.gguf, strict_identity=False)
    with pytest.raises(ValueError, match="worker GGUF identity differs"):
        binding.apply_binding(scene, frozen.inputs, gguf=replace(frozen.gguf, sha256="4" * 64), strict_identity=False)


def test_cli_refuses_enable_without_prerequisites_and_resume_mutation(tmp_path):
    with pytest.raises(SystemExit):
        adapter.main(['predict', '--output', str(tmp_path), '--selection', 'missing', '--final-output-selection'])
    with pytest.raises(SystemExit):
        adapter.main(['predict', '--output', str(tmp_path), '--resume', '--final-output-selection'])


def test_bound_model_serialization_preserves_canonical_owner(frozen):
    from heterollm_sim.config import model_from_dict
    from heterollm_sim.serde import to_primitive
    scene = binding.apply_binding(production_scene(frozen.gguf), frozen.inputs, gguf=frozen.gguf)
    restored = model_from_dict(to_primitive(scene.model))
    assert model_declaration(restored) == source_declaration()
    assert restored.graph.attributes['metadata'].get(SOURCE_KEY) is None


def test_wrong_gguf_or_scenario_identity_is_rejected(frozen):
    scene = production_scene(frozen.gguf)
    with pytest.raises(ValueError, match='GGUF identity'):
        binding.apply_binding(scene, frozen.inputs, gguf=replace(frozen.gguf, sha256='4' * 64))
    wrong = replace(scene.model, metadata={**scene.model.metadata,
                    'metadata': {**scene.model.metadata['metadata'], 'gguf_sha256': '4' * 64}})
    with pytest.raises(ValueError, match='bound GGUF'):
        binding.apply_binding(replace(scene, model=wrong), frozen.inputs, gguf=frozen.gguf)


def test_source_factory_qualification_cannot_be_upgraded_by_request(frozen):
    source = frozen.source
    assert source['qualification'] == 'conditional'
    assert source['historical_include_content_proven'] is False
    assert source['native_dispatch_proven'] is False
    assert source['declaration'] == source_declaration()
    assert len(source['source_refs']) == 9
    assert binding.checked(source) == source


def test_unknown_real_architecture_is_uncovered_without_model_name_fallback(frozen):
    frozen.refresh('display-name-Qwen2.5')
    assert frozen.inputs[binding.INPUT_KEY]['status'] == 'uncovered'
    assert 'gguf_architecture_unsupported' in frozen.inputs[binding.INPUT_KEY]['reasons']


def test_source_refs_need_all_reviewed_roles(frozen):
    source = copy.deepcopy(frozen.source)
    del source['source_refs']['qwen2']
    source = binding.seal({k: v for k, v in source.items() if k != 'content_sha256'})
    inputs = copy.deepcopy(frozen.inputs)
    proof = {k: v for k, v in inputs[binding.INPUT_KEY].items() if k != 'content_sha256'}
    proof['source_contract'] = source
    inputs[binding.INPUT_KEY] = binding.seal(proof)
    with pytest.raises(ValueError, match='coverage'):
        binding.verify_cell(inputs)


def test_any_mtp_presence_stays_uncovered(frozen):
    from heterollm_sim.ir import MTPPolicy
    scene = production_scene(frozen.gguf)
    scene = replace(scene, workload=replace(scene.workload, mtp=MTPPolicy(acceptance_rate=1.0)))
    candidate = binding.apply_binding(scene, frozen.inputs, gguf=frozen.gguf)
    assert model_declaration(candidate.model) is None
    assert 'mtp_execution_uncovered' in candidate.workload.metadata[binding.AUDIT_KEY]['reasons']


def test_frozen_configuration_mismatch_is_not_an_alias(frozen):
    scene = production_scene(frozen.gguf)
    scene = replace(scene, llama_cpp_config=replace(scene.llama_cpp_config, batch=128))
    candidate = binding.apply_binding(scene, frozen.inputs, gguf=frozen.gguf)
    assert model_declaration(candidate.model) is None
    assert 'scenario_runtime_shape_differs_from_frozen_configuration' in candidate.workload.metadata[binding.AUDIT_KEY]['reasons']


def test_output_rows_reenter_physical_dispatch_instead_of_scaling_old_m64_cost():
    from tests.test_llama_gpu_invocations import scenario, contract, projections
    from heterollm_sim.llama_gpu_invocations import apply_llama_gpu_invocation_contract
    device = {"available": True, "sm_count": 84, "max_shared_memory_per_block_optin_bytes": 101376}
    base = apply_llama_gpu_invocation_contract(scenario(m=64, fmt="Q5_K"),
        contract(device=device), enabled=True, enable_mmq_source_costs=True)
    selected = replace(base, model=replace(base.model,
        metadata={**base.model.metadata, SOURCE_KEY: source_declaration()}))
    tasks = projections(planner.compile_serving_cohort_schedule(selected, _cohort(64, 1)))
    attention = [t for t in tasks if t.metadata["projection_id"].startswith("attention.")]
    tails = [t for t in tasks if t.metadata["projection_id"].startswith("mlp.")]
    assert attention and tails
    assert all(t.metadata["gpu_native_invocation"]["m"] == 64 for t in attention)
    assert all(t.metadata["gpu_native_invocation"]["m"] == 1 for t in tails)
    down = next(t for t in tails if t.metadata["projection_id"] == "mlp.down")
    assert down.metadata["mmq_source_work"]["status"] == "mmvq_precedes_mmq"
    assert down.metadata["mmq_source_work"]["m"] == 1
    fused = next(t for t in tails if t.metadata["projection_id"] == "mlp.up_gate")
    assert fused.metadata["mmq_source_work"]["status"] == "uncovered"
    assert fused.metadata["mmq_source_work"]["reason"] == "one_physical_projection_not_proven"


@pytest.fixture
def normalized_freeze_fixture(frozen, tmp_path, monkeypatch):
    from tests.test_predict_stable_native_dataset import fixture, seal, document as save
    from tools import native_sampling_contract
    folder = tmp_path / "normalization"
    folder.mkdir()
    selection_path, selection, row, calls = fixture(folder, monkeypatch)
    row["cell_id"] = "unit"
    row["native_runtime_refs"] = frozen.module_refs
    selection["selected_cell_ids"] = ["unit"]
    # This is the production input shape: only flash_attention is present and
    # op_offload is supplied by verified host evidence, not a raw default.
    assert row["config"]["flash_attention"] is False
    assert "flash_attn" not in row["config"] and "op_offload" not in row["config"]
    save(selection_path, seal(selection))
    host = {**frozen.runtime, "cells": {"unit": {"status": "verified", "op_offload_enabled": True,
        "source_contract": frozen.runtime["contract"]}}, "evidence_refs": []}
    storage = {"contract": {"fixture": True}, "f32_hidden_storage_requested": True, "evidence_refs": []}
    monkeypatch.setattr(adapter, "verified_host_offload_source_contract", lambda *args, **kwargs: host)
    monkeypatch.setattr(adapter, "verified_tensor_storage_contract", lambda *args, **kwargs: storage)
    monkeypatch.setattr(adapter, "verify_gpu_invocation_source_links", lambda *args, **kwargs: frozen.linkage)
    monkeypatch.setattr(native_sampling_contract, "verify_sampling_contract", lambda *args, **kwargs: {**frozen.sampling, "evidence_refs": [frozen.sampling["contract_ref"]]})
    monkeypatch.setattr(adapter, "compiled_graph_evidence", lambda *args, **kwargs: {"compiled_cuda_graphs": False})
    monkeypatch.setattr(adapter, "read_retained_gguf_scope", lambda ref: {"architecture": "qwen2"})
    options = {"data_root": tmp_path, "host_offload_source_contract_path": tmp_path / "runtime.json",
        "sampling_contract_path": tmp_path / "sampling.json", "tensor_storage_contract_path": tmp_path / "storage.json",
        "tensor_storage_f32_hidden": True}
    return SimpleNamespace(path=selection_path, selection=selection, row=row, host=host,
        folder=folder, options=options, calls=calls)


def test_freeze_cell_proof_uses_actual_normalized_alias_and_verified_host_inputs(normalized_freeze_fixture):
    data = normalized_freeze_fixture
    off = adapter.freeze_selection(data.path, data.folder / "off", **data.options)
    on = adapter.freeze_selection(data.path, data.folder / "on", final_output_selection=True, **data.options)
    old, entry = off["cells"][0], on["cells"][0]
    assert old["preparation_error"] is None and entry["preparation_error"] is None
    inputs = entry["static_inputs"]
    proof = inputs[binding.INPUT_KEY]
    assert proof["config"]["flash_attn"] is False and proof["config"]["op_offload"] is True
    assert proof["config"] == {key: inputs["config"].get(key) for key in binding.CONFIG_KEYS}
    assert {key: value for key, value in inputs.items() if key not in (binding.FLAG, binding.INPUT_KEY)} == old["static_inputs"]
    assert binding.FLAG not in off and binding.INPUT_KEY not in off
    assert binding.FLAG not in old["static_inputs"] and binding.INPUT_KEY not in old["static_inputs"]
    assert "flash_attn" not in data.row["config"] and "op_offload" not in data.row["config"]
    assert binding.verify_cell(inputs) == proof
    binding.verify_freeze(on)
    assert not any(data.calls.values())  # No model building or simulation.


def test_normalized_binding_failure_remains_preparation_error(normalized_freeze_fixture, monkeypatch):
    data = normalized_freeze_fixture
    def failed_header(ref):
        raise ValueError("fixture model scope unavailable")
    monkeypatch.setattr(adapter, "read_retained_gguf_scope", failed_header)
    frozen = adapter.freeze_selection(data.path, data.folder / "failed", final_output_selection=True, **data.options)
    entry = frozen["cells"][0]
    assert entry["preparation_error"] == "ValueError: fixture model scope unavailable"
    assert entry["static_inputs"]["config"]["flash_attn"] is False
    assert entry["static_inputs"]["config"]["op_offload"] is True
    assert binding.INPUT_KEY not in entry["static_inputs"]
    assert frozen[binding.INPUT_KEY]["cells"] == {}
    binding.verify_freeze(frozen)  # Preserved failure, not campaign-wide abort.
    assert not any(data.calls.values())


def test_binding_does_not_guess_missing_normalized_flags(frozen):
    campaign = binding.freeze_binding(runtime_binding=frozen.runtime, sampling_binding=frozen.sampling,
        source_linkage=frozen.linkage)
    inputs = {key: copy.deepcopy(value) for key, value in frozen.inputs.items() if key not in (binding.FLAG, binding.INPUT_KEY)}
    inputs["config"].pop("flash_attn")
    inputs["config"].pop("op_offload")
    bound = binding.bind_static_inputs(campaign, inputs, model_scope_reader=lambda ref: {"architecture": "qwen2"})
    assert bound[binding.INPUT_KEY]["config"]["flash_attn"] is None
    assert bound[binding.INPUT_KEY]["config"]["op_offload"] is None
    assert "flash_attn" not in bound["config"] and "op_offload" not in bound["config"]
    changed = copy.deepcopy(bound)
    changed["config"]["op_offload"] = True
    with pytest.raises(ValueError, match="cell proof differs"):
        binding.verify_cell(changed)


def test_real131_normalized_static_inputs_and_final_output_qualification():
    import os
    if os.environ.get("FINAL_OUTPUT_REAL131_STATIC") != "1":
        pytest.skip("explicit real131 static-only qualification opt-in")
    main = Path(r"F:\codex_project\37_LLMsim\heterollm-sim-4.0.0")
    path = main / "artifacts/development/native_long_grid_135_20260915/optimization_loop/round_000/on/freeze.json"
    frozen_campaign = json.loads(path.read_text(encoding="utf-8-sig"))
    selection, _ = adapter.grid.read_document(frozen_campaign["selection_ref"]["path"], frozen_campaign["selection_sha256"])
    source = frozen_campaign[binding.INPUT_KEY]["source_contract"]
    assert binding.rederive_source(source) == source
    campaign = {"requested": True, "source_contract": source, "cells": {},
        "evidence_refs": source["evidence_refs"], "new_cost_coefficients": 0}
    saved = {entry["cell_id"]: entry for entry in frozen_campaign["cells"]}
    corrected = []
    normalization_mismatches = 0
    for row in adapter.selected_rows(selection):
        actual = adapter.static_inputs(row, selection, Path(frozen_campaign["data_root"]),
            model_snapshot_map=frozen_campaign["model_snapshot_map"], runtime_build_audit=frozen_campaign["runtime_build_audit"],
            recurrent_batching=frozen_campaign["recurrent_batching"], iq_panel=frozen_campaign["cpu_iq_panel_reuse"],
            slot_order=frozen_campaign["slot_order"], host_offload=frozen_campaign["host_offload_source"],
            tensor_storage=frozen_campaign["tensor_storage"], gpu_invocation=frozen_campaign["gpu_invocation"],
            sampling=frozen_campaign["sampling"], nonflash_kv_view=frozen_campaign["nonflash_kv_view"],
            mmvq_issue=frozen_campaign["mmvq_issue_bound"], retained_warmup=frozen_campaign["retained_kv_warmup"])
        adapter.configuration(actual)
        adapter.gpu_clock(actual)
        old = saved[row["cell_id"]]["static_inputs"]
        assert actual == {key: value for key, value in old.items() if key not in (binding.FLAG, binding.INPUT_KEY)}
        normalization_mismatches += old[binding.INPUT_KEY]["config"] != {key: actual["config"].get(key) for key in binding.CONFIG_KEYS}
        bound = binding.bind_static_inputs(campaign, actual, model_scope_reader=adapter.read_retained_gguf_scope)
        assert bound[binding.INPUT_KEY]["config"]["flash_attn"] is False
        assert bound[binding.INPUT_KEY]["config"]["op_offload"] is True
        assert binding.verify_cell(bound, verify_files=False) == bound[binding.INPUT_KEY]
        corrected.append({"cell_id": row["cell_id"], "static_inputs": bound, "preparation_error": None})
    assert len(corrected) == len(campaign["cells"]) == 131
    assert normalization_mismatches == 0
    assert all(entry["static_inputs"][binding.INPUT_KEY]["status"] == "conditional" for entry in corrected)
    binding.verify_freeze({binding.FLAG: True, binding.INPUT_KEY: campaign, "cells": corrected})
    print("real131:131 normalized inputs unchanged;0 baseline proof normalization mismatches;131 conditional proofs verified;no simulation/native/freeze writes")
