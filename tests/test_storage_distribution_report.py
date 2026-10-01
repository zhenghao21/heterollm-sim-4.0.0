"""Per-medium reporting must retain placement and physical traffic identities."""
from tools.qwen38_memory_scenario import load_model, build_scenario, execute_scenario


def test_real_qwen_reports_kv_weight_and_linear_state_owners_separately():
    model, _ = load_model()
    observed = execute_scenario(build_scenario(model, "hbf_active_kv"))
    stores = observed["storage_distribution"]
    assert stores["hbf0"]["kv_default_owner"]
    assert not stores["hbf0"]["linear_state_default_owner"]
    assert stores["hbm2"]["linear_state_default_owner"]
    # Read/write share the physical controller; direction is carried by the
    # transfer edge, not a fabricated legacy write-resource name.
    owner = observed["profile_owners"]["hbf0"]
    assert stores["hbf0"]["resource_bytes"][owner] > 0
    assert observed["resource_bytes"]["link.gpu-hbf0.gpu0->hbf0"] > 0
    for component, row in stores.items():
        assert row["persistent_weight_shard_bytes"] <= row["capacity_bytes"]
        for tensor in row["tensor_ids"]:
            assert observed["placement"]["tensor_to_component"][tensor] == component
    expected = sum(s["physical_bytes"] for shards in observed["placement"]["rank_weight_shards"].values() for s in shards)
    assert sum(row["persistent_weight_shard_bytes"] for row in stores.values()) == expected

def test_stale_model_sidecar_reparsed_without_overwrite(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import tools.qwen38_memory_scenario as tool
    path=tmp_path/'model.gguf';path.write_bytes(b'original model')
    sidecar=tmp_path/'model.gguf.metadata.json';sidecar.write_bytes(b'historical sidecar')
    calls=[]
    def reject(*args,**kwargs):raise tool.GGUFError('parser identity mismatch')
    gguf=SimpleNamespace(architecture='qwen35',n_embd=5120,sha256='new-hash',quantization='test',
                         tensor_directory=(),tensor_count=0,metadata={})
    model=SimpleNamespace(_execution_view=SimpleNamespace(layer_instances=tuple(
        SimpleNamespace(layer_id=str(i),layer=SimpleNamespace(sequence_mixer='full_attention' if i<16 else 'linear_attention'))
        for i in range(64))))
    monkeypatch.setattr(tool,'read_gguf_metadata_cache',reject)
    monkeypatch.setattr(tool,'read_gguf_metadata',lambda p:(calls.append(p),gguf)[1])
    monkeypatch.setattr(tool,'build_model_from_gguf',lambda g:model)
    monkeypatch.setattr(tool,'gguf_metadata_digest',lambda g:'digest')
    loaded,identity=tool.load_model(path)
    assert loaded is model and calls == [path.resolve()]
    assert identity['gguf_payload_rehashed'] and identity['metadata_source']=='fresh_gguf'
    assert sidecar.read_bytes()==b'historical sidecar'
