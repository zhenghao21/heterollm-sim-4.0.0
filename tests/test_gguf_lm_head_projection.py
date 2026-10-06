from heterollm_sim.gguf_parity import GGUFTensor, GGUFMetadata, build_model_from_gguf


def _tiny_gguf(*, tied: bool):
    tensors = []
    offset = 0

    def add(name, shape, type_name, n_bytes):
        nonlocal offset
        tensors.append(
            GGUFTensor(name, shape, 6 if type_name == "Q5_0" else 2,
                       type_name, 32, n_bytes, offset)
        )
        offset += n_bytes

    for name, shape, n_bytes in (
        ("attn_q", (32, 32), 576), ("attn_k", (32, 16), 288),
        ("attn_v", (32, 16), 288), ("attn_output", (32, 32), 576),
        ("ffn_gate", (32, 16), 288), ("ffn_up", (32, 16), 288),
        ("ffn_down", (16, 32), 576),
    ):
        add("blk.0." + name + ".weight", shape, "Q4_0", n_bytes)
    add("token_embd.weight", (32, 16), "Q5_0", 352)
    if not tied:
        add("output.weight", (32, 16), "Q5_0", 352)
    return GGUFMetadata(
        "tiny.gguf", "digest", 3, len(tensors), 0, "llama", 1, 32,
        2, 1, 16, 32, None, None, {}, tuple(tensors)
    )


def test_gguf_lm_head_keeps_independent_quantized_descriptor():
    model = build_model_from_gguf(_tiny_gguf(tied=False))
    descriptor = model.graph.attributes["metadata"]["weight_projection_descriptors"]
    segment = descriptor["projections"]["lm_head"]["segments"][0]
    assert segment["format"] == "Q5_0"
    assert (segment["k"], segment["n"]) == (32, 16)
    assert segment["physical_bytes"] == 352


def test_gguf_tied_embedding_descriptor_preserves_physical_geometry():
    model = build_model_from_gguf(_tiny_gguf(tied=True))
    segment = model.graph.attributes["metadata"]["weight_projection_descriptors"][
        "projections"
    ]["lm_head"]["segments"][0]
    assert segment["physical_tensor_name"] == "token_embd.weight"
    assert (segment["k"], segment["n"]) == (32, 16)
